"""`Dataset` metadata and prepared-data reads.

A `Dataset` handle (see `src/mojoboost/trainset.mojo`) owns a binned
feature matrix, the fitted `BinMapper` that produced it, and the columns
that describe its rows. `bindings/_mojoboost.mojo` already exposes its
shape (`dataset_num_data`, `dataset_num_feature`, `dataset_num_bin`); this
module exposes the rest, so the Python `Dataset` can answer from the
native object it constructed rather than only from what the caller handed
it.

Why that matters: `python/mojoboost/basic.py` keeps its own copy of the
label, the weights, the groups, the init scores, the names, and the
categorical declaration, and answers `get_field` from those. That is
correct for a dataset built in this process and cannot be right for one
that was not, and it says nothing about what the *binning* did with them.
The reads here come off the constructed dataset.

Nothing here bins, trains, or transforms. Every function reads state the
`Dataset` already holds, and every buffer address travels inward: a
column leaves as a Python list, or is written into a buffer the caller
preallocated and sized.
"""

from std.python import Python, PythonObject

from binding_support import (
    index_within,
    py_dict,
    py_f64_list,
    py_int_list,
    py_str_list,
    write_f64_buffer,
)

from mojoboost.efb import feature_bin_count
from mojoboost.trainset import Dataset


def _field_values(d: Dataset, field: String) raises -> List[Float64]:
    """One of the dataset's optional columns as float64.

    `group` holds per-query row counts, which are integers; they cross as
    float64 like every other integer column at this boundary (see
    `_int_list_from_f64` in `_mojoboost.mojo`), so one accessor serves all
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


def dataset_metadata(dataset: PythonObject) raises -> PythonObject:
    """Everything about the constructed dataset that is a scalar, in one
    call.

    Keys: `num_data`, `num_feature`, `num_bin` (the effective `max_bin`
    the binning reserved), `max_bin` (the one that was requested),
    `use_missing`, `n_groups`, `n_categorical`, `has_names`, and the
    lengths of the four optional columns (`n_label`, `n_weight`,
    `n_init_score`).

    One call rather than nine, because the Python `Dataset` builds its
    repr and its `get_field` answers from all of them at once.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    var out = py_dict()
    out["num_data"] = PythonObject(d[].num_data())
    out["num_feature"] = PythonObject(d[].num_feature())
    out["num_bin"] = PythonObject(d[].num_bin())
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
