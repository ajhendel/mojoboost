"""CPython extension module for mojotrees.

Built with `bindings/build.sh` into `python/mojotrees/_mojotrees.so`; the
public Python surface is the sklearn-style wrapper in `python/mojotrees/`.

Data crosses the boundary as raw buffer addresses (integers) plus lengths:
the wrapper passes float64-contiguous buffers (column-major for feature
matrices) and keeps them alive for the duration of each call. Copies into
Mojo Lists happen here, so no Python buffer is retained after a call
returns. Trained models are returned as opaque handles owned by Python.

Prediction comes in two shapes. The row-at-a-time entry points
(`predict`, `predict_range`, `predict_leaf`, and their multiclass and
sparse siblings) walk the trees on the host and are unchanged. The `_batch`
entry points below them take a device, hand the whole matrix to
`Model.predict_batch`, and reach the device walk in gpu_predict.mojo; they
report which backend ran rather than leaving a caller to assume. Where a
prediction runs is still decided in one place, `resolve_device` in
device.mojo, and nothing here decides it.
"""

from std.memory import unsafe_memcpy
from std.os import abort
from std.sys import has_accelerator
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

# `Dataset` construction beyond the dense case, and the reads that answer
# from a constructed one. They live in their own module because they are a
# coherent group and this file is long; they are registered here, in the one
# `PythonModuleBuilder`, because that is the only place a name becomes
# reachable from Python.
from dataset_bindings import (
    dataset_bin_upper_bounds,
    dataset_categorical_features,
    dataset_copy_field,
    dataset_create_csc,
    dataset_create_reference,
    dataset_feature_names,
    dataset_feature_num_bin,
    dataset_field,
    dataset_field_length,
    dataset_metadata,
    dataset_missing_bins,
    dataset_subset,
)

# The rest of the capability modules, on the same terms: each owns one
# subject, none of them decides anything, and all of them become reachable
# only here. `bindings/build.sh` puts this directory on the import path
# (`-I bindings`), which is what makes these top-level imports resolve.
from basic_bindings import (
    decide_device_workload,
    efb_check,
    efb_defaults,
    efb_settings_from_mapping,
    extra_option_supported,
    extra_params_check,
    extra_params_from_mapping,
    forced_splits_check,
    native_clock_ns,
    startup_environment,
    startup_phase_contract,
)
from distributed_bindings import (
    distributed_capability,
    distributed_check_machine_list,
    distributed_status_message,
    transport_status_message,
)
from inspection_bindings import (
    dump_leaf_index,
    dump_leaf_index_multiclass,
    dump_model,
    dump_model_json,
    dump_model_json_multiclass,
    dump_model_multiclass,
    dump_raw_scores,
    dump_raw_scores_multiclass,
    model_file_kind,
    model_format_versions,
    objective_code,
    split_values,
    split_values_multiclass,
)
from objective_bindings import (
    check_objective_param,
    metric_code_of_name,
    objective_code_of_name,
    objective_name_status_of,
    registry_metric_aliases,
    registry_metrics,
    registry_objective_aliases,
    registry_objective_unimplemented,
    registry_objectives,
    registry_vocabulary,
)

from mojotrees.bagging import BaggingParams
from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.contrib import ContribExplainer
from mojotrees.boosting import (
    BoosterParams,
    IterationRange,
    _softmax_inplace,
)
from mojotrees.callback import (
    BEFORE_ITERATION,
    CONTINUE,
    IterationEnv,
)
from mojotrees.custom_metric import (
    CustomMetric,
    MetricHistory,
    MetricSuite,
    RawValidSet,
    fit_multiclass_with_metrics as mojo_fit_multiclass_with_metrics,
    fit_ranker_with_metrics as mojo_fit_ranker_with_metrics,
    fit_with_callbacks as mojo_fit_with_callbacks,
    fit_with_metrics as mojo_fit_with_metrics,
    response_scale,
)
from mojotrees.metrics import (
    average_precision,
    binary_auc,
    binary_error,
    binary_log_loss,
    cross_entropy_loss,
    fair_loss,
    gamma_deviance,
    gamma_loss,
    huber_loss,
    kullback_leibler,
    l1,
    l2,
    mape,
    multiclass_error,
    multiclass_log_loss,
    poisson_loss,
    quantile_loss,
    rmse,
    tweedie_loss,
)
from mojotrees.trainset import (
    Dataset,
    train_dataset as mojo_train_dataset,
    train_dataset_multiclass as mojo_train_dataset_multiclass,
    train_dataset_ranker as mojo_train_dataset_ranker,
    update_dataset as mojo_update_dataset,
    update_dataset_multiclass as mojo_update_dataset_multiclass,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.device import (
    CPU_DEVICE,
    GPU_DEVICE,
    device_name as mojo_device_name,
    gpu_available as mojo_gpu_available,
    parse_device,
    resolve_device as mojo_resolve_device,
)
from mojotrees.efb import check_bundling_honored, check_bundling_supported
from mojotrees.gpu_predict import (
    RESPONSE_SOFTMAX,
    GpuPredictor,
    accumulate_booster_rounds,
    accumulate_multiclass_rounds,
    device_metric_code,
    device_metric_matches_host,
    gpu_predict_support,
    leaf_indices_gpu,
    leaf_indices_multiclass_gpu,
    response_for_objective,
    validation_host_metric,
)
from mojotrees.goss import GossParams
from mojotrees.importance import gain_importance, split_importance
from mojotrees.interaction import InteractionConstraints
from mojotrees.monotone import MonotoneConstraints
from mojotrees.model import Model, MulticlassModel
from mojotrees.model import fit as mojo_fit
from mojotrees.model_sparse import fit_csc as mojo_fit_csc
from mojotrees.model_sparse import (
    fit_multiclass_csc as mojo_fit_multiclass_csc,
)
from mojotrees.model_sparse import (
    predict_csr as mojo_predict_csr,
    predict_proba_csr as mojo_predict_proba_csr,
    predict_raw_csr as mojo_predict_raw_csr,
)
from mojotrees.sparse import CscMatrix, CsrMatrix
from mojotrees.model import fit_custom as mojo_fit_custom
from mojotrees.model import fit_multiclass as mojo_fit_multiclass
from mojotrees.objective import mean_label
from mojotrees.ranking import (
    RankerParams,
    fit_ranker as mojo_fit_ranker,
    groups_from_counts,
    mean_average_precision as mojo_map,
    ndcg as mojo_ndcg,
)
from mojotrees.serialize import (
    file_kind as mojo_file_kind,
    load_dataset as mojo_load_dataset,
    load_feature_names,
    load_model,
    load_multiclass_model,
    save_dataset as mojo_save_dataset,
    save_model,
    save_multiclass_model,
)
from mojotrees.growth_policy import parse_grow_policy
from mojotrees.tree import Tree, TreeParams


@export
def PyInit__mojotrees() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojotrees")
        _ = m.add_type[Model]("Model")
        _ = m.add_type[MulticlassModel]("MulticlassModel")
        _ = m.add_type[Dataset]("Dataset")
        _ = m.add_type[GpuValidation]("GpuValidation")
        m.def_function[dataset_create]("dataset_create")
        m.def_function[dataset_num_data]("dataset_num_data")
        m.def_function[dataset_num_feature]("dataset_num_feature")
        m.def_function[dataset_num_bin]("dataset_num_bin")
        # From dataset_bindings.mojo: the constructors the dense one has no
        # room for, and the reads that answer from the constructed object.
        m.def_function[dataset_create_csc]("dataset_create_csc")
        m.def_function[dataset_create_reference]("dataset_create_reference")
        m.def_function[dataset_subset]("dataset_subset")
        m.def_function[dataset_metadata]("dataset_metadata")
        m.def_function[dataset_feature_names]("dataset_feature_names")
        m.def_function[dataset_categorical_features](
            "dataset_categorical_features"
        )
        m.def_function[dataset_field]("dataset_field")
        m.def_function[dataset_field_length]("dataset_field_length")
        m.def_function[dataset_copy_field]("dataset_copy_field")
        m.def_function[dataset_feature_num_bin]("dataset_feature_num_bin")
        m.def_function[dataset_bin_upper_bounds]("dataset_bin_upper_bounds")
        m.def_function[dataset_missing_bins]("dataset_missing_bins")
        m.def_function[dataset_save]("dataset_save")
        m.def_function[dataset_load]("dataset_load")
        m.def_function[file_kind]("file_kind")
        m.def_function[train_dataset]("train_dataset")
        m.def_function[train_dataset_multiclass]("train_dataset_multiclass")
        m.def_function[train_dataset_ranker]("train_dataset_ranker")
        m.def_function[booster_update]("booster_update")
        m.def_function[booster_update_multiclass](
            "booster_update_multiclass"
        )
        m.def_function[copy_model]("copy_model")
        m.def_function[copy_multiclass_model]("copy_multiclass_model")
        m.def_function[fit]("fit")
        m.def_function[fit_csc]("fit_csc")
        m.def_function[fit_custom]("fit_custom")
        m.def_function[fit_with_metrics]("fit_with_metrics")
        m.def_function[fit_multiclass_with_metrics](
            "fit_multiclass_with_metrics"
        )
        m.def_function[fit_ranker_with_metrics]("fit_ranker_with_metrics")
        m.def_function[eval_metric]("eval_metric")
        m.def_function[fit_multiclass]("fit_multiclass")
        m.def_function[fit_multiclass_csc]("fit_multiclass_csc")
        m.def_function[fit_ranker]("fit_ranker")
        m.def_function[ndcg]("ndcg")
        m.def_function[predict_csr]("predict_csr")
        m.def_function[predict_raw_csr]("predict_raw_csr")
        m.def_function[predict_proba_csr]("predict_proba_csr")
        m.def_function[predict_range]("predict_range")
        m.def_function[predict_proba_range]("predict_proba_range")
        m.def_function[predict_leaf]("predict_leaf")
        m.def_function[predict_leaf_multiclass]("predict_leaf_multiclass")
        m.def_function[predict_batch]("predict_batch")
        m.def_function[predict_proba_batch]("predict_proba_batch")
        m.def_function[predict_leaf_batch]("predict_leaf_batch")
        m.def_function[predict_leaf_multiclass_batch](
            "predict_leaf_multiclass_batch"
        )
        m.def_function[gpu_predict_capability]("gpu_predict_capability")
        # Registering a function specializes its body.  On a CPU-only build,
        # specializing the real resident-validation functions reaches
        # `GpuPredictor` and asks the compiler for a GPU architecture that the
        # runner does not have.  Keep the Python surface present, but bind
        # stubs which fail explicitly without mentioning a device type.
        comptime if has_accelerator():
            m.def_function[gpu_validation_open]("gpu_validation_open")
            m.def_function[gpu_validation_open_multiclass](
                "gpu_validation_open_multiclass"
            )
            m.def_function[gpu_validation_shape]("gpu_validation_shape")
            m.def_function[gpu_validation_reset]("gpu_validation_reset")
            m.def_function[gpu_validation_accumulate](
                "gpu_validation_accumulate"
            )
            m.def_function[gpu_validation_accumulate_multiclass](
                "gpu_validation_accumulate_multiclass"
            )
            m.def_function[gpu_validation_metric]("gpu_validation_metric")
            m.def_function[gpu_validation_raw]("gpu_validation_raw")
        else:
            m.def_function[_gpu_validation_open_unavailable](
                "gpu_validation_open"
            )
            m.def_function[_gpu_validation_open_unavailable](
                "gpu_validation_open_multiclass"
            )
            m.def_function[_gpu_validation_shape_unavailable](
                "gpu_validation_shape"
            )
            m.def_function[_gpu_validation_reset_unavailable](
                "gpu_validation_reset"
            )
            m.def_function[_gpu_validation_accumulate_unavailable](
                "gpu_validation_accumulate"
            )
            m.def_function[_gpu_validation_accumulate_unavailable](
                "gpu_validation_accumulate_multiclass"
            )
            m.def_function[_gpu_validation_metric_unavailable](
                "gpu_validation_metric"
            )
            m.def_function[_gpu_validation_raw_unavailable](
                "gpu_validation_raw"
            )
        m.def_function[gpu_validation_metric_matches_host](
            "gpu_validation_metric_matches_host"
        )
        m.def_function[predict_contrib]("predict_contrib")
        m.def_function[predict_contrib_multiclass](
            "predict_contrib_multiclass"
        )
        m.def_function[num_trees]("num_trees")
        m.def_function[num_iterations]("num_iterations")
        m.def_function[num_iterations_multiclass]("num_iterations_multiclass")
        m.def_function[n_classes]("n_classes")
        m.def_function[n_features]("n_features")
        m.def_function[n_features_multiclass]("n_features_multiclass")
        m.def_function[categorical_features]("categorical_features")
        m.def_function[categorical_features_multiclass](
            "categorical_features_multiclass"
        )
        m.def_function[feature_importance]("feature_importance")
        m.def_function[feature_importance_multiclass](
            "feature_importance_multiclass"
        )
        m.def_function[save]("save")
        m.def_function[load]("load")
        m.def_function[save_multiclass]("save_multiclass")
        m.def_function[load_multiclass]("load_multiclass")
        m.def_function[model_feature_names]("model_feature_names")
        m.def_function[gpu_available]("gpu_available")
        m.def_function[resolve_device]("resolve_device")
        # The whole device decision, not just the backend name. One entry
        # point and one only: `resolve_device` above answers the shape-only
        # question and stays, and `device_selection.py` prefers this when it
        # is present. The workload crosses as a mapping rather than as ten
        # positional arguments, which is the shape `_parse_params` already
        # uses and which needs no bet on how many arguments `def_function`
        # accepts (eight are proven by `predict_range`; ten were never
        # tried). See handoffs/connect_14_bindings.md section 6.1(c).
        m.def_function[decide_device_workload]("decide_device")
        # -- structured inspection (migration_19_model_inspection.md) ----
        m.def_function[dump_model]("dump_model")
        m.def_function[dump_model_multiclass]("dump_model_multiclass")
        m.def_function[split_values]("split_values")
        m.def_function[split_values_multiclass]("split_values_multiclass")
        m.def_function[dump_raw_scores]("dump_raw_scores")
        m.def_function[dump_raw_scores_multiclass](
            "dump_raw_scores_multiclass"
        )
        m.def_function[dump_leaf_index]("dump_leaf_index")
        m.def_function[dump_leaf_index_multiclass](
            "dump_leaf_index_multiclass"
        )
        m.def_function[dump_model_json]("dump_model_json")
        m.def_function[dump_model_json_multiclass](
            "dump_model_json_multiclass"
        )
        # Takes a model handle and answers what it was trained for. The
        # name-to-code resolver is `objective_code_of_name` below; they are
        # two questions and only one of them may hold this name.
        m.def_function[objective_code]("objective_code")
        m.def_function[model_file_kind]("model_file_kind")
        m.def_function[model_format_versions]("model_format_versions")
        # -- objective and metric registry -------------------------------
        m.def_function[registry_objectives]("registry_objectives")
        m.def_function[registry_objective_aliases](
            "registry_objective_aliases"
        )
        m.def_function[registry_objective_unimplemented](
            "registry_objective_unimplemented"
        )
        m.def_function[registry_metrics]("registry_metrics")
        m.def_function[registry_metric_aliases]("registry_metric_aliases")
        m.def_function[registry_vocabulary]("registry_vocabulary")
        m.def_function[objective_code_of_name]("objective_code_of_name")
        m.def_function[metric_code_of_name]("metric_code_of_name")
        m.def_function[objective_name_status_of]("objective_name_status")
        m.def_function[check_objective_param]("check_objective_param")
        # -- run configuration -------------------------------------------
        m.def_function[extra_params_check]("extra_params_check")
        m.def_function[extra_option_supported]("extra_option_supported")
        m.def_function[forced_splits_check]("forced_splits_check")
        m.def_function[efb_check]("efb_check")
        m.def_function[efb_defaults]("efb_defaults")
        # -- distributed runtime -----------------------------------------
        m.def_function[distributed_capability]("distributed_capability")
        m.def_function[distributed_check_machine_list](
            "distributed_check_machine_list"
        )
        m.def_function[distributed_status_message](
            "distributed_status_message"
        )
        m.def_function[transport_status_message]("transport_status_message")
        # -- startup diagnostics -----------------------------------------
        m.def_function[startup_phase_contract]("startup_phase_contract")
        m.def_function[startup_environment]("startup_environment")
        m.def_function[native_clock_ns]("native_clock_ns")
        return m.finalize()
    except e:
        abort(String("failed to create _mojotrees module: ", e))


def _f64_list(addr: Int, n: Int) raises -> List[Float64]:
    """Copy a float64 buffer (NumPy's X, y, weights) into a Mojo list.

    One bulk copy, not an element-by-element append. The caller's buffer and
    the list hold the same bytes in the same order, so there is nothing per
    element to decide: `unsafe_uninit_length` skips the zero fill that
    `resize` would do, and the copy that follows writes every one of those
    bytes. Measured over 25 million elements, best of three alternating runs
    in one process: 0.0061 s appending, 0.0043 s copying.

    That is a small share of an ingest, and worth saying so here: for a
    C-ordered 250,000 x 100 array the NumPy-side `asfortranarray` transpose
    costs about 0.062 s, ten times this. The transpose is the price of a
    column-major layout and it belongs on the NumPy side, which does it
    blocked; handing the trainer a row-major buffer instead would only move
    the same work into the binner's per-column gather, strided and cold.
    """
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    if n == 0:
        return List[Float64]()
    var out = List[Float64](unsafe_uninit_length=n)
    unsafe_memcpy(dest=out.unsafe_ptr(), src=p, count=n)
    return out^


def _f64_view(addr: Int, n: Int) raises -> Span[Float64, ImmUntrackedOrigin]:
    """Borrow a float64 buffer (NumPy's X) instead of copying it.

    The feature matrix is the one input where the copy is worth avoiding,
    and not for the reason it looks like: `_f64_list` moves it at memory
    speed, a low single-digit percentage of an ingest. What the copy costs
    is *space*. It doubles the resident footprint of the matrix for as long
    as binning runs, so a 5,000,000 x 100 fit holds 4 GB of NumPy plus 4 GB
    of Mojo, and on a machine that can afford one of those but not both the
    difference is not a percentage.

    Borrowing is sound here because the matrix is read, never written, and
    is dead early: `fit_bins` and `BinMapper.transform` are the only things
    that look at it, and after transform the trainer works on the binned
    `UInt8` matrix. It stays alive throughout because the Python wrapper
    holds the array it took the address of (see `_arrays.column_major`,
    whose contract is exactly that) for the whole call.

    The origin is untracked because the owner is on the other side of the
    boundary and Mojo cannot see it. That is the same contract `_f64_list`
    already relies on for its source pointer; the difference is only how
    long it has to hold, which is the length of one call either way.

    Not every input can do this. A buffer that outlives the call must be
    copied, so `dataset_create` still takes `_f64_list` -- a `Dataset` keeps
    its matrix -- and so do the validation sets, which `RawValidSet` owns.
    """
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=addr)
    return Span[Float64, ImmUntrackedOrigin](unsafe_ptr=p, length=n)


def _int_list_from_f64(addr: Int, n: Int) raises -> List[Int]:
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(p.unsafe_load(i)))
    return out^


def _int_list(addr: Int, n: Int) raises -> List[Int]:
    """Copy an int64 buffer (SciPy's indices/indptr) into a Mojo list.

    Same bulk copy as `_f64_list`, for the same reason: int64 in, Int out,
    identical bytes.
    """
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=addr)
    if n == 0:
        return List[Int]()
    var out = List[Int](unsafe_uninit_length=n)
    unsafe_memcpy(dest=out.unsafe_ptr(), src=p, count=n)
    return out^


def _sparse_shape(params: PythonObject) raises -> List[Int]:
    """(n_rows, n_features, nnz) from the sparse keys of a params dict."""
    return [
        Int(py=params["n_rows"]),
        Int(py=params["n_features"]),
        Int(py=params["sparse_nnz"]),
    ]


def _csc(params: PythonObject) raises -> CscMatrix:
    """Rebuild a CSC matrix from the wrapper's buffer addresses. SciPy's
    arrays are normalized to float64 data and int64 indices on the Python
    side; the matrix itself is validated by the sparse binner."""
    var shape = _sparse_shape(params)
    return CscMatrix(
        _int_list(Int(py=params["sparse_indices_addr"]), shape[2]),
        _f64_list(Int(py=params["sparse_data_addr"]), shape[2]),
        _int_list(Int(py=params["sparse_indptr_addr"]), shape[1] + 1),
        shape[0],
        shape[1],
    )


def _csr(params: PythonObject) raises -> CsrMatrix:
    """Rebuild a CSR matrix from the wrapper's buffer addresses."""
    var shape = _sparse_shape(params)
    return CsrMatrix(
        _int_list(Int(py=params["sparse_indices_addr"]), shape[2]),
        _f64_list(Int(py=params["sparse_data_addr"]), shape[2]),
        _int_list(Int(py=params["sparse_indptr_addr"]), shape[0] + 1),
        shape[0],
        shape[1],
    )


def _row[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    r: Int,
) -> List[Float64]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def _parse_constraints(
    params: PythonObject, n_features: Int
) raises -> InteractionConstraints:
    """Interaction constraint groups from the params dict, flattened by the
    wrapper into two float64 buffers (group features, and one more offset
    than there are groups). A zero address means unconstrained. The wrapper
    validates too; this is the check that also covers direct callers."""
    var flat_addr = Int(py=params["interaction_flat_addr"])
    if flat_addr == 0:
        return InteractionConstraints()
    var flat = _int_list_from_f64(
        flat_addr, Int(py=params["interaction_flat_len"])
    )
    var offsets = _int_list_from_f64(
        Int(py=params["interaction_offsets_addr"]),
        Int(py=params["interaction_offsets_len"]),
    )
    return InteractionConstraints.from_flat(flat, offsets, n_features)


def _parse_monotone(
    params: PythonObject, n_features: Int
) raises -> MonotoneConstraints:
    """Monotonic constraints from the params dict: one float64 entry per
    feature (-1, 0, or 1) at `monotone_addr`, or a zero address for
    unconstrained. The wrapper rejects fractional entries before they get
    here, where the buffer is read as integers; the length and range checks
    below also cover direct callers."""
    var addr = Int(py=params["monotone_addr"])
    if addr == 0:
        return MonotoneConstraints()
    return MonotoneConstraints.from_signs(
        _int_list_from_f64(addr, n_features), n_features
    )


def _parse_params(
    params: PythonObject,
    n_features: Int,
    unbundled: String = "",
    cpu: Bool = True,
) raises -> BoosterParams:
    """The `BoosterParams` a fit runs under, from the params mapping.

    `unbundled` and `cpu` are how an entry point declares what it is about
    to call, because `BoosterParams.bundling` is only applied by some
    trainers (see efb.mojo) and this is the last place that knows which one
    is next. `unbundled` names the entry point when its trainer does not
    apply a bundling plan at all -- the custom-objective, custom-metric,
    and ranking trainers -- and an active switch is then refused by name
    rather than dropped. Leave it empty when the trainer does apply one,
    and pass `cpu=False` for a run that resolved to the GPU, which is the
    same device check `params.mojo` makes for a parameter string. The
    default is the CPU-honoring case because every entry point that leaves
    both alone (the sparse fits and continued training) is CPU-only by
    construction.
    """
    var tree = TreeParams(
        Int(py=params["num_leaves"]),
        Int(py=params["min_data_in_leaf"]),
        Float64(py=params["lambda_l2"]),
        Float64(py=params["min_child_hess"]),
        Float64(py=params["lambda_l1"]),
        _parse_constraints(params, n_features),
        feature_fraction=Float64(py=params["feature_fraction"]),
        feature_fraction_bynode=Float64(
            py=params["feature_fraction_bynode"]
        ),
        feature_fraction_seed=Int(py=params["feature_fraction_seed"]),
        max_depth=Int(py=params["max_depth"]),
        monotone=_parse_monotone(params, n_features),
        cat=_parse_cat_params(params),
        # The remaining LightGBM tree controls
        # (src/mojotrees/tree_parameters_extra.mojo). Until this was passed,
        # the bundle took its inactive default on every fit that came through
        # Python, so `min_gain_to_split`, `max_delta_step`, `path_smooth`,
        # `extra_trees`, `monotone_penalty`, the per-feature gain multipliers
        # and split costs, and forced splits were reachable from the C ABI and
        # the CLI (which parse a text spec through params.mojo) and from
        # nowhere else. `extra_params_from_mapping` is the same parser
        # `extra_params_check` validates with, so what is checked and what is
        # trained cannot come apart.
        extra=extra_params_from_mapping(params, n_features),
        # XGBoost's grow_policy (src/mojotrees/growth_policy.mojo); the
        # wrapper sends the name, and the same parser the parameter string
        # goes through resolves it, so the two front doors cannot disagree.
        grow_policy=parse_grow_policy(String(py=params["grow_policy"])),
    )
    # Exclusive feature bundling (src/mojotrees/efb.mojo). Until this was
    # passed, `BoosterParams.bundling` took its disabled default on every fit
    # that came through Python, so `enable_bundle` and the knobs it governs
    # were reachable from the CLI and the C ABI (params.mojo parses
    # `enable_bundle` and `max_conflict_rate`) and from nowhere else.
    # `efb_settings_from_mapping` is the same parser `efb_check` validates
    # with, so what is checked and what is trained cannot come apart.
    var bundling = efb_settings_from_mapping(params)
    # Reachability first, then ranges: the order params.mojo checks a
    # parameter string in. The ranges run whether or not the switch is on.
    if unbundled.byte_length() > 0:
        check_bundling_honored(bundling, unbundled)
    else:
        check_bundling_supported(bundling.enabled, cpu)
    bundling.check()
    return BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
        bundling^,
    )


def _parse_device(params: PythonObject) raises -> Int:
    """Device code from the params dict. The wrapper sends the name it
    already resolved ("cpu" or "gpu"); the trainer resolves it again, so
    the policy in device.mojo stays the only one."""
    return parse_device(String(py=params["device"]))


def _parse_bagging(params: PythonObject) raises -> BaggingParams:
    """Row bagging config from the params dict. The trainer validates it
    again, so the rules in bagging.mojo stay the only ones."""
    return BaggingParams(
        Float64(py=params["bagging_fraction"]),
        Int(py=params["bagging_freq"]),
        Int(py=params["bagging_seed"]),
    )


def _parse_goss(params: PythonObject) raises -> GossParams:
    """GOSS config from the params dict. `goss` arrives as an int so the
    boundary carries no Python bool conversion. The trainer validates the
    rates again, so the rules in goss.mojo stay the only ones."""
    return GossParams(
        Int(py=params["goss"]) != 0,
        Float64(py=params["top_rate"]),
        Float64(py=params["other_rate"]),
        Int(py=params["goss_seed"]),
        Int(py=params["goss_warmup_rounds"]),
    )


def _parse_categorical(params: PythonObject) raises -> List[Int]:
    """Categorical feature indices from the params dict, as one float64
    entry per index at `categorical_addr`, or a zero address for none. The
    binner validates the indices again, so the rules in categorical.mojo
    stay the only ones."""
    var addr = Int(py=params["categorical_addr"])
    if addr == 0:
        return List[Int]()
    return _int_list_from_f64(addr, Int(py=params["categorical_len"]))


def _parse_cat_params(params: PythonObject) raises -> CategoricalParams:
    """LightGBM's categorical hyperparameters from the params dict. The
    search validates nothing further; the rules in categorical.mojo stay the
    only ones."""
    return CategoricalParams(
        Int(py=params["max_cat_to_onehot"]),
        Int(py=params["max_cat_threshold"]),
        Float64(py=params["cat_smooth"]),
        Float64(py=params["cat_l2"]),
        Int(py=params["min_data_per_group"]),
    )


def _parse_use_missing(params: PythonObject) raises -> Bool:
    """LightGBM's use_missing, passed as an int so the boundary carries no
    Python bool conversion. The binner validates nothing further; the rules
    in binning.mojo stay the only ones."""
    return Int(py=params["use_missing"]) != 0


def _parse_weights(params: PythonObject, n_rows: Int) raises -> List[Float64]:
    var weight_addr = Int(py=params["sample_weight_addr"])
    if weight_addr == 0:
        return List[Float64]()
    return _f64_list(weight_addr, n_rows)


def fit(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    objective: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a single-output model. Buffers are float64; X is column-major."""
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    # The device is read before the parameters because bundling is applied
    # by the dense CPU trainer and not by the GPU one, so `_parse_params`
    # has to know which of the two `mojo_fit` will dispatch to. The wrapper
    # sends a device it has already resolved, so this is the backend.
    var device = _parse_device(params)
    var bp = _parse_params(params, nf, cpu=device == CPU_DEVICE)
    var weights = _parse_weights(params, nr)
    var model = mojo_fit(
        features,
        nr,
        nf,
        target,
        Int(py=objective),
        bp,
        Int(py=params["max_bin"]),
        weights,
        Float64(py=params["alpha"]),
        device,
        _parse_bagging(params),
        _parse_goss(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )
    return PythonObject(alloc=model^)


def fit_custom(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    bridge: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a single-output model against a Python objective callback.

    `bridge` is a zero-argument Python callable. Per boosting round this
    writes the current raw scores into the caller's `raw_addr` buffer, calls
    `bridge` once, and reads the gradients and hessians back out of the
    caller's `grad_addr` and `hess_addr` buffers. The Python side therefore
    sees whole arrays, once per round: no Python object crosses the boundary
    per row, and nothing Python-side runs inside tree growth. Three float64
    buffers of length n_rows must be alive at those addresses for the whole
    call. The remaining contract (validation, weights, base score, raw-score
    predictions) is the one in src/mojotrees/objective.mojo.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf, unbundled="fit_custom")
    var weights = _parse_weights(params, nr)

    var raw_addr = Int(py=params["raw_addr"])
    var grad_addr = Int(py=params["grad_addr"])
    var hess_addr = Int(py=params["hess_addr"])
    if raw_addr == 0 or grad_addr == 0 or hess_addr == 0:
        raise Error("invalid buffer")
    var raw_p = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=raw_addr
    )
    var grad_p = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=grad_addr
    )
    var hess_p = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=hess_addr
    )

    def py_grad_hess(
        raw: List[Float64],
        labels: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises {imm bridge, imm raw_p, imm grad_p, imm hess_p, imm nr}:
        for r in range(nr):
            raw_p.unsafe_store(r, raw[r])
        _ = bridge()
        grad.clear()
        hess.clear()
        for r in range(nr):
            grad.append(grad_p.unsafe_load(r))
            hess.append(hess_p.unsafe_load(r))

    # "mean" is resolved here rather than in the wrapper: the label mean has
    # to match the built-in objectives' base score bit for bit, so it has to
    # come from one summation order, and this is that one.
    var base_score = Float64(py=params["base_score"])
    if Int(py=params["base_score_mean"]) != 0:
        base_score = mean_label(target, weights)

    var model = mojo_fit_custom(
        features,
        nr,
        nf,
        target,
        py_grad_hess,
        bp,
        Int(py=params["max_bin"]),
        weights,
        base_score,
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )
    return PythonObject(alloc=model^)


def fit_with_metrics(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    objective: PythonObject,
    bridge: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a built-in objective while Python callbacks score validation
    sets (see src/mojotrees/custom_metric.mojo for the metric contract).

    `bridge(metric_index, valid_index) -> float` is called once per metric
    per validation set per round. Before each call the current raw
    validation predictions, which is what LightGBM's `feval` also receives,
    are written into the caller's `pred_addr` buffer; that buffer must be
    float64, alive for the whole call, and at least as long as the largest
    validation set. The Python side therefore sees whole arrays, once per
    metric per round: no Python object crosses the boundary per row.

    `params` additionally holds `valid_sets`, a sequence of
    `(name, x_addr, n_rows, y_addr)`, and `metrics`, a sequence of
    `(name, higher_is_better, use_for_early_stopping)` with the flags as
    ints. Returns
    `[model, values, n_rounds, best_iteration, best_score, stopped_early]`,
    where `values` is the flattened history, round-major, then validation
    set, then metric.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf, unbundled="fit_with_metrics")
    var weights = _parse_weights(params, nr)

    var pred_p = _pred_pointer(params)
    var valid_sets = _parse_valid_sets(params, nf)
    var metrics = _parse_metrics(params)

    def py_metric(
        metric: Int,
        valid: Int,
        pred: List[Float64],
        labels: List[Float64],
    ) raises {imm bridge, imm pred_p} -> Float64:
        for r in range(len(pred)):
            pred_p.unsafe_store(r, pred[r])
        return Float64(py=bridge(PythonObject(metric), PythonObject(valid)))

    var callback = params["callback"]
    var has_callback = Int(py=params["has_callback"]) != 0
    var reset_p = _reset_pointer(params)
    var evals_p = _evals_pointer(params)

    def py_callback(phase: Int, mut env: IterationEnv) raises {
        imm callback, imm has_callback, imm reset_p, imm evals_p
    } -> Int:
        # A run without callbacks must not pay for the boundary, and must
        # take the same path as `train_with_metrics` did before: no crossing,
        # no buffer traffic, no parameter round trip.
        if not has_callback:
            return CONTINUE
        if phase == BEFORE_ITERATION:
            _write_reset(reset_p, env.params)
        else:
            for i in range(len(env.evaluation)):
                evals_p.unsafe_store(i, env.evaluation[i])
        var code = Int(
            py=callback(PythonObject(phase), PythonObject(env.iteration))
        )
        if phase == BEFORE_ITERATION and code == CONTINUE:
            _read_reset(reset_p, env.params)
        return code

    var result = mojo_fit_with_callbacks(
        features,
        nr,
        nf,
        target,
        valid_sets^,
        Int(py=objective),
        bp,
        MetricSuite(
            metrics^, py_metric, Int(py=params["primary_metric"])
        ),
        py_callback,
        Int(py=params["early_stopping_rounds"]),
        Float64(py=params["min_delta"]),
        Int(py=params["max_bin"]),
        weights,
        Float64(py=params["alpha"]),
        _parse_bagging(params),
        _parse_goss(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )

    var model = result.model.copy()
    return _metric_output(
        PythonObject(alloc=model^),
        result.history,
        result.best_iteration,
        result.best_score,
        result.stopped_early,
    )


comptime RESET_SLOTS = 9
"""Length of the `reset_addr` buffer. The slot order below is the contract
the Python side mirrors in `_RESET_SLOTS`; changing either alone silently
reassigns hyperparameters, so change both together."""


def _write_reset(
    p: Pointer[Float64, MutUntrackedOrigin], params: BoosterParams
):
    """Publish the round's resettable hyperparameters for a before-iteration
    callback to read and, if it wants a schedule, overwrite."""
    p.unsafe_store(0, params.learning_rate)
    p.unsafe_store(1, Float64(params.tree.num_leaves))
    p.unsafe_store(2, Float64(params.tree.max_depth))
    p.unsafe_store(3, Float64(params.tree.min_data_in_leaf))
    p.unsafe_store(4, params.tree.min_child_hess)
    p.unsafe_store(5, params.tree.lambda_l1)
    p.unsafe_store(6, params.tree.lambda_reg)
    p.unsafe_store(7, params.tree.feature_fraction)
    p.unsafe_store(8, params.tree.feature_fraction_bynode)


def _read_reset(
    p: Pointer[Float64, MutUntrackedOrigin], mut params: BoosterParams
):
    """Read the buffer back after the callback ran. Values it did not touch
    round trip unchanged, so an untouched slot is not a reset; the loop's
    `check_resettable` sees equality and moves on."""
    params.learning_rate = p.unsafe_load(0)
    params.tree.num_leaves = Int(p.unsafe_load(1))
    params.tree.max_depth = Int(p.unsafe_load(2))
    params.tree.min_data_in_leaf = Int(p.unsafe_load(3))
    params.tree.min_child_hess = p.unsafe_load(4)
    params.tree.lambda_l1 = p.unsafe_load(5)
    params.tree.lambda_reg = p.unsafe_load(6)
    params.tree.feature_fraction = p.unsafe_load(7)
    params.tree.feature_fraction_bynode = p.unsafe_load(8)


def _reset_pointer(
    params: PythonObject,
) raises -> Pointer[Float64, MutUntrackedOrigin]:
    """The caller's `RESET_SLOTS`-long float64 buffer for parameter
    schedules. Always allocated, even with no callbacks, so the bridge never
    has to reason about a null pointer."""
    var addr = Int(py=params["reset_addr"])
    if addr == 0:
        raise Error("invalid buffer")
    return Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def _evals_pointer(
    params: PythonObject,
) raises -> Pointer[Float64, MutUntrackedOrigin]:
    """The caller's `n_valid * n_metrics` float64 buffer holding the round's
    metric values for an after-iteration callback."""
    var addr = Int(py=params["evals_addr"])
    if addr == 0:
        raise Error("invalid buffer")
    return Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def _pred_pointer(
    params: PythonObject,
) raises -> Pointer[Float64, MutUntrackedOrigin]:
    """The caller's scratch buffer for validation predictions. It must be
    float64, alive for the whole call, and long enough for the largest
    validation set (times n_classes for the softmax trainer)."""
    var pred_addr = Int(py=params["pred_addr"])
    if pred_addr == 0:
        raise Error("invalid buffer")
    return Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=pred_addr)


def _parse_valid_sets(
    params: PythonObject, n_features: Int
) raises -> List[RawValidSet]:
    """Validation sets from the params dict's `valid_sets`, a sequence of
    `(name, x_addr, n_rows, y_addr)`. Targets are float64 whatever the
    trainer makes of them: labels for the softmax trainer, relevance grades
    for the ranker."""
    var valid_specs = params["valid_sets"]
    var valid_sets = List[RawValidSet]()
    for v in range(Int(py=params["n_valid"])):
        var spec = valid_specs[v]
        var rows = Int(py=spec[2])
        valid_sets.append(
            RawValidSet(
                String(py=spec[0]),
                _f64_list(Int(py=spec[1]), rows * n_features),
                rows,
                _f64_list(Int(py=spec[3]), rows),
            )
        )
    return valid_sets^


def _parse_metrics(params: PythonObject) raises -> List[CustomMetric]:
    """Metric metadata from the params dict's `metrics`, a sequence of
    `(name, higher_is_better, use_for_early_stopping)` with the flags as
    ints so the boundary carries no Python bool conversion."""
    var metric_specs = params["metrics"]
    var metrics = List[CustomMetric]()
    for m in range(Int(py=params["n_metrics"])):
        var spec = metric_specs[m]
        metrics.append(
            CustomMetric(
                String(py=spec[0]),
                Int(py=spec[1]) != 0,
                Int(py=spec[2]) != 0,
            )
        )
    return metrics^


def _metric_output(
    var model: PythonObject,
    history: MetricHistory,
    best_iteration: Int,
    best_score: Float64,
    stopped_early: Bool,
) raises -> PythonObject:
    """The list every metric-scoring fit returns:
    `[model, values, n_rounds, best_iteration, best_score, stopped_early]`,
    where `values` is the flattened history, round-major, then validation
    set, then metric."""
    var values = Python.list()
    for i in range(len(history.values)):
        values.append(PythonObject(history.values[i]))
    var out = Python.list()
    out.append(model^)
    out.append(values)
    out.append(PythonObject(history.n_rounds()))
    out.append(PythonObject(best_iteration))
    out.append(PythonObject(best_score))
    out.append(PythonObject(Int(stopped_early)))
    return out^


def fit_multiclass_with_metrics(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    n_classes: PythonObject,
    bridge: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`fit_with_metrics` for the softmax trainer.

    The predictions written into `pred_addr` before each callback are
    row-major raw scores, `pred[r * n_classes + k]`, so that buffer needs
    `n_classes` entries per validation row. Labels arrive as float64 in
    0..n_classes-1, the same encoding `fit_multiclass` takes.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var nc = Int(py=n_classes)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf, unbundled="fit_multiclass_with_metrics")
    var weights = _parse_weights(params, nr)
    var pred_p = _pred_pointer(params)
    var valid_sets = _parse_valid_sets(params, nf)
    var metrics = _parse_metrics(params)

    def py_metric(
        metric: Int,
        valid: Int,
        pred: List[Float64],
        labels: List[Float64],
    ) raises {imm bridge, imm pred_p} -> Float64:
        for r in range(len(pred)):
            pred_p.unsafe_store(r, pred[r])
        return Float64(py=bridge(PythonObject(metric), PythonObject(valid)))

    var result = mojo_fit_multiclass_with_metrics(
        features,
        nr,
        nf,
        labels,
        nc,
        valid_sets^,
        bp,
        MetricSuite(metrics^, py_metric, Int(py=params["primary_metric"])),
        Int(py=params["early_stopping_rounds"]),
        Float64(py=params["min_delta"]),
        Int(py=params["max_bin"]),
        weights,
        _parse_bagging(params),
        _parse_goss(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )

    var model = result.model.copy()
    return _metric_output(
        PythonObject(alloc=model^),
        result.history,
        result.best_iteration,
        result.best_score,
        result.stopped_early,
    )


def fit_ranker_with_metrics(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    bridge: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`fit_with_metrics` for the LambdaRank trainer.

    Training query boundaries ride in the params dict as `group_addr` /
    `n_groups`, as they do for `fit_ranker`. A validation set's own
    boundaries are not passed here at all: the callback needs them, not the
    trainer, and the Python side already holds them.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf, unbundled="fit_ranker_with_metrics")
    var weights = _parse_weights(params, nr)
    var pred_p = _pred_pointer(params)
    var valid_sets = _parse_valid_sets(params, nf)
    var metrics = _parse_metrics(params)

    def py_metric(
        metric: Int,
        valid: Int,
        pred: List[Float64],
        labels: List[Float64],
    ) raises {imm bridge, imm pred_p} -> Float64:
        for r in range(len(pred)):
            pred_p.unsafe_store(r, pred[r])
        return Float64(py=bridge(PythonObject(metric), PythonObject(valid)))

    var result = mojo_fit_ranker_with_metrics(
        features,
        nr,
        nf,
        labels,
        _group_counts(params),
        valid_sets^,
        bp,
        MetricSuite(metrics^, py_metric, Int(py=params["primary_metric"])),
        Int(py=params["early_stopping_rounds"]),
        Float64(py=params["min_delta"]),
        _parse_rank_params(params),
        Int(py=params["max_bin"]),
        weights,
        _parse_bagging(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )

    var model = result.model.copy()
    return _metric_output(
        PythonObject(alloc=model^),
        result.history,
        result.best_iteration,
        result.best_score,
        result.stopped_early,
    )


# Built-in evaluation metric codes. The Python wrapper mirrors this table in
# python/mojotrees/_eval.py; they are one contract and must move together.
comptime _METRIC_L2 = 0
comptime _METRIC_RMSE = 1
comptime _METRIC_L1 = 2
comptime _METRIC_QUANTILE = 3
comptime _METRIC_HUBER = 4
comptime _METRIC_BINARY_LOGLOSS = 5
comptime _METRIC_BINARY_ERROR = 6
comptime _METRIC_AUC = 7
comptime _METRIC_MULTI_LOGLOSS = 8
comptime _METRIC_MULTI_ERROR = 9
comptime _METRIC_NDCG = 10
comptime _METRIC_MAPE = 11
comptime _METRIC_FAIR = 12
comptime _METRIC_POISSON = 13
comptime _METRIC_GAMMA = 14
comptime _METRIC_GAMMA_DEVIANCE = 15
comptime _METRIC_TWEEDIE = 16
comptime _METRIC_CROSS_ENTROPY = 17
comptime _METRIC_KLDIV = 18
comptime _METRIC_AVERAGE_PRECISION = 19
comptime _METRIC_MAP = 20


def eval_metric(
    code: PythonObject, params: PythonObject
) raises -> PythonObject:
    """One built-in metric (see the codes above) over buffers the caller
    owns, so the wrapper never reimplements a metric that metrics.mojo
    already defines.

    `params` holds `pred_addr`, `y_addr`, `weight_addr` (0 for unweighted),
    `n_rows`, `objective` (the trained model's objective code), and whatever
    the metric needs beyond that: `n_classes` for the multiclass metrics,
    `group_addr` / `n_groups` / `ndcg_at` for the ranking metrics, and
    `alpha` for the metrics that read the objective's scalar parameter
    (quantile, huber, fair, tweedie).

    Predictions arrive as raw scores, the metric contract in
    custom_metric.mojo. The *objective's* inverse link is applied here once,
    so every metric scores what a prediction would return: probabilities for
    a binary or cross-entropy model, expected values for a poisson, gamma,
    or tweedie one, and the raw score for the objectives with no link
    (including a custom one, which is LightGBM's `feval` contract). That is
    LightGBM's rule as well: the transform belongs to the objective, not to
    the metric, so `l2` on a poisson model scores the counts rather than
    their logarithms.
    """
    var kind = Int(py=code)
    var nr = Int(py=params["n_rows"])
    var objective = Int(py=params["objective"])
    var weight_addr = Int(py=params["weight_addr"])
    var weight = List[Float64]()
    if weight_addr != 0:
        weight = _f64_list(weight_addr, nr)

    if kind == _METRIC_NDCG or kind == _METRIC_MAP:
        var scores = _f64_list(Int(py=params["pred_addr"]), nr)
        var grades = _int_list_from_f64(Int(py=params["y_addr"]), nr)
        var groups = groups_from_counts(_group_counts(params))
        var cutoff = Int(py=params["ndcg_at"])
        if kind == _METRIC_MAP:
            return PythonObject(
                mojo_map(scores, grades, groups, cutoff)
            )
        return PythonObject(mojo_ndcg(scores, grades, groups, cutoff))

    if kind == _METRIC_MULTI_LOGLOSS or kind == _METRIC_MULTI_ERROR:
        var nc = Int(py=params["n_classes"])
        var raw = _f64_list(Int(py=params["pred_addr"]), nr * nc)
        var codes = _int_list_from_f64(Int(py=params["y_addr"]), nr)
        for r in range(nr):
            _softmax_inplace(raw, r * nc, nc)
        if kind == _METRIC_MULTI_LOGLOSS:
            return PythonObject(multiclass_log_loss(raw, codes, nc, weight))
        return PythonObject(multiclass_error(raw, codes, nc, weight))

    var raw = _f64_list(Int(py=params["pred_addr"]), nr)
    var target = _f64_list(Int(py=params["y_addr"]), nr)
    var alpha = Float64(py=params["alpha"])
    # One transform for every single-output metric: the objective's own.
    var pred = response_scale(objective, raw)
    if kind == _METRIC_L2:
        return PythonObject(l2(pred, target, weight))
    if kind == _METRIC_RMSE:
        return PythonObject(rmse(pred, target, weight))
    if kind == _METRIC_L1:
        return PythonObject(l1(pred, target, weight))
    if kind == _METRIC_QUANTILE:
        return PythonObject(quantile_loss(pred, target, alpha, weight))
    if kind == _METRIC_HUBER:
        return PythonObject(huber_loss(pred, target, alpha, weight))
    if kind == _METRIC_MAPE:
        return PythonObject(mape(pred, target, weight))
    if kind == _METRIC_FAIR:
        return PythonObject(fair_loss(pred, target, alpha, weight))
    if kind == _METRIC_POISSON:
        return PythonObject(poisson_loss(pred, target, weight))
    if kind == _METRIC_GAMMA:
        return PythonObject(gamma_loss(pred, target, weight))
    if kind == _METRIC_GAMMA_DEVIANCE:
        return PythonObject(gamma_deviance(pred, target, weight))
    if kind == _METRIC_TWEEDIE:
        return PythonObject(tweedie_loss(pred, target, alpha, weight))
    if kind == _METRIC_CROSS_ENTROPY:
        return PythonObject(cross_entropy_loss(pred, target, weight))
    if kind == _METRIC_KLDIV:
        return PythonObject(kullback_leibler(pred, target, weight))
    if kind == _METRIC_AUC:
        # AUC and average precision read only the score order, which every
        # link here preserves, so the transform above cannot change them.
        return PythonObject(binary_auc(pred, target, weight))
    if kind == _METRIC_AVERAGE_PRECISION:
        return PythonObject(average_precision(pred, target, weight))
    if kind == _METRIC_BINARY_LOGLOSS:
        return PythonObject(binary_log_loss(pred, target, weight))
    if kind == _METRIC_BINARY_ERROR:
        return PythonObject(binary_error(pred, target, 0.5, weight))
    raise Error(String("unknown metric code ", kind))


def fit_multiclass(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    n_classes: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a multiclass model. Labels arrive as float64 in 0..n_classes-1."""
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    # Read the device first, for the reason `fit` does.
    var device = _parse_device(params)
    var bp = _parse_params(params, nf, cpu=device == CPU_DEVICE)
    var weights = _parse_weights(params, nr)
    var model = mojo_fit_multiclass(
        features,
        nr,
        nf,
        labels,
        Int(py=n_classes),
        bp,
        Int(py=params["max_bin"]),
        weights,
        device,
        _parse_bagging(params),
        _parse_goss(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )
    return PythonObject(alloc=model^)


def fit_csc(
    y_addr: PythonObject,
    objective: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a single-output model on a sparse matrix, without densifying.

    The matrix arrives as CSC buffer addresses in `params`; the model that
    comes back is an ordinary `Model`, indistinguishable from a dense fit.
    """
    var csc = _csc(params)
    var nr = csc.n_rows
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, csc.n_features)
    var weights = _parse_weights(params, nr)
    var model = mojo_fit_csc(
        csc,
        target,
        Int(py=objective),
        bp,
        Int(py=params["max_bin"]),
        weights,
        Float64(py=params["alpha"]),
        _parse_bagging(params),
        _parse_goss(params),
        _parse_use_missing(params),
    )
    return PythonObject(alloc=model^)


def fit_multiclass_csc(
    y_addr: PythonObject,
    n_classes: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a multiclass model on a sparse matrix. Labels arrive as
    float64 in 0..n_classes-1."""
    var csc = _csc(params)
    var nr = csc.n_rows
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, csc.n_features)
    var weights = _parse_weights(params, nr)
    var model = mojo_fit_multiclass_csc(
        csc,
        labels,
        Int(py=n_classes),
        bp,
        Int(py=params["max_bin"]),
        weights,
        _parse_bagging(params),
        _parse_use_missing(params),
    )
    return PythonObject(alloc=model^)


def _store(values: List[Float64], out_addr: PythonObject) raises:
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for i in range(len(values)):
        out.unsafe_store(i, values[i])


def predict_csr(
    model: PythonObject, params: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Response-scale predictions for a sparse matrix, into a preallocated
    float64 buffer of length n_rows.

    The sparse walk is host-only. An explicit `device="gpu"` is refused
    rather than densified (see `_refuse_gpu_sparse`); a dict that names no
    device is the CPU, which is what every caller of this function has
    always meant."""
    var m = model.downcast_value_ptr[Model]()
    var shape = _sparse_shape(params)
    _refuse_gpu_sparse(params, shape[0], shape[1], 1)
    _store(mojo_predict_csr(m[], _csr(params)), out_addr)
    return PythonObject(None)


def predict_raw_csr(
    model: PythonObject, params: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Raw-score predictions for a sparse matrix (see `predict_csr` on the
    device)."""
    var m = model.downcast_value_ptr[Model]()
    var shape = _sparse_shape(params)
    _refuse_gpu_sparse(params, shape[0], shape[1], 1)
    _store(mojo_predict_raw_csr(m[], _csr(params)), out_addr)
    return PythonObject(None)


def predict_proba_csr(
    model: PythonObject, params: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Multiclass probabilities for a sparse matrix, row-major
    `[r * n_classes + k]`, into a buffer of length n_rows * n_classes (see
    `predict_csr` on the device)."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var shape = _sparse_shape(params)
    _refuse_gpu_sparse(params, shape[0], shape[1], m[].booster.n_classes)
    _store(mojo_predict_proba_csr(m[], _csr(params)), out_addr)
    return PythonObject(None)


def _parse_rank_params(params: PythonObject) raises -> RankerParams:
    """LambdaRank config from the params dict. `lambdarank_norm` arrives as
    an int so the boundary carries no Python bool conversion. The trainer
    validates again, so the rules in ranking.mojo stay the only ones."""
    return RankerParams(
        Int(py=params["lambdarank_truncation_level"]),
        Float64(py=params["sigmoid"]),
        Int(py=params["lambdarank_norm"]) != 0,
        Int(py=params["ndcg_eval_at"]),
    )


def _group_counts(params: PythonObject) raises -> List[Int]:
    """Per-query row counts (LightGBM's `group`) from the params dict. They
    travel as float64 like every other buffer at this boundary."""
    return _int_list_from_f64(
        Int(py=params["group_addr"]), Int(py=params["n_groups"])
    )


def fit_ranker(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a LambdaRank model. Relevance labels arrive as float64
    nonnegative integers and the query boundaries ride in the params dict,
    so this stays within the argument count the other fits use."""
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf, unbundled="fit_ranker")
    var weights = _parse_weights(params, nr)
    var model = mojo_fit_ranker(
        features,
        nr,
        nf,
        labels,
        _group_counts(params),
        bp,
        _parse_rank_params(params),
        Int(py=params["max_bin"]),
        weights,
        _parse_bagging(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )
    return PythonObject(alloc=model^)


def ndcg(
    scores_addr: PythonObject,
    y_addr: PythonObject,
    n_rows: PythonObject,
    k: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Mean NDCG@k over the queries described by the params dict's group
    counts. Exposed on its own so callers can score any set of scores, not
    only a mojotrees model's."""
    var nr = Int(py=n_rows)
    var scores = _f64_list(Int(py=scores_addr), nr)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var groups = groups_from_counts(_group_counts(params))
    return PythonObject(mojo_ndcg(scores, labels, groups, Int(py=k)))


def _iteration_slice(
    n_iterations: Int, start: PythonObject, stop: PythonObject
) raises -> IterationRange:
    """Clamp the wrapper's half-open iteration pair against the ensemble.

    The Python wrapper has already resolved LightGBM's
    `(start_iteration, num_iteration)` pair into an explicit `[start, stop)`,
    because it needs the resolved bounds to report output shapes. Clamping
    again here keeps the extension safe for a caller that reaches past it,
    and it is where an out-of-range pair becomes an empty range rather than
    an out-of-bounds tree index."""
    return IterationRange.slice(n_iterations, Int(py=start), Int(py=stop))


def predict_range(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    raw_score: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Single-output predictions from the boosting iterations in
    `[start, stop)` into a preallocated float64 buffer: raw scores when
    `raw_score` is nonzero, the response scale otherwise."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    var raw = Int(py=raw_score) != 0
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for r in range(nr):
        var row = _row(features, nr, nf, r)
        if raw:
            out.unsafe_store(r, m[].predict_raw_range(row, rng))
        else:
            out.unsafe_store(r, m[].predict_range(row, rng))
    return PythonObject(None)


def predict_proba_range(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    raw_score: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Multiclass output over `[start, stop)`, row-major
    `[r * n_classes + k]`, into a preallocated float64 buffer of size
    n_rows * n_classes: raw per-class scores when `raw_score` is nonzero,
    softmax probabilities otherwise."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    var raw = Int(py=raw_score) != 0
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var k = m[].booster.n_classes
    for r in range(nr):
        var row = _row(features, nr, nf, r)
        var values: List[Float64]
        if raw:
            values = m[].predict_raw_range(row, rng)
        else:
            values = m[].predict_proba_range(row, rng)
        for c in range(k):
            out.unsafe_store(r * k + c, values[c])
    return PythonObject(None)


def _leaf_host[
    features_origin: ImmOrigin, //
](
    model: PythonObject,
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    rng: IterationRange,
    out_addr: PythonObject,
) raises:
    """The host leaf walk for a single-output model, one row at a time.

    Split out of `predict_leaf` so the device-aware entry point below can
    fall back to exactly this code rather than to a second copy of it."""
    var m = model.downcast_value_ptr[Model]()
    var n_cols = rng.n_iterations()
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    # One ordinal table per tree in the range, built once and shared by every
    # row: the mapping from node id to leaf ordinal is a property of the tree.
    var tables = m[].booster.leaf_ordinals_range(rng)
    for r in range(n_rows):
        var bins = m[].mapper.bin_row(_row(features, n_rows, n_features, r))
        for i in range(n_cols):
            var node = m[].booster.trees[rng.start + i].leaf_index_bins(bins)
            out.unsafe_store(r * n_cols + i, Float64(tables[i][node]))


def _leaf_multiclass_host[
    features_origin: ImmOrigin, //
](
    model: PythonObject,
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    rng: IterationRange,
    out_addr: PythonObject,
) raises:
    """The host leaf walk for a multiclass model (see `_leaf_host`)."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var k = m[].booster.n_classes
    var n_rounds = rng.n_iterations()
    var n_cols = n_rounds * k
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var tables = m[].booster.leaf_ordinals_range(rng)
    for r in range(n_rows):
        var bins = m[].mapper.bin_row(_row(features, n_rows, n_features, r))
        for i in range(n_rounds):
            for c in range(k):
                var tree = (rng.start + i) * k + c
                var node = m[].booster.trees[tree].leaf_index_bins(bins)
                out.unsafe_store(
                    r * n_cols + i * k + c, Float64(tables[i * k + c][node])
                )


def predict_leaf(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Per-tree leaf ordinals over `[start, stop)` for a single-output model,
    row-major `[r * n_iterations + i]`, into a preallocated float64 buffer.

    The buffer is float64 because that is the only element type crossing this
    boundary. Leaf ordinals are small nonnegative integers, so they are
    exactly representable, and the wrapper casts them back. An empty range
    writes nothing."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    if rng.n_iterations() == 0 or nr == 0:
        return PythonObject(None)
    _leaf_host(model, _f64_view(Int(py=x_addr), nr * nf), nr, nf, rng, out_addr)
    return PythonObject(None)


def predict_leaf_multiclass(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Per-tree leaf ordinals over `[start, stop)` for a multiclass model,
    row-major and round-major within a row: column `i * n_classes + k` is
    class k's tree in the range's iteration i, so a row spans
    `n_iterations * n_classes` columns."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    if rng.n_iterations() == 0 or nr == 0:
        return PythonObject(None)
    _leaf_multiclass_host(
        model, _f64_view(Int(py=x_addr), nr * nf), nr, nf, rng, out_addr
    )
    return PythonObject(None)


def predict_contrib(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Exact TreeSHAP contributions over `[start, stop)` for a single-output
    model, row-major `[r * (n_features + 1) + f]`, into a preallocated
    float64 buffer.

    The last column of each row is the expected value, so a row's entries sum
    to its raw score over the same range. One explainer serves the whole
    batch: it holds the path scratch and validates the ensemble's node covers
    once rather than per row."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    if nr == 0:
        return PythonObject(None)
    var explainer = ContribExplainer.for_booster(m[].booster, nf)
    var width = explainer.width()
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var row_out = List[Float64](capacity=width)
    row_out.resize(width, 0.0)
    for r in range(nr):
        var bins = m[].mapper.bin_row(_row(features, nr, nf, r))
        explainer.contrib_bins_into(m[].booster, bins, row_out, 0, rng)
        for c in range(width):
            out.unsafe_store(r * width + c, row_out[c])
    return PythonObject(None)


def predict_contrib_multiclass(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    start: PythonObject,
    stop: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Exact TreeSHAP contributions over `[start, stop)` for a multiclass
    model, row-major with class-major blocks inside a row: column
    `k * (n_features + 1) + f` is feature f's contribution to class k, so a
    row spans `n_classes * (n_features + 1)` columns and each class's block
    sums to that class's raw score."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    if nr == 0:
        return PythonObject(None)
    var explainer = ContribExplainer.for_multiclass(m[].booster, nf)
    var width = explainer.width()
    var features = _f64_view(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var row_out = List[Float64](capacity=width)
    row_out.resize(width, 0.0)
    for r in range(nr):
        var bins = m[].mapper.bin_row(_row(features, nr, nf, r))
        explainer.contrib_bins_multiclass_into(
            m[].booster, bins, row_out, 0, rng
        )
        for c in range(width):
            out.unsafe_store(r * width + c, row_out[c])
    return PythonObject(None)


# -- GPU prediction ------------------------------------------------------
#
# Every prediction entry point above walks the trees on the host, one row at
# a time, whatever `device` the estimator was configured with: nothing in
# this module has ever reached gpu_predict.mojo, so `device="gpu"` has been
# a training-only setting. The entry points below are the batched forms that
# do reach it.
#
# None of them decides where to run. They hand the requested device to
# `Model.predict_batch` and `MulticlassModel.predict_batch`, which is the one
# place that chooses between the host walk and `GpuPredictor`, and they
# return the name of the backend that actually ran so a caller reporting
# `device="auto"` never has to assume which one it got. An explicit `gpu`
# that the prediction path cannot serve raises here, before any work, with
# the reason from `gpu_predict_support`; it is never quietly served by the
# CPU.
#
# The established path is untouched. `predict_range`,
# `predict_proba_range`, `predict_leaf`, and `predict_leaf_multiclass` keep
# their signatures and their host walk, so an estimator that has not moved
# over behaves exactly as before. (The older whole-model `predict`,
# `predict_raw`, and `predict_proba` entries, full-range twins of the
# `_range` forms that Python had stopped calling, were removed in the
# consolidation round.)


def _optional_device(params: PythonObject) raises -> Int:
    """The device a params dict requests, or the CPU when it names none.

    The sparse prediction dicts are built by `_arrays.py` and carry buffers
    only, so a device key is optional there and its absence has to mean the
    established behavior rather than an error."""
    return parse_device(String(py=params.get("device", PythonObject("cpu"))))


def _predict_device(
    params: PythonObject,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int,
    n_bins: Int,
) raises -> Int:
    """The backend a dense prediction call will run on: CPU or GPU.

    Two questions, each asked of the module that owns it. Whether the GPU
    prediction path covers a request of this shape at all is
    `gpu_predict_support` in gpu_predict.mojo, and it is asked only for an
    explicit `gpu`, so its refusal is the message an explicit request gets.
    Which backend runs is `resolve_device` in device.mojo, the same call
    `Model.predict_batch` makes on the way in, so what this returns is what
    the predictor will reach. Nothing here decides anything.
    """
    var requested = _parse_device(params)
    if requested == GPU_DEVICE:
        var support = gpu_predict_support(
            n_rows, n_features, n_outputs, n_bins
        )
        support.raise_if_blocked()
    return mojo_resolve_device(requested, n_rows, n_features, n_outputs)


def _refuse_gpu_sparse(
    params: PythonObject, n_rows: Int, n_features: Int, n_outputs: Int
) raises:
    """Refuse an explicit `gpu` for sparse input.

    The prediction kernels read a dense binned matrix, so a sparse request
    has no GPU path at all. Densifying behind the caller's back would be the
    silent fallback this whole vocabulary exists to prevent, so the refusal
    is explicit and carries `gpu_predict_support`'s message."""
    if _optional_device(params) != GPU_DEVICE:
        return
    var support = gpu_predict_support(
        n_rows, n_features, n_outputs, sparse=True
    )
    support.raise_if_blocked()


def _store_ints(values: List[Int], out_addr: PythonObject) raises:
    """Widen leaf ordinals into the caller's float64 buffer. Float64 is the
    only element type crossing this boundary and a leaf ordinal is a small
    nonnegative integer, so the conversion is exact and the wrapper casts
    them back."""
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for i in range(len(values)):
        out.unsafe_store(i, Float64(values[i]))


def gpu_predict_capability(params: PythonObject) raises -> PythonObject:
    """Whether the GPU prediction path covers a request, without asking for
    it.

    `params` carries `n_rows`, `n_features`, `n_outputs` (1 for a
    single-output model, the class count for multiclass), `n_bins` (the
    binner's reserved bin count, or 0 when the caller does not know it), and
    `sparse` as an int flag. The answer is
    `[supported, block_code, reason_name, message]`, where `block_code` and
    `reason_name` are device_policy.mojo's stable refusal vocabulary, so an
    estimator can branch on the code and show the prose.

    This is the question form. The refusal form is what the prediction entry
    points raise for an explicit `gpu`, and both come from the same record.
    """
    var support = gpu_predict_support(
        Int(py=params["n_rows"]),
        Int(py=params["n_features"]),
        Int(py=params["n_outputs"]),
        Int(py=params["n_bins"]),
        Int(py=params["sparse"]) != 0,
    )
    var out = Python.list()
    out.append(PythonObject(Int(support.supported)))
    out.append(PythonObject(support.block_code))
    out.append(PythonObject(support.reason_name()))
    out.append(PythonObject(support.message.copy()))
    return out^


def predict_batch(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Single-output predictions for a whole dense batch, on the device the
    params dict names, into a preallocated float64 buffer of length n_rows.

    `params` carries `device` ("cpu", "gpu", or "auto"), the resolved
    half-open iteration pair `start` and `stop`, and `raw_score` as an int
    flag. The buffer and the iteration semantics are `predict_range`'s; what
    is new is that the whole batch crosses at once, which is what the device
    walk needs, and that the backend that ran comes back as the return value.

    An empty batch returns None and writes nothing: no backend is selected
    for it, because the policy refuses to size a workload with no rows and a
    name for a backend that did not run would be a fiction.
    """
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    if nr == 0:
        return PythonObject(None)
    var rng = _iteration_slice(
        m[].n_iterations(), params["start"], params["stop"]
    )
    var device = _predict_device(params, nr, nf, 1, m[].mapper.n_bins)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    _store(
        m[].predict_batch(
            features, nr, rng, Int(py=params["raw_score"]) != 0, device
        ),
        out_addr,
    )
    return PythonObject(mojo_device_name(device))


def predict_proba_batch(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Multiclass output for a whole dense batch, row-major
    `[r * n_classes + k]`, into a preallocated float64 buffer of size
    n_rows * n_classes: softmax probabilities, or per-class raw scores when
    `params["raw_score"]` is nonzero.

    The multiclass shape is the ensemble's, not this function's: the device
    path walks one tree per class per iteration and softmaxes across the row,
    which is what `MulticlassBooster` does on the host. An empty batch
    returns None, as in `predict_batch`."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    if nr == 0:
        return PythonObject(None)
    var k = m[].booster.n_classes
    var rng = _iteration_slice(
        m[].n_iterations(), params["start"], params["stop"]
    )
    var device = _predict_device(params, nr, nf, k, m[].mapper.n_bins)
    var features = _f64_view(Int(py=x_addr), nr * nf)
    _store(
        m[].predict_batch(
            features, nr, rng, Int(py=params["raw_score"]) != 0, device
        ),
        out_addr,
    )
    return PythonObject(mojo_device_name(device))


def predict_leaf_batch(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`predict_leaf` for a whole dense batch, on the device the params dict
    names: row-major `[r * n_iterations + i]` float64 ordinals.

    The ordinal numbering is the same on both backends. The device walk
    reports the leaf's rank among its tree's leaves in node order, which is
    what `Tree.leaf_ordinals` counts and what the host table above indexes,
    so a saved model reports the same ordinals whichever device reads it.
    An empty batch or an empty iteration range selects no tree at all, so it
    writes nothing and returns None rather than naming a backend that did
    not run."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(
        m[].n_iterations(), params["start"], params["stop"]
    )
    if nr == 0 or rng.n_iterations() == 0:
        return PythonObject(None)
    var device = _predict_device(params, nr, nf, 1, m[].mapper.n_bins)
    var name = PythonObject(mojo_device_name(device))
    var features = _f64_view(Int(py=x_addr), nr * nf)
    if device == GPU_DEVICE:
        _store_ints(
            leaf_indices_gpu(
                m[].booster, m[].mapper.transform(features, nr), rng
            ),
            out_addr,
        )
        return name^
    _leaf_host(model, features, nr, nf, rng, out_addr)
    return name^


def predict_leaf_multiclass_batch(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`predict_leaf_multiclass` for a whole dense batch, on the device the
    params dict names: row-major and round-major within a row, column
    `i * n_classes + k`. An empty batch or range returns None, as in
    `predict_leaf_batch`."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var k = m[].booster.n_classes
    var rng = _iteration_slice(
        m[].n_iterations(), params["start"], params["stop"]
    )
    if nr == 0 or rng.n_iterations() == 0:
        return PythonObject(None)
    var device = _predict_device(params, nr, nf, k, m[].mapper.n_bins)
    var name = PythonObject(mojo_device_name(device))
    var features = _f64_view(Int(py=x_addr), nr * nf)
    if device == GPU_DEVICE:
        _store_ints(
            leaf_indices_multiclass_gpu(
                m[].booster, m[].mapper.transform(features, nr), rng
            ),
            out_addr,
        )
        return name^
    _leaf_multiclass_host(model, features, nr, nf, rng, out_addr)
    return name^


# -- resident validation scoring -----------------------------------------
#
# A training loop driven from Python scores its validation set once per
# round. Doing that through `predict_batch` would re-upload the whole matrix
# and re-walk the whole ensemble every round, which is the cost
# `GpuPredictor`'s resident validation path exists to remove: the matrix,
# its labels, its weights, and the running raw-score vector go to the device
# once, and a round uploads only the trees it grew.
#
# `GpuValidation` is that path with a Python-owned lifetime. The handle is
# opaque: nothing about a device buffer crosses the boundary, and the only
# way out is a metric value or a copy of the raw scores into a caller's
# float64 buffer. The model is not retained either. Each accumulate call
# reads the trees it needs and copies them, so a handle never keeps a
# model alive and a model can be updated, copied, or dropped underneath it.
#
# There is no CPU form of this handle, and it does not take a device: it is
# the device path by construction, so opening one on a build or a machine
# that cannot serve it raises `gpu_predict_support`'s refusal rather than
# silently scoring on the host. A caller that wants host validation scoring
# has `eval_metric` and the fits that carry their own metric suites.


def _gpu_validation_unavailable() raises:
    raise Error("GPU validation requires an accelerator-enabled build")


def _gpu_validation_open_unavailable(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    _gpu_validation_unavailable()
    return PythonObject(None)


def _gpu_validation_shape_unavailable(
    handle: PythonObject,
) raises -> PythonObject:
    _gpu_validation_unavailable()
    return PythonObject(None)


def _gpu_validation_reset_unavailable(
    handle: PythonObject, base_addr: PythonObject
) raises -> PythonObject:
    _gpu_validation_unavailable()
    return PythonObject(None)


def _gpu_validation_accumulate_unavailable(
    handle: PythonObject,
    model: PythonObject,
    start: PythonObject,
    stop: PythonObject,
) raises -> PythonObject:
    _gpu_validation_unavailable()
    return PythonObject(None)


def _gpu_validation_metric_unavailable(
    handle: PythonObject, metric: PythonObject, objective: PythonObject
) raises -> PythonObject:
    _gpu_validation_unavailable()
    return PythonObject(None)


def _gpu_validation_raw_unavailable(
    handle: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    _gpu_validation_unavailable()
    return PythonObject(None)


struct GpuValidation(Movable, Writable):
    """A validation set kept on the device across a training run.

    Holds a `GpuPredictor` with its validation buffers already sized and
    seeded, plus the shape a caller needs to size its own buffers. The
    predictor opens its own device context, which is right for a loop driven
    from outside Mojo: the trainer's context is not reachable from Python, so
    sharing one is not on the table here the way it is for
    `train_gpu_with_valid`.

    Lifetime is Python's. The device buffers and the context go away when
    the last Python reference to the handle does, so a loop that drops the
    handle releases the validation set without an explicit close.
    """

    var predictor: GpuPredictor
    var n_rows: Int
    var n_outputs: Int

    def __init__(
        out self,
        data: BinnedMatrix,
        target: List[Float64],
        weight: List[Float64],
        n_outputs: Int,
    ) raises:
        comptime if not has_accelerator():
            raise Error("GPU validation requires an accelerator")
        else:
            self.predictor = GpuPredictor(data.n_features, n_outputs)
            self.predictor.set_validation(data, target, weight)
            self.n_rows = data.n_rows
            self.n_outputs = n_outputs

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "GpuValidation(n_rows=",
            self.n_rows,
            ", n_outputs=",
            self.n_outputs,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _validation_columns(
    params: PythonObject, n_rows: Int
) raises -> List[List[Float64]]:
    """The label column and the optional weight column of a validation set,
    from the params dict's `y_addr` and `weight_addr` (0 for absent). The
    weight contract is metrics.mojo's, and `set_validation` validates it."""
    var target = _f64_list(Int(py=params["y_addr"]), n_rows)
    var weight = List[Float64]()
    if Int(py=params["weight_addr"]) != 0:
        weight = _f64_list(Int(py=params["weight_addr"]), n_rows)
    var out = List[List[Float64]]()
    out.append(target^)
    out.append(weight^)
    return out^


def gpu_validation_open(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Make a validation set device-resident for a single-output model and
    seed its raw scores with the model's base score.

    The matrix is binned by the model's own mapper, on the host, because bin
    edges are Float64 and a routing decision is discrete; only the bins go to
    the device. `params` carries `y_addr` and `weight_addr` (0 for no
    weights).

    Returns an opaque handle. Accumulate the rounds the model gains into it
    with `gpu_validation_accumulate`, then read a metric or the raw scores.
    """
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var m = model.downcast_value_ptr[Model]()
        var nr = Int(py=n_rows)
        var nf = Int(py=n_features)
        var support = gpu_predict_support(nr, nf, 1, m[].mapper.n_bins)
        support.raise_if_blocked()
        var features = _f64_view(Int(py=x_addr), nr * nf)
        var columns = _validation_columns(params, nr)
        var handle = GpuValidation(
            m[].mapper.transform(features, nr), columns[0], columns[1], 1
        )
        var base: List[Float64] = [m[].booster.base_score]
        handle.predictor.reset_validation(base)
        return PythonObject(alloc=handle^)


def gpu_validation_open_multiclass(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gpu_validation_open` for a softmax model. The labels are class codes
    in 0..n_classes-1, the same encoding `fit_multiclass` takes, and the
    resident score vector is `n_rows * n_classes` wide."""
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var m = model.downcast_value_ptr[MulticlassModel]()
        var nr = Int(py=n_rows)
        var nf = Int(py=n_features)
        var k = m[].booster.n_classes
        var support = gpu_predict_support(nr, nf, k, m[].mapper.n_bins)
        support.raise_if_blocked()
        var features = _f64_view(Int(py=x_addr), nr * nf)
        var columns = _validation_columns(params, nr)
        var handle = GpuValidation(
            m[].mapper.transform(features, nr), columns[0], columns[1], k
        )
        handle.predictor.reset_validation(m[].booster.base_scores)
        return PythonObject(alloc=handle^)


def gpu_validation_shape(handle: PythonObject) raises -> PythonObject:
    """`[n_rows, n_outputs]` for the resident validation set, so a caller can
    size the buffer `gpu_validation_raw` fills."""
    var h = handle.downcast_value_ptr[GpuValidation]()
    var out = Python.list()
    out.append(PythonObject(h[].n_rows))
    out.append(PythonObject(h[].n_outputs))
    return out^


def gpu_validation_reset(
    handle: PythonObject, base_addr: PythonObject
) raises -> PythonObject:
    """Set every resident raw score back to the per-output base score in the
    caller's float64 buffer of length n_outputs.

    Where a boosting run starts, and the only place the base score enters:
    `gpu_validation_accumulate` never adds it, exactly as `IterationRange`
    counts it as part of iteration 0 rather than of every round."""
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var h = handle.downcast_value_ptr[GpuValidation]()
        h[].predictor.reset_validation(
            _f64_list(Int(py=base_addr), h[].n_outputs)
        )
        return PythonObject(None)


def gpu_validation_accumulate(
    handle: PythonObject,
    model: PythonObject,
    start: PythonObject,
    stop: PythonObject,
) raises -> PythonObject:
    """Fold the single-output model's iterations in `[start, stop)` into the
    resident raw scores.

    A loop that appended one round calls this with that round's pair, and
    scoring the round costs one tree walk per validation row rather than a
    walk of the whole ensemble. An empty range does nothing."""
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var h = handle.downcast_value_ptr[GpuValidation]()
        var m = model.downcast_value_ptr[Model]()
        if h[].n_outputs != 1:
            raise Error(
                "this validation handle was opened for a multiclass model; use"
                " gpu_validation_accumulate_multiclass"
            )
        accumulate_booster_rounds(
            h[].predictor,
            m[].booster,
            _iteration_slice(m[].n_iterations(), start, stop),
        )
        return PythonObject(None)


def gpu_validation_accumulate_multiclass(
    handle: PythonObject,
    model: PythonObject,
    start: PythonObject,
    stop: PythonObject,
) raises -> PythonObject:
    """`gpu_validation_accumulate` for a softmax model. One iteration is one
    tree per class, so the range is taken in whole rounds."""
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var h = handle.downcast_value_ptr[GpuValidation]()
        var m = model.downcast_value_ptr[MulticlassModel]()
        if h[].n_outputs != m[].booster.n_classes:
            raise Error(
                "the model's class count does not match the validation handle"
            )
        accumulate_multiclass_rounds(
            h[].predictor,
            m[].booster,
            _iteration_slice(m[].n_iterations(), start, stop),
        )
        return PythonObject(None)


def gpu_validation_metric(
    handle: PythonObject, metric: PythonObject, objective: PythonObject
) raises -> PythonObject:
    """Score the resident validation set on the device.

    `metric` is a built-in metric code, the same numbering `eval_metric`
    takes, so a caller carries one metric vocabulary rather than a second
    device-side one. `objective` is the model's objective code; it selects
    the inverse link to apply to the raw scores before scoring, because
    every metric takes predictions on the response scale. A handle opened
    for a multiclass model softmaxes instead and ignores the objective,
    which is what a softmax ensemble's response is.

    A metric the device has no kernel for raises and names itself. The
    device metric set is the smaller one, and its log losses clamp at the
    Float32 floor rather than at 1e-15, so a caller whose stopping rule is
    defined by the host metric should read the raw scores with
    `gpu_validation_raw` and pass them to `eval_metric` instead. That is the
    same choice `train_gpu`'s device validation scorer makes per objective.
    """
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var h = handle.downcast_value_ptr[GpuValidation]()
        var response = RESPONSE_SOFTMAX
        if h[].n_outputs == 1:
            response = response_for_objective(Int(py=objective))
        return PythonObject(
            validation_host_metric(h[].predictor, Int(py=metric), response)
        )


def gpu_validation_metric_matches_host(
    metric: PythonObject,
) raises -> PythonObject:
    """`[has_kernel, matches_host]` for a built-in metric code.

    What an estimator needs to decide where to score without hardcoding the
    device's metric set: `has_kernel` says the device can compute it at all,
    and `matches_host` says its definition is metrics.mojo's term for term
    rather than merely close. Both answers come from gpu_predict.mojo, which
    owns the kernels."""
    var code = Int(py=metric)
    var out = Python.list()
    out.append(PythonObject(Int(device_metric_code(code) >= 0)))
    out.append(PythonObject(Int(device_metric_matches_host(code))))
    return out^


def gpu_validation_raw(
    handle: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Copy the resident raw scores into a preallocated float64 buffer of
    length n_rows * n_outputs, row-major `[r * n_outputs + k]`.

    The escape hatch, and the only way scores leave the device: it is what
    lets the host metric suite score a run whose metric the device has no
    kernel for."""
    comptime if not has_accelerator():
        raise Error("GPU validation requires an accelerator")
    else:
        var h = handle.downcast_value_ptr[GpuValidation]()
        _store(h[].predictor.validation_raw(), out_addr)
        return PythonObject(None)


def gpu_available() raises -> PythonObject:
    """True when training can run on an accelerator (see device.mojo)."""
    return PythonObject(mojo_gpu_available())


def resolve_device(
    device: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    n_outputs: PythonObject,
) raises -> PythonObject:
    """Resolve a requested device name ("cpu", "gpu", or "auto") to the
    backend that will run: "cpu" or "gpu". Raises on an unknown name, on
    "gpu" without an accelerator, and on "gpu" for a workload the GPU path
    does not cover. `n_outputs` is 1 for single-output training and the
    class count for multiclass."""
    var resolved = mojo_resolve_device(
        parse_device(String(py=device)),
        Int(py=n_rows),
        Int(py=n_features),
        Int(py=n_outputs),
    )
    return PythonObject(mojo_device_name(resolved))


def num_trees(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(len(m[].booster.trees))


def num_iterations(model: PythonObject) raises -> PythonObject:
    """Boosting iterations the fitted ensemble kept. One iteration is one
    tree for a single-output model, so this is the tree count."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(len(m[].booster.trees))


def num_iterations_multiclass(model: PythonObject) raises -> PythonObject:
    """Boosting iterations a multiclass ensemble kept. One iteration is one
    tree per class, so this is the tree count divided by the class count."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(len(m[].booster.trees) // m[].booster.n_classes)


def n_classes(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(m[].booster.n_classes)


def n_features(model: PythonObject) raises -> PythonObject:
    """Features the model was trained on, from its bin mapper."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(m[].mapper.n_features)


def n_features_multiclass(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(m[].mapper.n_features)


def _categorical_list(cats: CategoricalSpec) raises -> PythonObject:
    """The ascending feature indices a fitted bin mapper treats as
    categorical."""
    var out = Python.list()
    for f in range(len(cats.is_categorical)):
        if cats.is_categorical[f]:
            out.append(PythonObject(f))
    return out^


def categorical_features(model: PythonObject) raises -> PythonObject:
    """Which features the fitted model splits by category set, from its bin
    mapper. The serialized format carries the category tables, so a model
    read back from disk still knows this; what it cannot know is any label
    encoding the Python layer applied on top (see python/mojotrees)."""
    var m = model.downcast_value_ptr[Model]()
    return _categorical_list(m[].mapper.cats)


def categorical_features_multiclass(
    model: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return _categorical_list(m[].mapper.cats)


def _write_importance(
    trees: List[Tree], nf: Int, kind: Int, out_addr: Int
) raises:
    """Per-feature importance into a preallocated float64 buffer of length
    `nf`. `kind` is 0 for split counts and 1 for total gain."""
    if kind != 0 and kind != 1:
        raise Error("importance kind must be 0 (split) or 1 (gain)")
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=out_addr
    )
    if kind == 0:
        var counts = split_importance(trees, nf)
        for f in range(nf):
            out.unsafe_store(f, Float64(counts[f]))
    else:
        var gains = gain_importance(trees, nf)
        for f in range(nf):
            out.unsafe_store(f, gains[f])


def feature_importance(
    model: PythonObject,
    n_features: PythonObject,
    kind: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Split-count (kind 0) or total-gain (kind 1) importance for a
    single-output model. Gains are not part of the serialized format, so a
    model read back from disk reports zero gain importance."""
    var m = model.downcast_value_ptr[Model]()
    _write_importance(
        m[].booster.trees,
        Int(py=n_features),
        Int(py=kind),
        Int(py=out_addr),
    )
    return PythonObject(None)


def feature_importance_multiclass(
    model: PythonObject,
    n_features: PythonObject,
    kind: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Importance summed over every class's trees, LightGBM style."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    _write_importance(
        m[].booster.trees,
        Int(py=n_features),
        Int(py=kind),
        Int(py=out_addr),
    )
    return PythonObject(None)


def _saved_names(
    feature_names: PythonObject, n_names: PythonObject
) raises -> List[String]:
    """Feature names for a model about to be written, as a sequence plus
    its length like every other string sequence at this boundary (see
    `dataset_create`). An empty list writes no names section, which is what
    a model with none has always written."""
    var names = List[String]()
    var n = Int(py=n_names)
    for i in range(n):
        names.append(String(py=feature_names[i]))
    return names^


def save(
    model: PythonObject,
    path: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    """Write the model to `path`. Names travel with it from model format
    v4 on; `save_model` refuses a list that does not name this model's
    features rather than dropping it."""
    var m = model.downcast_value_ptr[Model]()
    save_model(m[], String(py=path), _saved_names(feature_names, n_names))
    return PythonObject(None)


def load(path: PythonObject) raises -> PythonObject:
    var model = load_model(String(py=path))
    return PythonObject(alloc=model^)


def save_multiclass(
    model: PythonObject,
    path: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    save_multiclass_model(
        m[], String(py=path), _saved_names(feature_names, n_names)
    )
    return PythonObject(None)


def load_multiclass(path: PythonObject) raises -> PythonObject:
    var model = load_multiclass_model(String(py=path))
    return PythonObject(alloc=model^)


def model_feature_names(path: PythonObject) raises -> PythonObject:
    """The feature names a saved model carries, empty for a file that has
    none: every file written before v4, and any model saved without them.

    A `Model` has no names field, so this reads the file header rather than
    a handle. It is what lets a `Booster` read back from disk report the
    names it was trained with instead of `Column_0`, `Column_1`, ...
    """
    var out = Python.list()
    var names = load_feature_names(String(py=path))
    for i in range(len(names)):
        out.append(PythonObject(names[i]))
    return out^


# -- datasets and booster-level training ---------------------------------
#
# A `Dataset` handle owns its binned matrix and the columns that describe
# its rows, so binning is paid for once however many models are trained on
# it (see src/mojotrees/trainset.mojo). The Python `Dataset` in
# python/mojotrees/basic.py validates every buffer before it gets here and
# keeps them alive for the duration of `dataset_create`; nothing on this
# side retains a Python buffer.


def dataset_create(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Bin a column-major float64 matrix into a reusable `Dataset`.

    `params` holds the optional columns as buffer addresses, 0 for absent:
    `label_addr`, `weight_addr`, `init_score_addr`, and `group_addr` with
    `n_groups`. It also holds the binning configuration (`max_bin`,
    `use_missing`, `categorical_addr` with `categorical_len`) and
    `feature_names`, a sequence of `n_names` strings.

    `keep_raw` retains the raw matrix inside the dataset, which is what
    `dataset_subset` needs and the only thing here that copies it; 0 is the
    default and drops it after binning, as this entry point always did.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)

    var label = List[Float64]()
    if Int(py=params["label_addr"]) != 0:
        label = _f64_list(Int(py=params["label_addr"]), nr)
    var weight = List[Float64]()
    if Int(py=params["weight_addr"]) != 0:
        weight = _f64_list(Int(py=params["weight_addr"]), nr)
    var init_score = List[Float64]()
    if Int(py=params["init_score_addr"]) != 0:
        init_score = _f64_list(Int(py=params["init_score_addr"]), nr)
    var group = List[Int]()
    if Int(py=params["group_addr"]) != 0:
        group = _group_counts(params)

    var names = List[String]()
    var n_names = Int(py=params["n_names"])
    if n_names != 0:
        var given = params["feature_names"]
        for i in range(n_names):
            names.append(String(py=given[i]))

    var dataset = Dataset(
        features,
        nr,
        nf,
        label^,
        weight^,
        group^,
        init_score^,
        names^,
        _parse_categorical(params),
        Int(py=params["max_bin"]),
        _parse_use_missing(params),
        Int(py=params["keep_raw"]) != 0,
    )
    return PythonObject(alloc=dataset^)


def dataset_save(dataset: PythonObject, path: PythonObject) raises:
    """Write a constructed dataset to `path` as a prepared table.

    The binning is what a run pays for before it can start, so a table
    written here is what lets the next process skip it. It is not a model
    file and cannot be loaded as one: see `serialize.save_dataset`.
    """
    var d = dataset.downcast_value_ptr[Dataset]()
    mojo_save_dataset(d[], String(py=path))


def dataset_load(path: PythonObject) raises -> PythonObject:
    """Read a prepared table written by `dataset_save`.

    The result carries no raw matrix, so `dataset_metadata` reports
    `has_raw` false for it and `dataset_subset` refuses it: bins cannot be
    refitted from bins.
    """
    var dataset = mojo_load_dataset(String(py=path))
    return PythonObject(alloc=dataset^)


def file_kind(path: PythonObject) raises -> PythonObject:
    """What a mojotrees file holds: "objective", "multiclass", or
    "dataset". Reads only the header."""
    return PythonObject(mojo_file_kind(String(py=path)))


def dataset_num_data(dataset: PythonObject) raises -> PythonObject:
    var d = dataset.downcast_value_ptr[Dataset]()
    return PythonObject(d[].num_data())


def dataset_num_feature(dataset: PythonObject) raises -> PythonObject:
    var d = dataset.downcast_value_ptr[Dataset]()
    return PythonObject(d[].num_feature())


def dataset_num_bin(dataset: PythonObject) raises -> PythonObject:
    """Bins the dataset's binning reserved per feature, which is `max_bin`
    unless the data had fewer distinct values."""
    var d = dataset.downcast_value_ptr[Dataset]()
    return PythonObject(d[].num_bin())


def train_dataset(
    dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Train a single-output model on a constructed dataset. The label,
    weights, and init scores are the dataset's, and so is the binning, so
    `max_bin`, `use_missing`, and the categorical declaration are not read
    from `params` here."""
    var d = dataset.downcast_value_ptr[Dataset]()
    # Read the device first, for the reason `fit` does.
    var device = _parse_device(params)
    var model = mojo_train_dataset(
        d[],
        Int(py=params["objective"]),
        _parse_params(params, d[].num_feature(), cpu=device == CPU_DEVICE),
        Float64(py=params["alpha"]),
        device,
        _parse_bagging(params),
        _parse_goss(params),
    )
    return PythonObject(alloc=model^)


def train_dataset_multiclass(
    dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Train a softmax model on a constructed dataset, whose label holds
    class codes in 0..n_classes-1."""
    var d = dataset.downcast_value_ptr[Dataset]()
    # Read the device first, for the reason `fit` does.
    var device = _parse_device(params)
    var model = mojo_train_dataset_multiclass(
        d[],
        Int(py=params["n_classes"]),
        _parse_params(params, d[].num_feature(), cpu=device == CPU_DEVICE),
        device,
        _parse_bagging(params),
        _parse_goss(params),
    )
    return PythonObject(alloc=model^)


def train_dataset_ranker(
    dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Train a LambdaRank model on a constructed dataset, whose `group`
    holds the per-query row counts."""
    var d = dataset.downcast_value_ptr[Dataset]()
    var model = mojo_train_dataset_ranker(
        d[],
        _parse_params(
            params,
            d[].num_feature(),
            unbundled="train_dataset_ranker",
        ),
        _parse_rank_params(params),
        _parse_bagging(params),
    )
    return PythonObject(alloc=model^)


def booster_update(
    model: PythonObject, dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Append `n_estimators` more rounds to a single-output model from the
    dataset it was trained on, returning how many trees were added. The
    dataset must be binned by the model's own mapper, which the Mojo side
    checks; see `trainset.update_dataset`."""
    var m = model.downcast_value_ptr[Model]()
    var d = dataset.downcast_value_ptr[Dataset]()
    var added = mojo_update_dataset(
        m[],
        d[],
        _parse_params(params, d[].num_feature()),
        Float64(py=params["alpha"]),
        _parse_bagging(params),
        _parse_goss(params),
    )
    return PythonObject(added)


def booster_update_multiclass(
    model: PythonObject, dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Append `n_estimators` more softmax rounds to a multiclass model,
    returning how many rounds were added (one round is one tree per class).
    See `trainset.update_dataset_multiclass`."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var d = dataset.downcast_value_ptr[Dataset]()
    var added = mojo_update_dataset_multiclass(
        m[],
        d[],
        _parse_params(params, d[].num_feature()),
        _parse_bagging(params),
        _parse_goss(params),
    )
    return PythonObject(added)


def copy_model(model: PythonObject) raises -> PythonObject:
    """An independent copy of a single-output model, split gains included.
    Continuing training from a model means appending trees to it, so the
    caller that wants to keep the original copies it first; a save/load
    round trip would not do, because the file format does not carry gains.
    """
    var m = model.downcast_value_ptr[Model]()
    var copy = m[].copy()
    return PythonObject(alloc=copy^)


def copy_multiclass_model(model: PythonObject) raises -> PythonObject:
    """An independent copy of a multiclass model (see `copy_model`)."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var copy = m[].copy()
    return PythonObject(alloc=copy^)
