"""CPython extension module for mojotrees.

Built with `bindings/build.sh` into `python/mojotrees/_mojotrees.so`; the
public Python surface is the sklearn-style wrapper in `python/mojotrees/`.

Data crosses the boundary as raw buffer addresses (integers) plus lengths:
the wrapper passes float64-contiguous buffers (column-major for feature
matrices) and keeps them alive for the duration of each call. Copies into
Mojo Lists happen here, so no Python buffer is retained after a call
returns. Trained models are returned as opaque handles owned by Python.

Every dense prediction entry point walks the trees over row blocks. It no
longer BINS the matrix first on the default path: `Model.predict_batch`
rewrites each node's `threshold_bin` into the Float64 bin EDGE it names and
compares the raw feature value, which is what LightGBM and CatBoost do and
is why this module stopped being the slowest of the three at inference. The
rewrite is exact (edges are strictly increasing and bin b is the half-open
interval `(e[b-1], e[b]]`, so `bin(v) <= T` if and only if `v <= e[T]`, with
missing handled separately because `NaN <= edge` is false), and the shapes
it cannot cover -- categorical splits, categorical features, CTR columns,
linear leaves, and fewer rows than `predict.RAW_MIN_ROWS` -- still bin the
whole matrix once and walk bin ids. `MOJOTREES_RAW_PREDICT=0` forces that
older path for every model. `predict_range` and `predict_proba_range`, which
predate the device vocabulary, do it by calling `Model.predict_batch` and
`MulticlassModel.predict_batch` with an explicit `CPU_DEVICE`; the `_batch`
entry points below them call the same two with the device the caller named,
reach the device walk in gpu_predict.mojo, and report which backend ran
rather than leaving a caller to assume. So there is one dense prediction
path, and where it runs is decided in one place, `resolve_device` in
device.mojo, which nothing here second-guesses.

Nothing that can raise may appear inside a parallel block: `dispatch_rows`
takes a non-raising `def (Int, Int) -> None`. `BinMapper.bin_row` raises,
and `Model.predict_range` and `Model.predict_raw_range` inherit it because
binning is their first statement, so a loop that bins per row cannot be
parallelized at all. Binning the matrix up front is what makes the fan-out
legal, and it keeps the validation rather than dropping it: `transform`
checks the whole matrix's shape once where `bin_row` checked each row's
shape `n_rows` times, which is the same question because every row of a
column-major matrix has `n_features` entries by construction.

The row axis is the only axis any of them splits, and that is a
correctness statement rather than a preference. A row reads the ensemble,
which no block writes, and writes its own output slots, which no other row
touches; nothing is accumulated across rows, so the per-row body runs on
exactly the values it ran on serially and the outputs are bit-identical at
every block count and at every `MOJOTREES_NUM_WORKERS`. Splitting the TREE
axis instead would reassociate a row's Float64 sum over trees and move the
last bits, so it is not done anywhere here.
"""

from std.os import abort
from std.sys import has_accelerator
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

# `Dataset` construction beyond the dense case, and the reads that answer
# from a constructed one. They live in their own module because they are a
# coherent group and this file is long; they are registered here, in the one
# `PythonModuleBuilder`, because that is the only place a name becomes
# reachable from Python.
from binding_support import (
    csc_from_params,
    csr_from_params,
    f64_buffer,
    f64_view,
    f64_view_mut,
    int_buffer,
    int_buffer_from_f64,
)
from sequence_bindings import (
    ChunkAccumulator,
    dataset_chunks_begin,
    dataset_chunks_finish,
    dataset_chunks_num_data,
    dataset_chunks_push,
)
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
    distributed_gpu_status,
    distributed_status_message,
    train_local_world,
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
from lgbm_bindings import (
    lgbm_export_file,
    lgbm_file_unsupported_reason,
    lgbm_import_file,
    lgbm_interop_status,
)
from model_editing_bindings import (
    get_leaf_output,
    get_leaf_output_multiclass,
    model_editing_status,
    refit,
    refit_multiclass,
    rollback_one_iter,
    rollback_one_iter_multiclass,
    rollback_to,
    rollback_to_multiclass,
    score_bounds,
    score_bounds_multiclass,
    set_leaf_output,
    set_leaf_output_multiclass,
    shuffle_models,
    shuffle_models_multiclass,
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
# The four capabilities that were built and reachable from nothing: ONNX
# export, MultiRMSE, text features, embedding features. See catalog A31 and
# the module docstring of catboost_reach_bindings.mojo.
from catboost_reach_bindings import (
    embedding_feature_count,
    embedding_features_into,
    multi_rmse_fit,
    multi_rmse_predict,
    multi_rmse_shape,
    onnx_export_refusals,
    onnx_export_refusals_multiclass,
    onnx_plan_text,
    onnx_plan_text_multiclass,
    text_features_open,
    text_features_shape,
    text_features_write,
)

from mojotrees.auto_learning_rate import (
    AUTO_LR_TASK_CPU,
    AUTO_LR_TASK_GPU,
    AutoLearningRateParams,
    catboost_boost_from_average_default,
    resolve_learning_rate,
)
from mojotrees.objective_registry import (
    CUSTOM as _CUSTOM_OBJECTIVE,
    LAMBDARANK as _LAMBDARANK_OBJECTIVE,
    MULTICLASS as _MULTICLASS_OBJECTIVE,
)
from mojotrees.bagging import BaggingParams
from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.contrib import ContribExplainer
from mojotrees.boosting import (
    BoosterParams,
    IterationRange,
    _softmax_inplace,
    catboost_leaf_estimation_iterations,
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
    has_infinite,
    transpose_to_column_major,
    train_dataset as mojo_train_dataset,
    train_dataset_multiclass as mojo_train_dataset_multiclass,
    train_dataset_ranker_advanced as mojo_train_dataset_ranker_advanced,
    update_dataset as mojo_update_dataset,
    update_dataset_multiclass as mojo_update_dataset_multiclass,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.ctr_columns import (
    CTR_SOURCE_ONE_HOT_MAX_SIZE,
    CTR_TARGET_BORDER_MIN_ENTROPY,
    CTR_TARGET_BORDER_MULTICLASS,
    SimpleCtrConfig,
)
from mojotrees.linear_tree import LinearParams
from mojotrees.ordered_boosting import OrderedBoostingParams
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
from mojotrees.sampling import (
    BootstrapParams,
    BootstrapRequest,
    canonical_bootstrap_type,
    check_mvs_bagging_temperature,
)
from mojotrees.alternate_boosting import (
    BOOSTING_DART,
    BOOSTING_RF,
    AlternateBoostingParams,
    boosting_name,
    fit_boosting,
    parse_boosting,
)
from mojotrees.boosting_dart import DartParams
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
from mojotrees.model import fit_custom as mojo_fit_custom
from mojotrees.model import fit_multiclass as mojo_fit_multiclass
from mojotrees.objective import mean_label
from mojotrees.parallel import dispatch_rows
from mojotrees.ranking_advanced import (
    AdvancedRankParams,
    LabelGain,
    PositionMap,
    advanced_ranking_requested,
    fit_ranker_advanced as mojo_fit_ranker_advanced,
    positions_from_codes,
)
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

# CatBoost's `score_function`. Parsed here rather than in
# `extra_params_from_mapping` for the reason `leaf_estimation_iterations` is:
# that parser is also `extra_params_check`'s, and this refusal needs the
# entry point, which only `_parse_params` knows.
from mojotrees.tree_parameters_extra import (
    CATBOOST_RANDOM_STRENGTH,
    SCORE_L2,
    derivative_precision_name,
    parse_derivative_precision,
    parse_score_function,
    score_function_name,
)


@export
def PyInit__mojotrees() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojotrees")
        _ = m.add_type[Model]("Model")
        _ = m.add_type[MulticlassModel]("MulticlassModel")
        _ = m.add_type[Dataset]("Dataset")
        _ = m.add_type[GpuValidation]("GpuValidation")
        # From sequence_bindings.mojo: LightGBM's Sequence path, a Dataset
        # built from row-major batches pushed one at a time.
        _ = m.add_type[ChunkAccumulator]("ChunkAccumulator")
        m.def_function[dataset_chunks_begin]("dataset_chunks_begin")
        m.def_function[dataset_chunks_push]("dataset_chunks_push")
        m.def_function[dataset_chunks_num_data]("dataset_chunks_num_data")
        m.def_function[dataset_chunks_finish]("dataset_chunks_finish")
        m.def_function[dataset_create]("dataset_create")
        # Ingestion: the transpose every C-ordered caller pays before
        # binning, and the infinity check that used to be a second pass.
        m.def_function[ingest_column_major]("ingest_column_major")
        m.def_function[buffer_has_infinite]("buffer_has_infinite")
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
        # -- LightGBM model-file interop (lgbm_bindings.mojo) -------------
        m.def_function[lgbm_interop_status]("lgbm_interop_status")
        m.def_function[lgbm_file_unsupported_reason](
            "lgbm_file_unsupported_reason"
        )
        m.def_function[lgbm_import_file]("lgbm_import_file")
        m.def_function[lgbm_export_file]("lgbm_export_file")
        # -- editing a fitted model (model_editing_bindings.mojo) ---------
        m.def_function[model_editing_status]("model_editing_status")
        m.def_function[rollback_one_iter]("rollback_one_iter")
        m.def_function[rollback_one_iter_multiclass](
            "rollback_one_iter_multiclass"
        )
        m.def_function[rollback_to]("rollback_to")
        m.def_function[rollback_to_multiclass]("rollback_to_multiclass")
        m.def_function[get_leaf_output]("get_leaf_output")
        m.def_function[get_leaf_output_multiclass](
            "get_leaf_output_multiclass"
        )
        m.def_function[set_leaf_output]("set_leaf_output")
        m.def_function[set_leaf_output_multiclass](
            "set_leaf_output_multiclass"
        )
        m.def_function[shuffle_models]("shuffle_models")
        m.def_function[shuffle_models_multiclass]("shuffle_models_multiclass")
        m.def_function[refit]("refit")
        m.def_function[refit_multiclass]("refit_multiclass")
        m.def_function[score_bounds]("score_bounds")
        m.def_function[score_bounds_multiclass]("score_bounds_multiclass")
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
        m.def_function[distributed_gpu_status]("distributed_gpu_status")
        m.def_function[distributed_train_local]("distributed_train_local")
        # -- startup diagnostics -----------------------------------------
        m.def_function[startup_phase_contract]("startup_phase_contract")
        m.def_function[startup_environment]("startup_environment")
        m.def_function[native_clock_ns]("native_clock_ns")
        # -- built but unreached until now (catboost_reach_bindings.mojo) --
        m.def_function[onnx_plan_text]("onnx_plan_text")
        m.def_function[onnx_plan_text_multiclass](
            "onnx_plan_text_multiclass"
        )
        m.def_function[onnx_export_refusals]("onnx_export_refusals")
        m.def_function[onnx_export_refusals_multiclass](
            "onnx_export_refusals_multiclass"
        )
        m.def_function[multi_rmse_fit]("multi_rmse_fit")
        m.def_function[multi_rmse_shape]("multi_rmse_shape")
        m.def_function[multi_rmse_predict]("multi_rmse_predict")
        m.def_function[text_features_open]("text_features_open")
        m.def_function[text_features_shape]("text_features_shape")
        m.def_function[text_features_write]("text_features_write")
        m.def_function[embedding_feature_count]("embedding_feature_count")
        m.def_function[embedding_features_into]("embedding_features_into")
        return m.finalize()
    except e:
        abort(String("failed to create _mojotrees module: ", e))


def ingest_column_major(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
) raises -> PythonObject:
    """Transpose a caller's row-major float64 matrix into a column-major one.

    `src_addr` is a C-ordered `(n_rows, n_features)` float64 buffer, which is
    what NumPy hands out by default, and `dst_addr` is a Fortran-ordered
    buffer of the same shape that the caller has already allocated. Returns 1
    when any value was `+inf` or `-inf` and 0 otherwise; `NaN` is the
    missing-value marker and is not reported.

    This exists because ingestion used to be two serial NumPy passes over the
    whole matrix and is now one parallel tiled pass. `np.asfortranarray`
    transposes on one thread with one side of the copy strided, and
    `np.isinf(Xa).any()` then reads the result again and materializes an
    `n_rows * n_features` byte array to reduce. At 1,000,000 x 50 that second
    pass is 400 MB read and 50 MB allocated to answer a question this one
    answers from a register.

    Neither the values nor their positions differ from what NumPy produced. A
    transpose does no arithmetic, so there is nothing here for a worker count
    to reassociate; see `trainset.transpose_to_column_major`.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var n = nr * nf
    var found = transpose_to_column_major(
        f64_view(Int(py=src_addr), n),
        f64_view_mut(Int(py=dst_addr), n),
        nr,
        nf,
    )
    return PythonObject(1 if found else 0)


def buffer_has_infinite(
    addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """1 when the float64 buffer holds `+inf` or `-inf`, 0 otherwise.

    The half of `ingest_column_major` that a caller whose matrix is already
    column-major needs on its own: that matrix is binned where it lies and
    never transposed, but it still has to be rejected if it holds an
    infinity. Parallel, and it allocates nothing, which is the difference
    from `np.isinf(buf).any()`.
    """
    return PythonObject(
        1 if has_infinite(f64_view(Int(py=addr), Int(py=n))) else 0
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
    var flat = int_buffer_from_f64(
        flat_addr, Int(py=params["interaction_flat_len"])
    )
    var offsets = int_buffer_from_f64(
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
        int_buffer_from_f64(addr, n_features), n_features
    )


# The reason the three eval-set entry points cannot derive CatBoost's rate.
# Shared because it is one reason and not three: what disqualifies them is
# the eval set itself, which is the only thing all three have in common and
# is exactly the input CatBoost's `use_best_model` is resolved from.
comptime _NO_CATBOOST_DEFAULTS = -1
"""`_parse_params`'s "this entry point resolves no CatBoost-mode default".

Negative because every objective code in `objective_registry` is nonnegative,
so no loss can ever collide with it and no call site can mean it by accident.
It is a distinct sentinel from `auto_lr_objective`'s `CUSTOM` default on
purpose: `CUSTOM` is a real objective that `fit_custom` really passes, so it
cannot also carry "nothing was declared here" without the two meanings landing
on one number.
"""

comptime _AUTO_LR_EVAL_SET_REASON = (
    "an eval_set is what makes CatBoost's use_best_model resolvable"
    " (UpdateUseBestModel, options_helper.cpp:100-113, which forces it false"
    " only when there is no eval set), and use_best_model is one of the four"
    " keys of the coefficient table, so with an eval set present the row to"
    " read is not the row a plain fit reads. mojotrees has no use_best_model"
    " parameter to resolve, so this would be a rate derived from a guess"
)

# The reason continued training cannot derive it either. `n_estimators` on a
# `booster_update` call is how many rounds to ADD, and CatBoost's formula
# reads `IterationCount`, the length of the whole run: deriving from the
# increment would give a 20-round top-up a rate fitted for a 20-round model.
comptime _AUTO_LR_CONTINUED_REASON = (
    "continued training adds n_estimators rounds to a model that already"
    " has some, and CatBoost's derivation reads the iteration count of the"
    " whole run (options_helper.cpp:272), which this call does not know."
    " The trees already in the model were grown at their own rate as well,"
    " so a rate derived here would apply to part of an ensemble. Derive it"
    " on the first fit, or pass an explicit learning_rate"
)


def _parse_params(
    params: PythonObject,
    n_features: Int,
    unbundled: String = "",
    cpu: Bool = True,
    entry: String = "",
    ordered_ok: Bool = False,
    leaf_estimation_ok: Bool = False,
    boost_from_average_ok: Bool = False,
    score_function_ok: Bool = False,
    random_strength_ok: Bool = False,
    derivative_precision_ok: Bool = False,
    auto_lr_ok: Bool = False,
    auto_lr_reason: String = "",
    auto_lr_rows: Int = 0,
    auto_lr_objective: Int = _CUSTOM_OBJECTIVE,
    catboost_defaults_objective: Int = _NO_CATBOOST_DEFAULTS,
    ctr_ok: Bool = False,
    boosting_ok: Bool = False,
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

    **`ordered_ok`, `leaf_estimation_ok`, `score_function_ok` and
    `random_strength_ok` are the same declaration for four CatBoost
    mechanisms, and they default to `False` on purpose.** The
    default of a reachability flag is the direction a forgotten call site
    fails in, and this repository has now shipped five mechanisms that were
    built and never reached because the default was "accept". A new entry
    point added below inherits the refusal and says so by name; it does not
    inherit a silent drop. `entry` is the name those refusals use, and falls
    back to `unbundled` when an entry point already declared one there.

    **A flag set at the one call site somebody looked at is the same defect
    wearing a different name**, and `random_strength_ok` has already been
    that once: it was True at `fit` alone while `bench/real_data` trains
    through `train_dataset`, so the benchmark arm lost the parameter. The
    rule for every flag here is that the verdict is a property of the
    ROUTING, not of the entry point's name, and an entry point whose routing
    forks (`fit` forks three ways on the CPU) needs the fork settled too.

    - `ordered_ok`: CatBoost's `boosting_type=Ordered`
      (src/mojotrees/ordered_boosting.mojo). Only `boosting.train` runs the
      fold ladder, so only `fit` on the CPU may pass True.
      `boosting.train_with_valid` and the multiclass trainers refuse it
      themselves with `check_ordered_honored`; the sparse, custom-objective,
      ranking and GPU trainers do not, which is why the refusal is here.
    - `leaf_estimation_ok`: CatBoost's `leaf_estimation_iterations`. Honored
      by `boosting.train`, `boosting.train_more`, `boosting.train_with_valid`,
      `train_gpu.train_gpu` and `train_gpu.train_gpu_with_valid`, and by
      nothing else. `fit_with_metrics` does not qualify, and the near miss is
      worth recording: it routes to `custom_metric.fit_with_metrics`, which is
      a different round loop from `boosting.train_with_valid` and reads the
      field nowhere.

      **This flag was True at `fit` alone until 2026-08-16, and that was the
      same defect `random_strength_ok` had before it**: `bench/real_data`
      trains through `mojotrees.train(params, Dataset)`, which is
      `train_dataset` below, so on every CatBoost-mode Logloss cell CatBoost
      took ten Newton steps per leaf and we took one -- while a comparison
      table transcribed from an RMSE fit, where CatBoost also takes one,
      reported the two engines as agreeing. The flag now follows the routing at
      all fifteen call sites, and the verdicts are:

      - `fit` -- True. Its plain fork reaches `boosting.train` or
        `train_gpu.train_gpu`. Its other two forks are refused by name below,
        beside `ordered`'s and `random_strength`'s: `alternate_boosting.
        fit_boosting` (dart and rf) and `custom_metric.fit_with_metrics`
        (`linear_tree`) run their own round loops and read the field nowhere.
      - `train_dataset` -- True on the dense arms only (`not is_sparse`).
        `trainset.train_dataset` forks three ways and two of them qualify:
        the dense CPU arm is `boosting.train` and the GPU arm is
        `train_gpu.train_gpu`. The sparse arm is `boosting_sparse.
        train_sparse`, which does not implement the extra steps, so the fork
        is settled here rather than left to the trainer. Note this is a
        WIDER condition than `random_strength_ok`'s `scale_is_computed` two
        lines above it, and the difference is real: the per-tree noise scale
        is computed by the dense CPU loops alone, while the extra Newton
        steps are implemented on the device too.
      - `booster_update` -- True, unconditional. `trainset.update_dataset`
        refuses a sparse dataset by name, takes no device argument, and calls
        `boosting.train_more`, which is `_boost_rounds` with a round offset
        and reads the field at `boosting.mojo:2345`. Unconditional rather
        than a sparse test for the reason its `random_strength_ok=True` is:
        a sparse continued fit should get `update_dataset`'s own message
        about continued training, not a message about leaf estimation.
      - `fit_with_metrics`, `fit_multiclass_with_metrics`,
        `fit_ranker_with_metrics` -- False. All three land in
        `custom_metric`'s round loops, none of which reads the field.
      - `fit_multiclass`, `train_dataset_multiclass`,
        `booster_update_multiclass` -- False. `boosting.train_multiclass`,
        `train_multiclass_gpu` and `boosting_sparse.train_multiclass_sparse`
        read the field nowhere. Nothing is lost by this: CatBoost resolves
        MultiClass to **1** as well (`catboost_options.cpp:106-112`; the 10
        in that block is the Gradient slot and is not the default), so the
        refusal fires only on a value CatBoost would not have chosen either.
      - `fit_csc`, `fit_multiclass_csc` -- False. `boosting_sparse` and
        `train_gpu_sparse` do not implement it; `train_gpu_sparse` already
        refuses it itself at `train_gpu_sparse.mojo:242`.
      - `fit_ranker`, `train_dataset_ranker` -- False. `ranking.train_ranker`
        reads the field nowhere. CatBoost resolves `LambdaMart` to 1
        (`catboost_options.cpp:199-205`), so again the refusal fires only
        above CatBoost's own value.
      - `fit_custom` -- False. A callback objective is a pair of derivative
        buffers; there is no loss to re-evaluate a leaf's rows at, which is
        what an extra Newton step is. `train_gpu` refuses the same pair at
        `train_gpu.mojo:3759`.
      - `distributed_train_local` -- False. `distributed_strategies` grows
        and shrinks its own trees and never calls
        `boosting._estimate_leaf_values`.
    - `boost_from_average_ok`: LightGBM's and CatBoost's
      `boost_from_average`. Only `False` is refusable, because `True` is what
      every trainer in this package does and has always done
      (`boosting._base_score`, called unconditionally by all fourteen of its
      callers before 2026-08-16). So this flag decides who may start from
      **zero**, and the honoring trainers are the four that thread the value
      into that call: `boosting.train`, `boosting.train_with_valid`,
      `train_gpu.train_gpu` and `train_gpu.train_gpu_with_valid`.

      **Its verdicts are the same as `leaf_estimation_ok`'s except at
      `booster_update`, and that exception is the reason it is a second flag
      rather than a reuse of the first.** `trainset.update_dataset` reaches
      `boosting.train_more`, which reads
      `leaf_estimation_iterations` and so may honor it, but which starts from
      the base score already stored on the model and never calls
      `boosting._base_score` at all. A `False` accepted there would be
      accepted and ignored, which is exactly the defect the rest of this
      docstring exists to prevent, so `booster_update` passes
      `leaf_estimation_ok=True` and `boost_from_average_ok` at its default.
      That leaves True at two call sites: `fit` (plain fork; its dart, rf and
      linear forks are refused by name below) and `train_dataset` on its
      dense arms.

      A continued fit therefore takes the ORIGINAL fit's starting point,
      which is the only coherent answer -- half an ensemble cannot start
      somewhere else -- and naming the parameter on an update says so instead
      of appearing to work.
    - `score_function_ok`: CatBoost's `score_function`
      (src/mojotrees/split.mojo, `SCORE_COSINE`). Honored by every trainer
      that elects a split through `tree._search` or
      `tree._grow_oblivious_levels`, which is every grower in this package
      except the distributed prototype: `tree.grow_tree`,
      `tree_sparse.grow_tree_sparse`, `train_gpu` and
      `train_gpu_sparse.grow_tree_gpu_sparse` all route through those two,
      and the device split searches decline or refuse on
      `ExtraTreeParams.is_active()`, which names the field. The one entry
      point that must pass False is `distributed_train_local`:
      `distributed_strategies` calls `split.find_best_split` itself, at its
      `SCORE_L2` default, so a Cosine fit there would be an L2 tree under a
      Cosine label.
    - `random_strength_ok`: CatBoost's `random_strength`
      (src/mojotrees/tree_parameters_extra.mojo). The per-split draw is
      implemented on both backends; what is scarce is the **per-tree scale**,
      `random_score_scale_from_gradients`, and exactly two round loops in the
      whole package compute it: `boosting._boost_rounds` (boosting.mojo:2593)
      and `boosting.train_with_valid`'s own loop (boosting.mojo:3196), both
      through `boosting._round_random_score_scale`. Every entry point that
      may pass True has to reach one of those two. That is `model.fit` on the
      CPU with plain gbdt and no `linear_tree` (the dart/rf and linear forks
      are refused below by name), `trainset.train_dataset` on its dense CPU
      arm, and `trainset.update_dataset`, which is CPU-and-dense by
      construction and routes to `boosting.train_more`. Multiclass, sparse,
      ranking, custom-objective, custom-metric, distributed and every device
      loop compute no scale and keep the refusal.
    - `derivative_precision_ok`: the precision a per-row gradient and hessian
      is CARRIED at (`src/mojotrees/histogram.mojo`,
      `DERIVATIVE_PRECISION_FLOAT32` / `_FLOAT64`). Only `float64` is
      refusable, for the reason only `boost_from_average=false` is: `float32`
      is what every trainer in this package does by default, so this flag
      decides who may ask for the WIDE derivative rather than who may ask for
      anything at all.

      The honoring trainers are the ones that thread
      `ExtraTreeParams.wants_float64_derivatives()` into
      `boosting._fill_grad_hess` or `boosting._fill_softmax_grad_hess`, and
      that is nearly all of them: `boosting` (`_boost_rounds`,
      `train_with_valid`, `_boost_rounds_multiclass`,
      `train_multiclass_with_valid`), `boosting_sparse` (all four loops),
      `boosting_rf`, `alternate_boosting` (dart and rf, single and
      multiclass), `custom_metric.train_with_callbacks` and
      `custom_metric.train_multiclass_with_metrics`, and
      `distributed.train_distributed_run`, which additionally folds the
      resolved value into its schema marker
      (`distributed._push_derivative_precision`). The two device growers
      refuse `float64` themselves, by name and at every shape, through
      `histogram.check_device_derivative_precision`: gradients reach an
      accelerator as Float32 and there is no Float64 there, so a GPU fork of
      an entry point that passes True here still raises rather than training
      the narrow answer under the wide label.

      **The four call sites that pass False are the ranking ones and the
      custom-objective one, and neither is a gap with a scheduled exit.**
      `ranking.train_ranker` computes its lambdas in Float64 and forwards no
      `float64_derivatives` to anything, so the value the caller typed
      selects nothing there; `custom_metric.train_ranker_with_metrics` is the
      same loop with a metric set around it. `fit_custom`'s derivatives are
      the caller's own buffers, and there is no narrowing site under this
      parameter's control between them and the histogram. Accepting `float64`
      at any of the four would be accepted-and-ignored, which is the defect
      the whole of this docstring exists to prevent.

      **There is no reachability question in the other direction and there
      cannot be**, because `float32` is the default: a caller who wants the
      narrow derivative on a ranking fit has no way to ask for it and no way
      to be told. That asymmetry is a property of the ranking loops and not of
      this flag.
    - `auto_lr_ok`: CatBoost's automatic `learning_rate`
      (src/mojotrees/auto_learning_rate.mojo, catalog A12/A38). Unlike the
      four above, this one needs **data** and not just a declaration: the
      derived rate is a function of the objective, the iteration count and
      the **train row count**, so an entry point that may honor it has to
      hand over `auto_lr_rows` and `auto_lr_objective` as well. `auto_lr_ok`
      alone would be a promise with nothing behind it, which is why the row
      count defaults to 0 and the objective to `CUSTOM`: a call site that
      sets the flag and forgets the data derives a rate from no rows, and
      `catboost_auto_learning_rate` raises on that rather than returning
      something.
      Every trainer applies `BoosterParams.learning_rate` the same way, so
      what decides the verdict here is not the round loop but whether the
      three inputs exist and mean what CatBoost means by them. Six entry
      points cannot supply them and refuse by name through `auto_lr_reason`:
      `fit_custom` (a callback objective is not a loss function, so there is
      no `ETargetType` to look up), the three `*_with_metrics` fits (an eval
      set is exactly what makes CatBoost's `use_best_model` resolvable, and
      `use_best_model` selects a different coefficient row; we have no
      parameter to resolve it from), and `booster_update` /
      `booster_update_multiclass` (`n_estimators` there is the increment,
      not the `IterationCount` the formula reads, and the trees already in
      the model were grown at another rate).
    - `auto_lr_reason` is the sentence such a refusal ends with. It is a
      per-call-site argument rather than a table here because the reason
      differs by entry point and a shared message would have to be vague
      enough to be true of all six.
    - `catboost_defaults_objective` is the objective code the CatBoost-mode
      per-objective defaults are resolved for, and `_NO_CATBOOST_DEFAULTS`
      means "this entry point resolves none". It is a separate argument from
      `auto_lr_objective`, which carries the same number at eight call sites,
      and the duplication is deliberate: `auto_lr_objective` defaults to
      `CUSTOM`, which is a legitimate value for it (`fit_custom` really does
      have a custom objective), so a mode-default resolver reusing it would
      silently resolve five entry points as CUSTOM instead of declining. Two
      arguments with two sentinels is the only shape in which a forgotten call
      site declines rather than resolves the wrong loss.

    - `ctr_ok`: CatBoost's ordered target statistics
      (src/mojotrees/ctr_columns.mojo, catalog A19/A36). The odd one out here,
      because what it declares is not which round loop runs but **which
      BINNING runs**: a CTR column is built from the label and a fixed
      permutation while the matrix is binned, and the only type in this
      package that binds a bundle to a matrix is `trainset.Dataset`. So the
      entry points that may pass True are the ones that can hand the bundle to
      a `Dataset`, and that is exactly one: `fit`, on its plain fork, which
      reroutes through `trainset.Dataset` + `trainset.train_dataset` when the
      bundle is active and takes the untouched `model.fit` call when it is
      not. Its dart/rf and linear forks are refused by name below.

      Every other entry point is False, and the reason is uniform rather than
      per-trainer: `model.fit`, `model.fit_multiclass`, `fit_csc`,
      `fit_ranker`, `fit_custom`, the three `*_with_metrics` fits and
      `distributed_train_local` all bin a raw matrix through `binning.fit_bins`
      and take no `SimpleCtrConfig`. The `train_dataset*` and `booster_update*`
      entry points are a different case, and until 2026-08-17 they were wrongly
      lumped in with the ones above. Their dataset was binned before they were
      called, so its bundle is already decided -- but "already decided" is not
      "absent", and the question `ctr_ok` asks is whether the CTR columns EXIST
      for this fit, not whether this entry point built them. `train_dataset`
      therefore passes `ctr_ok=d[].ctr.is_active()`: True when the dataset it
      was handed really carries the columns, which is exactly when honoring the
      key is truthful. A `ctr` key that disagrees with the dataset's own rule
      is a second answer arriving too late and gets its own refusal at that
      call site naming both rules. `Dataset(params={"ctr": ...})` remains where
      the rule is set, and that door is unchanged.

      **A mode default is only ever applied where the value can be honored.**
      `leaf_estimation_iterations` resolves through
      `boosting.catboost_leaf_estimation_iterations`, so the objectives whose
      CatBoost value is above 1 (`Logloss` 10, `CrossEntropy` 10, `Poisson` 10)
      only reach it at the three call sites that also pass
      `leaf_estimation_ok`; `boost_from_average` resolves through
      `auto_learning_rate.catboost_boost_from_average_default`, whose False
      answers only reach the same three; `random_strength` resolves to
      `tree_parameters_extra.CATBOOST_RANDOM_STRENGTH` and reaches only the
      three call sites that pass `random_strength_ok`, which is a NARROWER set
      again, because the per-tree scale it multiplies is computed by two round
      loops and not by five. Everywhere else the value stays at the
      LightGBM default this package has always had, and an EXPLICIT request is
      refused by name below.

      **All three are supplied with CatBoost's own `SetDefault` semantics**
      (`option.h:27-33`: assign the value, do not raise `IsSetFlag`). That is
      what lets `l2_leaf_reg = 3` and a per-objective
      `leaf_estimation_iterations` be supplied by CatBoost mode WITHOUT
      closing the automatic-learning-rate gate they are two of the four keys
      of (`options_helper.cpp:276-281`). The provenance flags this function
      reads -- `leaf_estimation_iterations_set`, `boost_from_average_set`,
      `random_strength_set`, and `auto_learning_rate_l2_set` further down --
      are all written by the estimator from `is not None` tests on what the
      CALLER passed, and no mode default writes any of them. Without that rule
      the mode would advertise a derived learning rate and ship the constant,
      which was live in this repository until 2026-08-16. That is the difference between a default and a
      request, and it is the same line `auto_lr_ok` / `auto_lr_required` draw:
      an inherited mode default an entry point cannot honor declines in
      silence, because that is what CatBoost itself does when its own table has
      no row; a value the caller typed is refused.
    """
    var who = entry.copy()
    if who.byte_length() == 0:
        who = unbundled.copy()
    if who.byte_length() == 0:
        who = String("this entry point")
    # CatBoost's `leaf_estimation_iterations`, folded onto the bundle
    # `extra_params_from_mapping` parsed rather than parsed there, because
    # that function is also `extra_params_check`'s and this refusal needs the
    # entry point, which only this function knows. Until this was passed the
    # field took its default of 1 on every fit that came through Python, so
    # `boosting._estimate_leaf_values` was reachable from the Mojo API and
    # from nowhere else -- `params.mojo` refuses the key on the string
    # surface for exactly the reason handled here, that a string reaches
    # trainers that do not implement it.
    # `boosting`, refused here rather than silently dropped. Added 2026-08-17.
    #
    # THE DEFECT THIS CLOSES. `params["boosting"]` was read at exactly one
    # place in this file, inside `_parse_boosting`, whose only caller is `fit`.
    # Every other entry point, `train_dataset`, `train_dataset_multiclass`,
    # `train_dataset_ranker`, `booster_update` and `booster_update_multiclass`,
    # took the key on the wire and dispatched no alternate loop, so
    # `mojotrees.train({'boosting': 'dart'}, ds)` trained a plain gbdt model
    # and said nothing about it.
    #
    # `rf` was worse than dart, because two wrongs did not cancel. The
    # estimator layer forces `learning_rate = 1.0` under `rf`, which is correct
    # for a forest and wrong for anything else, so a dropped `rf` left a
    # BOOSTED fit running at rate 1.0. Neither the mode the caller asked for
    # nor the rate they would have chosen survived.
    #
    # This is the route `bench/real_data` trains through. No arm sets
    # `boosting` today, so no published number is affected, which is luck
    # rather than design and is exactly the kind of luck this refusal removes.
    #
    # Refusing rather than dispatching, deliberately: five trainers have no
    # dart or forest loop at all, so wiring the mode here would mean building
    # five loops, while refusing it costs nothing and cannot mistrain anybody.
    # The rule is `docs/COMPATIBILITY_POLICY.md` and the shape is the
    # `derivative_precision` refusal below: a value the caller TYPED is
    # refused, never accepted and ignored.
    #
    # `fit` is unaffected. It calls `_parse_boosting` itself and passes
    # `boosting_ok=True`, so the one entry point that honors the key keeps
    # honoring it.
    if not boosting_ok:
        var boost_mode = parse_boosting(String(py=params["boosting"]))
        if boost_mode == BOOSTING_DART or boost_mode == BOOSTING_RF:
            raise Error(
                "boosting='",
                boosting_name(boost_mode),
                "' is not honored by ",
                who,
                ": the dart and forest round loops live in"
                " alternate_boosting.fit_boosting, which only the estimator"
                " entry point reaches. This entry point would have trained a"
                " plain gbdt model and reported nothing, and under 'rf' it"
                " would also have kept the learning_rate of 1.0 that the"
                " estimator layer sets for a forest. Use"
                " MojoTreesRegressor(boosting='",
                boosting_name(boost_mode),
                "') or MojoTreesClassifier, which reach that loop, or drop"
                " the key to train the gbdt model this entry point builds",
            )
    var extra = extra_params_from_mapping(params, n_features)
    # CatBoost mode resolves this per objective; `lossguide` does not resolve
    # it at all. `catboost_defaults` is 1 when the estimator's grow policy is
    # `symmetrictree` AND the caller did not name a value, which is the same
    # "unset is provenance, not a value" rule `auto_learning_rate` already
    # keeps: CatBoost's own gate reads `TOption::NotSet()` and not a
    # comparison against the default (`option.h:80-85`), so a caller who types
    # 1 has typed something and a caller who types nothing has not.
    var catboost_defaults = Int(py=params["catboost_mode_defaults"]) != 0
    var leaf_iters_named = (
        Int(py=params["leaf_estimation_iterations_set"]) != 0
    )
    extra.leaf_estimation_iterations = Int(
        py=params["leaf_estimation_iterations"]
    )
    if (
        catboost_defaults
        and not leaf_iters_named
        and leaf_estimation_ok
        and catboost_defaults_objective != _NO_CATBOOST_DEFAULTS
    ):
        # The mode default. Gated on `leaf_estimation_ok` as well as on the
        # mode, so a CatBoost-mode fit that landed on an entry point which
        # cannot take the extra steps keeps 1 and trains, rather than
        # resolving a 10 this function would then have to refuse. An inherited
        # default that an entry point cannot honor declines; only a value the
        # caller typed is refused. That is `auto_lr_required`'s rule, applied
        # to the second parameter that has a mode default.
        extra.leaf_estimation_iterations = (
            catboost_leaf_estimation_iterations(catboost_defaults_objective)
        )
    # CatBoost's `random_strength`, the third parameter with a CatBoost-mode
    # default and the third to be resolved HERE rather than in the estimator,
    # for the reason the two above it are: the mode default is only applied
    # where the value can be honored, and whether it can is a property of the
    # routing, which is what `random_strength_ok` carries.
    #
    # CatBoost ships 1.0 and mojotrees ships 0.0 (LightGBM's behavior, since
    # LightGBM has no such parameter), so the estimator sends both the value
    # and whether anybody typed it. A caller who types 0.0 has turned the
    # noise off and keeps it off in either mode; a caller who types nothing
    # gets 1.0 under `symmetrictree` and 0.0 under `lossguide`.
    #
    # Gated on `random_strength_ok` so that a CatBoost-mode fit which landed
    # on a loop that computes no per-tree scale keeps 0.0 and TRAINS, rather
    # than inheriting a 1.0 that `ExtraTreeParams.check_random_strength` would
    # then refuse. An inherited default an entry point cannot honor declines;
    # only a value the caller typed is refused, and a typed one still is,
    # below and in `check_scalars`.
    var strength_named = Int(py=params["random_strength_set"]) != 0
    if (
        catboost_defaults
        and not strength_named
        and random_strength_ok
        and catboost_defaults_objective != _NO_CATBOOST_DEFAULTS
    ):
        extra.random_strength = CATBOOST_RANDOM_STRENGTH

    if extra.leaf_estimation_active() and not leaf_estimation_ok:
        raise Error(
            "leaf_estimation_iterations > 1 is not implemented by ",
            who,
            "; it is implemented by boosting.train, boosting.train_more,"
            " boosting.train_with_valid, train_gpu.train_gpu and"
            " train_gpu.train_gpu_with_valid, which this estimator reaches"
            " through a dense, single-output fit without a custom objective."
            " 1 is LightGBM's behavior and mojotrees's, and is the default",
        )

    # CatBoost's ordered target statistics, catalog A19/A36. The verdict is
    # taken HERE, beside the other four, so the whole enumeration is one table
    # and a new entry point inherits a refusal rather than a silent drop.
    #
    # Unlike the four above, this one is not a field on `BoosterParams`: a CTR
    # bundle is a property of the BINNING, so it is applied where the dataset
    # is built and this function only decides whether the entry point about to
    # bin can carry one. `_parse_ctr` turns the same key into the bundle.
    #
    # The mode default is resolved in the estimator rather than here, and that
    # used to be the one difference from `leaf_estimation_iterations`: `ctr`
    # crosses the wire as a RULE NAME and not as a number, so "auto" arriving
    # from a CatBoost-mode default and "auto" typed by a caller were the same
    # six bytes, and the paragraph that stood here argued the provenance was
    # therefore not worth carrying because "both are refused identically".
    #
    # **THAT ARGUMENT WAS WRONG, AND IT COST EVERY CATBOOST-MODE FIT THAT DOES
    # NOT GO THROUGH `fit`.** Refusing both identically is exactly what a mode
    # default must not do, and the rule is already written down two parameters
    # up and in `sklearn.py:3195-3202`, for `random_strength`: "an inherited
    # default an entry point cannot honor must decline rather than refuse."
    # `random_strength` obeys it by sending `random_strength_set` and letting
    # this function apply the mode default only where the flag says the caller
    # was silent. `ctr` did not send that flag, so `grow_policy='symmetrictree'`
    # resolved `ctr='catboost'` in the estimator, every entry point but `fit`
    # read it as a request, and the fit raised. Measured 2026-08-17 on
    # `bench/real_data`, which trains through `mojotrees.train(params, Dataset)`
    # exclusively: the `mojotrees_catboost_mode` arm raised on 100 percent of
    # cells, on numeric scenarios with no categorical column and so with no CTR
    # to build in the first place. This is `random_strength_ok`'s defect a
    # second time, in the parameter directly below it, and the docstring above
    # already names that class: "A flag set at the one call site somebody
    # looked at is the same defect wearing a different name."
    #
    # THREE ANSWERS NOW, AND THEY DIFFER BY PROVENANCE AND BY THE DATASET.
    #
    # `ctr_named` is the estimator's `IsSet` flag, and it defaults to 1 rather
    # than 0. That direction is deliberate and it is the opposite of the
    # reachability flags above: those default to "refuse" so a forgotten ENTRY
    # POINT declines, and this defaults to "the caller asked" so a forgotten
    # CALLER is refused loudly instead of having its request dropped. A caller
    # that sends `ctr` without `ctr_set` is the C API, the CLI, or a hand-built
    # dict, none of which has a mode-defaults layer, so for all of them a
    # non-off rule really is a request.
    # WHAT DECLINING IS, MECHANICALLY, AND THE INVARIANT THAT MAKES IT FREE.
    # It is doing nothing. On every entry point that passes `ctr_ok=False`
    # there is no downstream reader of the `ctr` key at all, so the mode
    # default lapses by not being consulted rather than by being overwritten:
    # `_parse_ctr` has exactly two callers, `fit` (lines 2155, 2164, 2285),
    # which passes `ctr_ok=True`, and `dataset_create` (line 4659), which is
    # the binning door itself and is where the bundle is supposed to be set.
    # Nothing is assigned here because assigning would be dead code, and dead
    # code that looks like it enforces something is worse than a comment that
    # says what actually holds.
    #
    # **THE INVARIANT A NEW ENTRY POINT MUST NOT BREAK: if you pass
    # `ctr_ok=False` and then call `_parse_ctr(params)`, you will build the
    # bundle this guard just declined.** Pass the rule you intend to a
    # `trainset.Dataset` instead, or pass `ctr_ok=True` and mean it.
    #
    # Declining is SAFE, and not merely quiet. A CTR column is a column, so a
    # fit that built none is the fit this package shipped for its whole life,
    # bit for bit. What declining can still produce is a refusal further down,
    # and that is the point rather than a leak: the symmetric grower rejects a
    # matrix that still offers it a categorical column (`tree.mojo:1874`), so a
    # categorical fit through one of these routes fails with the grower's own
    # sentence naming the real problem, instead of with a sentence about a
    # parameter the caller never typed. `sklearn.py`'s `_CATBOOST_CTR` comment
    # has the measurement behind that grower refusal.
    var ctr_rule = String(py=params.get("ctr", PythonObject("off")))
    var ctr_named = Int(py=params.get("ctr_set", PythonObject(1))) != 0
    if ctr_rule != "off" and not ctr_ok and ctr_named:
        raise Error(
            "ctr='",
            ctr_rule,
            "' is not honored by ",
            who,
            ": ordered target statistics are built while the dataset is"
            " binned, from the label and a fixed permutation, and this entry"
            " point bins without a ctr_columns.SimpleCtrConfig. The routes"
            " that carry one are a dense single-output fit, and"
            " mojotrees.Dataset(params={'ctr': ...}) followed by"
            " mojotrees.train(params, dataset) -- and on that second route the"
            " bundle has to be on the DATASET, because by the time the train"
            " params are read the matrix is already binned. ctr='off' is what"
            " every fit made before this parameter existed did and is the"
            " default under every grow policy but symmetrictree",
        )

    # LightGBM's and CatBoost's `boost_from_average`, and the one parameter
    # here that NAMES behavior this package already had rather than reaching
    # something that was unreachable. `boosting._base_score` has always seeded
    # every row from the objective's optimal constant, so `true` -- LightGBM's
    # default (`config.h:948`) and the default under `lossguide` -- is
    # bit-identical to every fit made before this key existed. `false` starts
    # from 0.0 and is what CatBoost resolves for `Logloss`, `CrossEntropy` and
    # `MultiClass`.
    #
    # The mode default is resolved here rather than in the estimator for the
    # reason the leaf count is: the per-loss table is
    # `auto_learning_rate.catboost_boost_from_average_default`, transcribed
    # from `options_helper.cpp:353-374` with the objective codes this package
    # uses, and a Python copy of it would be a second table to keep true.
    var bfa_named = Int(py=params["boost_from_average_set"]) != 0
    extra.boost_from_average = Int(py=params["boost_from_average"]) != 0
    if (
        catboost_defaults
        and not bfa_named
        and boost_from_average_ok
        and catboost_defaults_objective != _NO_CATBOOST_DEFAULTS
    ):
        extra.boost_from_average = catboost_boost_from_average_default(
            catboost_defaults_objective
        )
    if extra.boost_from_average_disabled() and not boost_from_average_ok:
        raise Error(
            "boost_from_average=false is not honored by ",
            who,
            ": the trainers that thread it into boosting._base_score are"
            " boosting.train, boosting.train_with_valid, train_gpu.train_gpu"
            " and train_gpu.train_gpu_with_valid, which this estimator"
            " reaches through a dense, single-output fit. Every other round"
            " loop seeds its raw scores from the objective's optimal constant"
            " unconditionally -- the multiclass loops from the per-class log"
            " priors, the ranking loops from zero already -- so accepting"
            " false here would start from the label mean under a parameter"
            " that asked for zero. true is LightGBM's default"
            " (include/LightGBM/config.h:948) and is mojotrees's behavior on"
            " every trainer. For a per-row offset instead, pass init_score",
        )

    # CatBoost's `random_strength_seed`, the seed the per-split score noise is
    # keyed from. Parsed here beside the strength rather than in
    # `extra_params_from_mapping`, because the two are one mechanism and the
    # refusal below reads both.
    #
    # Until this line, `ExtraTreeParams.random_strength_seed` took
    # `DEFAULT_RANDOM_STRENGTH_SEED` on every fit that came through Python
    # while `random_state` fanned out to six other seeds, so a user who seeded
    # a run seeded everything about it EXCEPT the split-score noise. That is a
    # reproducibility hole rather than a wrong number -- the draw was always
    # deterministic, it just could not be moved -- and with `random_strength`
    # reachable it is the difference between a seeded fit and a fit that
    # repeats only because nothing asked it not to. `sklearn._SEEDS` now
    # carries the name, so `random_state` reaches it by the same rule as
    # `bagging_seed` and the rest: an explicitly named seed wins, an unnamed
    # one follows the global.
    #
    # No reachability flag. The draw is keyed by (seed, tree, node, feature,
    # bin) and by nothing else (`tree_parameters_extra.random_score_stream`),
    # the device backend keys it identically
    # (`gpu_split_search.gpu_random_score_stream`, same domain constant), and
    # the seed is inert whenever `random_strength` is 0. So the entry points
    # that may carry a seed are exactly the entry points that may carry a
    # strength, and `random_strength_ok` right below already decides that.
    extra.random_strength_seed = Int(py=params["random_strength_seed"])
    # CatBoost's `random_strength`. The noise and its per-tree scale are both
    # implemented; the dense CPU round loops compute the scale onto their own
    # copy of the bundle before growth, which is why the strength can arrive
    # here beside a zero `random_score_scale` and still be honest. A loop that
    # does NOT compute a scale must refuse rather than train a model that
    # ignored the setting, which is what `random_strength_ok` selects.
    extra.random_strength = Float64(py=params["random_strength"])
    if extra.random_strength > 0.0 and not random_strength_ok:
        raise Error(
            "random_strength is not honored by ",
            who,
            ": the per-split noise is added by split.find_best_split and"
            " staged on the device by GpuSplitSearcher, but its per-tree"
            " scale is computed only by the DENSE round loops -- the CPU's"
            " (boosting._round_random_score_scale) and, since 2026-08-17,"
            " both arms of the GPU's (train_gpu._train_gpu_rounds, the"
            " device-gradient arm reducing the squares on the device). The"
            " sparse, multiclass and distributed loops do not compute one,"
            " so accepting it here would train a model that silently ignored"
            " it. 0.0 is LightGBM's behavior and mojotrees's, and is"
            " accepted.",
        )

    # CatBoost's `score_function`, folded onto the same bundle for the same
    # reason. The wrapper sends the name lowercased, which is the contract
    # `parse_score_function` states and `device_policy.parse_device` and
    # `sampling.canonical_bootstrap_type` already keep, so the parameter
    # string and the estimator resolve one spelling to one code.
    extra.score_function = parse_score_function(
        String(py=params["score_function"])
    )
    if extra.score_function != SCORE_L2 and not score_function_ok:
        raise Error(
            "score_function=",
            score_function_name(extra.score_function),
            " is not implemented by ",
            who,
            "; the split search that reads it is tree._search and"
            " tree._grow_oblivious_levels, which every grower in this"
            " package reaches except the distributed prototype"
            " (distributed_strategies calls split.find_best_split at its"
            " SCORE_L2 default). 'L2' is what mojotrees has always scored"
            " with -- G^2/(H+lambda) -- and is the default",
        )

    # The precision a per-row derivative is carried at, folded onto the same
    # bundle for the same reason the two above it are. The wrapper sends the
    # name lowercased, which is the contract `parse_derivative_precision`
    # states and `parse_score_function` and `parse_device` already keep, so
    # the parameter string and the estimator resolve one spelling to one code.
    #
    # **Until this line the parameter had no Python door at all.** The field
    # existed, `params.parse_params` accepted the key on the string surface,
    # and every CPU round loop honored it, so a Python caller who wanted wide
    # derivatives had exactly one entry: exporting
    # MOJOTREES_DERIVATIVE_PRECISION. That is a process-wide switch that a
    # child process inherits without asking, and it does not travel in the
    # record that quotes the timing it changed -- which is how an A/B comes to
    # run one arm under the other's label. The parameter is the door;
    # docs/COMPATIBILITY_POLICY.md section 9.5.1 is the rule that says a
    # capability which ships becomes a named parameter.
    extra.derivative_precision = parse_derivative_precision(
        String(py=params["derivative_precision"])
    )
    if extra.wants_float64_derivatives() and not derivative_precision_ok:
        raise Error(
            "derivative_precision='",
            derivative_precision_name(extra.derivative_precision),
            "' is not honored by ",
            who,
            ": the wide derivative is carried by the round loops that thread"
            " ExtraTreeParams.wants_float64_derivatives() into"
            " boosting._fill_grad_hess or boosting._fill_softmax_grad_hess,"
            " which is every dense, sparse, multiclass, dart, rf,"
            " custom-metric and distributed loop in this package. The ranking"
            " loops (ranking.train_ranker,"
            " custom_metric.train_ranker_with_metrics) compute their lambdas"
            " in Float64 and forward no such flag, and a custom objective is"
            " the caller's own pair of derivative buffers, so at those entry"
            " points this value would select nothing and be silently dropped."
            " 'float32' is LightGBM's precision profile and mojotrees's, and"
            " is the default. Note that the accelerator refuses 'float64' at"
            " EVERY entry point and at every shape"
            " (histogram.check_device_derivative_precision): gradients are"
            " carried as Float32 on the device and there is no Float64 there",
        )
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
        extra=extra^,
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
    # LightGBM's `linear_tree` / `linear_lambda` (src/mojotrees/linear_tree.mojo).
    # The metric-path trainers fit the linear leaves; every binned-only
    # trainer refuses the switch by name rather than dropping it.
    var linear = LinearParams(
        enabled=Int(py=params["linear_tree"]) != 0,
        linear_lambda=Float64(py=params["linear_lambda"]),
    )
    # CatBoost's `boosting_type=Ordered` (src/mojotrees/ordered_boosting.mojo).
    # Until this was passed, `BoosterParams.ordered` took its disabled default
    # on every fit that came through Python AND on every parameter string, so
    # the fold ladder `boosting.train` grows was reachable from the Mojo API
    # and from nowhere else: the mechanism was merged, tested, and set by no
    # binding.
    #
    # `enable` rather than a field-by-field build, so the four knobs take the
    # module's defaults where the estimator sends nothing, and `validate` is
    # the module's own range check rather than a copy of it here.
    var ordered = OrderedBoostingParams.disabled()
    if Int(py=params["ordered"]) != 0:
        ordered = OrderedBoostingParams.enable(
            permutation_count=Int(py=params["permutation_count"]),
            fold_len_multiplier=Float64(py=params["fold_len_multiplier"]),
            permutation_block_size=Int(py=params["fold_permutation_block"]),
            seed=Int(py=params["ordered_seed"]),
        )
    ordered.validate()
    if ordered.enabled and not ordered_ok:
        raise Error(
            "boosting_type='ordered' is not implemented by ",
            who,
            "; only boosting.train honors the fold ladder, which this"
            " estimator reaches through a dense, single-output CPU fit"
            " without eval_set, without a custom objective, and without row"
            " sampling. 'plain' (an alias of 'gbdt') is the scheme every"
            " other path trains",
        )
    var bp = BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
        bundling^,
        linear^,
        ordered^,
    )
    _apply_auto_learning_rate(
        bp,
        params,
        who,
        auto_lr_ok,
        auto_lr_reason,
        auto_lr_rows,
        auto_lr_objective,
        cpu,
    )
    return bp^


def _apply_auto_learning_rate(
    mut bp: BoosterParams,
    params: PythonObject,
    who: String,
    ok: Bool,
    reason: String,
    n_rows: Int,
    objective: Int,
    cpu: Bool,
) raises:
    """Replace `bp.learning_rate` with CatBoost's derived rate, or refuse.

    The one call in the Python extension that reaches
    `auto_learning_rate.resolve_learning_rate`
    (src/mojotrees/auto_learning_rate.mojo). The C ABI and the CLI reach the
    same free function through `params.TrainConfig.resolved_learning_rate`;
    this path never builds a `TrainConfig`, and nothing had to move for it
    to get there.

    Four keys carry the request, and every fit sends all four because this
    parser subscripts the mapping rather than testing for a key:

    - `auto_learning_rate`: the derivation is wanted AND the user left
      `learning_rate` alone. That second half is folded in on the Python
      side because it is the one part of CatBoost's gate this side cannot
      see: the wire carries a resolved float and a float cannot say whether
      anybody typed it. `TOption::NotSet()` (`option.h:80-85`) is provenance,
      not a comparison against the default, and provenance stops at the
      estimator.
    - `auto_learning_rate_required`: the user asked in so many words
      (`auto_learning_rate=True`) rather than inheriting the default that
      `grow_policy='symmetrictree'` carries. It decides refuse versus
      decline, and the two are both right in their own case. An explicit
      request that cannot be honored must not be dropped, which is this
      repository's rule. A CatBoost-mode default that cannot be honored
      falls back to the given rate in silence, which is CatBoost's own
      behavior: handed a loss with no row in the coefficient table it keeps
      its constant 0.03 and prints nothing.
    - the two `*_set` provenance flags, which close CatBoost's gate on
      `l2_leaf_reg` and `leaf_estimation_iterations`
      (`options_helper.cpp:279-280`). They are handed to
      `AutoLearningRateParams` rather than tested here so that the gate has
      one implementation and this is not a second copy of it;
      `AutoLearningRateParams.fires` is what applies them.

    `use_best_model` is False and cannot be anything else here: every entry
    point that reaches this with `ok` true is a fit without an eval set,
    because the three eval-set entry points refuse by name above.
    `boost_from_average` is left to `catboost_boost_from_average_default`,
    which is what CatBoost would resolve for the loss -- notably **false**
    for Logloss, where mojotrees does start from the optimal constant. That
    is deliberate: the coefficient row has to be the row CatBoost would pick
    for the same run, or the two derived rates are not comparable, which is
    the whole point of deriving it.

    Determinism: `resolve_learning_rate` is scalar Float64 on one thread with
    no reduction and no parallel region, and its inputs here are an Int row
    count, an Int iteration count and an Int objective code. Nothing it reads
    depends on `MOJOTREES_NUM_WORKERS`, on the device, or on row order, and
    the result is narrowed to float32 before it is used.
    """
    if Int(py=params["auto_learning_rate"]) == 0:
        return
    var required = Int(py=params["auto_learning_rate_required"]) != 0
    if not ok:
        if required:
            var why = reason.copy()
            if why.byte_length() == 0:
                why = String(
                    "this entry point cannot supply the train row count, the"
                    " iteration count and the loss function that CatBoost's"
                    " derivation reads"
                )
            raise Error(
                "auto_learning_rate=True is not honored by ",
                who,
                ": ",
                why,
                ". Pass an explicit learning_rate instead",
            )
        return
    # A Booster constructed on a training set trains a zero-round model
    # before anything is boosted, and CatBoost's formula takes
    # log(iterationCount), which is undefined at zero rounds. Nothing
    # is fitted at zero rounds, so there is no rate to derive and no request
    # being dropped; the rounds that follow come through `booster_update`,
    # which refuses the parameter by name.
    if bp.n_estimators <= 0:
        return
    var auto = AutoLearningRateParams.catboost_defaults(
        AUTO_LR_TASK_CPU if cpu else AUTO_LR_TASK_GPU
    )
    auto.l2_leaf_reg_set = (
        Int(py=params["auto_learning_rate_l2_set"]) != 0
    )
    auto.leaf_estimation_iterations_set = (
        Int(py=params["auto_learning_rate_leaf_iters_set"]) != 0
    )
    if not auto.fires(objective):
        if required:
            raise Error(
                "auto_learning_rate=True has nothing to derive for this run"
                " at ",
                who,
                ": CatBoost's coefficient table is keyed by (target type,"
                " task type, use_best_model, boost_from_average) and has no"
                " row for this one. Its target types are Logloss, MultiClass"
                " and RMSE (options_helper.cpp:181-194), and the MultiClass"
                " rows exist only with boost_from_average false, so a"
                " ranking or survival objective, or a MultiClass run boosted"
                " from the average, leaves the rate alone. l2_leaf_reg and"
                " leaf_estimation_iterations close the same gate"
                " (options_helper.cpp:279-280). Pass an explicit"
                " learning_rate instead",
            )
        return
    bp.learning_rate = resolve_learning_rate(
        auto, objective, bp.n_estimators, n_rows, bp.learning_rate
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


def _parse_bootstrap(params: PythonObject) raises -> BootstrapParams:
    """CatBoost's `bootstrap_type` from the params dict, as one bundle.

    Modeled on `_parse_goss` above: the values are read, a bundle is built,
    and the trainer validates it again, so the rules in sampling.mojo stay the
    only ones. `canonical_bootstrap_type` is the resolver, the same one the
    parameter string and the CLI reach, so `Bernoulli` and `Poisson` are
    refused by name here for free and no copy of those two messages lives at
    this boundary.

    Three keys arrive as **sentinels rather than defaults**, and the reason is
    that a defaulted value and a value the user wrote down have to be
    distinguishable to refuse the wrong pairing at all:

    - `bootstrap_subsample` below 0 means "the user did not name `subsample`",
      and MVS takes `DEFAULT_MVS_SUBSAMPLE`. It is NOT spelled `subsample` on
      the wire, because `bagging_fraction` already carries LightGBM's meaning
      of that word and the two are different samplers; the estimator decides
      which meaning the user's `subsample` had while both are still visible
      (`_resolve_bootstrap` in python/mojotrees/sklearn.py) and sends the
      resolved number here.
    - `bagging_temperature` below 0 means "not set". CatBoost's own range is
      `>= 0`, so no real value is lost. This is `TOption::IsSet` on the wire,
      and `check_mvs_bagging_temperature` needs exactly that flag.
    - `mvs_reg` below 0 means "not set", which is a real state in CatBoost
      (`TMaybe<float>`) and not a magic number: unset derives the lambda from
      the data every tree.

    Every parameter that belongs to a bootstrap type other than the one
    selected is refused by name with the reason. A knob accepted and never
    read is a silent wrong answer to the user who set it, and CatBoost itself
    makes that mistake with `bagging_temperature` beside MVS (see
    `sampling.check_mvs_bagging_temperature`).
    """
    var kind = canonical_bootstrap_type(String(py=params["bootstrap_type"]))
    var subsample = Float64(py=params["bootstrap_subsample"])
    var subsample_is_set = subsample >= 0.0
    var temperature = Float64(py=params["bagging_temperature"])
    var temperature_is_set = temperature >= 0.0
    var reg = Float64(py=params["mvs_reg"])
    var reg_is_set = reg >= 0.0
    var seed = Int(py=params["bootstrap_seed"])

    if kind == "no":
        if subsample_is_set:
            raise Error(
                "subsample is the bootstrap rate only under"
                " bootstrap_type='mvs'; with no bootstrap type it is row"
                " bagging and reaches the fit as bagging_fraction"
            )
        if temperature_is_set:
            raise Error(
                "bagging_temperature belongs to bootstrap_type 'bayesian' and"
                " is read by no other type; remove it or set bootstrap_type"
            )
        if reg_is_set:
            raise Error(
                "mvs_reg belongs to bootstrap_type 'mvs' and is read by no"
                " other type; remove it or set bootstrap_type='mvs'"
            )
        return BootstrapParams.disabled()

    if kind == "mvs":
        var mvs: BootstrapParams
        if reg_is_set:
            if subsample_is_set:
                mvs = BootstrapParams.mvs_with_reg(reg, subsample, seed)
            else:
                mvs = BootstrapParams.mvs_with_reg(reg, seed=seed)
        elif subsample_is_set:
            mvs = BootstrapParams.mvs_at(subsample, seed)
        else:
            mvs = BootstrapParams.mvs_at(seed=seed)
        # The refusal that CatBoost does not make: `bagging_temperature`
        # beside MVS is accepted there, never read, and dropped in `Save`.
        check_mvs_bagging_temperature(mvs.mvs, temperature_is_set)
        return mvs^

    # `canonical_bootstrap_type` returns only "no", "mvs" and "bayesian"; the
    # other three spellings raise inside it.
    if subsample_is_set:
        raise Error(
            "bootstrap_type='bayesian' does not take subsample: the Bayesian"
            " bootstrap keeps every row and reweights it, so there is no"
            " fraction to set. CatBoost refuses the same pair"
        )
    if reg_is_set:
        raise Error(
            "mvs_reg belongs to bootstrap_type 'mvs' and is never read by the"
            " Bayesian bootstrap, which has no lambda at all"
        )
    if temperature_is_set:
        return BootstrapParams.bayesian_at(temperature, seed)
    return BootstrapParams.bayesian_at(seed=seed)


def _parse_bootstrap_request(
    params: PythonObject,
) raises -> BootstrapRequest:
    """`_parse_bootstrap`'s bundle together with **whether the user asked for
    it**, which is the sixth wire key of the group and the one that decides
    how an entry point that cannot honor a bootstrap behaves.

    `bootstrap_explicit` is 1 when the estimator saw a `bootstrap_type` the
    user wrote down and 0 when the value on the wire is the library's own
    default. The distinction cannot be recovered from the bundle -- a defaulted
    MVS and a typed MVS are the same five numbers -- and it has to survive the
    boundary because the two must degrade differently:

    - **A typed request is refused by name** on any path whose round loop does
      not call `sampling.bootstrap_round`. A user reading a benchmark of a
      "sampled" fit that was not sampled is the failure this whole group of
      refusals exists to prevent.
    - **A defaulted bundle is dropped, silently**, on those same paths. A
      library whose out-of-the-box `fit` raises on a multiclass problem, on a
      CSR matrix, or on the GPU is not shippable, and there is nothing to tell
      a user about a value they never set.

    `BootstrapRequest.resolve` and `resolve_or_defer` are the two ways to spend
    this; which one a call site wants depends on whether the trainer it is
    about to call has a better refusal of its own. **Every call site below
    picks one deliberately and none of them ignores the bundle**, which is the
    property this file has had to learn twice.
    """
    var bundle = _parse_bootstrap(params)
    if Int(py=params["bootstrap_explicit"]) != 0:
        return BootstrapRequest.named(bundle^)
    return BootstrapRequest.defaulted(bundle^)


def _parse_boosting(params: PythonObject) raises -> AlternateBoostingParams:
    """LightGBM's `boosting` from the params dict: the mode by name, and the
    DART bundle from the `drop_*` keys when the mode is dart. `goss` keeps
    arriving as its own flag (`_parse_goss`), so a `boosting` of "goss" or
    "gbdt" resolves to the plain bundle and the ordinary trainer; only dart
    and rf change which trainer runs. The dispatcher validates the bundle
    again, so the rules in alternate_boosting.mojo stay the only ones."""
    var mode = parse_boosting(String(py=params["boosting"]))
    if mode == BOOSTING_DART:
        return AlternateBoostingParams.dart_with(
            DartParams.enable(
                drop_rate=Float64(py=params["drop_rate"]),
                max_drop=Int(py=params["max_drop"]),
                skip_drop=Float64(py=params["skip_drop"]),
                uniform_drop=Int(py=params["uniform_drop"]) != 0,
                xgboost_dart_mode=Int(py=params["xgboost_dart_mode"]) != 0,
                seed=Int(py=params["drop_seed"]),
            )
        )
    if mode == BOOSTING_RF:
        return AlternateBoostingParams.rf()
    return AlternateBoostingParams()


def _parse_categorical(params: PythonObject) raises -> List[Int]:
    """Categorical feature indices from the params dict, as one float64
    entry per index at `categorical_addr`, or a zero address for none. The
    binner validates the indices again, so the rules in categorical.mojo
    stay the only ones."""
    var addr = Int(py=params["categorical_addr"])
    if addr == 0:
        return List[Int]()
    return int_buffer_from_f64(addr, Int(py=params["categorical_len"]))


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


def _apply_ctr_target_binarization(
    mut config: SimpleCtrConfig, params: PythonObject
) raises:
    """CatBoost's `target_binarization` pair, off the params dict onto a bundle.

    Both keys are optional and both fall back to what the bundle already holds,
    which is CatBoost's own default (`MinEntropy`, 1). `.get` rather than a
    subscript for the reason `one_hot_max_size` uses it: a direct caller of
    `dataset_create` that predates these keys must keep working.

    The type is a STRING and not an integer, because it names a rule rather than
    counting anything, which is the same argument `ctr` itself is passed as a
    string on. `ctr_columns` owns the codes and this is the only place the two
    spellings meet.
    """
    var count = Int(
        py=params.get(
            "ctr_target_border_count", PythonObject(config.target_border_count)
        )
    )
    if count < 1:
        raise Error(
            "ctr_target_border_count must be positive; got ",
            count,
        )
    config.target_border_count = count
    var type_name = String(
        py=params.get("ctr_target_border_type", PythonObject("minentropy"))
    )
    if type_name == "minentropy":
        config.target_border_type = CTR_TARGET_BORDER_MIN_ENTROPY
    elif type_name == "multiclass":
        config.target_border_type = CTR_TARGET_BORDER_MULTICLASS
    else:
        raise Error(
            "ctr_target_border_type must be 'minentropy' or 'multiclass'; got"
            " '",
            type_name,
            "'",
        )


def _parse_ctr(params: PythonObject) raises -> SimpleCtrConfig:
    """The dataset's ordered-target-statistic bundle, catalog A19.

    Three values, passed as a string because they name rules rather than
    counting anything:

    - `"auto"`, the default. `SimpleCtrConfig.auto()`: CTR columns for the
      categorical columns that filled their category table and so lost levels
      into `categorical.UNKNOWN_BIN`, and for no others. On a dataset with no
      categorical column, or none wide enough to overflow, this plans nothing
      and the binned matrix is the one it would have been.
    - `"off"`. `SimpleCtrConfig.disabled()`, the behavior of every release
      before this default moved.
    - `"catboost"`. `SimpleCtrConfig.catboost_defaults()`: CatBoost's own
      source rule, `uniqueValues > one_hot_max_size` at 2, which gives almost
      every categorical column four extra numeric columns. It is the faithful
      port and it is not the default, because away from the overflow boundary
      those columns cost histogram width without recovering anything the
      category table already holds.

    - `"on"`, the estimator's spelling of `"catboost"`. One rule, two words,
      because `Dataset(params={"ctr": ...})` shipped with `"catboost"` and the
      estimator parameter reads as a switch. Resolved here rather than in
      Python so that both surfaces reach one resolver.

    `one_hot_max_size` is CatBoost's fork between one-hot and CTR: a
    categorical column with at most that many levels is one-hot and every
    wider one is replaced by its CTR columns. It is read only by the
    `"catboost"` / `"on"` rule -- `"auto"` selects its source columns by
    whether the category table overflowed, which is a different question and
    reads a different number -- so it is applied only there, and a bundle that
    does not read it keeps `ctr_columns.CTR_ONE_HOT_MAX_SIZE` rather than
    recording a cutoff nothing consulted.

    `ctr_target_border_count` and `ctr_target_border_type` are the two halves of
    CatBoost's `target_binarization` option (`cat_feature_options.cpp:162`:
    `TBinarizationOptions(EBorderSelectionType::MinEntropy, 1)`), and they are
    read here for the same reason `one_hot_max_size` is: this is the door the
    bundle is built at. Both are optional and both default to CatBoost's, so a
    caller who names neither gets exactly what this function returned before
    they existed. `ctr_target_border_type` takes `"minentropy"` (CatBoost's
    default selection, run over the raw target) or `"multiclass"`
    (`GetMultiClassBorders`, `borders[i] = 0.5 + i`, which reads no target
    value). They are read on the `auto` arm too: `auto` differs from `catboost`
    in which SOURCE columns it selects, not in how the target is quantized, and
    a bundle that ignored a named option would be the silent-difference defect
    the `is_policy` split exists to avoid.

    Anything else raises here rather than resolving to a default, so a typo is
    an error and not a silently different model.
    """
    var name = String(py=params.get("ctr", PythonObject("off")))
    if name == "off":
        return SimpleCtrConfig.disabled()
    if name == "auto":
        var auto_out = SimpleCtrConfig.auto()
        _apply_ctr_target_binarization(auto_out, params)
        return auto_out^
    if name == "catboost" or name == "on":
        var out = SimpleCtrConfig.catboost_defaults()
        _apply_ctr_target_binarization(out, params)
        # `.get`, not a subscript: `Dataset` sends this key and the estimator
        # sends this key, but a direct caller of `dataset_create` predating
        # both does not, and a KeyError there would be a regression in a door
        # that was working. The fallback is the bundle's own default, which is
        # CatBoost's 2.
        var cutoff = Int(
            py=params.get(
                "one_hot_max_size", PythonObject(out.one_hot_max_size)
            )
        )
        if cutoff < 0:
            raise Error(
                "one_hot_max_size must be nonnegative; got ", cutoff
            )
        out.one_hot_max_size = cutoff
        return out^
    raise Error(
        "ctr must be 'off', 'auto', or 'on' (spelled 'catboost' on the"
        " Dataset door); got '",
        name,
        "'",
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
    return f64_buffer(weight_addr, n_rows)


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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var target = f64_buffer(Int(py=y_addr), nr)
    # The device is read before the parameters because bundling is applied
    # by the dense CPU trainer and not by the GPU one, so `_parse_params`
    # has to know which of the two `mojo_fit` will dispatch to. The wrapper
    # sends a device it has already resolved, so this is the backend.
    var device = _parse_device(params)
    # `ordered_ok` is the CPU test and nothing else: `model.fit` routes a CPU
    # run to `boosting.train`, which is the one trainer that grows the fold
    # ladder, and a GPU run to `train_gpu`, which reads `BoosterParams.ordered`
    # nowhere. `leaf_estimation_ok` is unconditional here because both of those
    # trainers implement the extra Newton steps.
    var bp = _parse_params(
        params,
        nf,
        cpu=device == CPU_DEVICE,
        entry=String("fit"),
        # THE ONE ENTRY POINT THAT HONORS `boosting`. It calls
        # `_parse_boosting` itself, twenty lines below, and forks into
        # `alternate_boosting.fit_boosting` for dart and rf. Every other entry
        # point took the key on the wire and dispatched no alternate loop, so
        # `_parse_params` refuses it for them; see the refusal there for what
        # that silence cost.
        boosting_ok=True,
        ordered_ok=device == CPU_DEVICE,
        # Same condition and the same reason: `model.fit` routes a CPU
        # run to `boosting.train`, whose round loop is the one that
        # computes `random_score_scale` per tree.
        #
        # It is the condition for the PLAIN CPU fork only, and the other two
        # CPU forks are refused a few statements below rather than here,
        # because they are selected by `boosting` and `bp.linear`, which are
        # not known until this call has returned. That is the same shape
        # `ordered_ok` already takes: a flag wide enough for the entry point
        # and two named refusals for the branches that leave the honoring
        # trainer.
        # **WIDENED 2026-08-17: the GPU dense fork computes the scale too.**
        # This was `device == CPU_DEVICE` because only the dense CPU round
        # loop computed `random_score_scale`. Both arms of
        # `train_gpu._train_gpu_rounds` compute it now -- the host-gradient
        # arm through `boosting._round_random_score_scale` from the round's
        # user-weighted derivatives, the device-gradient arm through
        # `_device_round_random_score_scale` over
        # `GpuObjectiveState.derivative_sum_squares`.
        #
        # `fit` is the DENSE SINGLE-OUTPUT entry, which is exactly the pair of
        # forks that compute it. The sparse arm and the multiclass arm still
        # do not, and they are refused at their own entry points rather than
        # here -- widening this one does not widen those, which is the whole
        # reason this is a per-entry flag rather than a device test.
        random_strength_ok=True,
        leaf_estimation_ok=True,
        # Same fork, same shape: the plain CPU and GPU forks both thread
        # `boost_from_average` into `boosting._base_score`, and the dart, rf
        # and linear forks are refused by name a few statements below,
        # because they are selected by `boosting` and `bp.linear` and neither
        # is known until this call has returned.
        boost_from_average_ok=True,
        score_function_ok=True,
        # CatBoost's automatic learning rate. Unconditional here, and the
        # fork does not have to be settled the way `ordered_ok`'s does: the
        # derived rate is a number in `BoosterParams`, and `boosting.train`,
        # `train_gpu` and `alternate_boosting.fit_boosting` all shrink by it.
        # The one fork that would discard it is `boosting='rf'`, which trains
        # at 1.0 whatever it was given, and the estimator refuses that pair
        # before it gets here rather than deriving a rate for a forest to
        # throw away.
        auto_lr_ok=True,
        auto_lr_rows=nr,
        auto_lr_objective=Int(py=objective),
        # CatBoost mode's per-objective defaults for
        # `leaf_estimation_iterations`, `boost_from_average` and
        # `random_strength`. The same code `auto_lr_objective` gets, declared
        # separately because the two sentinels mean different things; see
        # `_parse_params`.
        catboost_defaults_objective=Int(py=objective),
        # CatBoost's ordered target statistics. True for this entry point's
        # PLAIN fork, which reroutes through `trainset.Dataset` +
        # `trainset.train_dataset` when the bundle is active; the dart/rf and
        # linear forks are refused by name a few statements below, beside
        # `ordered`'s and `random_strength`'s and for the same reason --
        # `alternate_boosting.fit_boosting` and
        # `custom_metric.fit_with_metrics` take raw matrices and bin them
        # themselves, with no place to put a bundle.
        ctr_ok=True,
        # Every fork of this entry point either honors the wide derivative or
        # refuses it by name: the plain CPU fork is boosting.train, the dart and
        # rf forks are alternate_boosting, the linear_tree fork is
        # custom_metric.train_with_callbacks, and the GPU fork raises in
        # histogram.check_device_derivative_precision.
        derivative_precision_ok=True,
    )
    var weights = _parse_weights(params, nr)
    # CatBoost's `bootstrap_type`, with the flag that says whether the user
    # typed it. `model.fit` on the CPU with plain gbdt and no `linear_tree`
    # reaches `boosting.train`, whose round loop calls
    # `sampling.bootstrap_round`; the dart/rf and linear forks below leave that
    # path and settle themselves, and the GPU fork is settled where `model.fit`
    # is called. A fork is settled at the fork, not at the entry point, which
    # is the rule `_parse_params`'s reachability flags keep and the rule
    # `random_strength` was once shipped without.
    var boot_req = _parse_bootstrap_request(params)
    var boosting = _parse_boosting(params)
    if bp.ordered.enabled and (
        boosting.mode == BOOSTING_DART or boosting.mode == BOOSTING_RF
    ):
        # `alternate_boosting.fit_boosting` runs its own round loop and takes
        # no `ordered` bundle, so the pair would train a plain dart or forest
        # and report an ordered fit.
        raise Error(
            "boosting_type='ordered' cannot be combined with dart or rf:"
            " those run alternate_boosting's own round loop, which does not"
            " grow the fold ladder"
        )
    if bp.ordered.enabled and bp.linear.is_active():
        # linear_tree routes through `custom_metric.fit_with_metrics` below,
        # and that trainer refuses an ordered bundle itself; refusing here
        # names the combination rather than the entry point it landed on.
        raise Error(
            "boosting_type='ordered' cannot be combined with linear_tree:"
            " linear leaves are fitted by the metric-path trainer, which"
            " does not grow the fold ladder"
        )
    # `random_strength`'s two CPU forks, exactly beside `ordered`'s and for
    # the same reason. `random_strength_ok` above is True for every CPU fit,
    # because the plain fork reaches `boosting.train` and that is the common
    # case; these two branches leave it. Neither would train an unnoised
    # model in silence -- `split.find_best_split` refuses a positive strength
    # beside a zero scale -- but the message it raises tells a Mojo-API
    # caller to compute the scale themselves, which is not an answer a Python
    # caller can act on. Naming the combination here is.
    if bp.tree.extra.random_strength > 0.0 and (
        boosting.mode == BOOSTING_DART or boosting.mode == BOOSTING_RF
    ):
        raise Error(
            "random_strength cannot be combined with boosting='dart' or"
            " boosting='rf': those run alternate_boosting's own round loops,"
            " which never call boosting._round_random_score_scale, so the"
            " per-tree scale CatBoost's noise is drawn at would be zero."
            " Only the plain gbdt fork of this entry point reaches"
            " boosting.train, which computes it. Set random_strength=0, or"
            " train boosting='gbdt'"
        )
    if bp.tree.extra.random_strength > 0.0 and bp.linear.is_active():
        raise Error(
            "random_strength cannot be combined with linear_tree: linear"
            " leaves are fitted by custom_metric.fit_with_metrics, whose"
            " round loop never calls boosting._round_random_score_scale, so"
            " the per-tree scale CatBoost's noise is drawn at would be zero."
            " Set random_strength=0, or drop linear_tree"
        )
    # `leaf_estimation_iterations`'s and `boost_from_average`'s two CPU forks,
    # exactly beside `ordered`'s and `random_strength`'s and settled the same
    # way. `leaf_estimation_ok=True` above is right for this entry point's
    # PLAIN fork, which reaches `boosting.train` or `train_gpu.train_gpu`;
    # these two branches leave it, and until they were named here a
    # `boosting='dart'` fit with `leaf_estimation_iterations=5` was accepted
    # and silently took one Newton step per leaf. That is precisely the
    # accept-and-ignore this file exists to remove, and the flag being True at
    # the entry point rather than per fork is how it hid.
    #
    # Both parameters are refused together because both forks miss both: the
    # dart and rf loops live in `alternate_boosting`, the linear loop in
    # `custom_metric`, and none of the three reads
    # `TreeParams.extra.leaf_estimation_iterations` or threads
    # `boost_from_average` into `boosting._base_score`.
    if bp.tree.extra.leaf_estimation_active() and (
        boosting.mode == BOOSTING_DART or boosting.mode == BOOSTING_RF
    ):
        raise Error(
            "leaf_estimation_iterations > 1 cannot be combined with"
            " boosting='dart' or boosting='rf': those run"
            " alternate_boosting's own round loops, which never call"
            " boosting._estimate_leaf_values, so every leaf would take the"
            " single Newton step the grower wrote. Only the plain gbdt fork"
            " of this entry point reaches boosting.train. Set"
            " leaf_estimation_iterations=1, or train boosting='gbdt'"
        )
    if bp.tree.extra.leaf_estimation_active() and bp.linear.is_active():
        raise Error(
            "leaf_estimation_iterations > 1 cannot be combined with"
            " linear_tree: linear leaves are fitted by"
            " custom_metric.fit_with_metrics, whose round loop never calls"
            " boosting._estimate_leaf_values, and a linear leaf is a fitted"
            " model rather than a constant, so an extra Newton step on it is"
            " not the same operation. Set leaf_estimation_iterations=1, or"
            " drop linear_tree"
        )
    if bp.tree.extra.boost_from_average_disabled() and (
        boosting.mode == BOOSTING_DART or boosting.mode == BOOSTING_RF
    ):
        raise Error(
            "boost_from_average=false cannot be combined with"
            " boosting='dart' or boosting='rf': alternate_boosting's round"
            " loops call boosting._base_score without threading the"
            " parameter, so the fit would start from the label mean under a"
            " parameter that asked for zero. Set boost_from_average=true, or"
            " train boosting='gbdt'"
        )
    if bp.tree.extra.boost_from_average_disabled() and bp.linear.is_active():
        raise Error(
            "boost_from_average=false cannot be combined with linear_tree:"
            " custom_metric.fit_with_metrics calls boosting._base_score"
            " without threading the parameter, so the fit would start from"
            " the label mean under a parameter that asked for zero. Set"
            " boost_from_average=true, or drop linear_tree"
        )
    # `ctr`'s two CPU forks, exactly beside `ordered`'s, `random_strength`'s
    # and `leaf_estimation_iterations`'s, and settled the same way.
    # `ctr_ok=True` above is right for this entry point's PLAIN fork, which
    # reroutes through `trainset.Dataset`; these two branches leave it, and
    # both bin a raw matrix through `fit_bins` with nowhere to put a bundle.
    if _parse_ctr(params).is_active() and (
        boosting.mode == BOOSTING_DART or boosting.mode == BOOSTING_RF
    ):
        raise Error(
            "ctr cannot be combined with boosting='dart' or boosting='rf':"
            " alternate_boosting.fit_boosting bins the raw matrix itself and"
            " takes no ctr_columns.SimpleCtrConfig, so the CTR columns would"
            " never be built. Set ctr='off', or train boosting='gbdt'"
        )
    if _parse_ctr(params).is_active() and bp.linear.is_active():
        raise Error(
            "ctr cannot be combined with linear_tree: linear leaves are"
            " fitted by custom_metric.fit_with_metrics, which bins the raw"
            " matrix itself and takes no ctr_columns.SimpleCtrConfig, so the"
            " CTR columns would never be built. Set ctr='off', or drop"
            " linear_tree"
        )
    if boosting.mode == BOOSTING_DART or boosting.mode == BOOSTING_RF:
        # dart and rf run through alternate_boosting's dispatcher, which
        # bins with the same fit_bins and returns the same Model; it is CPU
        # only (train_gpu has no dropout or forest loop), and the wrapper
        # sends a device it already resolved, so a GPU here is refused
        # rather than downgraded.
        if device != CPU_DEVICE:
            raise Error(
                "boosting='dart' and boosting='rf' train on the CPU only;"
                " set device='cpu'"
            )
        # `resolve`, not `resolve_or_defer`: `alternate_boosting.fit_boosting`
        # takes no bootstrap argument at all, so there is no trainer-side
        # refusal to defer to and this is the last place a typed request can
        # be reported. A defaulted bundle is dropped here instead of raising.
        _ = boot_req.resolve(
            False,
            String(
                "boosting='dart' and boosting='rf'"
                " (alternate_boosting.fit_boosting)"
            ),
        )
        var routed = fit_boosting(
            features,
            nr,
            nf,
            target,
            Int(py=objective),
            bp,
            boosting,
            Int(py=params["max_bin"]),
            weights,
            Float64(py=params["alpha"]),
            _parse_bagging(params),
            GossParams.disabled(),
            use_missing=_parse_use_missing(params),
            categorical_features=_parse_categorical(params),
        )
        return PythonObject(alloc=routed^)
    if bp.linear.is_active():
        # linear_tree=True: the leaves are fitted on the raw rows, which
        # only the metric trainer keeps beside the binned matrix
        # (custom_metric.fit_with_metrics), so a plain fit is that trainer
        # scoring the training set with a metric that costs nothing and
        # never stops early. CPU only: train_gpu reads bins alone.
        if device != CPU_DEVICE:
            raise Error(
                "linear_tree=True trains on the CPU only; set device='cpu'"
            )
        # Same shape as the dart/rf fork: `custom_metric.fit_with_metrics`
        # takes no bundle, so a typed request is refused here and a default is
        # dropped here.
        _ = boot_req.resolve(
            False,
            String("linear_tree=True (custom_metric.fit_with_metrics)"),
        )
        var train_set = List[RawValidSet]()
        train_set.append(
            RawValidSet(
                "train", f64_buffer(Int(py=x_addr), nr * nf), nr, target.copy()
            )
        )

        def no_metric(
            metric: Int, valid: Int, pred: List[Float64], labels: List[Float64]
        ) raises -> Float64:
            return 0.0

        var metrics: List[CustomMetric] = [CustomMetric("linear_tree_fit")]
        var fitted = mojo_fit_with_metrics(
            features,
            nr,
            nf,
            target,
            train_set^,
            Int(py=objective),
            bp,
            MetricSuite(metrics^, no_metric, 0),
            0,
            0.0,
            Int(py=params["max_bin"]),
            weights,
            Float64(py=params["alpha"]),
            _parse_bagging(params),
            _parse_goss(params),
            use_missing=_parse_use_missing(params),
            categorical_features=_parse_categorical(params),
        )
        return PythonObject(alloc=fitted.model.copy())
    # The plain gbdt fork. `resolve_or_defer`, not `resolve`: `model.fit`
    # refuses a GPU bootstrap itself and names `train_gpu` while doing it,
    # which beats the generic sentence, so a typed request is handed on
    # unchanged and read there. A defaulted bundle is dropped here, because
    # deferring it would let a default raise.
    var bootstrap = boot_req.resolve_or_defer(device == CPU_DEVICE)
    # CatBoost's ordered target statistics, catalog A19/A36, and the ONE
    # reroute in this function.
    #
    # `model.fit` takes no `SimpleCtrConfig` and never will: the bundle is a
    # property of the binning and `model.fit` bins through `fit_bins`
    # directly, while the type that binds a bundle to a matrix is
    # `trainset.Dataset`. So a fit that asked for CTR columns builds the
    # dataset the mechanism already lives on and trains it through
    # `trainset.train_dataset`, which is the same trainer `model.fit` would
    # have reached -- `boosting.train` on the dense CPU arm and
    # `train_gpu.train_gpu` on the device one -- with the CTR columns present
    # in the binned matrix.
    #
    # **A fit that did not ask takes the untouched call below and nothing
    # about it moves.** The reroute is guarded on `is_active()`, which is
    # false for `ctr='off'`, the default under every grow policy but
    # `symmetrictree`; `_parse_params` has already refused an active bundle
    # for every entry point that is not this one.
    var ctr = _parse_ctr(params)
    if ctr.is_active():
        # `keep_raw=False`: nothing here calls `subset`, and the raw copy
        # would be a second `n_rows * n_features` buffer for a fit that has
        # the caller's matrix borrowed already.
        #
        # No group and no init_score: `fit` reads neither, so passing empty
        # lists is what this entry point already means, not a capability
        # dropped on the way through.
        var ctr_dataset = Dataset(
            features,
            nr,
            nf,
            target.copy(),
            weights.copy(),
            List[Int](),
            List[Float64](),
            List[String](),
            _parse_categorical(params),
            Int(py=params["max_bin"]),
            _parse_use_missing(params),
            False,
            ctr^,
        )
        var ctr_model = mojo_train_dataset(
            ctr_dataset,
            Int(py=objective),
            bp,
            Float64(py=params["alpha"]),
            device,
            _parse_bagging(params),
            _parse_goss(params),
            bootstrap,
        )
        return PythonObject(alloc=ctr_model^)
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
        bootstrap=bootstrap,
    )
    return PythonObject(alloc=model^)


def distributed_train_local(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    objective: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train over a world of `params["num_machines"]` ranks hosted in this
    process with LightGBM's `tree_learner` (`params["tree_learner"]`:
    serial, data, feature, voting) and `params["top_k"]`. Same buffers and
    params dict as `fit` otherwise; CPU only; returns the same Model handle
    `fit` returns."""
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = f64_view(Int(py=x_addr), nr * nf)
    var target = f64_buffer(Int(py=y_addr), nr)
    # `score_function_ok` is left at its default of False here and passed
    # True at every other entry point below, and this is the one place the
    # difference is real rather than cautious: the distributed prototype
    # elects a split through `distributed_strategies`, which calls
    # `split.find_best_split` itself and leaves `score_function` at its
    # `SCORE_L2` default, so a Cosine fit routed here would be an L2 tree
    # reported as a Cosine one. Refused by name instead.
    var bp = _parse_params(
        params,
        nf,
        cpu=True,
        entry=String("distributed_train_local"),
        # Honored, and this is the one flag on this call that is not a
        # refusal. `train_local_world` shrinks each rank's tree by
        # `BoosterParams.learning_rate` exactly as the serial trainer does,
        # and the world is hosted in this process, so `nr` is the whole
        # training set rather than a shard: the row count CatBoost's formula
        # reads is the one it would read for the same data trained serially.
        auto_lr_ok=True,
        auto_lr_rows=nr,
        auto_lr_objective=Int(py=objective),
        # distributed.train_distributed_run threads the flag into _fill_grad_hess
        # and folds the resolved value into its schema marker
        # (distributed._push_derivative_precision), so two workers at different
        # precisions cannot agree on a histogram schema by accident.
        derivative_precision_ok=True,
    )
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("distributed_train_local (tree_learner other than 'serial')"),
    )
    var weights = _parse_weights(params, nr)
    var model = train_local_world(
        features,
        nr,
        nf,
        target,
        Int(py=objective),
        bp,
        Int(py=params["max_bin"]),
        weights,
        Float64(py=params["alpha"]),
        Int(py=params["num_machines"]),
        String(py=params["tree_learner"]),
        Int(py=params["top_k"]),
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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var target = f64_buffer(Int(py=y_addr), nr)
    var bp = _parse_params(
        params,
        nf,
        unbundled="fit_custom",
        entry=String("fit_custom"),
        score_function_ok=True,
        auto_lr_reason=String(
            "a custom objective is a pair of derivative buffers, not a loss"
            " function, and CatBoost's coefficient table is keyed by one"
            " (GetTargetType, options_helper.cpp:181-194). There is no"
            " target type to look the coefficients up with"
        ),
    )
    # `boosting._check_bootstrap` refuses the pair outright rather than for
    # want of wiring: a callback's derivatives are the caller's, and a draw
    # would rescale them behind the caller's back.
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("fit_custom (a custom objective)"),
    )
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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var target = f64_buffer(Int(py=y_addr), nr)
    # Neither opt-in, and `leaf_estimation_ok` in particular is a correction:
    # this routes to `custom_metric.fit_with_metrics`, NOT to
    # `boosting.train_with_valid`. `train_with_valid` does implement the extra
    # Newton steps; that round loop does not read
    # `TreeParams.extra.leaf_estimation_iterations` at all. The two share a
    # name and not a body, and opting in on the strength of the name produced
    # bit-identical arms until `test_catboost_reachability` demanded they
    # differ.
    var bp = _parse_params(
        params,
        nf,
        unbundled="fit_with_metrics",
        score_function_ok=True,
        auto_lr_reason=String(_AUTO_LR_EVAL_SET_REASON),
        # custom_metric.train_with_callbacks passes the flag at its one
        # derivative site.
        derivative_precision_ok=True,
    )
    # An eval_set, a callback, or linear_tree routes here rather than to
    # `boosting.train_with_valid`, and `custom_metric`'s round loop does not
    # call `sampling.bootstrap_round`.
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("fit_with_metrics (an eval_set or callback fit)"),
    )
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
                f64_buffer(Int(py=spec[1]), rows * n_features),
                rows,
                f64_buffer(Int(py=spec[3]), rows),
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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var labels = int_buffer_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(
        params,
        nf,
        unbundled="fit_multiclass_with_metrics",
        score_function_ok=True,
        auto_lr_reason=String(_AUTO_LR_EVAL_SET_REASON),
        # custom_metric.train_multiclass_with_metrics passes it at both of its
        # derivative sites.
        derivative_precision_ok=True,
    )
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("fit_multiclass_with_metrics (a softmax eval_set fit)"),
    )
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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var labels = int_buffer_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(
        params,
        nf,
        unbundled="fit_ranker_with_metrics",
        score_function_ok=True,
        auto_lr_reason=String(_AUTO_LR_EVAL_SET_REASON),
    )
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("fit_ranker_with_metrics (a LambdaRank eval_set fit)"),
    )
    var advanced = _parse_advanced_rank_params(params)
    var positions = _parse_positions(params, nr)
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
        advanced.base,
        Int(py=params["max_bin"]),
        weights,
        _parse_bagging(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
        advanced=advanced,
        positions=positions,
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
        weight = f64_buffer(weight_addr, nr)

    if kind == _METRIC_NDCG or kind == _METRIC_MAP:
        var scores = f64_buffer(Int(py=params["pred_addr"]), nr)
        var grades = int_buffer_from_f64(Int(py=params["y_addr"]), nr)
        var groups = groups_from_counts(_group_counts(params))
        var cutoff = Int(py=params["ndcg_at"])
        if kind == _METRIC_MAP:
            return PythonObject(
                mojo_map(scores, grades, groups, cutoff)
            )
        return PythonObject(mojo_ndcg(scores, grades, groups, cutoff))

    if kind == _METRIC_MULTI_LOGLOSS or kind == _METRIC_MULTI_ERROR:
        var nc = Int(py=params["n_classes"])
        var raw = f64_buffer(Int(py=params["pred_addr"]), nr * nc)
        var codes = int_buffer_from_f64(Int(py=params["y_addr"]), nr)
        for r in range(nr):
            _softmax_inplace(raw, r * nc, nc)
        if kind == _METRIC_MULTI_LOGLOSS:
            return PythonObject(multiclass_log_loss(raw, codes, nc, weight))
        return PythonObject(multiclass_error(raw, codes, nc, weight))

    var raw = f64_buffer(Int(py=params["pred_addr"]), nr)
    var target = f64_buffer(Int(py=params["y_addr"]), nr)
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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var labels = int_buffer_from_f64(Int(py=y_addr), nr)
    # Read the device first, for the reason `fit` does.
    var device = _parse_device(params)
    var bp = _parse_params(
        params,
        nf,
        cpu=device == CPU_DEVICE,
        entry=String("the multiclass trainers"),
        score_function_ok=True,
        # Honored. CatBoost's table has MultiClass rows, on both task types,
        # for `boost_from_average = false` -- which is what
        # `catboost_boost_from_average_default` returns for MultiClass, so
        # the row exists for exactly the run this entry point makes.
        auto_lr_ok=True,
        auto_lr_rows=nr,
        auto_lr_objective=_MULTICLASS_OBJECTIVE,
        # boosting._boost_rounds_multiclass and
        # boosting_sparse.train_multiclass_sparse both pass the flag; the GPU
        # multiclass trainer refuses float64 by name.
        derivative_precision_ok=True,
    )
    # CatBoost's `bootstrap_type`, HONORED on the CPU arm since
    # `boosting._boost_rounds_multiclass` took a bundle: one draw per round,
    # shared by every class's tree. `model.fit_multiclass` refuses the GPU arm
    # itself and names `train_multiclass_gpu`, so this defers to it rather than
    # restating the routing. Parsing here also means `bagging_temperature`
    # beside MVS is still caught before anything else runs.
    var bootstrap = _parse_bootstrap_request(params).resolve_or_defer(
        device == CPU_DEVICE
    )
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
        bootstrap=bootstrap,
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
    var csc = csc_from_params(params)
    var nr = csc.n_rows
    var target = f64_buffer(Int(py=y_addr), nr)
    var bp = _parse_params(
        params,
        csc.n_features,
        entry=String("a sparse (CSC) fit"),
        score_function_ok=True,
        # Honored. `csc.n_rows` is the train row count whether or not the
        # matrix is stored compressed: CatBoost's formula reads how many
        # objects there are, not how they are laid out.
        auto_lr_ok=True,
        auto_lr_rows=nr,
        auto_lr_objective=Int(py=objective),
        # boosting_sparse.train_sparse passes the flag.
        derivative_precision_ok=True,
    )
    # HONORED on the CPU arm since `boosting_sparse.train_sparse`'s round loop
    # took a bundle and began calling `sampling.bootstrap_round`.
    # `model_sparse.fit_csc` refuses the device arm itself and names
    # `train_gpu_sparse`, so this defers to it.
    var device = _parse_device(params)
    var bootstrap = _parse_bootstrap_request(params).resolve_or_defer(
        device == CPU_DEVICE
    )
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
        _parse_categorical(params),
        device=device,
        bootstrap=bootstrap,
    )
    return PythonObject(alloc=model^)


def fit_multiclass_csc(
    y_addr: PythonObject,
    n_classes: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a multiclass model on a sparse matrix. Labels arrive as
    float64 in 0..n_classes-1."""
    var csc = csc_from_params(params)
    var nr = csc.n_rows
    var labels = int_buffer_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(
        params,
        csc.n_features,
        entry=String("a sparse (CSC) fit"),
        score_function_ok=True,
        # Honored, for the reasons `fit_csc` and `fit_multiclass` are.
        auto_lr_ok=True,
        auto_lr_rows=nr,
        auto_lr_objective=_MULTICLASS_OBJECTIVE,
        # boosting_sparse.train_multiclass_sparse passes the flag.
        derivative_precision_ok=True,
    )
    # HONORED on the CPU arm since `boosting_sparse.train_multiclass_sparse`
    # took a bundle; the sparse device trainer is refused inside
    # `model_sparse.fit_multiclass_csc`, which names it.
    var device = _parse_device(params)
    var bootstrap = _parse_bootstrap_request(params).resolve_or_defer(
        device == CPU_DEVICE
    )
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
        _parse_categorical(params),
        device=device,
        bootstrap=bootstrap,
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
    _refuse_gpu_sparse(
        params, Int(py=params["n_rows"]), Int(py=params["n_features"]), 1
    )
    _store(mojo_predict_csr(m[], csr_from_params(params)), out_addr)
    return PythonObject(None)


def predict_raw_csr(
    model: PythonObject, params: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Raw-score predictions for a sparse matrix (see `predict_csr` on the
    device)."""
    var m = model.downcast_value_ptr[Model]()
    _refuse_gpu_sparse(
        params, Int(py=params["n_rows"]), Int(py=params["n_features"]), 1
    )
    _store(mojo_predict_raw_csr(m[], csr_from_params(params)), out_addr)
    return PythonObject(None)


def predict_proba_csr(
    model: PythonObject, params: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Multiclass probabilities for a sparse matrix, row-major
    `[r * n_classes + k]`, into a buffer of length n_rows * n_classes (see
    `predict_csr` on the device)."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    _refuse_gpu_sparse(
        params,
        Int(py=params["n_rows"]),
        Int(py=params["n_features"]),
        m[].booster.n_classes,
    )
    _store(mojo_predict_proba_csr(m[], csr_from_params(params)), out_addr)
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


def _parse_advanced_rank_params(
    params: PythonObject,
) raises -> AdvancedRankParams:
    """The advanced ranking config from the params dict, on top of
    `_parse_rank_params`: `label_gain` (a float64 buffer of `n_label_gain`
    entries; 0 keeps LightGBM's default table),
    `lambdarank_position_bias_regularization`, `pair_sampling_rate`,
    `pair_sampling_seed`, `max_dcg_cutoff`. Every key is optional so a
    caller that never heard of them gets `AdvancedRankParams.default()` with
    the base parameters, which routes to `ranking.train_ranker` unchanged.
    The trainer validates; see ranking_advanced.check_advanced_rank_params.
    """
    var out = AdvancedRankParams.default()
    out.base = _parse_rank_params(params)
    var n_gain = Int(py=params.get("n_label_gain", PythonObject(0)))
    if n_gain > 0:
        out.gain = LabelGain(
            f64_buffer(Int(py=params["label_gain_addr"]), n_gain)
        )
    out.position_bias_regularization = Float64(
        py=params.get(
            "lambdarank_position_bias_regularization", PythonObject(0.0)
        )
    )
    out.pair_sampling_rate = Float64(
        py=params.get("pair_sampling_rate", PythonObject(1.0))
    )
    out.pair_sampling_seed = Int(
        py=params.get("pair_sampling_seed", PythonObject(5))
    )
    out.max_dcg_cutoff = Int(py=params.get("max_dcg_cutoff", PythonObject(0)))
    return out^


def _parse_positions(params: PythonObject, n_rows: Int) raises -> PositionMap:
    """LightGBM's `Dataset.position`: a per-row integer code buffer of
    `n_position_rows` entries (float64 at this boundary, like every other
    column), densified by first appearance. 0 rows means no position column,
    the ordinary LambdaRank case."""
    var n = Int(py=params.get("n_position_rows", PythonObject(0)))
    if n == 0:
        return PositionMap.absent()
    if n != n_rows:
        raise Error(
            "position must have one entry per row; got ", n, " for ", n_rows
        )
    var codes = int_buffer_from_f64(Int(py=params["position_addr"]), n)
    return positions_from_codes(codes).positions.copy()


def _group_counts(params: PythonObject) raises -> List[Int]:
    """Per-query row counts (LightGBM's `group`) from the params dict. They
    travel as float64 like every other buffer at this boundary."""
    return int_buffer_from_f64(
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
    var features = f64_view(Int(py=x_addr), nr * nf)
    var labels = int_buffer_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(
        params,
        nf,
        unbundled="fit_ranker",
        entry=String("fit_ranker"),
        score_function_ok=True,
        # `auto_lr_ok=True` on a ranker looks wrong and is not. The three
        # inputs all exist here, so the entry point is not what disqualifies
        # the run; what disqualifies it is that `GetTargetType` maps
        # LambdaRank to Unknown and the table has no row. Handing the
        # objective over and letting `AutoLearningRateParams.fires` say so
        # keeps that judgement in the module that owns the table, instead of
        # copying "ranking has no row" into this file where it would go
        # stale the day CatBoost adds one.
        auto_lr_ok=True,
        auto_lr_rows=nr,
        auto_lr_objective=_LAMBDARANK_OBJECTIVE,
    )
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("fit_ranker (the LambdaRank trainers)"),
    )
    var weights = _parse_weights(params, nr)
    var advanced = _parse_advanced_rank_params(params)
    var positions = _parse_positions(params, nr)
    if advanced_ranking_requested(advanced, positions):
        # Custom label_gain, position bias, pair sampling, or a decoupled
        # maxDCG cutoff: ranking_advanced's loop. The learned position
        # biases are training state and are not part of the model.
        var fitted = mojo_fit_ranker_advanced(
            features,
            nr,
            nf,
            labels,
            _group_counts(params),
            bp,
            advanced,
            positions,
            Int(py=params["max_bin"]),
            weights,
            _parse_bagging(params),
            use_missing=_parse_use_missing(params),
            categorical_features=_parse_categorical(params),
        )
        return PythonObject(alloc=fitted.model.copy())
    var model = mojo_fit_ranker(
        features,
        nr,
        nf,
        labels,
        _group_counts(params),
        bp,
        advanced.base,
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
    var scores = f64_buffer(Int(py=scores_addr), nr)
    var labels = int_buffer_from_f64(Int(py=y_addr), nr)
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


# How much work one row of a host prediction is worth, in the histogram-op
# equivalents `parallel.plan_tasks` compares against its grain: one indirect
# load plus a dependent walk down the tree, which is several loads deep for a
# 31-leaf tree.
#
# This is the same quantity as `_TRAVERSAL_ROW_OPS` in
# src/mojotrees/boosting.mojo and carries the same value. It is written out
# again rather than imported because that name is module-private there and
# this lane does not own that file; if the two ever have to move together,
# they are found by grepping for the number and for both names. Neither has
# been measured.
#
# It is a scheduling estimate and nothing more. Every block below writes only
# its own output slots, so the values are the same at one task and at sixty,
# and this can be retuned without changing an output.
comptime _PREDICT_TRAVERSAL_ROW_OPS = 8


def _predict_row_ops(n_rows: Int, n_features: Int, n_walks: Int) -> Int:
    """The work estimate the host row walks hand `dispatch_rows`.

    `n_walks` is the number of trees one row is walked through, which is the
    iteration count for a single-output model and the iteration count times
    the class count for a multiclass one. `n_features` is charged once per row
    for the bin reads the walk makes, which is the same term
    `Booster.predict_batch_range` charges for its gather; the binning itself
    is not in here, because it happens once for the whole matrix before the
    fan-out and is `dispatch_feature_rows`'s own workload with its own
    estimate.

    Only the leaf walks use this now. It is a scheduling estimate, so
    overstating or understating it moves no value; understating merely keeps a
    small batch on the serial path.
    """
    return n_rows * (n_features + n_walks * _PREDICT_TRAVERSAL_ROW_OPS)


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
    `raw_score` is nonzero, the response scale otherwise.

    One matrix, binned once, then the ensemble walk over row blocks. Both
    halves are `Model.predict_batch`'s, which this now calls with an explicit
    `CPU_DEVICE`, so the row-at-a-time entry point and the batch entry point
    are one path rather than two.

    WHY IT IS NOT A PER-ROW LOOP ANY MORE, which is a correctness argument
    before it is a speed one. `dispatch_rows` takes a NON-RAISING
    `def (Int, Int) -> None`, so nothing that can raise may appear inside a
    parallel block. `BinMapper.bin_row` raises (it checks the row's length),
    and `Model.predict_range` and `Model.predict_raw_range` inherit that
    because binning is their first statement. So a per-row loop that bins
    inside the block cannot be parallelized at all, whatever it is handed.
    Binning the whole matrix first moves the only raising call out, and it
    moves it to a place where it still runs: `BinMapper.transform` performs
    the same length validation once for the matrix that `bin_row` performed
    once per row, and it raises from here, on the caller's stack, rather than
    from inside a worker.

    THE MOVE IS LOOP INVARIANT, and this is the part worth stating rather
    than assuming. `bin_row` validates `len(row) == n_features`; every row of
    a column-major matrix has exactly `n_features` entries by construction,
    so the per-row check answered the same question `n_rows` times and
    `transform`'s `len(features) == n_rows * n_features` is that same
    question asked once. Nothing else in the per-row body depended on the
    row: the ensemble walk reads the bins and the trees, and the trees do not
    change between rows.

    Bit-identity. A row's bins are the same values either way (`transform`'s
    padded search and `bin_value`'s plain one are documented to agree, and
    `transform` appends the same CTR columns `bin_row` appends), and the walk
    is `Booster.predict_batch_range`, which computes each row's tree sum in
    ascending tree order exactly as `predict_raw_bins_range` did here. The
    row axis is the only axis split, so no Float64 sum is reassociated.

    The one consequence a caller should know: the binned matrix is now
    materialized, `n_rows * n_features` bytes of it, where the per-row loop
    held one row at a time. That is the same footprint `predict_batch` has
    always had on this input, and `build_view=False` inside
    `Model.predict_batch` keeps it to one copy rather than two.
    """
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    var raw = Int(py=raw_score) != 0
    var features = f64_view(Int(py=x_addr), nr * nf)
    # `CPU_DEVICE` explicitly, not `AUTO_DEVICE`: this entry point predates
    # the device vocabulary and has always run on the host, and
    # `decide_device` answers an explicit CPU request with CPU at every shape
    # on every machine, so routing through `predict_batch` cannot move a
    # legacy caller onto an accelerator behind its back.
    _store(m[].predict_batch(features, nr, rng, raw, CPU_DEVICE), out_addr)
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
    softmax probabilities otherwise.

    One matrix, binned once, then the walk over row blocks, on exactly the
    terms `predict_range` sets out: this calls
    `MulticlassModel.predict_batch` with an explicit `CPU_DEVICE`, because a
    parallel block may not contain the raising `bin_row` and binning is
    loop invariant with respect to the walk.

    `MulticlassBooster.predict_batch_range` writes the same row-major
    `[r * n_classes + k]` layout this buffer expects, and it takes each row's
    softmax over that row's own scores, so nothing crosses a row boundary."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    var raw = Int(py=raw_score) != 0
    var features = f64_view(Int(py=x_addr), nr * nf)
    _store(m[].predict_batch(features, nr, rng, raw, CPU_DEVICE), out_addr)
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
    """The host leaf walk for a single-output model, over row blocks.

    Split out of `predict_leaf` so the device-aware entry point below can
    fall back to exactly this code rather than to a second copy of it.

    Two things are hoisted out of the fan-out, and both have to be. The
    ordinal tables are a property of the trees, not of any row. The BINNING
    is hoisted because it must be: `dispatch_rows` takes a non-raising
    closure and `BinMapper.bin_row` raises on a row of the wrong length, a
    check that is loop invariant here because every row of a column-major
    matrix has `n_features` entries by construction. `transform` asks that
    same question once, for the whole matrix, from this stack.

    `Tree.leaf_index_row` is the bins-in variant of `leaf_index_bins` reading
    `data.bin_at(r, f)`, which is the value a gathered row would have held,
    so the node reached is the same one and the ordinal read out of the same
    table is the same ordinal. `build_view=False` because this matrix is
    scored and never histogrammed.

    Every row writes its own `n_cols` output slots and reads nothing another
    row wrote, so the ordinals are the same at every task count."""
    var m = model.downcast_value_ptr[Model]()
    var n_cols = rng.n_iterations()
    var out_base = Int(py=out_addr)
    # One ordinal table per tree in the range, built once and shared by every
    # row: the mapping from node id to leaf ordinal is a property of the tree.
    var tables = m[].booster.leaf_ordinals_range(rng)
    var data = m[].mapper.transform(features, n_rows, build_view=False)
    # Built here and captured, which is the shape `Booster.predict_batch_range`
    # uses. `MutUntrackedOrigin` carries no origin, so it cannot alias any of
    # the immutable captures, and constructing it outside keeps the closure to
    # calls that are known not to raise.
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=out_base
    )

    def apply(row_start: Int, row_end: Int) {imm}:
        for r in range(row_start, row_end):
            for i in range(n_cols):
                var node = m[].booster.trees[
                    rng.start + i
                ].leaf_index_row(data, r)
                out.unsafe_store(r * n_cols + i, Float64(tables[i][node]))

    dispatch_rows(
        apply, n_rows, _predict_row_ops(n_rows, n_features, n_cols)
    )


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
    """The host leaf walk for a multiclass model (see `_leaf_host`), over row
    blocks on the same terms and hoisting the same two things for the same two
    reasons."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var k = m[].booster.n_classes
    var n_rounds = rng.n_iterations()
    var n_cols = n_rounds * k
    var out_base = Int(py=out_addr)
    var tables = m[].booster.leaf_ordinals_range(rng)
    var data = m[].mapper.transform(features, n_rows, build_view=False)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=out_base
    )

    def apply(row_start: Int, row_end: Int) {imm}:
        for r in range(row_start, row_end):
            for i in range(n_rounds):
                for c in range(k):
                    var tree = (rng.start + i) * k + c
                    var node = m[].booster.trees[tree].leaf_index_row(data, r)
                    out.unsafe_store(
                        r * n_cols + i * k + c,
                        Float64(tables[i * k + c][node]),
                    )

    dispatch_rows(
        apply, n_rows, _predict_row_ops(n_rows, n_features, n_cols)
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
    _leaf_host(model, f64_view(Int(py=x_addr), nr * nf), nr, nf, rng, out_addr)
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
        model, f64_view(Int(py=x_addr), nr * nf), nr, nf, rng, out_addr
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
    once rather than per row.

    Still serial, and it is the one prediction-shaped loop in this file that
    is. The rows are as independent here as they are anywhere, and the same
    bit-identity argument is available to it, since a per-block explainer
    would run each row's TreeSHAP recursion on exactly the values a shared one
    runs it on. What stops it here is only that `explainer` and `row_out` are
    shared MUTABLE scratch, so each block needs its own
    `ContribExplainer.for_booster` and its own `row_out`, and that
    reconstruction re-walks every tree's node covers once per block. That is
    cheap and it is worth doing; it was left out of this lane to keep the
    change that closes the measured prediction gap as small as it could be.
    Not an oversight: measured, and next."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var rng = _iteration_slice(m[].n_iterations(), start, stop)
    if nr == 0:
        return PythonObject(None)
    var explainer = ContribExplainer.for_booster(m[].booster, nf)
    var width = explainer.width()
    var features = f64_view(Int(py=x_addr), nr * nf)
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
    var features = f64_view(Int(py=x_addr), nr * nf)
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
# The established path keeps its values and its signatures. `predict_range`,
# `predict_proba_range`, `predict_leaf`, and `predict_leaf_multiclass` are
# unchanged from a caller's side, so an estimator that has not moved over
# behaves exactly as before. Underneath, the first two now CALL
# `Model.predict_batch` and `MulticlassModel.predict_batch` with an explicit
# `CPU_DEVICE` and the second two bin up front and walk over row blocks; both
# are scheduling changes and not numerical ones (see the module docstring). (The older whole-model `predict`,
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
    var features = f64_view(Int(py=x_addr), nr * nf)
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
    var features = f64_view(Int(py=x_addr), nr * nf)
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
    var features = f64_view(Int(py=x_addr), nr * nf)
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
    var features = f64_view(Int(py=x_addr), nr * nf)
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
    var target = f64_buffer(Int(py=params["y_addr"]), n_rows)
    var weight = List[Float64]()
    if Int(py=params["weight_addr"]) != 0:
        weight = f64_buffer(Int(py=params["weight_addr"]), n_rows)
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
        var features = f64_view(Int(py=x_addr), nr * nf)
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
        var features = f64_view(Int(py=x_addr), nr * nf)
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
            f64_buffer(Int(py=base_addr), h[].n_outputs)
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


def _importance_tree_width(trees: List[Tree], nf: Int) -> Int:
    """The width `importance.*_importance` has to be run at for these trees.

    `nf` for every ensemble whose splits all land inside the caller's feature
    count, which is every ensemble this package produced before CTR columns
    could be reached from Python.

    **A CTR fit splits past it, legitimately.** Ordered target statistics
    append numeric columns to the binned matrix
    (`ctr_columns.build_ctr_train_columns`) and the trees split on them, so a
    tree can carry a feature id up to `mapper.n_total_features() - 1` while
    the caller's matrix is `mapper.n_features` wide. `split_importance` and
    `gain_importance` refuse a split past the width they are handed, and that
    refusal is right: a short buffer would otherwise be written past or a
    split silently dropped.

    So the width is taken from the TREES rather than from the mapper. That
    keeps this function correct for a model loaded from a file, whose mapper
    is reconstructed, and it needs no accessor that does not exist; the
    quantity actually wanted is "the largest id anybody splits on", and that
    is what this reads. One pass over the split arrays, only when the trees
    are already in memory, and it returns `nf` unchanged for the common case
    so nothing about an ordinary fit moves.
    """
    var width = nf
    for t in range(len(trees)):
        for i in range(len(trees[t].feature)):
            var f = trees[t].feature[i]
            if f >= width:
                width = f + 1
    return width


def _write_importance(
    trees: List[Tree], nf: Int, kind: Int, out_addr: Int
) raises:
    """Per-feature importance into a preallocated float64 buffer of length
    `nf`. `kind` is 0 for split counts and 1 for total gain.

    The buffer is the CALLER's width and the computation is the TREES' width,
    and on a CTR model those differ. Only the first `nf` entries are stored,
    so the caller gets importances for the columns it passed and the derived
    CTR columns are not reported. They are deliberately not folded back onto
    their source column either: a CTR column is a statistic of the TARGET
    keyed by that column, so the two carry different information and a sum
    would be a number that is neither.
    """
    if kind != 0 and kind != 1:
        raise Error("importance kind must be 0 (split) or 1 (gain)")
    var width = _importance_tree_width(trees, nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=out_addr
    )
    if kind == 0:
        var counts = split_importance(trees, width)
        for f in range(nf):
            out.unsafe_store(f, Float64(counts[f]))
    else:
        var gains = gain_importance(trees, width)
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

    The matrix is **borrowed**, not copied. It used to arrive through
    `f64_buffer`, which allocated and filled a second `n_rows * n_features`
    buffer that `Dataset.__init__` then only read: 400 MB at 1,000,000 x 50,
    on top of the caller's NumPy array and the column-major buffer the
    wrapper had already built, so a dataset construction held three copies of
    the matrix at once. It holds two now. Under `keep_raw` it used to hold
    four, because the constructor copies its own retained matrix regardless;
    that copy is the one that has to exist and it is now the only one.

    The borrow is sound on the same contract `fit` relies on: `basic.Dataset`
    holds the array it took the address of in `self._x`, and the call is
    synchronous, so the buffer outlives the binning that reads it.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = f64_view(Int(py=x_addr), nr * nf)

    var label = List[Float64]()
    if Int(py=params["label_addr"]) != 0:
        label = f64_buffer(Int(py=params["label_addr"]), nr)
    var weight = List[Float64]()
    if Int(py=params["weight_addr"]) != 0:
        weight = f64_buffer(Int(py=params["weight_addr"]), nr)
    var init_score = List[Float64]()
    if Int(py=params["init_score_addr"]) != 0:
        init_score = f64_buffer(Int(py=params["init_score_addr"]), nr)
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
        _parse_ctr(params),
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
    # CatBoost's `random_strength`, and this is the entry point the benchmark
    # arm actually trains through: `mojotrees.train(params, Dataset)` never
    # touches `model.fit`. `trainset.train_dataset` forks three ways and only
    # one of them computes the per-tree noise scale --
    #
    #   sparse            -> boosting_sparse.train_sparse, no scale
    #   dense + GPU       -> train_gpu.train_gpu, no scale
    #   dense + CPU       -> boosting.train -> _boost_rounds, WHICH COMPUTES IT
    #
    # so the declaration is the conjunction rather than the device test alone.
    # A sparse dataset resolves its device to the CPU like any other, so
    # testing the device by itself would have declared the sparse arm honored.
    # **The device test dropped 2026-08-17, the sparse test kept.** It read
    # `device == CPU_DEVICE and not d[].is_sparse` because only the dense CPU
    # round loop computed `random_score_scale`. Both arms of
    # `train_gpu._train_gpu_rounds` compute it now, so the device is no longer
    # what decides; `train_gpu_sparse` still does not, so `is_sparse` still is.
    var scale_is_computed = not d[].is_sparse
    # CatBoost's ordered target statistics. **THE VERDICT IS THE DATASET'S, NOT
    # THIS ENTRY POINT'S NAME.** The refusal in `_parse_params` advertises this
    # route as one of the two that can carry a bundle, and until 2026-08-17 it
    # could not, because the flag was a constant False here and the guard read
    # the train params without ever asking the dataset what it had built. So
    # `Dataset(params={"ctr": "catboost"})` followed by `train(params, dataset)`
    # raised, on the route its own error message tells the caller to use.
    #
    # `d[].ctr` is the honest answer and its docstring (`trainset.mojo:711`)
    # says why: it records what HAPPENED, not what was offered, and is set back
    # to `disabled()` when the policy declined. So `is_active()` True means the
    # CTR columns are IN the binned matrix this fit is about to train on, which
    # is precisely the condition `ctr_ok` names. Training then honors them by
    # construction, because they are columns and the grower sees columns.
    #
    # The inverse reading is what makes this narrow rather than a blanket
    # True: `is_active()` False on a numeric matrix means the bundle built
    # nothing, so there is nothing for this route to honor and an EXPLICIT
    # request should still be told so. That case is the guard's, not ours.
    var ctr_on_dataset = d[].ctr.is_active()
    # A dataset binned under one source rule and trained under another is a
    # distinct error from either, so it gets its own sentence naming both
    # rather than falling through to a message about entry points. It can only
    # be reached deliberately, since both values have to be non-off and
    # different, and the two rules select DIFFERENT COLUMNS
    # (`ctr_columns.mojo:426`): CatBoost's `one_hot_max_size` boundary against
    # our `bin overflow` one. The model would carry inference tables for the
    # columns the dataset built while the caller believed it asked for others.
    #
    # **GUARDED ON THE CALLER HAVING NAMED IT, for the same reason the guard in
    # `_parse_params` is.** A dense `Dataset` built with no `ctr` key takes
    # `SimpleCtrConfig.auto()`, which is `CTR_SOURCE_BIN_OVERFLOW`, and a
    # `symmetrictree` fit resolves the mode default to
    # `CTR_SOURCE_ONE_HOT_MAX_SIZE`. Those differ, so without this guard every
    # categorical fit through `Dataset` + `train` under the shipped default
    # would raise on a mismatch between two values the caller never typed. When
    # the caller named neither, the DATASET'S rule stands and stands silently:
    # it is the one that actually built the columns, and per the invariant
    # documented at `_parse_params`'s ctr guard nothing on this route reads the
    # train params' rule to contradict it.
    if ctr_on_dataset and Int(py=params.get("ctr_set", PythonObject(1))) != 0:
        var asked = _parse_ctr(params)
        if asked.is_active() and asked.source_rule != d[].ctr.source_rule:
            # Named rather than printed as the raw code. The two rules are
            # `comptime` Ints and a message reading "source rule 0" tells a
            # caller nothing it can act on.
            var built = (
                String("CTR_SOURCE_ONE_HOT_MAX_SIZE")
                if d[].ctr.source_rule == CTR_SOURCE_ONE_HOT_MAX_SIZE
                else String("CTR_SOURCE_BIN_OVERFLOW")
            )
            var wanted = (
                String("CTR_SOURCE_ONE_HOT_MAX_SIZE")
                if asked.source_rule == CTR_SOURCE_ONE_HOT_MAX_SIZE
                else String("CTR_SOURCE_BIN_OVERFLOW")
            )
            raise Error(
                "this dataset was binned with ctr source rule ",
                built,
                " and the train params ask for ",
                wanted,
                ": the CTR columns are built while the dataset is binned, so"
                " the rule in the train params arrives after the columns"
                " exist and cannot change them. The two rules select"
                " different categorical columns"
                " (ctr_columns.CTR_SOURCE_ONE_HOT_MAX_SIZE is CatBoost's"
                " one_hot_max_size boundary, CTR_SOURCE_BIN_OVERFLOW is the"
                " columns the category table could not hold), so this fit"
                " would train on one set and report the other. Pass the rule"
                " you want to mojotrees.Dataset(params={'ctr': ...}), or drop"
                " it from the train params and let the dataset's stand",
            )
    var model = mojo_train_dataset(
        d[],
        Int(py=params["objective"]),
        _parse_params(
            params,
            d[].num_feature(),
            cpu=device == CPU_DEVICE,
            entry=String("a Dataset fit"),
            ctr_ok=ctr_on_dataset,
            score_function_ok=True,
            random_strength_ok=scale_is_computed,
            # CatBoost's `leaf_estimation_iterations` and, on the same flag,
            # `boost_from_average`. **`not is_sparse`, which is WIDER than
            # `scale_is_computed` right above, and the difference is the
            # device.** `trainset.train_dataset` forks three ways: the dense
            # CPU arm is `boosting.train` and the GPU arm is
            # `train_gpu.train_gpu`, and both implement the extra Newton steps
            # and both thread `boost_from_average` into
            # `boosting._base_score`. Only the sparse arm
            # (`boosting_sparse.train_sparse`) implements neither, so only it
            # is refused. The per-tree noise scale one line up is narrower
            # because it is computed by the dense CPU round loops alone.
            #
            # This is the entry point the whole item is for.
            # `mojotrees.train(params, Dataset)` is what `bench/real_data`
            # trains through, so until this argument existed every
            # CatBoost-mode Logloss cell had CatBoost taking ten Newton steps
            # per leaf and mojotrees taking one, under a parity table that had
            # been transcribed from an RMSE fit where both take one.
            leaf_estimation_ok=not d[].is_sparse,
            # The same condition and the same three-way fork:
            # `boosting.train` and `train_gpu.train_gpu` both thread the value
            # into `boosting._base_score`, `boosting_sparse.train_sparse` does
            # not.
            boost_from_average_ok=not d[].is_sparse,
            catboost_defaults_objective=Int(py=params["objective"]),
            # Honored, and this is the entry point that makes the capability
            # worth having: `mojotrees.train(params, Dataset)` is what
            # `bench/real_data` trains through, and a CatBoost-mode arm whose
            # rate is not CatBoost's rate is not a comparison of defaults.
            # No fork to settle, unlike `random_strength_ok` right above:
            # all three of the sparse, dense-CPU and dense-GPU arms shrink by
            # `BoosterParams.learning_rate`.
            auto_lr_ok=True,
            auto_lr_rows=d[].num_data(),
            auto_lr_objective=Int(py=params["objective"]),
            # UNCONDITIONAL, and WIDER than either flag beside it. Every fork of
            # trainset.train_dataset honors the wide derivative or refuses it: the
            # dense CPU arm is boosting.train, the sparse arm is
            # boosting_sparse.train_sparse -- which DOES pass the flag, unlike
            # leaf_estimation_iterations -- and the GPU arm raises in
            # histogram.check_device_derivative_precision. This is the entry point
            # bench/real_data trains through, and a flag that was True at `fit`
            # alone would lose the parameter here, which is the defect
            # random_strength_ok and leaf_estimation_ok have each been once.
            derivative_precision_ok=True,
        ),
        Float64(py=params["alpha"]),
        device,
        _parse_bagging(params),
        _parse_goss(params),
        # The second reachable path, and the one `bench/real_data`'s dense
        # arm actually takes: `mojotrees.train(params, Dataset)` never
        # touches `model.fit`. `trainset.train_dataset` honors the bundle on
        # BOTH CPU arms now, dense and sparse, and refuses only the GPU arm --
        # by name, which is why this defers rather than restating the routing.
        _parse_bootstrap_request(params).resolve_or_defer(
            device == CPU_DEVICE
        ),
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
    # CatBoost's `bootstrap_type`. This entry point REFUSED the parameter
    # outright until `trainset.train_dataset_multiclass` grew a bundle to take,
    # because until then nothing behind it could draw. Both CPU arms honor it
    # now (dense through `boosting._boost_rounds_multiclass`, sparse through
    # `boosting_sparse.train_multiclass_sparse`), and the GPU arm refuses it by
    # name inside the trainer, which is what this defers to.
    var bootstrap = _parse_bootstrap_request(params).resolve_or_defer(
        device == CPU_DEVICE
    )
    var model = mojo_train_dataset_multiclass(
        d[],
        Int(py=params["n_classes"]),
        _parse_params(
            params,
            d[].num_feature(),
            cpu=device == CPU_DEVICE,
            entry=String("a Dataset fit"),
            score_function_ok=True,
            # Honored. The objective key on a multiclass Dataset fit is not
            # read by the trainer (it takes the class count instead), so the
            # code is named here rather than taken from the mapping.
            auto_lr_ok=True,
            auto_lr_rows=d[].num_data(),
            auto_lr_objective=_MULTICLASS_OBJECTIVE,
            # boosting._boost_rounds_multiclass passes the flag; the GPU multiclass
            # trainer refuses float64 by name.
            derivative_precision_ok=True,
        ),
        device,
        _parse_bagging(params),
        _parse_goss(params),
        bootstrap,
    )
    return PythonObject(alloc=model^)


def train_dataset_ranker(
    dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Train a LambdaRank model on a constructed dataset, whose `group`
    holds the per-query row counts."""
    var d = dataset.downcast_value_ptr[Dataset]()
    _ = _parse_bootstrap_request(params).resolve(
        False,
        String("a Dataset ranker fit (trainset.train_dataset_ranker)"),
    )
    var model = mojo_train_dataset_ranker_advanced(
        d[],
        _parse_params(
            params,
            d[].num_feature(),
            unbundled="train_dataset_ranker",
            score_function_ok=True,
            # `True` for the reason `fit_ranker` passes True: the inputs are
            # all here, and it is the coefficient table that has no
            # LambdaRank row, which is the module's judgement to make.
            auto_lr_ok=True,
            auto_lr_rows=d[].num_data(),
            auto_lr_objective=_LAMBDARANK_OBJECTIVE,
        ),
        _parse_advanced_rank_params(params),
        _parse_positions(params, d[].num_data()),
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
    # CatBoost's `bootstrap_type` on a continued run. `boosting.train_more` is
    # `_boost_rounds` with a `round_offset`, so it draws at the ABSOLUTE round
    # index and a continued run draws what an uninterrupted one would have --
    # honored, not refused. Two configurations it cannot continue, and both
    # already refuse themselves with a better sentence than the generic one:
    # a sparse dataset ("continued training has no sparse path") and MVS with
    # a DERIVED mvs_reg ("the derived value is the squared mean leaf-value norm
    # of the tree before"). `resolve_or_defer` hands a TYPED request on to
    # those messages and drops a DEFAULTED bundle here, so a default cannot
    # turn a working `booster_update` into a raise.
    var boot_req = _parse_bootstrap_request(params)
    var boot_can_continue = not d[].is_sparse and not (
        boot_req.params.mvs.enabled and not boot_req.params.mvs.reg_is_set
    )
    var added = mojo_update_dataset(
        m[],
        d[],
        _parse_params(
            params,
            d[].num_feature(),
            entry=String("a Dataset fit"),
            score_function_ok=True,
            # `trainset.update_dataset` has no fork to settle: it refuses a
            # sparse dataset by name ("continued training has no sparse
            # path"), takes no device argument at all, and calls
            # `boosting.train_more`, which is `_boost_rounds` with a
            # `round_offset`. So the scale IS computed, once per tree, at the
            # absolute round index -- which is the reason `_boost_rounds`
            # takes the offset: a continued run computes the model length an
            # uninterrupted run would have had. Unconditional True rather
            # than a sparse test, so a sparse continued fit gets
            # `update_dataset`'s own message about continued training instead
            # of a message about random_strength, which is not its problem.
            random_strength_ok=True,
            # CatBoost's `leaf_estimation_iterations`. Unconditional for the
            # reason `random_strength_ok` is unconditional here:
            # `trainset.update_dataset` refuses a sparse dataset by name,
            # takes no device argument, and calls `boosting.train_more`, which
            # is `_boost_rounds` with a round offset and reads the field at
            # boosting.mojo:2345. A sparse continued fit should hear about
            # continued training, not about leaf estimation.
            leaf_estimation_ok=True,
            # No `catboost_defaults_objective`. Continued training resolves no
            # mode default, and the reason is the one
            # `_AUTO_LR_CONTINUED_REASON` gives for the learning rate: the
            # trees already in the model were grown under whatever this
            # resolved on the first fit, and resolving it again here could
            # append ten-step leaves to a one-step ensemble without either
            # half being wrong on its own.
            #
            # And no `boost_from_average_ok`, which is the one place its
            # verdict parts company with `leaf_estimation_ok`'s.
            # `boosting.train_more` reads `leaf_estimation_iterations`, so the
            # extra steps are honest here; it starts from the base score
            # already stored on the model and never calls
            # `boosting._base_score`, so a `false` would be accepted and
            # ignored. Left at its refusing default, so a continued fit that
            # names the parameter hears that the original fit decided it.
            auto_lr_reason=String(_AUTO_LR_CONTINUED_REASON),
            # boosting.train_more is boosting._boost_rounds with a round offset, and
            # that loop passes the flag. A top-up at a different precision from the
            # original fit is a caveat about the ensemble and not an ignored
            # parameter: the rounds this call adds really are computed at the
            # precision named here.
            derivative_precision_ok=True,
        ),
        Float64(py=params["alpha"]),
        _parse_bagging(params),
        _parse_goss(params),
        boot_req.resolve_or_defer(boot_can_continue),
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
    # `booster_update`'s rule, minus one clause: the derived-`mvs_reg` case
    # cannot arise on a softmax round because every softmax round refuses it
    # (`sampling.check_mvs_reg_is_set`), so the only thing left that cannot
    # continue is a sparse dataset, which `trainset.update_dataset_multiclass`
    # refuses by name.
    var boot_req = _parse_bootstrap_request(params)
    var added = mojo_update_dataset_multiclass(
        m[],
        d[],
        _parse_params(
            params,
            d[].num_feature(),
            entry=String("a Dataset fit"),
            score_function_ok=True,
            auto_lr_reason=String(_AUTO_LR_CONTINUED_REASON),
            # trainset.update_dataset_multiclass reaches
            # boosting.train_multiclass_more, i.e. _boost_rounds_multiclass, which
            # passes the flag.
            derivative_precision_ok=True,
        ),
        _parse_bagging(params),
        _parse_goss(params),
        boot_req.resolve_or_defer(not d[].is_sparse),
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
