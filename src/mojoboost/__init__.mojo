"""mojoboost: gradient boosted decision trees in Mojo."""

from .bagging import (
    DEFAULT_BAGGING_SEED,
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
    sample_rows,
)
from .categorical import (
    CAT_BITSET_WORDS,
    CategoricalParams,
    CategoricalSpec,
    cat_add,
    cat_contains,
    cat_empty,
    cat_pool_contains,
    find_best_categorical_split,
    fit_categorical_spec,
)
from .gain import leaf_score
from .binning import BinMapper, BinnedMatrix, bin_equal_width, fit_bins
from .histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
)
from .goss import (
    GossParams,
    GossSelection,
    apply_goss_scaling,
    goss_importance,
    goss_round,
    goss_select,
)
from .interaction import InteractionConstraints, extend_branch
from .monotone import (
    MONOTONE_DECREASING,
    MONOTONE_FREE,
    MONOTONE_INCREASING,
    MonotoneConstraints,
    OutputBounds,
)
from .sampling import (
    DEFAULT_FEATURE_FRACTION_SEED,
    check_feature_fraction,
    check_feature_fractions,
    sample_without_replacement,
    select_node_features,
    select_tree_features,
    selection_count,
)
from .split import SplitInfo, find_best_split, soft_threshold_l1
from .tree import Tree, TreeParams, grow_tree, node_bounds
from .boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    DEFAULT_FAIR_C,
    DEFAULT_TWEEDIE_VARIANCE_POWER,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
    Booster,
    BoosterParams,
    IterationRange,
    MulticlassBooster,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
    train,
    train_more,
    train_multiclass,
    train_multiclass_more,
    train_multiclass_with_valid,
    train_with_valid,
)
from .contrib import (
    ContribExplainer,
    predict_contrib,
    predict_contrib_bins,
    predict_contrib_bins_multiclass,
    predict_contrib_multiclass,
    tree_expected_value,
)
from .trainset import (
    Dataset,
    train_dataset,
    train_dataset_multiclass,
    train_dataset_ranker,
    update_dataset,
    update_dataset_multiclass,
)
from .collective import (
    STATUS_INVALID_PARAM,
    STATUS_INVALID_TARGET,
    STATUS_INVALID_WEIGHT,
    STATUS_LAYOUT_MISMATCH,
    STATUS_OK,
    STATUS_SHAPE_MISMATCH,
    STATUS_UNSUPPORTED,
    Collective,
    LocalCollective,
    agree_equal_ints,
    agree_status,
    status_message,
)
from .distributed import (
    DataShard,
    allreduce_histogram,
    grow_tree_distributed,
    partition_rows,
    partition_values,
    train_distributed,
)
from .device import (
    AUTO_DEVICE,
    CPU_DEVICE,
    GPU_DEVICE,
    device_name,
    gpu_available,
    parse_device,
    resolve_device,
)
from .objective import (
    EvalLossFn,
    GradHessFn,
    check_custom_grad_hess,
    mean_label,
    squared_error_grad_hess,
    squared_error_loss,
    train_custom,
    train_custom_with_valid,
)
from .custom_metric import (
    MetricFn,
    MetricSetFn,
    CustomMetric,
    MetricFitResult,
    MetricHistory,
    MetricMulticlassFitResult,
    MetricMulticlassTrainResult,
    MetricSuite,
    MetricTrainResult,
    RawValidSet,
    ValidSet,
    fit_multiclass_with_metrics,
    fit_ranker_with_metrics,
    fit_with_metrics,
    response_scale,
    train_custom_with_metrics,
    train_multiclass_with_metrics,
    train_ranker_with_metrics,
    train_with_metric,
    train_with_metrics,
)
from .histogram_gpu import GpuHistogramBuilder, build_histogram_gpu
from .gpu_runtime import (
    GpuSession,
    HazardTracker,
    MatrixIdentity,
    NoLifecycle,
    PhaseCounters,
    PoolLedger,
    ResidencyLedger,
    RoundLifecycle,
    SessionLifecycle,
    StagingRing,
    audit_round,
    bins_fingerprint,
)
from .gpu_predict import (
    GpuPredictor,
    flatten_booster,
    flatten_multiclass,
    flatten_trees,
    predict_gpu,
    predict_proba_gpu,
    predict_raw_gpu,
    predict_raw_multiclass_gpu,
    response_for_objective,
)
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
)
from .train_gpu import (
    OBJECTIVE_SOURCE_AUTO,
    OBJECTIVE_SOURCE_DEVICE,
    OBJECTIVE_SOURCE_HOST,
    SPLIT_SEARCH_AUTO,
    SPLIT_SEARCH_DEVICE,
    SPLIT_SEARCH_HOST,
    VALID_SCORE_AUTO,
    VALID_SCORE_DEVICE,
    VALID_SCORE_HOST,
    device_gradients,
    grow_tree_gpu,
    resolve_objective_source,
    resolve_split_search,
    resolve_valid_scoring,
    train_custom_gpu,
    train_gpu,
    train_gpu_with_valid,
    train_multiclass_gpu,
)
from .importance import gain_importance, split_importance
from .class_weight import (
    balanced_class_weights,
    balanced_sample_weight,
    binary_labels_to_codes,
    check_class_balance_params,
    check_class_weights,
    class_counts,
    class_weight_rows,
    scale_pos_weight_rows,
    unbalance_scale,
    unbalanced_sample_weight,
)
from .metrics import (
    average_precision,
    binary_accuracy,
    binary_auc,
    binary_error,
    binary_log_loss,
    check_metric_weight,
    cross_entropy_loss,
    fair_loss,
    gamma_deviance,
    gamma_loss,
    huber_loss,
    kullback_leibler,
    l1,
    l2,
    mape,
    multiclass_accuracy,
    multiclass_error,
    multiclass_log_loss,
    poisson_loss,
    quantile_loss,
    rmse,
    tweedie_loss,
)
from .model import Model, MulticlassModel, fit, fit_custom, fit_multiclass
from .sparse import (
    CscMatrix,
    CsrMatrix,
    SparseBinnedMatrix,
    SparseBinnedRows,
    csc_from_dense,
    default_bins,
    fit_bins_csc,
    transform_csc,
)
from .histogram_sparse import (
    NodeTotals,
    SparseEntryOrder,
    SparseNodeEntries,
    build_histogram_sparse,
    build_histogram_sparse_node,
    build_histogram_sparse_subset,
)
from .tree_sparse import (
    SparseTreeResult,
    grow_tree_sparse,
    predict_row_sparse,
)
from .boosting_sparse import (
    train_multiclass_sparse,
    train_sparse,
    train_sparse_with_valid,
)
from .model_sparse import (
    fit_csc,
    fit_multiclass_csc,
    predict_class_csr,
    predict_csr,
    predict_proba_csr,
    predict_raw_csr,
)
from .ranking import (
    DEFAULT_NDCG_EVAL_AT,
    DEFAULT_SIGMOID,
    DEFAULT_TRUNCATION_LEVEL,
    LAMBDARANK,
    MAX_RELEVANCE_LABEL,
    RankGroups,
    RankerParams,
    check_groups,
    check_labels,
    check_ranker_params,
    fit_ranker,
    groups_from_counts,
    groups_from_query_ids,
    label_gain,
    lambdarank_gradients,
    map_at_cutoffs,
    max_dcg,
    mean_average_precision,
    ndcg,
    ndcg_at_cutoffs,
    train_ranker,
    train_ranker_with_valid,
)
from .params import (
    MULTICLASS,
    SUPPORTED_KEYS,
    TrainConfig,
    objective_default_alpha,
    objective_display_name,
    objective_from_name,
    params_names_mojo_api_only,
    parse_params,
)
from .serialize import (
    load_model,
    load_multiclass_model,
    model_file_kind,
    save_model,
    save_multiclass_model,
)
