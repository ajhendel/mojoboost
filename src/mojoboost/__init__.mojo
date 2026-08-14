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
    CUSTOM,
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    IterationRange,
    MulticlassBooster,
    train,
    train_multiclass,
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
from .train_gpu import (
    grow_tree_gpu,
    train_custom_gpu,
    train_gpu,
    train_multiclass_gpu,
)
from .importance import gain_importance, split_importance
from .metrics import (
    binary_accuracy,
    binary_auc,
    binary_error,
    binary_log_loss,
    check_metric_weight,
    huber_loss,
    l1,
    l2,
    multiclass_accuracy,
    multiclass_error,
    multiclass_log_loss,
    quantile_loss,
    rmse,
)
from .model import Model, MulticlassModel, fit, fit_custom, fit_multiclass
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
    max_dcg,
    ndcg,
    ndcg_at_cutoffs,
    train_ranker,
    train_ranker_with_valid,
)
from .params import (
    MULTICLASS,
    SUPPORTED_KEYS,
    TrainConfig,
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
