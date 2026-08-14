"""mojoboost: gradient boosted decision trees in Mojo."""

from .bagging import (
    DEFAULT_BAGGING_SEED,
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
    sample_rows,
)
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
    MulticlassBooster,
    train,
    train_multiclass,
    train_multiclass_with_valid,
    train_with_valid,
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
    binary_log_loss,
    multiclass_accuracy,
    multiclass_log_loss,
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
from .serialize import (
    load_model,
    load_multiclass_model,
    save_model,
    save_multiclass_model,
)
