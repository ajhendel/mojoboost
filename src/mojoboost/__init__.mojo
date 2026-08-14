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
from .interaction import InteractionConstraints, extend_branch
from .split import SplitInfo, find_best_split, soft_threshold_l1
from .tree import Tree, TreeParams, grow_tree
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
from .serialize import (
    load_model,
    load_multiclass_model,
    save_model,
    save_multiclass_model,
)
