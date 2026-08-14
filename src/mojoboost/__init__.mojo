"""mojoboost: gradient boosted decision trees in Mojo."""

from .binning import BinMapper, BinnedMatrix, bin_equal_width, fit_bins
from .histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
)
from .split import SplitInfo, find_best_split
from .tree import Tree, TreeParams, grow_tree
from .boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_multiclass,
    train_multiclass_with_valid,
    train_with_valid,
)
from .importance import split_importance
from .metrics import (
    binary_accuracy,
    binary_auc,
    binary_log_loss,
    multiclass_accuracy,
    multiclass_log_loss,
    rmse,
)
from .model import Model, fit
from .serialize import load_model, save_model
