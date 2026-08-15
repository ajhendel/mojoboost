"""End-to-end sparse models: CSC in, predictions out.

`fit_csc` bins a sparse matrix with sparse quantile binning, trains on it
without ever densifying, and returns an ordinary `Model`. The model carries
nothing sparse-specific: it is byte-for-byte the same structure a dense fit
produces, so it serializes with `save_model`, loads with `load_model`, and
predicts on dense rows through `Model.predict`. Nothing in the serialization
format changes for sparse.

`predict_csr` is the sparse prediction path. It takes CSR because prediction
is row-oriented, and it never materializes a dense row: at each node it looks
up that one feature in the row's own entries by binary search, falling back
to the feature's zero bin when the entry is absent. Cost per row is
O(depth * log nnz_in_row), against O(n_features) to densify a row.

Categorical features are carried end to end, not excluded: `fit_csc` fits
their tables with `fit_categorical_spec_csc`, the grower searches category
partitions through the matrix's own `cats`, and `predict_csr` resolves a
stored code through `BinMapper.bin_value` (so an unseen code takes the
unknown bin) and an absent entry through the feature's zero bin, which for a
categorical column is category code 0's bin -- or the unknown bin when code 0
was not kept. `sparse.absent_is_unknown` is how a caller finds out which of
the two a fitted column got.
"""

from std.math import exp

from .binning import BinMapper
from .boosting import (
    BINARY_LOGISTIC,
    POISSON,
    BoosterParams,
    _sigmoid,
    _softmax_inplace,
)
from .bagging import BaggingParams
from .boosting_sparse import (
    prepare_bundling_csc,
    train_multiclass_sparse,
    train_sparse,
)
from .goss import GossParams
from .sampling import ClassBaggingParams
from .model import Model, MulticlassModel
from .sparse import (
    CscMatrix,
    CsrMatrix,
    default_bins,
    fit_bins_csc,
    transform_csc,
)
from .tree import Tree


def fit_csc(
    csc: CscMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
    init_score: List[Float64] = [],
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
) raises -> Model:
    """Fit on a sparse matrix without densifying it. Same arguments and
    semantics as `fit`; the implicit zeros are numerical zeros (see
    sparse.mojo).

    `categorical_features` names the columns to bin as category tables rather
    than quantiles, exactly as in `fit`: `fit_categorical_spec_csc` counts an
    absent entry as category code 0, which is what it is, and the fitted
    tables, tie-breaking, and unknown-category bin are the dense ones.
    `init_score` and `class_bagging` carry their `train_sparse` meanings.

    `params.bundling` is exclusive feature bundling, and this is where it is
    fitted: `prepare_bundling_csc` either replaces the binned matrix with a
    bundled one and returns the view that reads it, or declines and returns
    the matrix untouched (which is the default, since `enable_bundle`
    defaults to off). Either way the model returned is the same shape -- the
    plan is training-time scaffolding and is dropped here, so `mapper` and
    the trees are in the original feature space and nothing about
    `save_model` changes."""
    var mapper = fit_bins_csc(
        csc, max_bins, categorical_features, use_missing
    )
    var prepared = prepare_bundling_csc(
        mapper, transform_csc(mapper, csc), params.bundling
    )
    var booster = train_sparse(
        prepared.data,
        target,
        objective,
        params,
        sample_weight,
        alpha,
        bagging,
        goss,
        init_score,
        class_bagging,
        prepared.bundling,
    )
    return Model(mapper^, booster^)


def fit_multiclass_csc(
    csc: CscMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassModel:
    """Fit a softmax multiclass model on a sparse matrix, labels in
    0..n_classes-1. `goss` draws one gradient-based sample per round, shared
    by every class's tree in that round, as in `fit_multiclass`.
    `params.bundling` carries its `fit_csc` meaning, and the one plan fitted
    here is shared by every class's tree."""
    var mapper = fit_bins_csc(
        csc, max_bins, categorical_features, use_missing
    )
    var prepared = prepare_bundling_csc(
        mapper, transform_csc(mapper, csc), params.bundling
    )
    var booster = train_multiclass_sparse(
        prepared.data,
        labels,
        n_classes,
        params,
        sample_weight,
        bagging,
        goss,
        prepared.bundling,
    )
    return MulticlassModel(mapper^, booster^)


def _check_csr(mapper: BinMapper, csr: CsrMatrix) raises:
    csr.validate()
    if csr.n_features != mapper.n_features:
        raise Error("csr n_features must equal the model's n_features")


def _row_bin(
    mapper: BinMapper,
    zero_bin: List[UInt8],
    csr: CsrMatrix,
    row: Int,
    feature: Int,
) -> Int:
    """Bin of (row, feature) from CSR, by binary search over the row's own
    entries. An absent entry is the implicit zero, whose bin is precomputed
    once per feature."""
    var lo = csr.row_offsets[row]
    var end = csr.row_offsets[row + 1]
    var hi = end
    while lo < hi:
        var mid = (lo + hi) // 2
        if csr.col_index[mid] < feature:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and csr.col_index[lo] == feature:
        return mapper.bin_value(feature, csr.values[lo])
    return Int(zero_bin[feature])


def _tree_value_csr(
    tree: Tree,
    mapper: BinMapper,
    zero_bin: List[UInt8],
    csr: CsrMatrix,
    row: Int,
) -> Float64:
    var node = 0
    while tree.feature[node] >= 0:
        var bin = _row_bin(mapper, zero_bin, csr, row, tree.feature[node])
        if tree.goes_left(node, bin):
            node = tree.left[node]
        else:
            node = tree.right[node]
    return tree.value[node]


def predict_raw_csr(model: Model, csr: CsrMatrix) raises -> List[Float64]:
    """Raw scores for every row of a sparse matrix (log-odds for
    BINARY_LOGISTIC)."""
    _check_csr(model.mapper, csr)
    var zero_bin = default_bins(model.mapper)
    var out = List[Float64](capacity=csr.n_rows)
    for r in range(csr.n_rows):
        var s = model.booster.base_score
        for t in range(len(model.booster.trees)):
            s += model.booster.learning_rate * _tree_value_csr(
                model.booster.trees[t], model.mapper, zero_bin, csr, r
            )
        out.append(s)
    return out^


def predict_csr(model: Model, csr: CsrMatrix) raises -> List[Float64]:
    """Response-scale predictions for every row of a sparse matrix
    (probability for logistic, expected count for poisson)."""
    var raw = predict_raw_csr(model, csr)
    if model.booster.objective == BINARY_LOGISTIC:
        for r in range(len(raw)):
            raw[r] = _sigmoid(raw[r])
    elif model.booster.objective == POISSON:
        for r in range(len(raw)):
            raw[r] = exp(raw[r])
    return raw^


def predict_raw_proba_csr(
    model: MulticlassModel, csr: CsrMatrix
) raises -> List[Float64]:
    """Per-class raw scores before the softmax, row-major
    `[r * n_classes + k]`."""
    _check_csr(model.mapper, csr)
    var zero_bin = default_bins(model.mapper)
    var k = model.booster.n_classes
    var n_rounds = len(model.booster.trees) // k
    var out = List[Float64](capacity=csr.n_rows * k)
    for r in range(csr.n_rows):
        for c in range(k):
            out.append(model.booster.base_scores[c])
        for i in range(n_rounds):
            for c in range(k):
                out[r * k + c] += (
                    model.booster.learning_rate
                    * _tree_value_csr(
                        model.booster.trees[i * k + c],
                        model.mapper,
                        zero_bin,
                        csr,
                        r,
                    )
                )
    return out^


def predict_proba_csr(
    model: MulticlassModel, csr: CsrMatrix
) raises -> List[Float64]:
    """Class probabilities for every row, row-major `[r * n_classes + k]`."""
    var raw = predict_raw_proba_csr(model, csr)
    for r in range(csr.n_rows):
        _softmax_inplace(raw, r * model.booster.n_classes,
                         model.booster.n_classes)
    return raw^


def predict_class_csr(
    model: MulticlassModel, csr: CsrMatrix
) raises -> List[Int]:
    """The argmax class for every row."""
    var raw = predict_raw_proba_csr(model, csr)
    var k = model.booster.n_classes
    var out = List[Int](capacity=csr.n_rows)
    for r in range(csr.n_rows):
        var argmax = 0
        for c in range(1, k):
            if raw[r * k + c] > raw[r * k + argmax]:
                argmax = c
        out.append(argmax)
    return out^
