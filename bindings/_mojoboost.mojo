"""CPython extension module for mojoboost.

Built with `bindings/build.sh` into `python/mojoboost/_mojoboost.so`; the
public Python surface is the sklearn-style wrapper in `python/mojoboost/`.

Data crosses the boundary as raw buffer addresses (integers) plus lengths:
the wrapper passes float64-contiguous buffers (column-major for feature
matrices) and keeps them alive for the duration of each call. Copies into
Mojo Lists happen here, so no Python buffer is retained after a call
returns. Trained models are returned as opaque handles owned by Python.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from mojoboost.bagging import BaggingParams
from mojoboost.categorical import CategoricalParams, CategoricalSpec
from mojoboost.contrib import ContribExplainer
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    BoosterParams,
    IterationRange,
    _softmax_inplace,
)
from mojoboost.custom_metric import (
    CustomMetric,
    MetricHistory,
    MetricSuite,
    RawValidSet,
    fit_multiclass_with_metrics as mojo_fit_multiclass_with_metrics,
    fit_ranker_with_metrics as mojo_fit_ranker_with_metrics,
    fit_with_metrics as mojo_fit_with_metrics,
    response_scale,
)
from mojoboost.metrics import (
    binary_auc,
    binary_error,
    binary_log_loss,
    huber_loss,
    l1,
    l2,
    multiclass_error,
    multiclass_log_loss,
    quantile_loss,
    rmse,
)
from mojoboost.device import (
    device_name as mojo_device_name,
    gpu_available as mojo_gpu_available,
    parse_device,
    resolve_device as mojo_resolve_device,
)
from mojoboost.goss import GossParams
from mojoboost.importance import gain_importance, split_importance
from mojoboost.interaction import InteractionConstraints
from mojoboost.monotone import MonotoneConstraints
from mojoboost.model import Model, MulticlassModel
from mojoboost.model import fit as mojo_fit
from mojoboost.model import fit_custom as mojo_fit_custom
from mojoboost.model import fit_multiclass as mojo_fit_multiclass
from mojoboost.objective import mean_label
from mojoboost.ranking import (
    RankerParams,
    fit_ranker as mojo_fit_ranker,
    groups_from_counts,
    ndcg as mojo_ndcg,
)
from mojoboost.serialize import (
    load_model,
    load_multiclass_model,
    save_model,
    save_multiclass_model,
)
from mojoboost.tree import Tree, TreeParams


@export
def PyInit__mojoboost() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojoboost")
        _ = m.add_type[Model]("Model")
        _ = m.add_type[MulticlassModel]("MulticlassModel")
        m.def_function[fit]("fit")
        m.def_function[fit_custom]("fit_custom")
        m.def_function[fit_with_metrics]("fit_with_metrics")
        m.def_function[fit_multiclass_with_metrics](
            "fit_multiclass_with_metrics"
        )
        m.def_function[fit_ranker_with_metrics]("fit_ranker_with_metrics")
        m.def_function[eval_metric]("eval_metric")
        m.def_function[fit_multiclass]("fit_multiclass")
        m.def_function[fit_ranker]("fit_ranker")
        m.def_function[ndcg]("ndcg")
        m.def_function[predict]("predict")
        m.def_function[predict_raw]("predict_raw")
        m.def_function[predict_proba]("predict_proba")
        m.def_function[predict_range]("predict_range")
        m.def_function[predict_proba_range]("predict_proba_range")
        m.def_function[predict_leaf]("predict_leaf")
        m.def_function[predict_leaf_multiclass]("predict_leaf_multiclass")
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
        m.def_function[gpu_available]("gpu_available")
        m.def_function[resolve_device]("resolve_device")
        return m.finalize()
    except e:
        abort(String("failed to create _mojoboost module: ", e))


def _f64_list(addr: Int, n: Int) raises -> List[Float64]:
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


def _int_list_from_f64(addr: Int, n: Int) raises -> List[Int]:
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(p.unsafe_load(i)))
    return out^


def _row(
    features: List[Float64], n_rows: Int, n_features: Int, r: Int
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
    params: PythonObject, n_features: Int
) raises -> BoosterParams:
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
    )
    return BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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
        _parse_device(params),
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
    predictions) is the one in src/mojoboost/objective.mojo.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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
    sets (see src/mojoboost/custom_metric.mojo for the metric contract).

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
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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

    var result = mojo_fit_with_metrics(
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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
# python/mojoboost/_eval.py; they are one contract and must move together.
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


def eval_metric(code: PythonObject, params: PythonObject) raises -> PythonObject:
    """One built-in metric (see the codes above) over buffers the caller
    owns, so the wrapper never reimplements a metric that metrics.mojo
    already defines.

    `params` holds `pred_addr`, `y_addr`, `weight_addr` (0 for unweighted),
    `n_rows`, and whatever the metric needs beyond that: `n_classes` for the
    multiclass metrics, `group_addr` / `n_groups` / `ndcg_at` for NDCG, and
    `alpha` for the quantile and Huber losses. Predictions are raw scores,
    the metric contract in custom_metric.mojo; the link a metric wants is
    applied here, so the binary metrics see probabilities and the multiclass
    ones see softmax rows.
    """
    var kind = Int(py=code)
    var nr = Int(py=params["n_rows"])
    var weight_addr = Int(py=params["weight_addr"])
    var weight = List[Float64]()
    if weight_addr != 0:
        weight = _f64_list(weight_addr, nr)

    if kind == _METRIC_NDCG:
        var scores = _f64_list(Int(py=params["pred_addr"]), nr)
        var grades = _int_list_from_f64(Int(py=params["y_addr"]), nr)
        return PythonObject(
            mojo_ndcg(
                scores,
                grades,
                groups_from_counts(_group_counts(params)),
                Int(py=params["ndcg_at"]),
            )
        )

    if kind == _METRIC_MULTI_LOGLOSS or kind == _METRIC_MULTI_ERROR:
        var nc = Int(py=params["n_classes"])
        var raw = _f64_list(Int(py=params["pred_addr"]), nr * nc)
        var codes = _int_list_from_f64(Int(py=params["y_addr"]), nr)
        for r in range(nr):
            _softmax_inplace(raw, r * nc, nc)
        if kind == _METRIC_MULTI_LOGLOSS:
            return PythonObject(multiclass_log_loss(raw, codes, nc, weight))
        return PythonObject(multiclass_error(raw, codes, nc, weight))

    var pred = _f64_list(Int(py=params["pred_addr"]), nr)
    var target = _f64_list(Int(py=params["y_addr"]), nr)
    if kind == _METRIC_L2:
        return PythonObject(l2(pred, target, weight))
    if kind == _METRIC_RMSE:
        return PythonObject(rmse(pred, target, weight))
    if kind == _METRIC_L1:
        return PythonObject(l1(pred, target, weight))
    if kind == _METRIC_QUANTILE:
        return PythonObject(
            quantile_loss(pred, target, Float64(py=params["alpha"]), weight)
        )
    if kind == _METRIC_HUBER:
        return PythonObject(
            huber_loss(pred, target, Float64(py=params["alpha"]), weight)
        )
    if kind == _METRIC_AUC:
        # AUC only reads the score order, so the raw margin serves.
        return PythonObject(binary_auc(pred, target, weight))
    var probs = response_scale(BINARY_LOGISTIC, pred)
    if kind == _METRIC_BINARY_LOGLOSS:
        return PythonObject(binary_log_loss(probs, target, weight))
    if kind == _METRIC_BINARY_ERROR:
        return PythonObject(binary_error(probs, target, 0.5, weight))
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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
        _parse_device(params),
        _parse_bagging(params),
        _parse_goss(params),
        use_missing=_parse_use_missing(params),
        categorical_features=_parse_categorical(params),
    )
    return PythonObject(alloc=model^)


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
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
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
    only a mojoboost model's."""
    var nr = Int(py=n_rows)
    var scores = _f64_list(Int(py=scores_addr), nr)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var groups = groups_from_counts(_group_counts(params))
    return PythonObject(mojo_ndcg(scores, labels, groups, Int(py=k)))


def predict(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Response-scale predictions into a preallocated float64 buffer."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for r in range(nr):
        out.unsafe_store(r, m[].predict(_row(features, nr, nf, r)))
    return PythonObject(None)


def predict_raw(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Raw-score predictions into a preallocated float64 buffer."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for r in range(nr):
        out.unsafe_store(r, m[].predict_raw(_row(features, nr, nf, r)))
    return PythonObject(None)


def predict_proba(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Multiclass probabilities, row-major `[r * n_classes + k]`, into a
    preallocated float64 buffer of size n_rows * n_classes."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var k = m[].booster.n_classes
    for r in range(nr):
        var proba = m[].predict_proba(_row(features, nr, nf, r))
        for c in range(k):
            out.unsafe_store(r * k + c, proba[c])
    return PythonObject(None)


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
    var features = _f64_list(Int(py=x_addr), nr * nf)
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
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
    var n_cols = rng.n_iterations()
    if n_cols == 0 or nr == 0:
        return PythonObject(None)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    # One ordinal table per tree in the range, built once and shared by every
    # row: the mapping from node id to leaf ordinal is a property of the tree.
    var tables = m[].booster.leaf_ordinals_range(rng)
    for r in range(nr):
        var bins = m[].mapper.bin_row(_row(features, nr, nf, r))
        for i in range(n_cols):
            var node = m[].booster.trees[rng.start + i].leaf_index_bins(bins)
            out.unsafe_store(r * n_cols + i, Float64(tables[i][node]))
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
    var k = m[].booster.n_classes
    var n_rounds = rng.n_iterations()
    var n_cols = n_rounds * k
    if n_cols == 0 or nr == 0:
        return PythonObject(None)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var tables = m[].booster.leaf_ordinals_range(rng)
    for r in range(nr):
        var bins = m[].mapper.bin_row(_row(features, nr, nf, r))
        for i in range(n_rounds):
            for c in range(k):
                var tree = (rng.start + i) * k + c
                var node = m[].booster.trees[tree].leaf_index_bins(bins)
                out.unsafe_store(
                    r * n_cols + i * k + c, Float64(tables[i * k + c][node])
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
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
    var features = _f64_list(Int(py=x_addr), nr * nf)
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
    encoding the Python layer applied on top (see python/mojoboost)."""
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


def save(model: PythonObject, path: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    save_model(m[], String(py=path))
    return PythonObject(None)


def load(path: PythonObject) raises -> PythonObject:
    var model = load_model(String(py=path))
    return PythonObject(alloc=model^)


def save_multiclass(
    model: PythonObject, path: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    save_multiclass_model(m[], String(py=path))
    return PythonObject(None)


def load_multiclass(path: PythonObject) raises -> PythonObject:
    var model = load_multiclass_model(String(py=path))
    return PythonObject(alloc=model^)
