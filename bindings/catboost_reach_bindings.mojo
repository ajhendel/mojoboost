"""The Python surface for four native modules that no entry point reached.

`tools/connectivity_audit.py` reports a module no root import chain gets to,
and on 2026-08-16 it reported eleven. Four of them were unreachable for the
same reason and not for four different ones: the capability was written
against a native input type that only the native API can build -- a
`BinnedMatrix`, a `TextBag`, an `EmbeddingMatrix`, a fitted `Model` -- and
nothing at the Python boundary ever built one. This file is that boundary.
Catalog A31 records what it closed and what it did not.

    onnx_plan_text(model, raw_score)            -> str
    onnx_plan_text_multiclass(model, raw_score)  -> str
    onnx_export_refusals(model, raw_score)       -> list[str]
    multi_rmse_fit(...)                          -> model handle
    multi_rmse_shape(handle)                     -> list[int]
    multi_rmse_predict(handle, ...)              -> writes a buffer
    text_features_open(...)                      -> columns handle
    text_features_shape(handle)                  -> list[int]
    text_features_write(handle, addr, capacity)  -> writes a buffer
    embedding_feature_count(...)                 -> int
    embedding_features_into(...)                 -> writes a buffer

Conventions are `binding_support.mojo`'s and are not restated here, with one
addition that is worth stating. **Text features are an open/write pair and
embedding features are a single call**, and the difference is not a style
choice: the embedding column count is a function of `(n_classes, dim,
params)` and is known before the pass, so a caller can size its buffer and
pay one crossing; the text column count depends on the FITTED dictionary
size, which nothing knows until the corpus has been read, so the count has to
come back before the values can be written anywhere.

Nothing here holds a device buffer or hands a pointer to Python.
"""

from std.python import Python, PythonObject

from binding_support import (
    f64_buffer,
    f64_view,
    flag,
    nonnegative,
    py_int_list,
    py_str_list,
    str_sequence,
    write_f64_buffer,
)

from mojotrees.embedding import (
    EmbeddingEstimatorParams,
    EmbeddingMatrix,
    KnnParams,
    LdaParams,
    compute_online_features,
    online_feature_count,
)
from mojotrees.model import Model, MulticlassModel
from mojotrees.multi_target import (
    MultiTargetModel,
    fit_multi_rmse,
    predict_multi_rmse,
)
from mojotrees.onnx_export import (
    MULTICLASS_OBJECTIVE,
    onnx_plan,
    onnx_plan_multiclass,
    onnx_plan_text as mojo_onnx_plan_text,
    onnx_refusals,
)
from mojotrees.ordered_boosting import (
    default_permutation_block_size,
    ordered_permutation,
)
from mojotrees.target_matrix import TargetMatrix
from mojotrees.text_features import (
    TextColumns,
    TextFeatureSpec,
    text_column_features,
)
from mojotrees.text_processing import DictionaryParams, TokenizerParams
from mojotrees.tree import TreeParams
from mojotrees.boosting import BoosterParams


# -- ONNX export ---------------------------------------------------------


def onnx_plan_text(
    model: PythonObject, raw_score: PythonObject
) raises -> PythonObject:
    """The export plan for a single-output model, as the token stream
    `python/mojotrees/onnx_export.py` reads.

    Raises, naming every reason at once, when the model holds anything
    `ai.onnx.ml` opset 3 cannot reproduce. The plan crosses as text rather
    than as a file path so that a caller can convert a live model without
    choosing a temporary directory first; `save_model` is not on this path
    and does not need to be.
    """
    var m = model.downcast_value_ptr[Model]()
    var plan = onnx_plan(m[], flag(raw_score, "raw_score"))
    return PythonObject(mojo_onnx_plan_text(plan))


def onnx_plan_text_multiclass(
    model: PythonObject, raw_score: PythonObject
) raises -> PythonObject:
    """The same for a softmax multiclass model, whose plan carries one
    target per class and a per-class base value."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var plan = onnx_plan_multiclass(m[], flag(raw_score, "raw_score"))
    return PythonObject(mojo_onnx_plan_text(plan))


def onnx_export_refusals(
    model: PythonObject, raw_score: PythonObject
) raises -> PythonObject:
    """Every reason this single-output model cannot be exported exactly, all
    of them rather than the first. An empty list is the only thing that
    permits an export, and asking is cheaper than catching."""
    var m = model.downcast_value_ptr[Model]()
    return py_str_list(
        onnx_refusals(
            m[].mapper,
            m[].booster.trees,
            m[].booster.objective,
            m[].booster.linear,
            flag(raw_score, "raw_score"),
        )
    )


def onnx_export_refusals_multiclass(
    model: PythonObject, raw_score: PythonObject
) raises -> PythonObject:
    """The multiclass twin. The objective code is `MULTICLASS_OBJECTIVE`
    rather than a field read, because a `MulticlassBooster` carries none;
    `onnx_export.mojo` states that and this passes the constant it states."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    return py_str_list(
        onnx_refusals(
            m[].mapper,
            m[].booster.trees,
            MULTICLASS_OBJECTIVE,
            m[].booster.linear,
            flag(raw_score, "raw_score"),
        )
    )


# -- MultiRMSE -----------------------------------------------------------


def _multi_target_params(params: PythonObject) raises -> BoosterParams:
    """The `BoosterParams` a multi-output fit runs under.

    **This is deliberately not `_parse_params`.** That function lives in
    `bindings/_mojotrees.mojo`, it resolves twenty mechanisms against the
    single-output trainers, and most of those mechanisms have no multi-output
    trainer to be resolved against: bagging, GOSS, EFB bundling, linear
    leaves, ordered boosting, dart and rf all reach `boosting.train` or
    `alternate_boosting`, and `train_multi_rmse` calls neither. Reusing it
    here would accept every one of those keys and honor none, which is the
    exact defect this round exists to remove.

    So the multi-output entry point takes the named subset it can honor and
    the Python wrapper refuses the rest by name before it ever gets here.
    Every key below is required rather than defaulted: a missing key is a
    `KeyError` at this boundary, which is the convention `_ORDERED_DEFAULTS`
    already established.
    """
    var tree = TreeParams(
        Int(py=params["num_leaves"]),
        Int(py=params["min_data_in_leaf"]),
        Float64(py=params["lambda_l2"]),
        Float64(py=params["min_child_hess"]),
        Float64(py=params["lambda_l1"]),
    )
    tree.max_depth = Int(py=params["max_depth"])
    return BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
    )


def multi_rmse_fit(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    n_targets: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train `MultiRMSE` on a column-major float64 X and a row-major float64
    `(n_rows, n_targets)` y.

    Returns a `MultiTargetModel` handle, which is NOT a `Model`: it holds one
    tree per target per round and no single-output entry point in this
    extension will accept it. That is why the shape and predict functions
    below exist rather than reusing `num_trees` and `predict_batch`.

    CPU only, and it is refused above rather than downgraded here: `train_gpu`
    has no multi-output round loop.
    """
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var t = Int(py=n_targets)
    if t < 1:
        raise Error("n_targets must be at least 1")
    var features = f64_view(Int(py=x_addr), nr * nf)
    var targets = TargetMatrix(f64_buffer(Int(py=y_addr), nr * t), t)
    var weights = List[Float64]()
    var w_addr = Int(py=params["weight_addr"])
    if w_addr != 0:
        weights = f64_buffer(w_addr, nr)
    var model = fit_multi_rmse(
        features,
        nr,
        nf,
        targets,
        _multi_target_params(params),
        Int(py=params["max_bin"]),
        weights,
        flag(params["with_missing_values"], "with_missing_values"),
        flag(params["use_missing"], "use_missing"),
    )
    return PythonObject(alloc=model^)


def multi_rmse_shape(model: PythonObject) raises -> PythonObject:
    """`[n_targets, n_iterations, n_features, n_trees]`.

    `n_iterations` and `n_trees` are both here and they are different
    numbers: `n_trees` is `n_iterations * n_targets`, and the one comparable
    to CatBoost's `tree_count_` is `n_iterations`.
    """
    var m = model.downcast_value_ptr[MultiTargetModel]()
    var out = List[Int]()
    out.append(m[].n_targets())
    out.append(m[].n_iterations())
    out.append(m[].n_features())
    out.append(len(m[].booster.trees))
    return py_int_list(out)


def multi_rmse_predict(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Score a column-major float64 X into a caller-allocated
    `n_rows * n_targets` buffer, row-major: `out[r * T + t]`."""
    var m = model.downcast_value_ptr[MultiTargetModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var scores = predict_multi_rmse(
        m[], f64_view(Int(py=x_addr), nr * nf), nr, nf
    )
    write_f64_buffer(scores, Int(py=out_addr), nr * m[].n_targets())
    return PythonObject(None)


# -- text features -------------------------------------------------------


def _text_spec(options: PythonObject) raises -> TextFeatureSpec:
    return TextFeatureSpec(
        flag(options["bow"], "bow"),
        Int(py=options["top_tokens_count"]),
        flag(options["naive_bayes"], "naive_bayes"),
        flag(options["bm25"], "bm25"),
    )


def _text_dictionaries(options: PythonObject) raises -> List[DictionaryParams]:
    """The dictionaries as three parallel sequences plus their length, which
    is how a list of records crosses here: one sequence per field, so the
    boundary reads no Python object it did not build."""
    var n = nonnegative(options["n_dictionaries"], "n_dictionaries")
    var orders = options["gram_orders"]
    var bounds = options["occurrence_lower_bounds"]
    var sizes = options["max_dictionary_sizes"]
    var out = List[DictionaryParams](capacity=n)
    for i in range(n):
        out.append(
            DictionaryParams(
                Int(py=orders[i]),
                Int(py=bounds[i]),
                Int(py=sizes[i]),
            )
        )
    return out^


def text_features_open(
    docs: PythonObject,
    n_docs: PythonObject,
    y_addr: PythonObject,
    num_classes: PythonObject,
    permutation_seed: PythonObject,
    options: PythonObject,
) raises -> PythonObject:
    """Generate the text columns for one text column and hold them.

    Returns a `TextColumns` handle. The column COUNT is not knowable before
    the corpus is read -- it is a function of the fitted dictionary size --
    so the caller reads `text_features_shape` off the handle, allocates, and
    then calls `text_features_write`.

    The permutation is built HERE, from `permutation_seed`, by
    `ordered_boosting.ordered_permutation` at the block size that module's
    own default rule gives. It is deliberately not built by the caller and
    deliberately not built by `text_features.mojo`: A21 says one permutation
    layer, and this is a consumer of it. A `BoW`-only spec never looks at it.

    `y_addr` may be 0 for a `BoW`-only spec, which is the one configuration
    with no target in it and therefore no leakage question.
    """
    var n = nonnegative(n_docs, "n_docs")
    var spec = _text_spec(options)
    var classes = List[Int]()
    var permutation = List[Int]()
    if spec.target_aware():
        var addr = Int(py=y_addr)
        if addr == 0:
            raise Error(
                "naive_bayes and bm25 read the target; a class vector is"
                " required for either"
            )
        var labels = f64_buffer(addr, n)
        for i in range(n):
            classes.append(Int(labels[i]))
        permutation = ordered_permutation(
            Int(py=permutation_seed),
            0,
            n,
            default_permutation_block_size(n),
        )
    var columns = text_column_features(
        str_sequence(docs, n),
        classes,
        Int(py=num_classes),
        permutation,
        TokenizerParams(
            String(py=options["delimiter"]),
            flag(options["split_by_set"], "split_by_set"),
            flag(options["skip_empty"], "skip_empty"),
            flag(options["lowercasing"], "lowercasing"),
        ),
        _text_dictionaries(options),
        spec,
    )
    return PythonObject(alloc=columns^)


def text_features_shape(columns: PythonObject) raises -> PythonObject:
    """`[n_rows, n_features, derived_bytes]`. The third is
    `text_features.text_column_memory_bound_bytes`, a derived bound and not a
    measurement, so a caller can refuse a 16 GB `BoW` before allocating it."""
    var c = columns.downcast_value_ptr[TextColumns]()
    var out = List[Int]()
    out.append(c[].n_rows)
    out.append(c[].n_features)
    out.append(c[].memory_bound_bytes())
    return py_int_list(out)


def text_features_write(
    columns: PythonObject, out_addr: PythonObject, capacity: PythonObject
) raises -> PythonObject:
    """Copy the generated columns into a caller-allocated buffer,
    COLUMN-MAJOR (`out[f * n_rows + r]`), which is the layout the binner
    reads and the layout `numpy.asfortranarray` produces."""
    var c = columns.downcast_value_ptr[TextColumns]()
    write_f64_buffer(c[].values, Int(py=out_addr), Int(py=capacity))
    return PythonObject(None)


# -- embedding features --------------------------------------------------


def _embedding_params(options: PythonObject) raises -> EmbeddingEstimatorParams:
    return EmbeddingEstimatorParams(
        LdaParams(
            flag(options["lda"], "lda"),
            Int(py=options["lda_components"]),
            Float64(py=options["lda_reg"]),
            flag(
                options["lda_catboost_final_flush_only"],
                "lda_catboost_final_flush_only",
            ),
            Int(py=options["lda_jacobi_max_sweeps"]),
        ),
        KnnParams(
            flag(options["knn"], "knn"),
            Int(py=options["knn_k"]),
            Int(py=options["knn_max_rows"]),
        ),
    )


def embedding_feature_count(
    n_classes: PythonObject, dim: PythonObject, options: PythonObject
) raises -> PythonObject:
    """How many columns `embedding_features_into` will produce, without
    producing them, so the caller can size the buffer first."""
    return PythonObject(
        online_feature_count(
            Int(py=n_classes), Int(py=dim), _embedding_params(options)
        )
    )


def embedding_features_into(
    emb_addr: PythonObject,
    n_rows: PythonObject,
    dim: PythonObject,
    y_addr: PythonObject,
    n_classes: PythonObject,
    permutation_seed: PythonObject,
    options: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """LDA's and KNN's online columns for one embedding column, written
    COLUMN-MAJOR into a caller-allocated buffer.

    The embedding itself crosses ROW-major (`emb[r * dim + j]`), because that
    is the layout `EmbeddingMatrix` documents and the layout every consumer
    of it walks; the OUTPUT is column-major because it is a feature matrix
    from that point on. The two layouts in one function are the reason this
    paragraph exists.

    Both estimators read the target and both are computed strictly before
    write over the permutation, which is built here from `permutation_seed`
    for the reason `text_features_open` gives.
    """
    var nr = nonnegative(n_rows, "n_rows")
    var d = Int(py=dim)
    var params = _embedding_params(options)
    var embeddings = EmbeddingMatrix(
        nr, d, f64_buffer(Int(py=emb_addr), nr * d)
    )
    var cols = compute_online_features(
        embeddings,
        f64_buffer(Int(py=y_addr), nr),
        Int(py=n_classes),
        ordered_permutation(
            Int(py=permutation_seed),
            0,
            nr,
            default_permutation_block_size(nr),
        ),
        params,
    )
    var flat = List[Float64](capacity=len(cols) * nr)
    for f in range(len(cols)):
        for r in range(nr):
            flat.append(cols[f][r])
    write_f64_buffer(flat, Int(py=out_addr), len(cols) * nr)
    return PythonObject(None)
