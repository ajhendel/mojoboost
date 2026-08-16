"""CatBoost's generated features: `text_features` and `embedding_features`.

Two column types mojotrees cannot ingest and CatBoost can. CatBoost takes a
raw text column or a raw embedding column, runs an estimator over it, and
hands the resulting NUMERIC columns to the same quantizer every float column
goes through. Nothing downstream of the quantizer knows a text feature from
any other float feature. That is CatBoost's design and it is the reason this
module exists at the shape it does: it is a TRANSFORM, not a trainer.

    text_features(docs, ...)      -> (n_rows, n_features) float64 columns
    embedding_features(emb, ...)  -> (n_rows, n_features) float64 columns

`numpy.hstack` the result onto `X` and fit as usual. The estimator has no
`text_features=` or `embedding_features=` keyword and this module is why it
does not need one: the columns are ordinary floats from the moment they are
returned, so a `text_features=` keyword would buy exactly one `hstack` and
would owe a fitted-state contract that the model format has no section for.
See catalog A31 and the A29 note on what the model file carries.

**What this does NOT give you.** A fitted dictionary, a fitted NaiveBayes
calcer or a fitted LDA projection is FITTED STATE. It is not written into a
mojotrees model file, so a model trained on these columns cannot regenerate
them for new data on its own; the caller keeps the transform and re-runs it,
and a caller who does not is scoring a model against columns that mean
something else. That is stated here rather than discovered later.

**The leakage rule, which is not optional.** `naive_bayes`, `bm25`, `lda` and
`knn` all read the target. Each is computed strictly before-write over a
permutation of the rows, which is what keeps row `i`'s feature out of row
`i`'s own statistic, and the permutation is built natively from
`permutation_seed` by the one permutation layer this package has
(`ordered_boosting.ordered_permutation`, catalog A21). Running any of them
over a validation split's own targets is a leak this module cannot detect:
generate the columns on the LEARN rows, once, and score the rest with the
same fitted transform if and when one exists.

`bow` is the one estimator with no target in it at all, and it is the one to
start with.
"""

from __future__ import annotations

from . import _arrays, _compat

_mojotrees = _compat.import_extension()

__all__ = [
    "DEFAULT_TOP_TOKENS_COUNT",
    "CATBOOST_DICTIONARIES",
    "text_features",
    "embedding_features",
    "embedding_feature_count",
]

#: `TBagOfWordsEstimator`'s `TopTokensCount`, and the reason `bow` is the one
#: estimator whose default should not be shipped as-is: 2000 float64 columns
#: is 16 KB per row, so a million rows is a derived bound of 16 GB. The
#: native side returns that bound with the shape and this module refuses
#: nothing on your behalf; it is your allocation.
DEFAULT_TOP_TOKENS_COUNT = 2000

#: CatBoost's untouched default `text_processing` dictionaries: a BIGRAM and
#: a UNIGRAM, in that order, both consumed by `BoW`
#: (`TTextProcessingOptions::SetDefault`). The order is not cosmetic: it
#: fixes which generated column lands where.
CATBOOST_DICTIONARIES = (
    {"gram_order": 2, "occurrence_lower_bound": 3, "max_dictionary_size": 50000},
    {"gram_order": 1, "occurrence_lower_bound": 3, "max_dictionary_size": 50000},
)


def _classes_buffer(y, n_rows, needed):
    """A float64 buffer of integer class codes, or `(None, 0)` when the spec
    reads no target."""
    if not needed:
        return None, 0
    if y is None:
        raise ValueError(
            "naive_bayes and bm25 read the target; pass y with one integer "
            "class code per document"
        )
    yb = _arrays.f64_vector(y, n_rows, "y")
    return yb, _arrays.addr(yb)


def text_features(
    docs,
    y=None,
    num_classes=0,
    *,
    bow=False,
    naive_bayes=False,
    bm25=False,
    top_tokens_count=DEFAULT_TOP_TOKENS_COUNT,
    dictionaries=CATBOOST_DICTIONARIES,
    delimiter=" ",
    split_by_set=False,
    skip_empty=True,
    lowercasing=False,
    permutation_seed=0,
):
    """Numeric columns for one raw text column.

    `docs` is a sequence of `n_rows` strings. Returns a column-major float64
    array of shape `(n_rows, n_features)`; `n_features` depends on the FITTED
    dictionary size and is therefore not knowable before the call, which is
    why nothing here takes an output buffer.

    Every estimator is off by default. CatBoost's own default for a
    classification pool is `bow=True, naive_bayes=True`; BM25 is not a
    CatBoost default and a comparison that enables it is not a comparison
    against CatBoost's defaults.

    The tokenizer defaults are CatBoost's and they do almost nothing: split
    on the literal one-character string `" "`, drop empty pieces, stop. No
    lowercasing, no punctuation stripping. `"Cat,"` and `"cat"` are two
    tokens at the defaults, and anyone surprised by a comparison should read
    that sentence first.
    """
    docs = list(docs)
    n_rows = len(docs)
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            raise TypeError(f"docs[{i}] is not a str")
    target_aware = bool(naive_bayes or bm25)
    yb, y_addr = _classes_buffer(y, n_rows, target_aware)
    options = {
        "bow": int(bool(bow)),
        "top_tokens_count": int(top_tokens_count),
        "naive_bayes": int(bool(naive_bayes)),
        "bm25": int(bool(bm25)),
        "delimiter": str(delimiter),
        "split_by_set": int(bool(split_by_set)),
        "skip_empty": int(bool(skip_empty)),
        "lowercasing": int(bool(lowercasing)),
        "n_dictionaries": len(dictionaries),
        "gram_orders": [int(d["gram_order"]) for d in dictionaries],
        "occurrence_lower_bounds": [
            int(d["occurrence_lower_bound"]) for d in dictionaries
        ],
        "max_dictionary_sizes": [
            int(d["max_dictionary_size"]) for d in dictionaries
        ],
    }
    handle = _mojotrees.text_features_open(
        docs, n_rows, y_addr, int(num_classes), int(permutation_seed), options
    )
    del yb
    rows, n_features, _bound = _mojotrees.text_features_shape(handle)
    out = _arrays.out_buffer(rows * n_features)
    _mojotrees.text_features_write(
        handle, _arrays.addr(out), rows * n_features
    )
    return _reshape_column_major(out, rows, n_features)


def _embedding_options(
    lda,
    lda_components,
    lda_reg,
    lda_catboost_final_flush_only,
    lda_jacobi_max_sweeps,
    knn,
    knn_k,
    knn_max_rows,
):
    return {
        "lda": int(bool(lda)),
        "lda_components": int(lda_components),
        "lda_reg": float(lda_reg),
        "lda_catboost_final_flush_only": int(
            bool(lda_catboost_final_flush_only)
        ),
        "lda_jacobi_max_sweeps": int(lda_jacobi_max_sweeps),
        "knn": int(bool(knn)),
        "knn_k": int(knn_k),
        "knn_max_rows": int(knn_max_rows),
    }


def embedding_feature_count(
    num_classes,
    dim,
    *,
    lda=False,
    lda_components=-1,
    lda_reg=0.00005,
    lda_catboost_final_flush_only=False,
    lda_jacobi_max_sweeps=100,
    knn=False,
    knn_k=5,
    knn_max_rows=100000,
):
    """How many columns `embedding_features` will produce, without producing
    them. Unlike the text side this IS knowable up front, because it is a
    function of `(num_classes, dim, params)` alone."""
    return _mojotrees.embedding_feature_count(
        int(num_classes),
        int(dim),
        _embedding_options(
            lda,
            lda_components,
            lda_reg,
            lda_catboost_final_flush_only,
            lda_jacobi_max_sweeps,
            knn,
            knn_k,
            knn_max_rows,
        ),
    )


def embedding_features(
    embeddings,
    y,
    num_classes,
    *,
    lda=False,
    lda_components=-1,
    lda_reg=0.00005,
    lda_catboost_final_flush_only=False,
    lda_jacobi_max_sweeps=100,
    knn=False,
    knn_k=5,
    knn_max_rows=100000,
    permutation_seed=0,
):
    """Numeric columns for one raw embedding column.

    `embeddings` is `(n_rows, dim)`, ROW-major, which is the layout every
    consumer of an embedding walks. The returned feature matrix is
    `(n_rows, n_features)` like any other design matrix.

    LDA projects onto the leading generalized eigenvectors of (between-class
    scatter, within-class scatter); KNN emits per-class neighbour counts for
    classification or the neighbour target mean for regression. Both read the
    target, both are online over the permutation, and both are off by
    default. CatBoost runs BOTH for every embedding column by default, which
    is the whole difference in posture between the two packages.
    """
    if not _arrays.have_numpy():
        raise ImportError(
            "embedding_features needs numpy: an embedding column is a 2-D "
            "float array and there is no stdlib spelling of one worth having"
        )
    import numpy as np  # noqa: PLC0415

    emb = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float64))
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2-D, (n_rows, dim)")
    n_rows, dim = emb.shape
    yb = _arrays.f64_vector(y, n_rows, "y")
    options = _embedding_options(
        lda,
        lda_components,
        lda_reg,
        lda_catboost_final_flush_only,
        lda_jacobi_max_sweeps,
        knn,
        knn_k,
        knn_max_rows,
    )
    n_features = _mojotrees.embedding_feature_count(
        int(num_classes), int(dim), options
    )
    out = _arrays.out_buffer(n_rows * n_features)
    _mojotrees.embedding_features_into(
        emb.ctypes.data,
        n_rows,
        dim,
        _arrays.addr(yb),
        int(num_classes),
        int(permutation_seed),
        options,
        _arrays.addr(out),
    )
    del yb
    return _reshape_column_major(out, n_rows, n_features)


def _reshape_column_major(buf, n_rows, n_features):
    """The flat column-major buffer as an `(n_rows, n_features)` array, or as
    a list of rows when numpy is absent."""
    if _arrays.have_numpy():
        import numpy as np  # noqa: PLC0415

        return np.asarray(buf, dtype=np.float64).reshape(
            (n_features, n_rows)
        ).T
    return [
        [buf[f * n_rows + r] for f in range(n_features)]
        for r in range(n_rows)
    ]
