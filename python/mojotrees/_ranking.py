"""Query-group helpers for ranking: `group_from_query_ids`, `ndcg_score`.

Moved here from the package `__init__` in the consolidation round. Both
public names are bound back into the package namespace, as are the two
private helpers `MojoTreesRanker` and `Dataset` (basic.py) call.
"""

from . import _arrays, _compat
from ._fit_args import _MAX_RELEVANCE_LABEL

_mojotrees = _compat.import_extension()
_np = _arrays.np
_as_f64_vector = _arrays.f64_vector
_addr = _arrays.addr


def group_from_query_ids(query_ids):
    """LightGBM's `group` array (per-query row counts) from a per-row query
    id column.

    Ids may be anything hashable-by-equality and need not be sorted, but
    each query's rows must be consecutive: an id that reappears after a
    different one raises, because splitting a query in two would change
    every NDCG it takes part in.
    """
    ids = list(query_ids)
    if not ids:
        raise ValueError("query_ids must not be empty")
    counts = []
    seen = []
    for i, qid in enumerate(ids):
        if i and qid == ids[i - 1]:
            counts[-1] += 1
            continue
        if qid in seen:
            raise ValueError(
                f"rows of query {qid!r} are not consecutive; a query's rows "
                "must form one unbroken run"
            )
        seen.append(qid)
        counts.append(1)
    return counts


def _group_buffer(group, n_rows):
    """Validated float64 buffer of per-query row counts. Must stay
    referenced while its address is in use."""
    if group is None:
        raise ValueError(
            "a ranker needs `group`: the number of rows in each query, in "
            "row order (LightGBM's `group` parameter)"
        )
    try:
        values = list(group)
    except TypeError:
        raise ValueError("group must be a sequence of row counts") from None
    if not values:
        raise ValueError("group must contain at least one query")
    counts = []
    for value in values:
        count = float(value)
        if count != int(count):
            raise ValueError(f"group counts must be integers, got {value!r}")
        if count <= 0:
            raise ValueError(f"group counts must be positive, got {value!r}")
        counts.append(count)
    if int(sum(counts)) != n_rows:
        raise ValueError(
            f"group counts sum to {int(sum(counts))} but X has {n_rows} rows"
        )
    return _as_f64_vector(counts, len(counts), "group")


def _check_relevance(yb, n_rows):
    """Relevance labels must be integers in [0, 30], LightGBM's default
    `label_gain` range."""
    if _np is not None:
        arr = _np.asarray(yb)
        bad = (
            bool((arr < 0).any())
            or bool((arr > _MAX_RELEVANCE_LABEL).any())
            or not bool(_np.array_equal(arr, _np.floor(arr)))
        )
    else:
        bad = any(
            v < 0 or v > _MAX_RELEVANCE_LABEL or v != int(v) for v in yb
        )
    if bad:
        raise ValueError(
            "relevance labels must be integers in "
            f"[0, {_MAX_RELEVANCE_LABEL}]"
        )


def ndcg_score(scores, y, group, at=5):
    """Mean NDCG@`at` of `scores` against relevance labels `y`, averaged
    over the queries `group` describes.

    Documents are ranked within their own query and never across queries.
    A query whose labels are all 0 counts as 1.0, which is what LightGBM's
    ndcg metric does with a query that has no attainable DCG.
    """
    n_rows = len(scores)
    sb = _as_f64_vector(scores, n_rows, "scores")
    yb = _as_f64_vector(y, n_rows)
    _check_relevance(yb, n_rows)
    gb = _group_buffer(group, n_rows)
    at = int(at)
    if at < 1:
        raise ValueError("at must be positive")
    return float(
        _mojotrees.ndcg(
            _addr(sb),
            _addr(yb),
            n_rows,
            at,
            {"group_addr": _addr(gb), "n_groups": len(gb)},
        )
    )
