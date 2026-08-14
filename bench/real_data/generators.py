"""Deterministic data generators, one per scenario.

Every scenario in this harness can run without a download. The generators
here are the offline option, and they exist for three reasons: the harness
has to be runnable in CI and on a machine with no network, a generator is
the only way to control a property exactly (a 0.5 percent positive rate, a
known nonlinearity, a known missingness mechanism), and a differential test
needs data that is identical on both sides of the comparison to the bit.

Determinism comes from counter-based splitmix64, the same construction the
rest of bench/ uses, so a row's values depend only on its index and the
seed. No global random state, no sequence dependence, no difference between
generating 10000 rows and generating the first 10000 of 100000.

Data is generated once, in Python, and handed to both engines as the same
arrays. Nothing is regenerated per engine, so a generator bug cannot
advantage either side.

These are synthetic. Result records built on them carry
`data_kind: "synthetic"`, and reporting them as real-data results is a
misrepresentation, not a shortcut.
"""

import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
INV_2_53 = 1.0 / 9007199254740992.0


def splitmix64(counter):
    """splitmix64 applied elementwise to a uint64 array of counters."""
    with np.errstate(over="ignore"):
        z = (np.asarray(counter, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)) & MASK
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK
        return z ^ (z >> np.uint64(31))


def uniform(counter):
    """Uniform [0, 1) doubles, one per counter, using the top 53 bits."""
    return (splitmix64(counter) >> np.uint64(11)).astype(np.float64) * INV_2_53


def _stream(seed, tag, count):
    """`count` uniforms from an independent stream of `seed`.

    `tag` separates the streams within one generator. Streams are spaced
    2**40 apart in counter space, which is far more room than any scenario
    here needs, so two streams of one seed never overlap.
    """
    base = np.uint64(seed) * np.uint64(1_000_003) + np.uint64(tag) * np.uint64(1 << 40)
    return uniform(base + np.arange(count, dtype=np.uint64))


def _normal(seed, tag, count):
    """Standard normals by Box-Muller on two uniform streams. Deterministic
    and vectorised; the tail beyond about 6.6 sigma is unreachable, which
    matters to nobody here."""
    u1 = _stream(seed, tag, count)
    u2 = _stream(seed, tag + 1, count)
    u1 = np.maximum(u1, np.float64(2.0**-53))
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _hash_split(n_rows, train_fraction, seed):
    """A row mask that depends only on the row index, so the same rows land
    on the same side however many rows are generated."""
    keep = _stream(seed, 999, n_rows) < float(train_fraction)
    return keep


def dense_regression(n_rows=200_000, n_features=50, seed=1901, noise=0.30):
    """Dense continuous features, a target with interactions and a mild
    nonlinearity, and half the features pure noise.

    The signal is deliberately not a sum of univariate terms. A model that
    only ever finds axis-aligned marginal structure scores visibly worse
    here, which is what makes the two engines' RMSE comparable rather than
    both trivially optimal.
    """
    x = _stream(seed, 1, n_rows * n_features).reshape(n_features, n_rows).T.copy()
    signal = (
        3.0 * x[:, 0]
        + 2.0 * x[:, 1] * x[:, 2]
        + 1.5 * np.sin(6.0 * x[:, 3])
        - 2.0 * (x[:, 4] > 0.7).astype(np.float64)
        + 0.8 * x[:, 5] ** 2
    )
    y = signal + noise * _normal(seed, 11, n_rows)
    return {"X": np.ascontiguousarray(x), "y": y}


def imbalanced_binary(
    n_rows=300_000, n_features=40, seed=1902, positive_rate=0.005
):
    """Binary labels at a chosen positive rate, drawn from a logistic model
    whose intercept is solved for that rate rather than guessed.

    Extreme imbalance is the point. At half a percent positive, accuracy is
    meaningless, log loss is dominated by the negatives, and the only
    metrics worth comparing are ranking metrics and average precision. The
    quality module scores it accordingly.
    """
    x = _stream(seed, 1, n_rows * n_features).reshape(n_features, n_rows).T.copy()
    latent = (
        2.5 * x[:, 0]
        + 1.8 * x[:, 1] * x[:, 2]
        - 1.2 * x[:, 3]
        + 0.9 * (x[:, 4] > 0.8).astype(np.float64)
    )
    latent = latent - latent.mean()
    # Solve the intercept for the requested rate by bisection on the mean
    # probability. Twelve halvings put it inside 1e-3 of the target rate,
    # which is closer than the sampling noise at these row counts.
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _logistic(latent + mid).mean() < positive_rate:
            lo = mid
        else:
            hi = mid
    p = _logistic(latent + 0.5 * (lo + hi))
    y = (_stream(seed, 21, n_rows) < p).astype(np.float64)
    return {"X": np.ascontiguousarray(x), "y": y}


def multiclass(n_rows=200_000, n_features=30, seed=1903, n_classes=7):
    """A softmax over class-specific linear scores with unequal priors, so
    the class distribution is skewed the way a real multiclass problem is."""
    x = _stream(seed, 1, n_rows * n_features).reshape(n_features, n_rows).T.copy()
    weights = _normal(seed, 31, n_classes * n_features).reshape(n_classes, n_features)
    # Only the first eight features carry class signal; the rest are noise.
    weights[:, 8:] = 0.0
    prior = np.log(np.linspace(1.0, 0.15, n_classes))
    scores = x @ weights.T * 2.0 + prior
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    draws = _stream(seed, 41, n_rows)
    y = (probs.cumsum(axis=1) < draws[:, None]).sum(axis=1).astype(np.float64)
    np.clip(y, 0, n_classes - 1, out=y)
    return {"X": np.ascontiguousarray(x), "y": y, "n_classes": n_classes}


def ranking(n_queries=20_000, n_features=30, seed=1904, docs_lo=6, docs_hi=40):
    """Queries of varying length, graded 0 to 4 within the query.

    Grading within the query is what makes NDCG a real measurement here:
    every query has a spread of labels, so a ranker that gets the ordering
    wrong is punished rather than saved by a query that was all zeros.
    """
    sizes = docs_lo + (
        splitmix64(
            np.uint64(seed) * np.uint64(1_000_003)
            + np.arange(n_queries, dtype=np.uint64)
        )
        % np.uint64(docs_hi - docs_lo + 1)
    ).astype(np.int64)
    n_rows = int(sizes.sum())

    x = _stream(seed, 1, n_rows * n_features).reshape(n_features, n_rows).T.copy()
    utility = (
        3.0 * x[:, 0]
        + 2.0 * x[:, 1] * x[:, 2]
        - 1.0 * x[:, 3]
        + 0.35 * (_stream(seed, 51, n_rows) - 0.5)
    )

    y = np.zeros(n_rows, dtype=np.float64)
    starts = np.concatenate(([0], np.cumsum(sizes)))
    for q in range(n_queries):
        lo, hi = int(starts[q]), int(starts[q + 1])
        order = np.argsort(-utility[lo:hi], kind="stable")
        y[lo + order] = np.rint(np.linspace(4.0, 0.0, hi - lo))
    return {
        "X": np.ascontiguousarray(x),
        "y": y,
        "group": sizes.astype(np.int64),
    }


def categorical_missing(
    n_rows=200_000,
    n_numeric=20,
    n_categorical=8,
    seed=1905,
    missing_rate=0.15,
    cardinalities=(3, 7, 12, 40, 120, 500, 2, 25),
):
    """Mixed numeric and integer-coded categorical columns, with values
    missing not at random.

    The missingness mechanism matters. Values are dropped with a
    probability that depends on the target, so a model that routes NaN by
    learned direction beats one that imputes, and the two engines' default
    directions are actually under test. A missing-completely-at-random hole
    pattern would make this scenario measure nothing.

    Categorical effects are per level, drawn once, so a high-cardinality
    column carries real signal that only a category-set split can capture.
    """
    cards = list(cardinalities)[:n_categorical]
    while len(cards) < n_categorical:
        cards.append(16)

    numeric = _stream(seed, 1, n_rows * n_numeric).reshape(n_numeric, n_rows).T.copy()
    codes = np.empty((n_rows, n_categorical), dtype=np.float64)
    effect = np.zeros(n_rows)
    for j, card in enumerate(cards):
        raw = (_stream(seed, 100 + j, n_rows) * card).astype(np.int64)
        np.clip(raw, 0, card - 1, out=raw)
        codes[:, j] = raw.astype(np.float64)
        levels = _normal(seed, 200 + 2 * j, card)
        effect += levels[raw] * (1.5 if j < 3 else 0.5)

    target = (
        2.0 * numeric[:, 0]
        + 1.5 * numeric[:, 1] * numeric[:, 2]
        + effect
        + 0.3 * _normal(seed, 61, n_rows)
    )

    x = np.hstack([numeric, codes])
    cat_indices = list(range(n_numeric, n_numeric + n_categorical))

    # Not missing at random: rows in the upper tail of the target lose
    # values more often. Columns 0, 1 and the first two categoricals are
    # the ones with holes, so the rest stay clean as a control.
    tail = (target - target.mean()) / (target.std() + 1e-12)
    p_missing = missing_rate * _logistic(tail)
    for k, col in enumerate([0, 1, n_numeric, n_numeric + 1]):
        drop = _stream(seed, 300 + k, n_rows) < p_missing
        x[drop, col] = np.nan

    return {
        "X": np.ascontiguousarray(x),
        "y": target,
        "categorical_feature": cat_indices,
    }


def sparse_highdim(
    n_rows=100_000, n_features=50_000, seed=1906, nnz_per_row=60
):
    """A high-dimensional sparse binary problem, returned as CSC.

    Structural zeros are the majority of the matrix and are the value zero,
    not missing values: that is this library's documented CSC semantics and
    LightGBM's default (`zero_as_missing=false`), so both engines see the
    same thing. A run that wants zero treated as missing is a different
    experiment and is not in this harness.

    Nonzero positions follow a power law over the feature index, so the
    matrix has the head-and-tail column density that real text and
    identifier features have, rather than uniform noise.
    """
    from scipy import sparse

    total = n_rows * nnz_per_row
    # Zipf-ish column choice by inverse transform on u**alpha.
    u = _stream(seed, 1, total)
    cols = (np.power(u, 3.0) * n_features).astype(np.int64)
    np.clip(cols, 0, n_features - 1, out=cols)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), nnz_per_row)
    vals = 0.5 + _stream(seed, 2, total)

    weights = np.zeros(n_features)
    signal_cols = 200
    weights[:signal_cols] = _normal(seed, 71, signal_cols) * 1.2

    matrix = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_rows, n_features), dtype=np.float64
    )
    matrix.sum_duplicates()
    latent = np.asarray(matrix @ weights).ravel()
    latent = (latent - latent.mean()) / (latent.std() + 1e-12)
    p = _logistic(1.5 * latent)
    y = (_stream(seed, 81, n_rows) < p).astype(np.float64)
    return {"X": matrix.tocsc(), "y": y, "sparse": True}


def split(data, train_fraction=0.8, seed=1900):
    """Split a generated dataset into train and test.

    Ranking data splits by query, never by row, so no query has documents
    on both sides. Everything else splits by a per-row hash, so the
    assignment does not depend on row order or on the row count.
    """
    y = data["y"]
    if "group" in data:
        group = data["group"]
        keep_q = _hash_split(len(group), train_fraction, seed)
        starts = np.concatenate(([0], np.cumsum(group)))
        row_mask = np.zeros(len(y), dtype=bool)
        for q, keep in enumerate(keep_q):
            if keep:
                row_mask[int(starts[q]) : int(starts[q + 1])] = True
        train = dict(data)
        test = dict(data)
        train["X"], test["X"] = data["X"][row_mask], data["X"][~row_mask]
        train["y"], test["y"] = y[row_mask], y[~row_mask]
        train["group"] = group[keep_q]
        test["group"] = group[~keep_q]
        return train, test

    mask = _hash_split(len(y), train_fraction, seed)
    train, test = dict(data), dict(data)
    x = data["X"]
    if data.get("sparse"):
        # CSC slices by row through CSR and back, which is cheaper than
        # fancy-indexing a CSC matrix by row.
        csr = x.tocsr()
        train["X"] = csr[mask].tocsc()
        test["X"] = csr[~mask].tocsc()
    else:
        train["X"], test["X"] = x[mask], x[~mask]
    train["y"], test["y"] = y[mask], y[~mask]
    return train, test


#: Generator per scenario id. Sizes here are the defaults; scenarios.py
#: overrides them per size tier.
GENERATORS = {
    "dense_regression": dense_regression,
    "imbalanced_binary": imbalanced_binary,
    "multiclass": multiclass,
    "ranking": ranking,
    "categorical_missing": categorical_missing,
    "sparse_highdim": sparse_highdim,
}
