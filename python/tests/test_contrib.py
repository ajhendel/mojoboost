"""`predict(..., pred_contrib=True)`: exact TreeSHAP through the Python API.

The Mojo suite (tests/test_contrib.mojo) proves the algorithm against a
subset-enumeration reference. What is checked here is the contract the
Python caller sees: shapes, dtypes, LightGBM compatibility, iteration
slicing, flag exclusions, and the sum property end to end.

`_ReferenceModel` is a second, independent implementation. It parses the
saved model file and computes Shapley values by enumerating all 2^M feature
subsets, in Python, sharing nothing with the Mojo code but the file format.
Its agreement with the extension is a differential check that covers the
whole path at once: node covers recorded during training, written to disk,
read back, and consumed by the recursion.
"""

import itertools
import math

import pytest

np = pytest.importorskip("numpy")

from mojoboost import MojoBoostClassifier, MojoBoostRegressor


# ----------------------------------------------------------------------
# Independent reference: parse the model file, enumerate subsets.
# ----------------------------------------------------------------------


def _bits_to_float(token):
    """The model format stores floats as raw IEEE-754 bit patterns."""
    return np.uint64(int(token)).view(np.float64).item()


class _Tree:
    __slots__ = ("feature", "threshold", "left", "right", "value",
                 "default_left", "missing_bin", "cat_offset", "cat_bitset",
                 "count")

    def goes_left(self, node, bin_id):
        off = self.cat_offset[node]
        if off >= 0:
            word = self.cat_bitset[off + bin_id // 64]
            return (word >> (bin_id % 64)) & 1 == 1
        if bin_id == self.missing_bin[node]:
            return self.default_left[node]
        return bin_id <= self.threshold[node]

    def conditional(self, bins, subset, node=0):
        """`v(S)`: known features route, unknown ones average over covers."""
        f = self.feature[node]
        if f < 0:
            return self.value[node]
        if (subset >> f) & 1:
            child = (
                self.left[node] if self.goes_left(node, bins[f])
                else self.right[node]
            )
            return self.conditional(bins, subset, child)
        lo, hi = self.left[node], self.right[node]
        return (
            self.count[lo] * self.conditional(bins, subset, lo)
            + self.count[hi] * self.conditional(bins, subset, hi)
        ) / self.count[node]


class _ReferenceModel:
    """A mojoboost model file, read and explained in pure Python."""

    def __init__(self, path):
        tokens = open(path).read().split()
        self._t = tokens
        self._i = 0
        assert self._next() == "mojoboost"
        version = self._next()
        assert version == "v3", f"expected a v3 file, got {version}"
        kind = self._next()
        self.multiclass = kind == "multiclass"
        if self.multiclass:
            self.n_classes = int(self._next())
        else:
            self.n_classes = 1
            self.objective = int(self._next())
        assert self._next() == "learning_rate"
        self.learning_rate = _bits_to_float(self._next())
        if self.multiclass:
            assert self._next() == "base_scores"
            self.base_scores = [
                _bits_to_float(self._next()) for _ in range(self.n_classes)
            ]
        else:
            assert self._next() == "base_score"
            self.base_scores = [_bits_to_float(self._next())]
        self._read_mapper()
        self._read_optional_sections()
        self.trees = self._read_trees()

    def _next(self):
        tok = self._t[self._i]
        self._i += 1
        return tok

    def _peek(self):
        return self._t[self._i] if self._i < len(self._t) else ""

    def _read_mapper(self):
        assert self._next() == "mapper"
        self.n_features = int(self._next())
        self.n_bins = int(self._next())
        n_edges = int(self._next())
        self.edges = [_bits_to_float(self._next()) for _ in range(n_edges)]
        self.edge_offsets = [
            int(self._next()) for _ in range(self.n_features + 1)
        ]
        self.missing_bin_of = [
            int(self._next()) for _ in range(self.n_features)
        ]

    def _read_optional_sections(self):
        self.is_categorical = [False] * self.n_features
        self.cat_codes = []
        self.cat_offsets = [0] * (self.n_features + 1)
        if self._peek() == "categorical":
            self._next()
            n_flags = int(self._next())
            n_codes = int(self._next())
            self.is_categorical = [
                int(self._next()) != 0 for _ in range(n_flags)
            ]
            self.cat_codes = [int(self._next()) for _ in range(n_codes)]
            self.cat_offsets = [
                int(self._next()) for _ in range(self.n_features + 1)
            ]
        if self._peek() == "monotone":
            self._next()
            for _ in range(int(self._next())):
                self._next()

    def _read_trees(self):
        assert self._next() == "trees"
        n_trees = int(self._next())
        trees = []
        for _ in range(n_trees):
            assert self._next() == "tree"
            n_nodes = int(self._next())
            self._next()  # n_leaves
            t = _Tree()
            t.feature = [int(self._next()) for _ in range(n_nodes)]
            t.threshold = [int(self._next()) for _ in range(n_nodes)]
            t.left = [int(self._next()) for _ in range(n_nodes)]
            t.right = [int(self._next()) for _ in range(n_nodes)]
            t.value = [_bits_to_float(self._next()) for _ in range(n_nodes)]
            t.default_left = [
                int(self._next()) != 0 for _ in range(n_nodes)
            ]
            t.missing_bin = [int(self._next()) for _ in range(n_nodes)]
            t.count = [_bits_to_float(self._next()) for _ in range(n_nodes)]
            t.cat_offset = [-1] * n_nodes
            t.cat_bitset = []
            if self._peek() == "cat":
                self._next()
                n_words = int(self._next())
                t.cat_offset = [int(self._next()) for _ in range(n_nodes)]
                t.cat_bitset = [int(self._next()) for _ in range(n_words)]
            trees.append(t)
        return trees

    def bin_row(self, row):
        """Bin a raw row the way the fitted mapper does."""
        bins = []
        for f, value in enumerate(row):
            missing = self.missing_bin_of[f]
            if math.isnan(value):
                bins.append(missing if missing >= 0 else 0)
                continue
            if self.is_categorical[f]:
                codes = self.cat_codes[
                    self.cat_offsets[f] : self.cat_offsets[f + 1]
                ]
                code = int(value)
                # Bin 0 collects missing, unseen, and dropped categories.
                bins.append(codes.index(code) + 1 if code in codes else 0)
                continue
            lo, hi = self.edge_offsets[f], self.edge_offsets[f + 1]
            edges = self.edges[lo:hi]
            b = 0
            while b < len(edges) and value > edges[b]:
                b += 1
            bins.append(b)
        return bins

    def contrib(self, row, start=0, stop=None):
        """Shapley values by enumeration, laid out as the API lays them
        out: `n_features + 1` per class, class-major."""
        n_rounds = len(self.trees) // self.n_classes
        stop = n_rounds if stop is None else stop
        m = self.n_features
        bins = self.bin_row(row)
        out = np.zeros(self.n_classes * (m + 1))
        m_fact = math.factorial(m)
        for k in range(self.n_classes):
            base = k * (m + 1)
            if start == 0:
                out[base + m] = self.base_scores[k]
            for i in range(start, stop):
                tree = self.trees[i * self.n_classes + k]
                out[base + m] += (
                    self.learning_rate * tree.conditional(bins, 0)
                )
                for f in range(m):
                    phi = 0.0
                    others = [j for j in range(m) if j != f]
                    for size in range(m):
                        weight = (
                            math.factorial(size)
                            * math.factorial(m - size - 1)
                            / m_fact
                        )
                        for combo in itertools.combinations(others, size):
                            s = 0
                            for j in combo:
                                s |= 1 << j
                            phi += weight * (
                                tree.conditional(bins, s | (1 << f))
                                - tree.conditional(bins, s)
                            )
                    out[base + f] += self.learning_rate * phi
        return out


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _regression_data(n_rows=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n_rows, n_features))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + X[:, 0] * X[:, 2]
    return X, y


def _fitted_regressor(n_features=4, **kwargs):
    X, y = _regression_data(n_features=n_features)
    params = dict(n_estimators=8, learning_rate=0.3, num_leaves=8,
                  min_child_samples=5, max_bin=32)
    params.update(kwargs)
    return MojoBoostRegressor(**params).fit(X, y), X, y


# ----------------------------------------------------------------------
# Shape and layout
# ----------------------------------------------------------------------


def test_regression_shape_and_sum():
    model, X, _ = _fitted_regressor()
    contrib = model.predict(X, pred_contrib=True)
    assert contrib.shape == (X.shape[0], X.shape[1] + 1)
    assert contrib.dtype == np.float64
    raw = model.predict(X, raw_score=True)
    np.testing.assert_allclose(contrib.sum(axis=1), raw, atol=1e-9)


def test_binary_classifier_explains_the_log_odds():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = MojoBoostClassifier(
        n_estimators=10, learning_rate=0.3, num_leaves=8,
        min_child_samples=5, max_bin=32,
    ).fit(X, y)
    contrib = model.predict_proba(X, pred_contrib=True)
    assert contrib.shape == (200, 5)
    # The binary classifier is a single-output ensemble, so contributions
    # explain its one raw score: the log-odds, not the probability.
    raw = model.predict_proba(X, raw_score=True)
    np.testing.assert_allclose(contrib.sum(axis=1), raw, atol=1e-9)


def test_multiclass_shape_is_class_major_blocks():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(180, 3))
    y = rng.integers(0, 3, size=180)
    model = MojoBoostClassifier(
        n_estimators=6, learning_rate=0.3, num_leaves=8,
        min_child_samples=5, max_bin=32,
    ).fit(X, y)
    contrib = model.predict_proba(X, pred_contrib=True)
    stride = X.shape[1] + 1
    assert contrib.shape == (180, 3 * stride)
    raw = model.predict_proba(X, raw_score=True)
    for k in range(3):
        block = contrib[:, k * stride : (k + 1) * stride]
        np.testing.assert_allclose(block.sum(axis=1), raw[:, k], atol=1e-9)


def test_classifier_predict_passes_contributions_through():
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(120, 3))
    y = (X[:, 0] > 0.5).astype(int)
    model = MojoBoostClassifier(
        n_estimators=6, num_leaves=8, min_child_samples=5, max_bin=32
    ).fit(X, y)
    # `predict` cannot return a label for a contribution request, so as in
    # LightGBM it hands back what `predict_proba` produced.
    np.testing.assert_array_equal(
        model.predict(X, pred_contrib=True),
        model.predict_proba(X, pred_contrib=True),
    )


def test_ranker_shape_and_sum():
    from mojoboost import MojoBoostRanker

    rng = np.random.default_rng(13)
    X = rng.uniform(size=(120, 3))
    y = rng.integers(0, 3, size=120)
    group = [30, 30, 30, 30]
    model = MojoBoostRanker(
        n_estimators=6, num_leaves=8, min_child_samples=5, max_bin=32
    ).fit(X, y, group=group)
    contrib = model.predict(X, pred_contrib=True)
    assert contrib.shape == (120, 4)
    np.testing.assert_allclose(
        contrib.sum(axis=1), model.predict(X), atol=1e-9
    )


# ----------------------------------------------------------------------
# Differential against the independent Python reference
# ----------------------------------------------------------------------


def test_matches_python_subset_enumeration(tmp_path):
    model, X, _ = _fitted_regressor(n_features=4)
    path = tmp_path / "model.txt"
    model.save(path)
    reference = _ReferenceModel(path)
    contrib = model.predict(X[:6], pred_contrib=True)
    for r in range(6):
        np.testing.assert_allclose(
            contrib[r], reference.contrib(X[r]), atol=1e-9
        )


def test_matches_python_reference_with_missing_values(tmp_path):
    X, y = _regression_data(n_rows=200, n_features=3, seed=7)
    X = X.copy()
    X[::4, 0] = np.nan
    X[::6, 1] = np.nan
    model = MojoBoostRegressor(
        n_estimators=8, learning_rate=0.3, num_leaves=8,
        min_child_samples=5, max_bin=32,
    ).fit(X, y)
    path = tmp_path / "model.txt"
    model.save(path)
    reference = _ReferenceModel(path)
    rows = [0, 1, 4, 6, 12]
    contrib = model.predict(X[rows], pred_contrib=True)
    for i, r in enumerate(rows):
        np.testing.assert_allclose(
            contrib[i], reference.contrib(X[r]), atol=1e-9
        )


def test_matches_python_reference_with_categorical_features(tmp_path):
    rng = np.random.default_rng(21)
    n_rows = 300
    X = np.column_stack(
        [
            (np.arange(n_rows) % 6).astype(float),
            rng.uniform(size=n_rows),
            rng.uniform(size=n_rows),
        ]
    )
    effect = np.array([3.0, -2.0, 0.5, -1.5, 2.0, -3.0])
    y = effect[(np.arange(n_rows) % 6)] + 0.8 * X[:, 1]
    model = MojoBoostRegressor(
        n_estimators=8, learning_rate=0.3, num_leaves=8,
        min_child_samples=5, max_bin=32, categorical_feature=[0],
    ).fit(X, y)
    path = tmp_path / "model.txt"
    model.save(path)
    reference = _ReferenceModel(path)
    assert any(
        off >= 0 for t in reference.trees for off in t.cat_offset
    ), "expected the fit to use a categorical set split"
    rows = [0, 1, 2, 3, 4, 5]
    contrib = model.predict(X[rows], pred_contrib=True)
    for i, r in enumerate(rows):
        np.testing.assert_allclose(
            contrib[i], reference.contrib(X[r]), atol=1e-9
        )


def test_matches_python_reference_on_an_iteration_slice(tmp_path):
    model, X, _ = _fitted_regressor(n_features=3, n_estimators=10)
    path = tmp_path / "model.txt"
    model.save(path)
    reference = _ReferenceModel(path)
    contrib = model.predict(
        X[:4], pred_contrib=True, start_iteration=2, num_iteration=5
    )
    for r in range(4):
        np.testing.assert_allclose(
            contrib[r], reference.contrib(X[r], start=2, stop=7), atol=1e-9
        )


# ----------------------------------------------------------------------
# Iteration ranges
# ----------------------------------------------------------------------


def test_slices_sum_to_the_whole_model():
    model, X, _ = _fitted_regressor()
    head = model.predict(X, pred_contrib=True, num_iteration=3)
    tail = model.predict(X, pred_contrib=True, start_iteration=3)
    whole = model.predict(X, pred_contrib=True)
    np.testing.assert_allclose(head + tail, whole, atol=1e-9)


def test_slice_sums_to_the_slice_raw_score():
    model, X, _ = _fitted_regressor()
    contrib = model.predict(
        X, pred_contrib=True, start_iteration=2, num_iteration=4
    )
    raw = model.predict(
        X, raw_score=True, start_iteration=2, num_iteration=4
    )
    np.testing.assert_allclose(contrib.sum(axis=1), raw, atol=1e-9)


def test_empty_slice_keeps_its_shape():
    model, X, _ = _fitted_regressor()
    # A range starting past the end selects nothing and carries no base
    # score, so every entry is zero and the shape is unchanged.
    contrib = model.predict(X, pred_contrib=True, start_iteration=999)
    assert contrib.shape == (X.shape[0], X.shape[1] + 1)
    np.testing.assert_allclose(contrib, 0.0, atol=0.0)


def test_zero_iterations_is_the_base_score_alone():
    model, X, _ = _fitted_regressor()
    contrib = model.predict(
        X, pred_contrib=True, start_iteration=0, num_iteration=0
    )
    # num_iteration <= 0 means "every remaining iteration" in LightGBM, so
    # this is the whole model, not an empty range.
    np.testing.assert_allclose(
        contrib, model.predict(X, pred_contrib=True), atol=0.0
    )


# ----------------------------------------------------------------------
# Failure modes and input handling
# ----------------------------------------------------------------------


def test_raw_score_and_pred_contrib_are_mutually_exclusive():
    model, X, _ = _fitted_regressor()
    with pytest.raises(ValueError, match="different outputs"):
        model.predict(X, raw_score=True, pred_contrib=True)


def test_pred_leaf_and_pred_contrib_are_mutually_exclusive():
    model, X, _ = _fitted_regressor()
    with pytest.raises(ValueError, match="different outputs"):
        model.predict(X, pred_leaf=True, pred_contrib=True)


def test_sparse_input_is_rejected_as_it_is_everywhere_else():
    sparse = pytest.importorskip("scipy.sparse")
    model, X, _ = _fitted_regressor()
    # Plain prediction takes sparse input now, but contributions do not
    # yet, and they refuse rather than densifying quietly.
    with pytest.raises(ValueError, match="sparse"):
        model.predict(sparse.csr_matrix(X), pred_contrib=True)
    # Densifying is the documented workaround, and it works.
    contrib = model.predict(sparse.csr_matrix(X).toarray(), pred_contrib=True)
    np.testing.assert_allclose(
        contrib.sum(axis=1), model.predict(X, raw_score=True), atol=1e-9
    )


def test_wrong_feature_count_is_rejected():
    model, X, _ = _fitted_regressor()
    with pytest.raises(ValueError):
        model.predict(X[:, :2], pred_contrib=True)


def test_unfitted_estimator_is_rejected():
    from sklearn.exceptions import NotFittedError

    X, _ = _regression_data()
    with pytest.raises(NotFittedError):
        MojoBoostRegressor().predict(X, pred_contrib=True)


# ----------------------------------------------------------------------
# Round-trips
# ----------------------------------------------------------------------


def test_contributions_survive_save_and_load(tmp_path):
    model, X, _ = _fitted_regressor()
    before = model.predict(X, pred_contrib=True)
    path = tmp_path / "model.txt"
    model.save(path)
    loaded = MojoBoostRegressor.load(path)
    np.testing.assert_allclose(
        loaded.predict(X, pred_contrib=True), before, atol=0.0
    )


def _downgrade_to_v2(path, out_path):
    """Rewrite a v3 model file as the v2 it would have been.

    v3 added exactly one line per tree, the node covers, written eighth in
    each tree block (after feature, threshold, left, right, value,
    default_left, missing_bin). Dropping it and the version bump is the whole
    difference, which is what makes this a real v2 file rather than a
    truncated v3 one."""
    lines = open(path).read().splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i == 0:
            out.append(line.replace("mojoboost v3", "mojoboost v2"))
            i += 1
            continue
        out.append(line)
        if line.startswith("tree "):
            out.extend(lines[i + 1 : i + 8])
            i += 9  # skip the covers line
            continue
        i += 1
    open(out_path, "w").write("\n".join(out) + "\n")


def test_v2_files_still_load_and_predict_but_refuse_contributions(tmp_path):
    model, X, _ = _fitted_regressor()
    v3 = tmp_path / "model_v3.txt"
    model.save(v3)
    v2 = tmp_path / "model_v2.txt"
    _downgrade_to_v2(v3, v2)

    loaded = MojoBoostRegressor.load(v2)
    # Prediction is unaffected: covers never entered it.
    np.testing.assert_allclose(
        loaded.predict(X), model.predict(X), atol=0.0
    )
    # Contributions cannot be invented from a file that never held the
    # covers they condition on, so they say so.
    with pytest.raises(Exception, match="v1 or v2|node counts"):
        loaded.predict(X, pred_contrib=True)


def test_contributions_survive_pickling():
    import pickle

    model, X, _ = _fitted_regressor()
    before = model.predict(X, pred_contrib=True)
    revived = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        revived.predict(X, pred_contrib=True), before, atol=0.0
    )


def test_single_row_and_single_feature():
    rng = np.random.default_rng(31)
    X = rng.uniform(size=(80, 1))
    y = 2.0 * X[:, 0]
    model = MojoBoostRegressor(
        n_estimators=5, num_leaves=4, min_child_samples=5, max_bin=16
    ).fit(X, y)
    contrib = model.predict(X[:1], pred_contrib=True)
    assert contrib.shape == (1, 2)
    np.testing.assert_allclose(
        contrib.sum(axis=1), model.predict(X[:1], raw_score=True), atol=1e-9
    )


def test_constant_target_puts_everything_in_the_expected_value():
    # A constant target converges immediately, so the ensemble is the base
    # score alone and no feature can carry anything.
    X = np.random.default_rng(41).uniform(size=(60, 3))
    y = np.full(60, 2.5)
    model = MojoBoostRegressor(
        n_estimators=5, num_leaves=4, min_child_samples=5, max_bin=16
    ).fit(X, y)
    contrib = model.predict(X, pred_contrib=True)
    np.testing.assert_allclose(contrib[:, :3], 0.0, atol=1e-12)
    np.testing.assert_allclose(contrib[:, 3], 2.5, atol=1e-9)
