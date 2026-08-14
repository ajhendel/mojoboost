"""Model inspection: the dump schema and what is derived from it.

The schema is only worth having if it describes the model that actually
predicts, so most of what is checked here is that claim: walking the dump's
thresholds must land on the same leaf, and sum to the same raw score, as
the model's own `predict`. The rest checks the shapes `trees_to_dataframe`
and the split value histogram produce, and the six LightGBM attributes this
module wires up.

`docs/MODEL_INSPECTION_SCHEMA.md` is the schema these tests hold the
implementation to.
"""

import json

import pytest

np = pytest.importorskip("numpy")

from mojoboost import (  # noqa: E402
    Booster,
    MojoBoostClassifier,
    MojoBoostRegressor,
)
from mojoboost import inspection  # noqa: E402


@pytest.fixture(scope="module")
def fitted(regression):
    X, y = regression
    return MojoBoostRegressor(n_estimators=8, num_leaves=7).fit(X, y), X


@pytest.fixture(scope="module")
def dump(fitted):
    model, _ = fitted
    return inspection.dump_model(model)


# -- the schema ----------------------------------------------------------


def test_dump_reports_its_own_version_and_source(dump):
    assert dump["dump_format_version"] == inspection.DUMP_FORMAT_VERSION
    assert dump["producer"] == "mojoboost"
    assert dump["model_format_version"] == 3
    # No native hook is bound, so the model text is the source and the one
    # thing it cannot carry is the split gains.
    assert dump["source"].startswith("model_to_string")
    assert dump["has_split_gain"] is False
    assert dump["has_node_count"] is True


def test_dump_describes_the_ensemble(dump, fitted):
    model, X = fitted
    assert dump["objective"] == "regression"
    assert dump["objective_code"] == 0
    assert dump["num_class"] == 1
    assert dump["num_tree_per_iteration"] == 1
    assert dump["num_iteration"] == model.n_iter_ == 8
    assert len(dump["tree_info"]) == 8
    assert dump["num_feature"] == X.shape[1]
    assert dump["max_feature_idx"] == X.shape[1] - 1
    assert dump["feature_names"] == [f"Column_{i}" for i in range(X.shape[1])]
    assert dump["learning_rate"] == pytest.approx(0.1)
    assert len(dump["base_score"]) == 1
    # mojoboost stores unshrunk leaf values and multiplies by the shrinkage
    # when it predicts; LightGBM folds the shrinkage into the leaf.
    assert dump["leaf_value_is_shrunk"] is False
    assert all(
        tree["shrinkage"] == dump["learning_rate"]
        for tree in dump["tree_info"]
    )


def test_dump_is_json(dump):
    """Every value is a plain Python scalar, list, dict, or None, so the
    schema can travel as JSON without a custom encoder."""
    assert json.loads(json.dumps(dump)) == dump


def test_feature_infos_describe_the_binning(dump):
    for index, info in enumerate(dump["feature_infos"]):
        assert info["index"] == index
        assert info["type"] == "numerical"
        assert info["categories"] is None
        assert info["monotone"] == 0
        edges = info["bin_upper_bounds"]
        assert edges == sorted(edges)
        assert len(set(edges)) == len(edges)
        # No NaN in this training matrix, so no feature reserves a bin.
        assert info["missing_bin"] == -1
        assert info["missing_type"] == "None"
        assert info["num_bin"] == len(edges) + 1


def test_nodes_carry_the_documented_keys(dump):
    for tree in dump["tree_info"]:
        assert tree["num_nodes"] == 2 * tree["num_leaves"] - 1
        assert tree["num_cat"] == 0
        for node in _walk_nodes(tree["tree_structure"]):
            if "leaf_index" in node:
                assert set(node) == {
                    "node_index",
                    "leaf_index",
                    "leaf_value",
                    "leaf_count",
                    "depth",
                }
            else:
                assert node["decision_type"] == "<="
                assert node["threshold"] is not None
                assert node["categories"] is None
                assert node["category_bins"] is None
                assert node["split_gain"] is None
                assert node["missing_type"] == "None"


def test_leaf_ordinals_are_dense_and_ordered(dump):
    """The dump's `leaf_index` is `Tree.leaf_ordinals`: leaves ranked in
    node-array order, so the ordinals of a tree are exactly 0..n_leaves-1
    and rise with the node index."""
    for tree in dump["tree_info"]:
        leaves = [
            (n["node_index"], n["leaf_index"])
            for n in _walk_nodes(tree["tree_structure"])
            if "leaf_index" in n
        ]
        leaves.sort()
        assert [ordinal for _, ordinal in leaves] == list(
            range(tree["num_leaves"])
        )


def test_node_counts_split_between_children(dump):
    """A node's cover is the training rows that reached it, so an internal
    node's cover is exactly its two children's."""
    for tree in dump["tree_info"]:
        for node in _walk_nodes(tree["tree_structure"]):
            if "leaf_index" in node:
                continue
            children = node["left_child"], node["right_child"]
            total = sum(_count_of(child) for child in children)
            assert node["internal_count"] == pytest.approx(total)
            assert node["internal_count"] > 0


def test_depths_agree_with_the_tree(dump):
    for tree in dump["tree_info"]:
        depths = [n["depth"] for n in _walk_nodes(tree["tree_structure"])]
        assert tree["tree_structure"]["depth"] == 0
        assert tree["max_depth"] == max(depths)


# -- the dump describes the model that predicts --------------------------


def test_thresholds_reproduce_raw_predictions(fitted, dump):
    """The load-bearing claim. A bin's upper edge is the exact boundary
    routing uses, so walking the dump with raw feature values has to land
    on the same leaves the model does and sum to the same score."""
    model, X = fitted
    expected = model.booster_.predict(X, raw_score=True)
    got = [inspection.raw_scores(dump, row)[0] for row in X]
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_leaf_indices_match_pred_leaf(fitted, dump):
    model, X = fitted
    expected = model.predict(X[:40], pred_leaf=True)
    for row_index, row in enumerate(X[:40]):
        for tree_index in range(dump["num_iteration"]):
            assert (
                inspection.leaf_index_of(dump, tree_index, row)
                == expected[row_index][tree_index]
            )


def test_missing_values_route_as_the_model_routes_them():
    """With NaN in the training matrix a feature reserves a missing bin,
    and the dump has to carry both the reservation and each node's
    direction, or a NaN row would take the wrong branch."""
    gen = np.random.default_rng(3)
    X = gen.random((300, 3))
    X[::5, 0] = np.nan
    y = np.where(np.isnan(X[:, 0]), 5.0, X[:, 0]) + 0.5 * X[:, 1]
    model = MojoBoostRegressor(n_estimators=6, num_leaves=5).fit(X, y)
    dump = inspection.dump_model(model)

    info = dump["feature_infos"][0]
    assert info["missing_bin"] >= 0
    assert info["missing_type"] == "NaN"
    assert info["num_bin"] == len(info["bin_upper_bounds"]) + 2

    expected = model.booster_.predict(X, raw_score=True)
    got = [inspection.raw_scores(dump, row)[0] for row in X]
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_categorical_splits_report_their_category_codes():
    gen = np.random.default_rng(5)
    codes = gen.integers(0, 6, size=400).astype(float)
    X = np.column_stack([codes, gen.random(400)])
    y = np.where(codes % 2 == 0, 1.0, -1.0) + 0.1 * X[:, 1]
    model = MojoBoostRegressor(
        n_estimators=5, num_leaves=5, categorical_feature=[0]
    ).fit(X, y)
    dump = inspection.dump_model(model)

    info = dump["feature_infos"][0]
    assert info["type"] == "categorical"
    assert info["categories"] == sorted(set(int(c) for c in codes))
    assert info["bin_upper_bounds"] is None
    # Bin 0 collects the missing, unseen, and dropped codes.
    assert info["num_bin"] == len(info["categories"]) + 1

    splits = [
        node
        for tree in dump["tree_info"]
        for node in _walk_nodes(tree["tree_structure"])
        if "split_index" in node and node["split_feature"] == 0
    ]
    assert splits, "the model never split on the categorical feature"
    for node in splits:
        assert node["decision_type"] == "=="
        assert node["threshold"] is None
        assert node["categories"] == sorted(node["categories"])
        assert set(node["categories"]) <= set(info["categories"])
        # Bin 0 is never a set member, so an unseen code always goes right.
        assert 0 not in node["category_bins"]

    expected = model.booster_.predict(X, raw_score=True)
    got = [inspection.raw_scores(dump, row)[0] for row in X]
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_multiclass_dump_is_round_major(multiclass):
    X, y = multiclass
    model = MojoBoostClassifier(n_estimators=4).fit(X, y)
    dump = inspection.dump_model(model)

    assert dump["objective"] == "multiclass"
    assert dump["objective_code"] is None
    assert dump["num_class"] == 3
    assert dump["num_tree_per_iteration"] == 3
    assert dump["num_iteration"] == 4
    assert len(dump["tree_info"]) == 12
    assert len(dump["base_score"]) == 3
    for index, tree in enumerate(dump["tree_info"]):
        assert tree["tree_index"] == index
        assert tree["iteration"] == index // 3
        assert tree["class_id"] == index % 3

    expected = model.booster_.predict(X[:50], raw_score=True)
    got = [inspection.raw_scores(dump, row) for row in X[:50]]
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def test_dump_from_the_model_text_alone(fitted, dump):
    """The dump's only source today is the model text, so handing that text
    in directly has to give the same schema."""
    model, _ = fitted
    text = model.booster_.model_to_string()
    assert inspection.dump_model(text) == dump


def test_a_booster_read_back_dumps_the_same_trees(fitted, dump):
    model, _ = fitted
    revived = Booster(model_str=model.booster_.model_to_string())
    assert inspection.dump_model(revived) == dump


def test_feature_names_can_be_supplied(fitted):
    model, X = fitted
    names = [f"f{i}" for i in range(X.shape[1])]
    named = inspection.dump_model(model, feature_names=names)
    assert named["feature_names"] == names
    root = named["tree_info"][0]["tree_structure"]
    assert root["split_feature_name"] in names
    with pytest.raises(ValueError, match="entries for a model with"):
        inspection.dump_model(model, feature_names=names[:-1])


def test_dump_rejects_text_that_is_not_a_model():
    with pytest.raises(ValueError, match="not a mojoboost model"):
        inspection.dump_model("lightgbm v3\n")
    with pytest.raises(ValueError, match="newer than this build reads"):
        inspection.dump_model("mojoboost v99\n")


# -- derived shapes ------------------------------------------------------


def test_trees_to_records_has_lightgbm_columns(fitted, dump):
    model, _ = fitted
    rows = inspection.trees_to_records(model, dump=dump)
    assert rows
    assert all(
        tuple(row) == inspection.TREE_FRAME_COLUMNS for row in rows
    )
    assert len(rows) == sum(t["num_nodes"] for t in dump["tree_info"])
    # LightGBM counts the root as depth 1 and names nodes per tree.
    root = rows[0]
    assert root["node_depth"] == 1
    assert root["node_index"] == "0-S0"
    assert root["parent_index"] is None
    assert root["decision_type"] == "<="
    assert root["missing_direction"] in ("left", "right")
    # mojoboost records covers, not hessian sums, so there is no weight.
    assert all(row["weight"] is None for row in rows)
    assert all(row["split_gain"] is None for row in rows)

    by_name = {row["node_index"]: row for row in rows}
    for row in rows:
        if row["left_child"] is None:
            assert row["node_index"].split("-")[1].startswith("L")
            continue
        for child in (row["left_child"], row["right_child"]):
            assert by_name[child]["parent_index"] == row["node_index"]
            assert by_name[child]["node_depth"] == row["node_depth"] + 1


def test_trees_to_dataframe(fitted, dump):
    pandas = pytest.importorskip("pandas")
    model, _ = fitted
    frame = inspection.trees_to_dataframe(model, dump=dump)
    assert isinstance(frame, pandas.DataFrame)
    assert list(frame.columns) == list(inspection.TREE_FRAME_COLUMNS)
    assert len(frame) == sum(t["num_nodes"] for t in dump["tree_info"])
    assert frame["tree_index"].nunique() == dump["num_iteration"]


def test_split_values_are_the_thresholds_the_model_uses(fitted, dump):
    model, _ = fitted
    values = inspection.split_values(model, 0, dump=dump)
    edges = set(dump["feature_infos"][0]["bin_upper_bounds"])
    assert values
    assert set(values) <= edges
    assert values == inspection.split_values(model, "Column_0", dump=dump)


def test_split_value_histogram(fitted, dump):
    model, _ = fitted
    values = inspection.split_values(model, 0, dump=dump)
    counts, edges = inspection.get_split_value_histogram(
        model, 0, dump=dump
    )
    assert sum(counts) == len(values)
    assert len(edges) == len(counts) + 1
    assert edges == sorted(edges)
    assert edges[0] <= min(values)
    assert edges[-1] >= max(values)

    coarse_counts, coarse_edges = inspection.get_split_value_histogram(
        model, 0, bins=3, dump=dump
    )
    assert len(coarse_counts) == min(3, len(set(values)))
    assert sum(coarse_counts) == len(values)
    assert len(coarse_edges) == len(coarse_counts) + 1


def test_split_value_histogram_as_frame(fitted, dump):
    pytest.importorskip("pandas")
    model, _ = fitted
    frame = inspection.get_split_value_histogram(
        model, 0, bins=4, as_frame=True, dump=dump
    )
    assert list(frame.columns) == ["Count", "SplitValue"]
    assert frame["Count"].sum() == len(
        inspection.split_values(model, 0, dump=dump)
    )


def test_split_value_histogram_refuses_what_it_cannot_bin(fitted, dump):
    model, _ = fitted
    with pytest.raises(ValueError, match="outside the model's"):
        inspection.get_split_value_histogram(model, 99, dump=dump)
    with pytest.raises(ValueError, match="no feature named"):
        inspection.get_split_value_histogram(model, "nope", dump=dump)


def test_categorical_features_have_no_split_value_histogram():
    gen = np.random.default_rng(5)
    codes = gen.integers(0, 6, size=300).astype(float)
    X = np.column_stack([codes, gen.random(300)])
    y = np.where(codes % 2 == 0, 1.0, -1.0)
    model = MojoBoostRegressor(
        n_estimators=3, num_leaves=5, categorical_feature=[0]
    ).fit(X, y)
    with pytest.raises(ValueError, match="is categorical"):
        inspection.get_split_value_histogram(model, 0)


# -- the LightGBM attributes ---------------------------------------------


def test_objective_names_track_the_estimator_table():
    """`OBJECTIVE_NAMES` is a second copy of the trainer's objective codes,
    so it is checked against the table the estimators validate against."""
    by_code = {}
    for name, code in MojoBoostRegressor._OBJECTIVES.items():
        by_code.setdefault(code, set()).add(name)
    for code, names in by_code.items():
        assert inspection.OBJECTIVE_NAMES[code] in names
    # The two objectives no regressor spells, one per non-regression task.
    assert inspection.OBJECTIVE_NAMES[1] == "binary"
    assert inspection.OBJECTIVE_NAMES[7] == "lambdarank"


def test_attribute_wiring(fitted):
    model, X = fitted
    booster = inspection.booster_of(model)
    assert booster is not None
    assert inspection.booster_of(booster) is booster
    assert inspection.feature_name_of(model) == [
        f"Column_{i}" for i in range(X.shape[1])
    ]
    assert inspection.n_features_of(model) == X.shape[1]
    assert inspection.n_iter_of(model) == model.n_iter_ == 8
    assert inspection.n_iter_of(booster) == 8
    assert inspection.objective_of(model) == "regression"
    assert inspection.objective_of(booster) == "regression"


def test_objective_is_resolved_not_echoed(regression):
    """An alias in, the canonical name out: the answer comes from the code
    the model carries, not from the string the estimator was given."""
    X, y = regression
    model = MojoBoostRegressor(objective="mae", n_estimators=3).fit(X, y)
    assert model.objective == "mae"
    assert inspection.objective_of(model) == "regression_l1"


def test_best_score_is_a_fit_time_record(fitted, regression):
    model, _ = fitted
    with pytest.raises(AttributeError, match="fitted without a validation"):
        inspection.best_score_of(model)
    X, y = regression
    scored = MojoBoostRegressor(n_estimators=5).fit(
        X, y, eval_set=[(X, y)], eval_metric="l2"
    )
    assert inspection.best_score_of(scored) == pytest.approx(
        scored.best_score_
    )


def test_unfitted_estimators_have_nothing_to_inspect():
    from mojoboost import NotFittedError

    with pytest.raises(NotFittedError):
        inspection.dump_model(MojoBoostRegressor())


def test_leaf_editing_is_not_offered():
    """Stated as a test so that adding it is a deliberate act with a place
    to state its invariants, and not an accident."""
    for name in ("set_leaf_output", "set_leaf_value", "edit_leaf"):
        assert not hasattr(inspection, name)
    assert "set_leaf_output" not in inspection.__all__


# -- helpers -------------------------------------------------------------


def _walk_nodes(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if "leaf_index" not in current:
            stack.append(current["left_child"])
            stack.append(current["right_child"])


def _count_of(node):
    if "leaf_index" in node:
        return node["leaf_count"]
    return node["internal_count"]
