"""The v5 `ctr` section: a CTR model that saves, loads and scores the same.

Catalog A19 (the tables) and A31 (this section). The claim under test is the
one the old refusal existed to protect: a model carrying fitted ordered
target statistics must come back off disk **identical**, not approximately.
The tables are read off the target, so a model that lost or rounded them
keeps every tree referencing its CTR columns and bins those columns from
different numbers -- a wrong answer that looks like a right one.

What each test pins:

- **The round trip is exact, at the layer where the tables are consumed.**
  `BinMapper.bin_row` is where a fitted table turns into a bin index, so the
  saved and loaded mappers are compared there, row by row, as well as through
  `CtrTables.matches` and through the model's predictions. Comparing only
  predictions would pass a model whose trees never split on a CTR column.
- **A CTR tree really is written and read.** A tree may split on ids up to
  `n_total_features() - 1`, which the loader's topology check refused before
  this lane widened it. The test asserts such a split exists, so the previous
  point is not vacuous.
- **Version 4 still loads, and still gets written.** A model without CTR
  tables and without linear leaves declares v4 byte for byte as before, and a
  v4 file loads to `CtrTables.none()`, which is what its absence means.
- **An unknown version is refused rather than guessed at.** That is the
  property that makes an older binary meeting a newer file safe.
- **What still refuses, refuses by name**: the prepared-table writer, the
  model dump, and a table set whose derived lookup is not the one its counts
  imply.
"""

from std.os import remove
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.boosting import BINARY_LOGISTIC, BoosterParams
from mojotrees.ctr_columns import (
    CtrTables,
    SimpleCtrConfig,
    check_ctr_dataset_serializable,
    check_ctr_serializable,
    plan_ctr_columns,
)
from mojotrees.model_dump import build_dump
from mojotrees.raw_data import RawData
from mojotrees.serialize import (
    CTR_SECTION_REVISION,
    CURRENT_FORMAT_VERSION,
    load_model,
    load_multiclass_model,
    save_dataset,
    save_model,
    save_multiclass_model,
)
from mojotrees.trainset import Dataset, train_dataset, train_dataset_multiclass
from mojotrees.tree import TreeParams

comptime _CTR_PATH = "./.test_ctr_model.tmp"
comptime _CTR_MC_PATH = "./.test_ctr_multiclass.tmp"
comptime _PLAIN_PATH = "./.test_ctr_plain_model.tmp"
comptime _BAD_PATH = "./.test_ctr_bad_version.tmp"
comptime _DS_PATH = "./.test_ctr_dataset.tmp"

comptime _N_ROWS = 240
comptime _N_CATEGORIES = 8


def _rows() raises -> RawData:
    """Two columns, column-major: a noise feature and a wide categorical one.

    The categorical column has eight levels, comfortably past
    `one_hot_max_size = 2`, so `ctr_source_features` gives it CTRs. The noise
    column exists so that the split search has something to reject.
    """
    var values = List[Float64](capacity=_N_ROWS * 2)
    var state: UInt64 = 987654321
    for _ in range(_N_ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        values.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    for r in range(_N_ROWS):
        values.append(Float64(r % _N_CATEGORIES))
    return RawData.dense(values^, _N_ROWS, 2)


def _labels() -> List[Float64]:
    """A 0/1 target that is a function of the category and of nothing else,
    so the ordered statistic of the category is the informative column."""
    var label = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        label.append(1.0 if (r % _N_CATEGORIES) >= 5 else 0.0)
    return label^


def _raw_row(r: Int) raises -> List[Float64]:
    """One row in raw feature order, which stays two wide however many CTR
    columns the binned matrix gained."""
    var raw = _rows()
    return [raw.values[r], raw.values[_N_ROWS + r]]


def _ctr_dataset() raises -> Dataset:
    var cats: List[Int] = [1]
    return Dataset.from_raw(
        _rows(),
        label=_labels(),
        categorical_features=cats^,
        max_bin=64,
        ctr=SimpleCtrConfig.catboost_defaults(),
    )


def _plain_dataset() raises -> Dataset:
    var cats: List[Int] = [1]
    return Dataset.from_raw(
        _rows(), label=_labels(), categorical_features=cats^, max_bin=64
    )


def _params() -> BoosterParams:
    return BoosterParams(12, 0.2, TreeParams(8, 5, 1.0, 1e-3))


def _first_token_pair(path: String) raises -> String:
    """The magic and the version token of a written file, joined by a space,
    which is what the header's first line says."""
    var content = open(path, "r").read()
    var tokens = List[String]()
    for tok in content.split():
        tokens.append(String(tok))
        if len(tokens) == 2:
            break
    if len(tokens) < 2:
        raise Error("file has no header")
    return tokens[0] + " " + tokens[1]


# ---------------------------------------------------------------------------
# The round trip, which is the deliverable
# ---------------------------------------------------------------------------


def test_a_ctr_model_round_trips_exactly() raises:
    var dataset = _ctr_dataset()
    # Four numeric columns from one wide categorical column, A19's headline
    # count: Borders x 3 priors + Counter x 1.
    assert_true(dataset.has_ctr())
    assert_equal(dataset.n_ctr_columns(), 4)
    assert_equal(dataset.n_total_features(), 6)

    var model = train_dataset(dataset, BINARY_LOGISTIC, _params())
    assert_true(model.mapper.has_ctr())

    # Not vacuous: some tree splits on a column that only exists because the
    # CTR tables do. Before this lane the loader's topology check refused
    # exactly this file, because it validated feature ids against
    # `n_features` rather than `n_total_features()`.
    var splits_on_ctr = False
    for t in range(len(model.booster.trees)):
        ref tree = model.booster.trees[t]
        for i in range(len(tree.feature)):
            if tree.feature[i] >= model.mapper.n_features:
                splits_on_ctr = True
    assert_true(splits_on_ctr)

    save_model(model, _CTR_PATH)
    # The file says v5, because it carries a section a v4 reader cannot.
    assert_equal(_first_token_pair(_CTR_PATH), String("mojotrees v5"))
    var loaded = load_model(_CTR_PATH)
    remove(_CTR_PATH)

    # 1. The tables themselves, field by field, including the derived lookup
    #    the file does not carry and the reader rebuilds.
    assert_true(loaded.mapper.ctr.matches(model.mapper.ctr))

    # PRODUCT DEFECT, pinned here rather than asserted away. `usable` is the
    # one BinMapper field the v5 file does not carry, and since the
    # CTR-replacement lane a CatBoost-mode fit MUTATES it: `trainset.
    # _build_ctr` calls `mapper.drop_usable(replaced)` to take the replaced
    # source column out of the split-search pool. `serialize` writes no
    # `usable` section, and the loader's BinMapper constructor reads an empty
    # `usable` as "nothing was prefiltered" and rebuilds a FULL pool. So a
    # saved CatBoost-mode CTR model loads back with a pool it did not have.
    #
    # `BinMapper.matches` compares the field, and `matches` is on the LOAD
    # path -- trainset, model_editing and external_memory all call
    # `model.mapper.matches(dataset.mapper)` on a model that may have come
    # off disk. So `save -> load -> train_more` on such a model now refuses
    # with "this one is binned differently", which is not why it refused:
    # the binning is bit-identical and only the pool differs.
    #
    # Bounded honestly: predictions are NOT affected. `bin_row`, `predict`,
    # `predict_raw`, dumps and contributions never read `usable`, and part 2
    # and part 3 below still pass bit for bit. This fails CLOSED, as a
    # spurious refusal, not open as a wrong score. The remedy is a `usable`
    # section in serialize.mojo, which that file's own comment already calls
    # for; it is not this lane's to make. When it lands, this whole block
    # collapses back to `assert_true(loaded.mapper.matches(model.mapper))`
    # and the `assert_true(not ...)` line below is what will say so.
    assert_equal(len(model.mapper.usable), 1)
    assert_equal(model.mapper.usable[0], 0)
    assert_equal(len(loaded.mapper.usable), 2)
    assert_true(not loaded.mapper.matches(model.mapper))

    # And the round-trip claim itself is kept, not dropped: restore the one
    # field the file does not carry and the mappers must be equal. This still
    # discriminates over every other field -- edges, edge offsets, missing
    # bins, category codes, the CTR tables -- so a SECOND field failing to
    # round trip fails here.
    var restored = loaded.mapper.copy()
    restored.drop_usable([1])
    assert_true(restored.matches(model.mapper))

    assert_equal(loaded.mapper.ctr.n_columns(), 4)
    assert_equal(
        loaded.mapper.n_total_features(), model.mapper.n_total_features()
    )

    # 2. The layer where a table becomes a bin. This is the comparison that
    #    cannot be passed by a model whose trees ignore the CTR columns: every
    #    appended bin here came out of the fitted tables.
    for r in range(_N_ROWS):
        var row = _raw_row(r)
        var want = model.mapper.bin_row(row)
        var got = loaded.mapper.bin_row(row)
        assert_equal(len(got), len(want))
        assert_equal(len(got), 6)
        for f in range(len(want)):
            assert_equal(got[f], want[f])

    # 3. The score, bit for bit. Floats travel as their IEEE-754 bit
    #    patterns, so `==` here is the whole claim and not a tolerance.
    for r in range(_N_ROWS):
        var row = _raw_row(r)
        assert_true(loaded.predict(row) == model.predict(row))
        assert_true(loaded.predict_raw(row) == model.predict_raw(row))


def test_a_ctr_multiclass_model_round_trips_exactly() raises:
    var dataset = _ctr_dataset()
    var model = train_dataset_multiclass(dataset, 2, _params())
    assert_true(model.mapper.has_ctr())

    save_multiclass_model(model, _CTR_MC_PATH)
    assert_equal(_first_token_pair(_CTR_MC_PATH), String("mojotrees v5"))
    var loaded = load_multiclass_model(_CTR_MC_PATH)
    remove(_CTR_MC_PATH)

    assert_true(loaded.mapper.ctr.matches(model.mapper.ctr))
    for r in range(0, _N_ROWS, 3):
        var row = _raw_row(r)
        var want = model.predict_raw(row)
        var got = loaded.predict_raw(row)
        assert_equal(len(got), len(want))
        for k in range(len(want)):
            assert_true(got[k] == want[k])


# ---------------------------------------------------------------------------
# Both directions across the version boundary
# ---------------------------------------------------------------------------


def test_a_model_without_ctr_still_writes_and_reads_v4() raises:
    var dataset = _plain_dataset()
    assert_true(not dataset.has_ctr())
    var model = train_dataset(dataset, BINARY_LOGISTIC, _params())

    save_model(model, _PLAIN_PATH)
    # The bump is conditional, so nothing that did not gain a section moved.
    assert_equal(_first_token_pair(_PLAIN_PATH), String("mojotrees v4"))
    var loaded = load_model(_PLAIN_PATH)
    remove(_PLAIN_PATH)

    # A v4 file has no ctr section, and its absence means exactly this.
    assert_true(not loaded.mapper.has_ctr())
    assert_true(not loaded.mapper.ctr.is_active())
    assert_true(loaded.mapper.ctr.matches(CtrTables.none()))
    for r in range(0, _N_ROWS, 5):
        var row = _raw_row(r)
        assert_true(loaded.predict(row) == model.predict(row))


def test_an_unknown_version_token_is_refused() raises:
    # The other direction of the cross-version question, stated as the
    # property that makes it safe: a reader that meets a version it does not
    # know stops, rather than reading the fields it does know and guessing at
    # the rest. A v4 build meeting a v5 CTR file stops for the same kind of
    # reason one token later, on `ctr` where it expects `trees`.
    with open(_BAD_PATH, "w") as f:
        f.write(String("mojotrees v9\nobjective 0\n"))
    with assert_raises():
        _ = load_model(_BAD_PATH)
    remove(_BAD_PATH)


def test_the_version_constants_say_what_the_writer_writes() raises:
    assert_equal(CURRENT_FORMAT_VERSION, 5)
    # Revision 2 since 2026-08-16: the section carries `slot_codes` /
    # `slot_code_offsets`, because a CTR bucket became an index into the
    # source column's complete category table rather than a binned category
    # id. Nothing else in the file carries that table, so a reader that
    # skipped it would map every raw value to bucket 0 and score every row
    # from the pure prior. `_read_ctr` refuses an unrecognized revision by
    # number, which is the next test.
    assert_equal(CTR_SECTION_REVISION, 2)


# ---------------------------------------------------------------------------
# What still refuses, and each one by its own reason
# ---------------------------------------------------------------------------


def test_a_prepared_table_still_refuses_ctr_columns() raises:
    # Not the model's blocker any more, and a different blocker: the binned
    # matrix is n_base + n_ctr wide while its mapper counts the base features,
    # and `Dataset.from_binned_dense` refuses that pair on the way back in.
    var dataset = _ctr_dataset()
    with assert_raises():
        save_dataset(dataset, _DS_PATH)
    # A dataset without ctr columns writes as it always did.
    save_dataset(_plain_dataset(), _DS_PATH)
    remove(_DS_PATH)
    check_ctr_dataset_serializable(CtrTables.none())


def test_the_model_dump_still_refuses_ctr_columns() raises:
    # Every field of that schema is sized by `mapper.n_features`, which counts
    # base features only. Refusing beats describing the wrong features.
    var model = train_dataset(_ctr_dataset(), BINARY_LOGISTIC, _params())
    with assert_raises():
        _ = build_dump(model)
    # The same model saves and loads, which is the point of the distinction.
    save_model(model, _CTR_PATH)
    var loaded = load_model(_CTR_PATH)
    remove(_CTR_PATH)
    assert_true(loaded.mapper.ctr.matches(model.mapper.ctr))


def test_a_table_set_whose_lookup_is_not_derived_refuses() raises:
    # `predict_lut` is the one part of `CtrTables` the file leaves out, and it
    # is safe to leave out only while it is the lookup the counts imply. A set
    # planned by `plan_ctr_columns` and never fitted has no lookup at all, so
    # saving it would silently produce a different model on reload.
    check_ctr_serializable(CtrTables.none())
    var flags: List[Bool] = [False, True]
    var counts: List[Int] = [0, 6]
    var borders: List[Float64] = [0.5]
    var config = SimpleCtrConfig.catboost_defaults(borders^)
    var planned = plan_ctr_columns(config, flags, counts, 255)
    assert_true(planned.is_active())
    with assert_raises():
        check_ctr_serializable(planned)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
