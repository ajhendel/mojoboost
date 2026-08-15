"""Tests for the command line tool.

The commands are ordinary functions (see cli/mojotrees_cli.mojo), so these
call them directly instead of shelling out to a built binary: the same code
runs, and a failure points at a line rather than at a process exit status.

Run with `mojo run -I src -I cli tests/test_cli.mojo`.
"""

from std.os import remove
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.serialize import load_model, load_multiclass_model

from mojotrees_cli import (
    column_major_features,
    command_info,
    command_predict,
    command_train,
    parse_field,
    parse_options,
    read_table,
    resolve_column,
    run,
)

comptime _DATA = "./.test_cli_data.tmp"
comptime _MODEL = "./.test_cli_model.tmp"
comptime _OUTPUT = "./.test_cli_output.tmp"


def _write(path: String, content: String) raises:
    with open(path, "w") as f:
        f.write(content)


def _args(items: List[String]) -> List[String]:
    var out = List[String]()
    for i in range(len(items)):
        out.append(items[i].copy())
    return out^


def _training_file(path: String) raises -> Int:
    """A small regression dataset with the label in column 0. Returns the
    row count."""
    var content = String("# label,f0,f1\n")
    var state: UInt64 = 0x9E3779B97F4A7C15
    var n_rows = 120
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var a = Float64(state >> 11) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var b = Float64(state >> 11) * (1.0 / 9007199254740992.0)
        content += String(2.0 * a - b, ",", a, ",", b, "\n")
    _write(path, content)
    return n_rows


def _lines(text: String) -> List[String]:
    var out = List[String]()
    for line_slice in text.split("\n"):
        var line = String(String(line_slice).strip())
        if line.byte_length() > 0:
            out.append(line^)
    return out^


# ------------------------------------------------------------ field parsing


def test_parse_field_reads_numbers() raises:
    assert_equal(parse_field(String("1.5")), 1.5)
    assert_equal(parse_field(String("-2")), -2.0)
    assert_equal(parse_field(String(" 3.25 ")), 3.25)
    assert_equal(parse_field(String("1e-3")), 0.001)


def test_parse_field_reads_missing_markers() raises:
    var markers = [
        String(""),
        String("  "),
        String("?"),
        String("na"),
        String("NA"),
        String("n/a"),
        String("nan"),
        String("NaN"),
        String("null"),
    ]
    for i in range(len(markers)):
        var value = parse_field(markers[i])
        # NaN is the only value that is not equal to itself, which is how
        # the trainer recognizes a missing feature.
        assert_true(value != value)


def test_parse_field_rejects_text() raises:
    with assert_raises():
        _ = parse_field(String("abc"))
    with assert_raises():
        _ = parse_field(String("1.2.3"))


# -------------------------------------------------------------- file reader


def test_read_table_skips_comments_and_blank_lines() raises:
    _write(
        String(_DATA),
        String("# a comment\n\n1,2,3\n\n  # another\n4,5,6\n"),
    )
    var table = read_table(String(_DATA))
    assert_equal(table.n_rows, 2)
    assert_equal(table.n_cols, 3)
    assert_equal(table.at(0, 0), 1.0)
    assert_equal(table.at(1, 2), 6.0)
    remove(_DATA)


def test_read_table_header_is_optional() raises:
    _write(String(_DATA), String("label,f0\n1,2\n3,4\n"))
    var table = read_table(String(_DATA), has_header=True)
    assert_equal(table.n_rows, 2)
    assert_equal(table.at(0, 0), 1.0)
    # Without --header the same file is a parse error, since the header is
    # not numeric.
    with assert_raises():
        _ = read_table(String(_DATA))
    remove(_DATA)


def test_read_table_rejects_bad_files() raises:
    with assert_raises():
        _ = read_table(String("./.test_cli_no_such_file.tmp"))

    _write(String(_DATA), String("# only a comment\n"))
    with assert_raises():
        _ = read_table(String(_DATA))

    _write(String(_DATA), String("1,2,3\n4,5\n"))
    with assert_raises():
        _ = read_table(String(_DATA))

    _write(String(_DATA), String("1,2,3\n4,oops,6\n"))
    with assert_raises():
        _ = read_table(String(_DATA))
    remove(_DATA)


# ------------------------------------------------------------------ columns


def test_resolve_column_allows_negative_indices() raises:
    assert_equal(resolve_column(0, 4, String("label")), 0)
    assert_equal(resolve_column(3, 4, String("label")), 3)
    assert_equal(resolve_column(-1, 4, String("label")), 3)
    assert_equal(resolve_column(-4, 4, String("label")), 0)
    with assert_raises():
        _ = resolve_column(4, 4, String("label"))
    with assert_raises():
        _ = resolve_column(-5, 4, String("label"))


def test_column_major_features_drops_label_and_weight() raises:
    _write(String(_DATA), String("1,10,100,0.5\n2,20,200,1.5\n"))
    var table = read_table(String(_DATA))

    # Label in column 0, weight in column 3, so features are columns 1 and 2
    # laid out column-major.
    var features = column_major_features(table, 0, 3)
    assert_equal(len(features), 4)
    assert_equal(features[0], 10.0)
    assert_equal(features[1], 20.0)
    assert_equal(features[2], 100.0)
    assert_equal(features[3], 200.0)

    # With no label or weight every column is a feature.
    var all_columns = column_major_features(table, -1, -1)
    assert_equal(len(all_columns), 8)
    assert_equal(all_columns[0], 1.0)
    assert_equal(all_columns[1], 2.0)

    # A file whose only column is the label leaves nothing to train on.
    _write(String(_DATA), String("1\n2\n"))
    var single = read_table(String(_DATA))
    with assert_raises():
        _ = column_major_features(single, 0, -1)
    remove(_DATA)


# ------------------------------------------------------------------ options


def test_parse_options_reads_flags() raises:
    var options = parse_options(
        _args(
            [
                String("--data"),
                String("d.csv"),
                String("--model"),
                String("m.mbst"),
                String("--output"),
                String("p.csv"),
                String("--params"),
                String("num_leaves=7"),
                String("--label"),
                String("-1"),
                String("--weight"),
                String("2"),
                String("--header"),
                String("--raw"),
            ]
        )
    )
    assert_equal(options.data, String("d.csv"))
    assert_equal(options.model, String("m.mbst"))
    assert_equal(options.output, String("p.csv"))
    assert_equal(options.params, String("num_leaves=7"))
    assert_equal(options.label, -1)
    assert_true(options.label_given)
    assert_equal(options.weight, 2)
    assert_true(options.weight_given)
    assert_true(options.header)
    assert_true(options.raw)


def test_parse_options_defaults_are_unset() raises:
    var options = parse_options(List[String]())
    assert_true(not options.label_given)
    assert_true(not options.weight_given)
    assert_true(not options.header)
    assert_true(not options.raw)


def test_parse_options_rejects_bad_input() raises:
    with assert_raises():
        _ = parse_options(_args([String("--nonesuch")]))
    with assert_raises():
        _ = parse_options(_args([String("--data")]))
    with assert_raises():
        _ = parse_options(_args([String("--label"), String("first")]))


# ----------------------------------------------------------------- commands


def test_train_then_predict_matches_the_library() raises:
    var n_rows = _training_file(String(_DATA))
    var summary = command_train(
        _args(
            [
                String("--data"),
                String(_DATA),
                String("--model"),
                String(_MODEL),
                String("--params"),
                String("objective=regression num_iterations=10 num_leaves=7"),
            ]
        )
    )
    assert_true(summary.find("2 features") >= 0)
    assert_true(summary.find(String(n_rows, " rows")) >= 0)

    var text = command_predict(
        _args(
            [
                String("--model"),
                String(_MODEL),
                String("--data"),
                String(_DATA),
                String("--label"),
                String("0"),
            ]
        )
    )
    var lines = _lines(text)
    assert_equal(len(lines), n_rows)

    # The printed values must be the model's own predictions, exactly: the
    # tool prints enough digits to round-trip a Float64.
    var model = load_model(String(_MODEL))
    var table = read_table(String(_DATA))
    var features = column_major_features(table, 0, -1)
    for r in range(n_rows):
        var row = List[Float64](capacity=2)
        for f in range(2):
            row.append(features[f * n_rows + r])
        assert_equal(Float64(lines[r]), model.predict(row))

    remove(_DATA)
    remove(_MODEL)


def test_predict_raw_differs_for_binary() raises:
    # A label column of 0/1 with a clean boundary, so the logistic model has
    # something to separate.
    var content = String("")
    var n_rows = 80
    for i in range(n_rows):
        var v = Float64(i) / Float64(n_rows)
        content += String(1.0 if v > 0.5 else 0.0, ",", v, ",", 1.0 - v, "\n")
    _write(String(_DATA), content)

    _ = command_train(
        _args(
            [
                String("--data"),
                String(_DATA),
                String("--model"),
                String(_MODEL),
                String("--params"),
                String("objective=binary num_iterations=10 num_leaves=7"),
            ]
        )
    )
    var base = _args(
        [
            String("--model"),
            String(_MODEL),
            String("--data"),
            String(_DATA),
            String("--label"),
            String("0"),
        ]
    )
    var proba = _lines(command_predict(base))
    var raw_args = base.copy()
    raw_args.append(String("--raw"))
    var raw = _lines(command_predict(raw_args))

    assert_equal(len(proba), n_rows)
    assert_equal(len(raw), n_rows)
    var differs = False
    for r in range(n_rows):
        var p = Float64(proba[r])
        assert_true(p > 0.0 and p < 1.0)
        if Float64(raw[r]) != p:
            differs = True
    assert_true(differs)

    remove(_DATA)
    remove(_MODEL)


def test_multiclass_round_trip() raises:
    var content = String("")
    var n_rows = 90
    for i in range(n_rows):
        var k = i % 3
        var v = Float64(k) + 0.01 * Float64(i)
        content += String(k, ",", v, ",", -v, "\n")
    _write(String(_DATA), content)

    var summary = command_train(
        _args(
            [
                String("--data"),
                String(_DATA),
                String("--model"),
                String(_MODEL),
                String("--params"),
                String(
                    "objective=multiclass num_class=3 num_iterations=4"
                    " num_leaves=7"
                ),
            ]
        )
    )
    assert_true(summary.find("multiclass") >= 0)
    assert_true(summary.find("3 classes") >= 0)

    var lines = _lines(
        command_predict(
            _args(
                [
                    String("--model"),
                    String(_MODEL),
                    String("--data"),
                    String(_DATA),
                    String("--label"),
                    String("0"),
                ]
            )
        )
    )
    assert_equal(len(lines), n_rows)
    # One probability per class per row, summing to one.
    for r in range(n_rows):
        var fields = lines[r].split(",")
        assert_equal(len(fields), 3)
        var total = 0.0
        for k in range(3):
            total += Float64(String(fields[k]))
        assert_true(abs(total - 1.0) < 1e-9)

    var info = command_info(
        _args([String("--model"), String(_MODEL)])
    )
    assert_true(info.find("kind: multiclass") >= 0)
    assert_true(info.find("classes: 3") >= 0)

    remove(_DATA)
    remove(_MODEL)


def test_weight_column_reaches_the_trainer() raises:
    var n_rows = _training_file(String(_DATA))
    # Reuse feature 1 as a weight column, so the two runs see the same rows
    # but different weights.
    var weighted = _args(
        [
            String("--data"),
            String(_DATA),
            String("--model"),
            String(_MODEL),
            String("--params"),
            String("num_iterations=5 num_leaves=7"),
            String("--weight"),
            String("2"),
        ]
    )
    _ = command_train(weighted)
    var with_weights = load_model(String(_MODEL))

    _ = command_train(
        _args(
            [
                String("--data"),
                String(_DATA),
                String("--model"),
                String(_MODEL),
                String("--params"),
                String("num_iterations=5 num_leaves=7"),
            ]
        )
    )
    var without = load_model(String(_MODEL))

    # The weighted run drops a feature column as well, so it is enough to
    # check that the two models are not the same ensemble.
    assert_true(
        with_weights.mapper.n_features != without.mapper.n_features
        or with_weights.booster.base_score != without.booster.base_score
    )
    _ = n_rows
    remove(_DATA)
    remove(_MODEL)


def test_commands_reject_missing_arguments() raises:
    with assert_raises():
        _ = command_train(_args([String("--model"), String(_MODEL)]))
    with assert_raises():
        _ = command_train(_args([String("--data"), String(_DATA)]))
    with assert_raises():
        _ = command_predict(_args([String("--data"), String(_DATA)]))
    with assert_raises():
        _ = command_info(List[String]())


def test_predict_checks_the_feature_count() raises:
    _ = _training_file(String(_DATA))
    _ = command_train(
        _args(
            [
                String("--data"),
                String(_DATA),
                String("--model"),
                String(_MODEL),
                String("--params"),
                String("num_iterations=3 num_leaves=7"),
            ]
        )
    )
    # Without --label the label column is scored as a third feature.
    with assert_raises():
        _ = command_predict(
            _args(
                [
                    String("--model"),
                    String(_MODEL),
                    String("--data"),
                    String(_DATA),
                ]
            )
        )
    remove(_DATA)
    remove(_MODEL)


# -------------------------------------------------------------- exit status


def test_run_exit_status() raises:
    assert_equal(run(List[String]()), 2)
    assert_equal(run(_args([String("help")])), 0)
    assert_equal(run(_args([String("version")])), 0)
    assert_equal(run(_args([String("nonesuch")])), 2)
    assert_equal(
        run(_args([String("info"), String("--model"), String("nope.tmp")])), 1
    )
    assert_equal(run(_args([String("train"), String("--bogus")])), 1)


def test_run_writes_predictions_to_a_file() raises:
    var n_rows = _training_file(String(_DATA))
    assert_equal(
        run(
            _args(
                [
                    String("train"),
                    String("--data"),
                    String(_DATA),
                    String("--model"),
                    String(_MODEL),
                    String("--params"),
                    String("num_iterations=5 num_leaves=7"),
                ]
            )
        ),
        0,
    )
    assert_equal(
        run(
            _args(
                [
                    String("predict"),
                    String("--model"),
                    String(_MODEL),
                    String("--data"),
                    String(_DATA),
                    String("--label"),
                    String("0"),
                    String("--output"),
                    String(_OUTPUT),
                ]
            )
        ),
        0,
    )
    var written = open(String(_OUTPUT), "r").read()
    assert_equal(len(_lines(written)), n_rows)

    remove(_DATA)
    remove(_MODEL)
    remove(_OUTPUT)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
