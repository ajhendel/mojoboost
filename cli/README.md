# mojoboost command line tool

Train a model and score data without writing any code.

```sh
pixi run build-cli      # builds cli/mojoboost
pixi run test-cli       # the Mojo tests for it
```

```sh
cli/mojoboost train --data train.csv --model model.mbst \
    --params "objective=binary num_iterations=200 num_leaves=31"
cli/mojoboost predict --model model.mbst --data test.csv --output pred.csv
cli/mojoboost info --model model.mbst --json
```

It is a thin layer over the Mojo API, and hyperparameters use the same
parameter string as the C ABI, documented in
[capi/README.md](../capi/README.md). Anything the tool can do is reachable
from `mojoboost.model` and `mojoboost.serialize` directly: prediction is
`Model.predict_batch`, `--json` is `dump_model`, `--device` is
`parse_device`, and the file format is `serialize.mojo`. Nothing about a
model is computed in this directory. See [docs/CLI.md](../docs/CLI.md) for
the reference.

This is a mojoboost-native command interface, not a LightGBM-compatible
one. There is no config-file mode and no `task=` verb: commands are
subcommands, options are flags, and the parameter string carries
hyperparameters only.

## Data format

Plain text, one example per line:

- Fields are separated by commas.
- Every field is a decimal number. An empty field, or `?`, `na`, `n/a`,
  `nan`, or `null` in any case, is a missing value, which the trainer
  handles as LightGBM's `use_missing` does.
- Blank lines and lines whose first character is `#` are ignored.
- Every line must have the same number of fields; the first line that is
  not a comment fixes the count.
- `--header` skips the first line that is not a comment, for files with
  column names.

```
# label,age,income,score
1,34,58000,0.42
0,51,,0.10
1,29,44000,?
```

Column roles:

- `--label INDEX` is the label column. `train` defaults to column 0, which
  is where LightGBM's own text format puts it. `predict` defaults to no
  label column, so scoring data needs no placeholder column; pass `--label`
  when the file still carries one so it is not scored as a feature.
- `--weight INDEX` is a per-row weight column for `train`, none by default.
- Negative indices count from the last column, so `--label -1` is the last
  one.
- Every remaining column is a feature, in file order. That order is what
  the model records, so scoring data must present the same columns in the
  same order.

## Output

`predict` writes one line per input row to `--output`, or to stdout when
that is absent. A single-output model writes one value per line; a
multiclass model writes one comma separated value per class. Values are
printed with enough digits to round-trip a 64-bit float exactly.

`--raw` switches to raw scores: log-odds for `binary`, the log mean for
`poisson`, and per-class scores before the softmax for `multiclass`. For
the regression objectives the two are the same.

## Scoring part of an ensemble

`--start-iteration N` and `--num-iteration N` restrict scoring to a slice of
the boosting iterations, which is how you see what a shorter run would have
predicted without retraining:

```sh
cli/mojoboost predict --model model.mbst --data test.csv --num-iteration 50
```

The unit is boosting iterations, not trees; for a multiclass model one
iteration is one tree per class, and `info` reports both counts. The base
score belongs to iteration 0, so a range starting there includes it and a
later range does not. `--num-iteration` at or below 0 means every iteration
from the start on, which is the default and LightGBM's convention.

## Devices

`--device cpu|gpu|auto` chooses where `predict` runs. `gpu` fails, with the
reason, when no accelerator is available or the workload is outside what the
accelerated path covers; it never falls back silently. `auto` uses the
accelerator only when it is available, covers the workload, and evidence
selects it.

Training has no `--device`, deliberately: it reads `device=` from the
parameter string, so there is exactly one place to look.

```sh
cli/mojoboost train --data train.csv --model model.mbst \
    --params "objective=binary device=gpu"
cli/mojoboost predict --model model.mbst --data test.csv --device gpu
```

## Inspecting a model

`info` prints a short summary; `info --json` prints the whole model in the
inspection schema documented in
[docs/MODEL_INSPECTION_SCHEMA.md](../docs/MODEL_INSPECTION_SCHEMA.md). It
comes from `dump_model` in `inspection.mojo`, the one implementation of that
schema, which is also what `mojoboost.inspection.dump_model` in Python
reaches when it takes its native path. Features have no names in a model
file, so the dump names them `Column_0`, `Column_1`, ... as LightGBM does;
the CLI has no flag to override that, while the Python entry point does.

## Exit status

`0` on success, `1` when the command failed (a bad file, a shape mismatch,
an unusable parameter), and `2` for a usage error such as an unknown
command. Failures print to stderr, results to stdout.
