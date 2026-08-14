# Command line tool

`mojoboost` trains a model and scores data from text files, without writing
any code. `cli/README.md` is the practical guide; this document is the
reference for the interface itself and for what each command reaches inside
mojoboost.

```sh
mojoboost train --data train.csv --model model.mbst \
    --params "objective=binary num_iterations=200"
mojoboost predict --model model.mbst --data test.csv --output pred.csv
mojoboost info --model model.mbst --json
```

## It is a mojoboost interface, not a LightGBM one

LightGBM's command line tool is driven by a config file of `key=value`
lines, with the verb itself as one of them (`task=train`), the data path as
another, and no distinction between a hyperparameter and an instruction
about where a file lives. mojoboost does not reproduce that surface, and
adding it later would be a mistake rather than a compatibility win:

- **Commands are subcommands.** `mojoboost train`, not `task=train`. The
  verb decides which options are meaningful, so it belongs where a reader
  and a shell completion can both see it first.
- **Options are flags.** Where a file is, which column is the label, where
  output goes: these are arguments, and spelling them as flags is what lets
  the tool reject an unknown one instead of ignoring it.
- **The parameter string carries hyperparameters only.** `--params` takes
  LightGBM's names and aliases, because those names are the genuinely
  valuable part of the compatibility surface, and it is the same string the
  C ABI takes. What it does *not* carry is anything about files or tasks.
- **No config file mode.** A config file that mixes the three categories
  above is what makes LightGBM's tool hard to reason about. A shell already
  has a way to reuse a long command line.

The single genuine loss is that a LightGBM config file cannot be handed to
`mojoboost` unchanged. The hyperparameters inside it can, as a `--params`
string.

## Everything is computed elsewhere

Nothing about a model is implemented in `cli/`:

| Command | Implementation |
|---|---|
| `train` | `fit` / `fit_multiclass` (`model.mojo`) |
| `predict` | `Model.predict_batch` / `MulticlassModel.predict_batch` (`model.mojo`) |
| `info --json` | `dump_model` / `dump_multiclass_model` (`inspection.mojo`) |
| `--params` | `parse_params` (`params.mojo`) |
| `--device` | `parse_device` (`device.mojo`) |
| reading and writing model files | `serialize.mojo` |

The tool parses a command line, reads a text table, and formats numbers.
That is the whole of it, which is why the CLI, the C ABI, and the Python
package cannot disagree about what a model predicts.

## Commands

### train

| Option | Meaning |
|---|---|
| `--data FILE` | training data, required |
| `--model FILE` | where to write the fitted model, required |
| `--params STRING` | `"key=value key=value"`, LightGBM parameter names |
| `--label INDEX` | label column, default 0, negative counts from the end |
| `--weight INDEX` | per-row weight column, none by default |
| `--header` | skip the first non-comment line |

Column 0 is the label by default, which is where LightGBM's own text format
puts it. Every column that is not the label or the weight is a feature, in
file order, and that order is what the model records.

There is no `--device` for `train`. Training reads `device=` from the
parameter string, so there is exactly one place to look, and the summary
line reports which device was requested.

Objectives, aliases, defaults, and the intentional differences from
LightGBM are documented in `capi/README.md`, since it is the same parameter
string. The features that are reachable only from the Mojo API — bagging,
GOSS, monotone and interaction constraints, categorical features, custom
objectives, ranking — are rejected by name with a message saying where to
find them, rather than ignored.

### predict

| Option | Meaning |
|---|---|
| `--model FILE` | saved model, required |
| `--data FILE` | data to score, required |
| `--output FILE` | where to write predictions, default stdout |
| `--label INDEX` | label column to ignore, none by default |
| `--raw` | raw scores instead of response-scale predictions |
| `--start-iteration N` | first boosting iteration to score with, default 0 |
| `--num-iteration N` | how many from there, `<= 0` for all, the default |
| `--device NAME` | `cpu`, `gpu`, or `auto`, default `cpu` |
| `--header` | skip the first non-comment line |

Scoring data usually has no label column, so `predict` drops one only when
`--label` says where it is.

One line per input row: one value for a single-output model, one comma
separated value per class for a multiclass one, printed with enough digits
to round-trip a 64-bit float exactly.

`--raw` gives log-odds for `binary`, the log mean for `poisson`, and
per-class scores before the softmax for `multiclass`. For the regression
objectives the two are the same.

### info

| Option | Meaning |
|---|---|
| `--model FILE` | saved model, required |
| `--json` | the whole model as inspection-schema JSON |

Without `--json`: the kind, the objective, the feature count, the tree
count, the iteration count, and the learning rate. With `--json`: the model
in the schema documented in
[MODEL_INSPECTION_SCHEMA.md](MODEL_INSPECTION_SCHEMA.md), from the same Mojo
implementation the Python `mojoboost.inspection.dump_model` reaches on its
native path.

A model file carries no feature names, so the dump names features
`Column_0`, `Column_1`, ... as LightGBM does. The CLI has no flag to
override that; the Python entry point does.

### version, help

`version` prints the mojoboost version. `help`, `--help`, and `-h` print
usage.

## Scoring part of an ensemble

`--start-iteration` and `--num-iteration` restrict scoring to a slice of the
boosting iterations, which is how to see what a shorter run would have
predicted without retraining:

```sh
mojoboost predict --model model.mbst --data test.csv --num-iteration 50
```

The unit is boosting iterations, not trees. For a multiclass model one
iteration is one tree per class, and `info` reports both counts. Ranges are
clamped to the ensemble, so slicing past the end is an empty range rather
than an error.

The base score belongs to iteration 0, so a range starting there includes it
and a later range does not, and `[0, k)` and `[k, n)` sum to the full raw
score. For response-scale multiclass output the softmax is taken over the
sliced scores, so the result is the truncated model's probabilities rather
than a slice of the full model's.

## Devices

`--device cpu|gpu|auto` chooses where `predict` runs, using the same
vocabulary as everything else in mojoboost; see
[DEVICE_SELECTION.md](DEVICE_SELECTION.md).

`gpu` fails with the reason when no accelerator is available or the workload
is outside what the accelerated path covers. It never falls back silently.
`auto` uses the accelerator only when it is available, covers the workload,
and evidence selects it.

Binning always runs on the host regardless of device, so both devices route
every row to the same leaf and differ only in the precision of the leaf-value
accumulation.

## Data format

Plain text, one example per line:

- Fields separated by commas, each a decimal number.
- An empty field, or `?`, `na`, `n/a`, `nan`, or `null` in any case, is a
  missing value, handled as LightGBM's `use_missing` does.
- Blank lines and lines whose first character is `#` are ignored.
- Every line must have the same number of fields; the first non-comment line
  fixes the count.
- `--header` skips the first non-comment line, for files with column names.

```
# label,age,income,score
1,34,58000,0.42
0,51,,0.10
1,29,44000,?
```

A field with two decimal points is rejected rather than silently truncated,
because a data file is exactly where that kind of typo hides.

## Exit status

| Status | Meaning |
|---|---|
| 0 | success |
| 1 | the command failed: a bad file, a shape mismatch, an unusable parameter, a refused device |
| 2 | a usage error: no command, an unknown command |

Failures print to stderr, results to stdout, so `mojoboost predict` without
`--output` composes in a pipeline.

## Building

```sh
pixi run build-cli      # builds cli/mojoboost
pixi run test-cli       # the Mojo tests for it
```

The binary needs the Mojo runtime on its search path. For a distributed
artifact, read `packaging/native/README.md` first: what `cli/build.sh`
produces today is not relocatable, and the reasons are listed there.
