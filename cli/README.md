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
cli/mojoboost info --model model.mbst
```

It is a thin layer over the Mojo API, and hyperparameters use the same
parameter string as the C ABI, documented in
[capi/README.md](../capi/README.md). Anything the tool can do is reachable
from `mojoboost.model` and `mojoboost.serialize` directly.

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

## Exit status

`0` on success, `1` when the command failed (a bad file, a shape mismatch,
an unusable parameter), and `2` for a usage error such as an unknown
command. Failures print to stderr, results to stdout.
