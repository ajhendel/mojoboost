# mojoboost C ABI

A small, stable C interface to mojoboost: train on a dense matrix, predict,
save, load, read an error back, and free. It is meant as the base for
bindings in any language that speaks C (R, Julia, Go), which is why it is
deliberately narrower than the Mojo API.

- `mojoboost.h` is the contract. Read it first; it documents every
  function.
- `mojoboost_capi.mojo` is the implementation.
- `build.sh` produces `libmojoboost.dylib` (macOS) or `libmojoboost.so`.
- `test_capi.c` is the C test; `run_c_tests.sh` builds and runs it.

```sh
pixi run build-capi     # build the shared library
pixi run test-c         # compile and run the C tests
pixi run test-capi      # the Mojo side of the same tests
```

## What crosses the boundary

Only C scalars, C strings, buffers the caller owns, and opaque handles. No
mojoboost struct layout is exposed, so a mojoboost release can change any
internal type without breaking a compiled caller. Two consequences worth
stating plainly:

- Hyperparameters travel as a parameter string, not a struct, so adding a
  hyperparameter never changes a signature or a layout.
- Model handles are opaque. There is no way to reach into a model from C;
  the accessors report what a caller actually needs to size a buffer.

## Ownership

| Thing | Owner | Freed by |
|---|---|---|
| `data`, `labels`, `weights`, `parameters`, `path` | caller | caller, any time after the call returns |
| `MojoBoostModel *` | library | `mojoboost_model_free` |
| `MojoBoostError *` | library | `mojoboost_error_free` |
| `mojoboost_error_message` result | error object | nothing; valid until the next call passed that error object |

The library copies whatever it needs from your buffers during the call and
retains nothing afterward. Both free functions accept `NULL`.

## Errors

Every fallible function returns `MOJOBOOST_OK` (0) or a negative code, and
writes the reason into the `MojoBoostError *` you pass. Passing `NULL`
discards the message but keeps the status code.

| Code | Meaning |
|---|---|
| `MOJOBOOST_ERROR_INVALID_ARGUMENT` | a NULL, a nonpositive dimension, a buffer too small, a shape that does not match the model, or a parameter string that is wrong |
| `MOJOBOOST_ERROR_TRAINING` | the arguments were well formed but training failed, for example poisson with a negative label |
| `MOJOBOOST_ERROR_IO` | a model file could not be read or written |
| `MOJOBOOST_ERROR_UNSUPPORTED` | the parameter string named a real mojoboost feature that only the Mojo API exposes |

An error object is cleared at the start of every call that receives it,
including calls that then succeed, so a stale message never survives.

## Parameter strings

Whitespace separated `key=value` pairs using LightGBM's names, which is the
shape LightGBM itself accepts in config files:

```
objective=binary num_leaves=31 learning_rate=0.05 num_iterations=200
```

Supported keys, with LightGBM's common aliases accepted for each:

| Key | Aliases | Default |
|---|---|---|
| `objective` | `application` | `regression` |
| `num_class` | `num_classes` | required for `multiclass`, invalid otherwise |
| `num_iterations` | `n_estimators`, `num_round`, `num_boost_round` | 100 |
| `learning_rate` | `eta`, `shrinkage_rate` | 0.1 |
| `num_leaves` | `num_leaf` | 31 |
| `min_data_in_leaf` | `min_data`, `min_child_samples` | 20 |
| `min_sum_hessian_in_leaf` | `min_child_weight`, `min_sum_hessian` | 1e-3 |
| `lambda_l1` | `reg_alpha` | 0 |
| `lambda_l2` | `reg_lambda`, `lambda` | 1.0 |
| `max_depth` | | -1 (unlimited) |
| `feature_fraction` | `sub_feature`, `colsample_bytree` | 1.0 |
| `feature_fraction_bynode` | `colsample_bynode` | 1.0 |
| `feature_fraction_seed` | | mojoboost's default seed |
| `max_bin` | | 255 |
| `alpha` | | 0.9 |
| `device` | `device_type` | `cpu` |
| `use_missing` | | `true` |

Objectives: `regression` (aliases `regression_l2`, `l2`, `mse`), `binary`,
`multiclass` (alias `softmax`), `poisson`, `huber`, `quantile`, and `mae`
(aliases `regression_l1`, `l1`).

Intentional differences from LightGBM:

- An unknown key is an error. LightGBM warns and ignores it, which silently
  drops typos.
- `lambda_l2` defaults to 1.0, matching the rest of mojoboost rather than
  LightGBM's 0.
- `num_class` without `objective=multiclass` is an error rather than being
  ignored.
- Bagging, GOSS, monotone, interaction, and categorical settings, custom
  objectives, and ranking are reachable from the Mojo API only. Naming one
  of them returns `MOJOBOOST_ERROR_UNSUPPORTED` rather than being ignored,
  so the message can say where to find it.

## Data layout

Training and prediction matrices are column-major: feature `f` of row `r`
is `data[f * n_rows + r]`. `NaN` is a missing value. Predictions are written
row-major, `out[r * k + c]` for `k = mojoboost_model_num_classes(model)`,
which is 1 for every single-output model.

## Example

```c
#include "mojoboost.h"

MojoBoostError *err = mojoboost_error_create();
MojoBoostModel *model = NULL;

if (mojoboost_train_dense(x, n_rows, n_features, y, NULL,
                          "objective=binary num_iterations=200",
                          &model, err) != MOJOBOOST_OK) {
    fprintf(stderr, "train failed: %s\n", mojoboost_error_message(err));
    mojoboost_error_free(err);
    return 1;
}

int64_t k = 0;
mojoboost_model_num_classes(model, &k, err);
double *pred = malloc(sizeof(double) * n_rows * k);
mojoboost_predict(model, x, n_rows, n_features, pred, n_rows * k, err);

mojoboost_save_model(model, "model.mbst", err);
mojoboost_model_free(model);
mojoboost_error_free(err);
```

Linking: the shared library links the Mojo runtime from the pixi
environment, so a caller needs that directory on its runtime search path.
`run_c_tests.sh` shows the `-rpath` flags.

## Notes for Mojo callers

Calling this ABI from Mojo works (that is what `tests/test_capi.mojo`
does), with one trap: Mojo destroys a value at its last use, and handing a
buffer's address to the ABI is not a use of the buffer afterward. Name the
buffer again after the call, `_ = buffer^`, or it can be freed while the
call is still reading it. C callers have no such problem.

## Saving

`mojoboost_save_model` creates missing parent directories, which is the
behavior of Mojo's `open`. A path that cannot be a directory at all, such
as one under `/dev/null`, is still an `MOJOBOOST_ERROR_IO`.

## Stability

`MOJOBOOST_ABI_VERSION` changes only when a declaration in `mojoboost.h`
changes in a way that breaks a compiled caller. `mojoboost_abi_version()`
reports what the loaded library was built with, for callers that load it
dynamically. `tests/test_capi.mojo` checks that the header and the
implementation still agree on every constant.
