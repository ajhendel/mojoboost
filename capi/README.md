# mojotrees C ABI

A small, stable C interface to mojotrees: train on a dense matrix, predict,
save, load, read an error back, and free. It is meant as the base for
bindings in any language that speaks C (R, Julia, Go), which is why it is
deliberately narrower than the Mojo API.

- `mojotrees.h` is the contract. Read it first; it documents every
  function.
- `mojotrees_capi.mojo` is the implementation.
- `build.sh` produces `libmojotrees.dylib` (macOS) or `libmojotrees.so`.
- `test_capi.c` is the C test; `run_c_tests.sh` builds and runs it.
- `docs/C_API.md` is the reference, including the version history.
- `packaging/native/` is how the header and library are laid out in a
  release.

The implementation reimplements nothing. Training is `fit`, prediction is
`Model.predict_batch`, inspection is `dump_model`, the device decision is
`resolve_device`, and the file format is `serialize.mojo` — the same code
the Python package and the command line tool call. A change to how
mojotrees bins, walks trees, or picks a device reaches C callers without
anyone touching this directory.

```sh
pixi run build-capi     # build the shared library
pixi run test-c         # compile and run the C tests
pixi run test-capi      # the Mojo side of the same tests
```

## What crosses the boundary

Only C scalars, C strings, buffers the caller owns, and opaque handles. No
mojotrees struct layout is exposed, so a mojotrees release can change any
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
| `MojoTreesModel *` | library | `mojotrees_model_free` |
| `MojoTreesError *` | library | `mojotrees_error_free` |
| `mojotrees_error_message` result | error object | nothing; valid until the next call passed that error object |
| `mojotrees_model_dump_json` result | caller | `mojotrees_string_free` |

The library copies whatever it needs from your buffers during the call and
retains nothing afterward. Every free function accepts `NULL`.

Note the one asymmetry: `mojotrees_error_message` hands back a pointer the
error object still owns, so it must not be freed and does not outlive the
next call on that object, whereas `mojotrees_model_dump_json` hands over a
fresh allocation the caller must release with `mojotrees_string_free`.
Never `free()` a library allocation directly: the library may not share the
caller's allocator.

## Errors

Every fallible function returns `MOJOTREES_OK` (0) or a negative code, and
writes the reason into the `MojoTreesError *` you pass. Passing `NULL`
discards the message but keeps the status code.

| Code | Meaning |
|---|---|
| `MOJOTREES_ERROR_INVALID_ARGUMENT` | a NULL, a nonpositive dimension, a buffer too small, a shape that does not match the model, or a parameter string that is wrong |
| `MOJOTREES_ERROR_TRAINING` | the arguments were well formed but training failed, for example poisson with a negative label |
| `MOJOTREES_ERROR_IO` | a model file could not be read or written |
| `MOJOTREES_ERROR_UNSUPPORTED` | the parameter string named a real mojotrees feature that only the Mojo API exposes, or a device request the policy refused |

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
| `feature_fraction_seed` | | mojotrees's default seed |
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
- `lambda_l2` defaults to 1.0, matching the rest of mojotrees rather than
  LightGBM's 0.
- `num_class` without `objective=multiclass` is an error rather than being
  ignored.
- Bagging, GOSS, monotone, interaction, and categorical settings, custom
  objectives, and ranking are reachable from the Mojo API only. Naming one
  of them returns `MOJOTREES_ERROR_UNSUPPORTED` rather than being ignored,
  so the message can say where to find it.

## Data layout

Training and prediction matrices are column-major: feature `f` of row `r`
is `data[f * n_rows + r]`. `NaN` is a missing value. Predictions are written
row-major, `out[r * k + c]` for `k = mojotrees_model_num_classes(model)`,
which is 1 for every single-output model.

## Iteration ranges and devices

`mojotrees_predict` and `mojotrees_predict_raw` score the whole ensemble on
the CPU. `mojotrees_predict_ex` is the same call with the rest of the
prediction surface exposed:

```c
/* the first 50 iterations only, on whichever device the policy picks */
mojotrees_predict_ex(model, x, n_rows, n_features,
                     /* start_iteration */ 0, /* num_iteration */ 50,
                     MOJOTREES_PREDICT_RESPONSE, MOJOTREES_DEVICE_AUTO,
                     pred, n_rows * k, err);
```

Ranges are in boosting iterations, not trees, which for a multiclass model
differ by a factor of `num_class`; `mojotrees_model_num_iterations` reports
the right unit. `num_iteration <= 0` means every iteration from the start
on, LightGBM's convention.

`MOJOTREES_DEVICE_GPU` fails with `MOJOTREES_ERROR_UNSUPPORTED` and the
policy's own reason rather than falling back to the CPU. Check
`mojotrees_gpu_available()` before asking for it, and note that a 1 there
means the request is worth making, not that every workload is covered.

## Inspecting a model

`mojotrees_model_dump_json` returns the whole model in the inspection
schema (`docs/MODEL_INSPECTION_SCHEMA.md`), which is versioned separately
from this ABI:

```c
char *json = NULL;
if (mojotrees_model_dump_json(model, &json, err) == MOJOTREES_OK) {
    puts(json);
    mojotrees_string_free(json);
}
```

That is the whole model-reading surface. There is no way to walk trees node
by node from C, on purpose: the schema is a documented, versioned format,
while a node-level API would freeze the tree representation.

## Example

```c
#include "mojotrees.h"

MojoTreesError *err = mojotrees_error_create();
MojoTreesModel *model = NULL;

if (mojotrees_train_dense(x, n_rows, n_features, y, NULL,
                          "objective=binary num_iterations=200",
                          &model, err) != MOJOTREES_OK) {
    fprintf(stderr, "train failed: %s\n", mojotrees_error_message(err));
    mojotrees_error_free(err);
    return 1;
}

int64_t k = 0;
mojotrees_model_num_classes(model, &k, err);
double *pred = malloc(sizeof(double) * n_rows * k);
mojotrees_predict(model, x, n_rows, n_features, pred, n_rows * k, err);

mojotrees_save_model(model, "model.mbst", err);
mojotrees_model_free(model);
mojotrees_error_free(err);
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

`mojotrees_save_model` creates missing parent directories, which is the
behavior of Mojo's `open`. A path that cannot be a directory at all, such
as one under `/dev/null`, is still an `MOJOTREES_ERROR_IO`.

## Stability

`MOJOTREES_ABI_VERSION` is incremented whenever `mojotrees.h` gains
declarations. Versions are cumulative: each one adds and none removes or
changes, so a caller built against version N works unchanged against any
library reporting at least N. Test for the version that introduced the
newest symbol you call, never for equality.

| Version | Adds |
|---|---|
| 1 | train, predict, save, load, accessors, errors |
| 2 | `mojotrees_predict_ex`, `mojotrees_model_num_iterations`, `mojotrees_gpu_available`, `mojotrees_model_dump_json`, `mojotrees_string_free`, the `MOJOTREES_DEVICE_*` and `MOJOTREES_PREDICT_*` constants |

`mojotrees_abi_version()` reports what the loaded library was built with,
for callers that load it dynamically. `tests/test_capi.mojo` checks that the
header and the implementation still agree on every constant. A change that
did break a compiled caller would ship as a renamed library rather than as a
version number that silently means something else.
