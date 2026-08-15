# mojotrees for R

R bindings for mojotrees, wrapping the stable C ABI in `capi/` rather than
binding to Mojo internals. The syntax follows the `lightgbm` R package.

Nothing here reimplements mojotrees. Every call forwards through
`capi/mojotrees.h` to the same Mojo implementation the Python package and
the command line tool use, so the trainer, the tree walk, and the file
format are shared rather than mirrored. The package needs ABI version 3 or
newer, which `configure` checks before generating a makefile.

## Build and install

The package links against the mojotrees C ABI, so build that first:

```sh
capi/build.sh                    # -> capi/libmojotrees.{dylib,so}
R CMD INSTALL r/mojotrees
```

The library is found by rpath rather than copied, so rebuilding the C ABI
takes effect without reinstalling the package.

`r/mojotrees/configure` finds the library by walking up from the package
directory. Set `MOJOTREES_HOME` to point at a checkout somewhere else.

Run the tests with:

```sh
cd r/mojotrees/tests && Rscript testthat.R
```

## Quick start

```r
library(mojotrees)

model <- mb.train(
  params = list(objective = "regression", num_leaves = 31L,
                learning_rate = 0.05),
  data = mb.Dataset(X_train, label = y_train),
  nrounds = 200L
)

pred <- predict(model, X_test)
mb.importance(model)
mb.save(model, "model.mbst")
```

`inst/examples/regression.R` and `inst/examples/classification.R` are
runnable end-to-end versions of the above.

## Coming from lightgbm

| lightgbm             | mojotrees                |
| -------------------- | ------------------------ |
| `lgb.Dataset()`      | `mb.Dataset()`           |
| `lgb.train()`        | `mb.train()`             |
| `lightgbm()`         | `mojotrees()`            |
| `predict()`          | `predict()`              |
| `lgb.save()`         | `mb.save()`              |
| `lgb.load()`         | `mb.load()`              |
| `lgb.importance()`   | `mb.importance()`        |

Parameters keep LightGBM's names and aliases, so a `params` list generally
carries over unchanged. `mb.parameter.keys()` lists the engine's primary key
names, straight from the engine, and `mb.params.string()` shows what a list
renders to. Aliases are accepted but not listed: `eta` trains the same model
as `learning_rate`, and the engine reports the canonical name.

### Deliberate differences

These are choices, not gaps.

- **An unknown parameter name is an error**, where LightGBM warns and ignores
  it. A typo must not quietly train a different model. `verbosity`, `verbose`,
  and `num_threads` are the only exceptions: `mb.params.string()` drops them
  before the engine sees them, because LightGBM users pass them reflexively
  and mojotrees has no use for them. (Thread count comes from the
  `MOJOTREES_NUM_WORKERS` environment variable.)
- **No bagging through `params`.** `bagging_fraction`, `bagging_freq`, and
  `bagging_seed` exist in the engine but are reachable only from the Mojo
  API, not from a parameter string, so passing one raises and says so.
  `feature_fraction` and its aliases do work.
- **`nrounds` and `params$num_iterations` cannot both be set.** LightGBM
  resolves the collision with a warning; this raises.
- **No `lgb.Dataset` engine handle.** `mb.Dataset()` is a plain R object that
  validates and carries `data`/`label`/`weight`. mojotrees bins inside
  training, so there is nothing to construct early and nothing to free.
- **No incremental training.** There is no `UpdateOneIter` equivalent, no
  `init_model`, and no `valids`/early stopping through this binding: the
  engine trains an ensemble in one call. Early stopping exists in the Mojo API
  (`train_with_valid`) and is not reachable from R yet.
- **`mb.importance()` has no `Cover` column.** mojotrees does not record
  per-split hessian sums.
- **Saving creates missing parent directories**, where LightGBM fails.
- **Class labels are 0-based integer codes.** The model format stores no label
  mapping, so `predict(type = "class")` returns codes and translating them
  back is yours to do. `mb.Dataset()` refuses a factor label rather than
  guessing an encoding.

### Not yet supported

- **Sparse matrices.** `dgCMatrix` and friends are rejected with a message
  rather than silently densified. The engine has sparse binning, sparse
  histograms, and sparse tree growth (`src/mojotrees/sparse.mojo`,
  `histogram_sparse.mojo`, `tree_sparse.mojo`), but no sparse boosting loop
  and no sparse `fit`, so there is nothing for a CSC path here to call. When
  that lands, the ABI gains a sparse training entry point and `mb.Dataset()`
  stops refusing sparse input.
- **`predict(type = "leaf")` and `type = "contrib")`.** The engine has both,
  including exact TreeSHAP contributions, and the Python package exposes them
  as `pred_leaf` and `pred_contrib`. What is missing is the C ABI surface:
  `mojotrees_predict_ex` returns scores only, so there is nothing for R to
  call yet. This is a gap in the binding, not in mojotrees.
- **Categorical features, monotone constraints, interaction constraints.**
  Present in the Mojo API; not yet plumbed through the C ABI's parameter
  string, because they take vectors rather than scalars.

## Model lifetime

A model is an external pointer into the engine with a registered finalizer, so
an unreferenced one is freed at the next garbage collection. It does not
survive `saveRDS()`, `save.image()`, or a session restart: use `mb.save()` and
`mb.load()`. `mb.is.valid()` reports whether a handle is still usable, and
using a dead one raises an R error naming the fix rather than crashing.
`mb.free()` releases a model immediately; calling it twice is harmless.
