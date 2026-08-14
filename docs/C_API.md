# C API

A small, stable C interface to mojoboost: train on a dense matrix, predict,
inspect, save, load, read an error back, and free. It is the intended base
for bindings in any language that speaks C, which is why it is deliberately
narrower than the Mojo API.

- `capi/mojoboost.h` is the contract, and documents every function.
- `capi/README.md` is the practical guide: parameter strings, worked
  examples, linking.
- `packaging/native/` is how the header and library are laid out in a
  release.

This document is the reference for the parts a binding author has to get
right: what the ABI guarantees, how versions move, who owns what, and what
each call actually reaches inside mojoboost.

## The ABI reimplements nothing

Every call forwards to the same Mojo implementation the Python package and
the command line tool use:

| C entry point | Implementation |
|---|---|
| `mojoboost_train_dense` | `fit` / `fit_multiclass` (`model.mojo`) |
| `mojoboost_predict`, `mojoboost_predict_raw`, `mojoboost_predict_ex` | `Model.predict_batch` / `MulticlassModel.predict_batch` (`model.mojo`) |
| `mojoboost_save_model`, `mojoboost_load_model` | `serialize.mojo` |
| `mojoboost_model_dump_json` | `dump_model` / `dump_multiclass_model` (`inspection.mojo`) |
| `mojoboost_feature_importance` | `split_importance` / `gain_importance` (`importance.mojo`) |
| `mojoboost_parameter_keys` | `SUPPORTED_KEYS` (`params.mojo`) |
| the `device` argument | `resolve_device` (`device.mojo`, over `device_policy.mojo`) |
| `parameters` | `parse_params` (`params.mojo`) |
| `mojoboost_gpu_available` | `gpu_available` (`device.mojo`) |

The ABI translates C arguments into those calls and their failures into
status codes. It does not bin, walk trees, apply a link function, or decide
where anything runs. A change to how mojoboost does any of that reaches C
callers without anyone editing `capi/`.

The one place this is load-bearing rather than tidy: prediction goes through
`predict_batch`, which bins the whole matrix once and dispatches to the
device, rather than looping over the per-row `predict`. That is what lets a
C caller reach the accelerator and the iteration range at all, and it is why
those two arrived without a new prediction implementation.

## Versioning

`MOJOBOOST_ABI_VERSION` is incremented whenever the header gains
declarations. Versions are **cumulative**: each adds and none removes or
changes.

| Version | Adds |
|---|---|
| 1 | train, predict, save, load, accessors, errors |
| 2 | `mojoboost_predict_ex`, `mojoboost_model_num_iterations`, `mojoboost_gpu_available`, `mojoboost_model_dump_json`, `mojoboost_string_free`, `MOJOBOOST_DEVICE_*`, `MOJOBOOST_PREDICT_*` |
| 3 | `mojoboost_feature_importance`, `mojoboost_parameter_keys`, `MOJOBOOST_IMPORTANCE_*` |

A caller built against version N works unchanged against any library
reporting at least N. Test accordingly:

```c
if (mojoboost_abi_version() < 2) { /* no predict_ex here */ }
```

Never test for equality. A change that genuinely broke a compiled caller
would ship under a different library name rather than as a version number
that silently means something else.

The library version, from `mojoboost_library_version`, moves independently
and says nothing about the ABI.

Two things are versioned separately and must not be inferred from
`MOJOBOOST_ABI_VERSION`:

- The **model file format**, which `serialize.mojo` versions itself, and
  which `mojoboost_load_model` reads. A file written by one release loads in
  another under the compatibility policy in
  [COMPATIBILITY_POLICY.md](COMPATIBILITY_POLICY.md), not under this
  version.
- The **inspection schema** that `mojoboost_model_dump_json` returns, which
  is versioned inside the JSON. See
  [MODEL_INSPECTION_SCHEMA.md](MODEL_INSPECTION_SCHEMA.md).

## What crosses the boundary

Only C scalars, C strings, buffers the caller owns, and opaque handles. No
mojoboost struct layout is exposed, so a release can change any internal
type without breaking a compiled caller. Two consequences:

- Hyperparameters travel as a parameter string, not a struct, so adding a
  hyperparameter never changes a signature or a layout.
- Model handles are opaque. The accessors report what a caller needs to size
  a buffer, and `mojoboost_model_dump_json` covers everything else. There is
  no node-level tree API on purpose: the JSON schema is documented and
  versioned, while a node API would freeze the tree representation.

## Ownership

| Thing | Owner | Released by |
|---|---|---|
| `data`, `labels`, `weights`, `parameters`, `path` | caller | caller, any time after the call returns |
| `MojoBoostModel *` | library | `mojoboost_model_free` |
| `MojoBoostError *` | library | `mojoboost_error_free` |
| `mojoboost_error_message` result | the error object | nothing |
| `mojoboost_model_dump_json` result | caller | `mojoboost_string_free` |
| `mojoboost_parameter_keys` result | caller | `mojoboost_string_free` |

The library copies whatever it needs during a call and retains nothing
afterward. Every free function accepts `NULL`.

The two string results differ and the difference matters. The error message
belongs to the error object: it must not be freed, and it stops being valid
at the next call passed that object. The JSON dump is a fresh allocation the
caller must release. Neither may go to `free()` — the library may not share
the caller's allocator, which is why `mojoboost_string_free` exists at all.

Every out parameter is left untouched on failure, so a failed call cannot
orphan a handle or leave a stale pointer behind.

## Errors

Every fallible function returns `MOJOBOOST_OK` (0) or a negative code and
writes the reason into the `MojoBoostError *` passed to it. A `NULL` error
object discards the message and keeps the status code.

| Code | Meaning |
|---|---|
| `MOJOBOOST_ERROR_INVALID_ARGUMENT` | a `NULL`, a nonpositive dimension, a buffer too small, a shape that does not match the model, an undefined flag or device constant, or a parameter string that is wrong |
| `MOJOBOOST_ERROR_TRAINING` | well-formed arguments, but training failed — poisson with a negative label, for instance |
| `MOJOBOOST_ERROR_IO` | a model file could not be read or written |
| `MOJOBOOST_ERROR_UNSUPPORTED` | a real mojoboost feature asked for the wrong way: a parameter string naming something only the Mojo API exposes, or a device request the policy refused |

The separation between the last two is the useful one. `UNSUPPORTED` always
means the capability exists and the request has to change; the message says
how. `INVALID_ARGUMENT` means the request itself is wrong.

An error object is cleared at the start of every call that receives it,
including calls that then succeed, so a stale message never survives.

Nothing unwinds into C. Every exported function catches everything, so a
failure is always a status code, never a crash and never a Mojo error
crossing the boundary.

## Devices

`MOJOBOOST_DEVICE_CPU`, `_GPU`, and `_AUTO` are the same three values the
Mojo API and the `device=` parameter use, so the vocabulary is one thing
across every front end. See [DEVICE_SELECTION.md](DEVICE_SELECTION.md).

- **CPU** is the dependable path: Float64 throughout, every objective, every
  entry point.
- **GPU** fails with `MOJOBOOST_ERROR_UNSUPPORTED` and the policy's own
  reason when no accelerator is present or the workload is outside what the
  accelerated path covers. It never falls back silently, because a silent
  fallback turns a performance request into an invisible non-event.
- **AUTO** uses the accelerator only when it is available, covers the
  workload, and evidence selects it, and runs on the CPU otherwise.

`mojoboost_gpu_available()` answers whether a GPU request is worth making at
all. It is a property of the build as well as the machine: a library built
without accelerator support returns 0 on a machine that has one. A 1 is not
a promise that every workload is covered — that is decided per call.

Training reads its device from the parameter string (`device=gpu`), not from
an argument, so there is one place to look. Prediction takes it as an
argument to `mojoboost_predict_ex`, because a model is trained once and
scored in many places.

`mojoboost_predict` and `mojoboost_predict_raw` are fixed to the CPU and to
the whole ensemble. They behave exactly as they did in ABI version 1, and
reaching the accelerator is an explicit choice a caller makes rather than
something a rebuild changes underneath it.

## Iteration ranges

`mojoboost_predict_ex` takes LightGBM's `(start_iteration, num_iteration)`
pair, clamped to the ensemble: a negative start is 0, a start past the end
gives an empty range, and `num_iteration <= 0` means every iteration from
the start on.

The unit is **boosting iterations, not trees**. For a multiclass model one
iteration is one tree per class, so the two differ by a factor of
`num_class`; `mojoboost_model_num_iterations` reports the unit ranges are
expressed in, and `mojoboost_model_num_trees` reports the other one.

The base score belongs to iteration 0, exactly as in LightGBM, where it is
folded into the first iteration's leaf values rather than stored apart. So:

- A range starting at 0 includes the base score; a later one does not.
- `[0, k)` and `[k, n)` sum to the full raw score for any `k`.
- `[0, 0)` is the base-score-only model.

For response-scale multiclass output the softmax is taken over the sliced
raw scores, so what you get is the truncated model's probabilities, not a
slice of the full model's.

## Data layout

Training and prediction matrices are column-major: feature `f` of row `r` is
`data[f * n_rows + r]`. `NaN` is a missing value.

Predictions are written row-major, `out[r * k + c]` where `k` is
`mojoboost_model_num_classes(model)`, which is 1 for every single-output
model. `out_len` must be at least `n_rows * k`. Nothing is written unless
the whole call succeeds, so a failure leaves the caller's buffer untouched.

## Thread safety

No global state is involved, so calls on distinct handles may run
concurrently. A single `MojoBoostError`, or a single model being freed, must
not be used from two threads at once. Model handles are immutable once
trained or loaded, so concurrent prediction on one model is safe as long as
each thread passes its own error object and its own output buffer.

## Notes for binding authors

- Load the library dynamically and gate on `mojoboost_abi_version()` if you
  want one binding to work across mojoboost releases. Everything in this ABI
  is reachable by symbol name; nothing requires a compiled-in struct.
- Size every prediction buffer from `mojoboost_model_num_classes`, not from
  what you believe the objective is. A single-output model reports 1 and a
  multiclass one reports its class count, and that is the only correct
  source.
- Do not parse error messages. They are for humans; branch on the status
  code.
- `r/mojoboost` is the first in-tree consumer of this ABI and the worked
  example of the paragraphs above: `r/mojoboost/src/mojoboost_r.c` sizes its
  buffers from the accessors, creates one error object per call, and copies
  the message out before raising, because R's error mechanism is a long
  jump. `r/mojoboost/configure` shows how to fail at configure time on an
  ABI older than a binding needs.
- Keep a binding's parameter list from drifting by reading
  `mojoboost_parameter_keys` rather than hardcoding one. The list is primary
  names only; aliases are accepted by `parse_params` but not reported, so a
  binding that checks membership before calling must allow for them.
- The shared library needs the Mojo runtime on its search path. In a
  development checkout that is the pixi environment's `lib` directory, and
  `capi/run_c_tests.sh` shows the `-rpath` flags. For a distributed
  artifact, read `packaging/native/README.md` first: the library
  `capi/build.sh` produces today is **not** relocatable, and the reasons are
  listed there.
