/* mojoboost C ABI.
 *
 * A small, stable C interface to mojoboost: train a model from a dense
 * matrix, predict, save, load, read back an error, and free. It is the
 * intended base for bindings in languages that speak C (R, Julia, Go, and
 * so on), so it is deliberately narrower than the Mojo API.
 *
 * Design rules, which callers may rely on:
 *
 * 1. Only C scalar types, C strings, and opaque handles cross this
 *    boundary. No mojoboost struct layout is exposed, so a mojoboost
 *    release can change any internal type without breaking compiled
 *    callers.
 * 2. Hyperparameters are passed as a LightGBM style parameter string, not
 *    as a struct, so adding a hyperparameter never changes a signature or
 *    a struct layout.
 * 3. Every fallible function returns int32_t: MOJOBOOST_OK (0) or a
 *    negative MOJOBOOST_ERROR_* code. The human readable reason goes into
 *    the caller's error object.
 * 4. The library never takes ownership of caller memory and never retains
 *    a pointer past the call it was passed to. Everything it needs is
 *    copied.
 * 5. Every handle the library returns is freed by exactly one
 *    mojoboost_*_free call. Those free functions accept NULL.
 * 6. Pointer arguments are checked. A NULL where a value is required is
 *    MOJOBOOST_ERROR_INVALID_ARGUMENT, never a crash.
 *
 * Thread safety: no global state is involved, so calls on distinct handles
 * may run concurrently. A single MojoBoostError, or a single model being
 * freed, must not be used from two threads at once. Model handles are
 * immutable once trained or loaded, so concurrent prediction on one model
 * is safe as long as each thread passes its own error object and its own
 * output buffer.
 *
 * Differences from LightGBM's C API, on purpose:
 *
 * - Errors are retrieved from an explicit error object passed to the call,
 *   not from a thread local LGBM_GetLastError(). Explicit ownership is
 *   what makes concurrent use well defined here.
 * - Status codes distinguish invalid arguments, training failures, I/O
 *   failures, and unsupported requests; LightGBM returns -1 for all.
 * - There is no separate dataset handle. Training takes the matrix
 *   directly, which is all a dense trainer needs.
 *
 * Build: capi/build.sh, which produces capi/libmojoboost.{dylib,so}.
 * See capi/README.md for the parameter string and worked examples.
 *
 * Copyright the mojoboost authors. Apache-2.0.
 */

#ifndef MOJOBOOST_H
#define MOJOBOOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of this ABI. Incremented only for a breaking change to the
 * declarations in this header. Check it with mojoboost_abi_version() when
 * loading the library dynamically. */
#define MOJOBOOST_ABI_VERSION 1

/* Status codes. Success is 0; every failure is negative. */
#define MOJOBOOST_OK 0
#define MOJOBOOST_ERROR_INVALID_ARGUMENT (-1)
#define MOJOBOOST_ERROR_TRAINING (-2)
#define MOJOBOOST_ERROR_IO (-3)
#define MOJOBOOST_ERROR_UNSUPPORTED (-4)

/* A trained or loaded model. Opaque: the layout is a mojoboost internal. */
typedef struct MojoBoostModel MojoBoostModel;

/* Receives the message for the most recent failed call it was passed to. */
typedef struct MojoBoostError MojoBoostError;

/* ---------------------------------------------------------------- version */

/* The ABI version this library was built with. */
int32_t mojoboost_abi_version(void);

/* The library version. Any of the three out pointers may be NULL. */
void mojoboost_library_version(int32_t *major, int32_t *minor,
                               int32_t *patch);

/* ------------------------------------------------------------------ error */

/* A new, empty error object. Returns NULL only if allocation fails. */
MojoBoostError *mojoboost_error_create(void);

/* Free an error object. NULL is accepted and does nothing. */
void mojoboost_error_free(MojoBoostError *error);

/* The message from the most recent failed call that was passed `error`,
 * as a NUL terminated UTF-8 string, or "" when no such call has failed.
 * Returns NULL if `error` is NULL.
 *
 * The pointer is owned by the error object. It stays valid until the next
 * call that is passed the same error object or until it is freed, so copy
 * the message if it must outlive that. Every call that takes an error
 * object clears it on entry, including calls that go on to succeed. */
const char *mojoboost_error_message(const MojoBoostError *error);

/* ----------------------------------------------------------------- train */

/* Train a model on a dense matrix.
 *
 *   data        column-major, n_rows * n_features doubles, so feature f of
 *               row r is data[f * n_rows + r]. NaN means missing.
 *   labels      n_rows doubles. Regression target, {0, 1} for binary,
 *               nonnegative counts for poisson, or class indices in
 *               0..num_class-1 for multiclass.
 *   weights     NULL for unweighted, or n_rows nonnegative doubles.
 *   parameters  NULL or "" for the defaults, or a whitespace separated
 *               LightGBM style "key=value" string; see capi/README.md.
 *   out_model   receives the new model handle on success, and is left
 *               untouched on failure, so a failed call leaks nothing and
 *               cannot orphan a handle.
 *   error       NULL to discard the message, else an error object.
 *
 * The caller keeps ownership of every buffer; none is retained after the
 * call returns. Free the model with mojoboost_model_free. */
int32_t mojoboost_train_dense(const double *data, int64_t n_rows,
                              int64_t n_features, const double *labels,
                              const double *weights, const char *parameters,
                              MojoBoostModel **out_model,
                              MojoBoostError *error);

/* --------------------------------------------------------------- predict */

/* Response-scale predictions: probability for binary, expected count for
 * poisson, class probabilities for multiclass, and the raw score for the
 * regression objectives.
 *
 * n_features must equal mojoboost_model_num_features(model). Values are
 * written row-major as out_values[r * k + c] where k is
 * mojoboost_model_num_classes(model), so out_len must be at least
 * n_rows * k. Nothing is written unless the whole call succeeds. */
int32_t mojoboost_predict(const MojoBoostModel *model, const double *data,
                          int64_t n_rows, int64_t n_features,
                          double *out_values, int64_t out_len,
                          MojoBoostError *error);

/* Raw scores, before the link function: log-odds for binary, the log mean
 * for poisson, per-class scores before the softmax for multiclass. Shape
 * and validation are exactly as for mojoboost_predict. */
int32_t mojoboost_predict_raw(const MojoBoostModel *model,
                              const double *data, int64_t n_rows,
                              int64_t n_features, double *out_values,
                              int64_t out_len, MojoBoostError *error);

/* ----------------------------------------------------------- save / load */

/* Write the model to `path` in the versioned mojoboost text format. */
int32_t mojoboost_save_model(const MojoBoostModel *model, const char *path,
                             MojoBoostError *error);

/* Load a model written by mojoboost_save_model, or by the Mojo or Python
 * save functions. Single-output and multiclass files are both accepted;
 * the file says which it is. `out_model` is untouched on failure. */
int32_t mojoboost_load_model(const char *path, MojoBoostModel **out_model,
                             MojoBoostError *error);

/* ------------------------------------------------------------- accessors */

/* Number of features the model expects in a prediction matrix. */
int32_t mojoboost_model_num_features(const MojoBoostModel *model,
                                     int64_t *out_value,
                                     MojoBoostError *error);

/* Number of values the model predicts per row: the class count for a
 * multiclass model, 1 for every single-output model. */
int32_t mojoboost_model_num_classes(const MojoBoostModel *model,
                                    int64_t *out_value,
                                    MojoBoostError *error);

/* Number of trees in the ensemble. A multiclass model grows one tree per
 * class per iteration, so this is iterations * num_class there. */
int32_t mojoboost_model_num_trees(const MojoBoostModel *model,
                                  int64_t *out_value,
                                  MojoBoostError *error);

/* --------------------------------------------------------------- destroy */

/* Free a model handle. NULL is accepted and does nothing. Using a handle
 * after freeing it, or freeing it twice, is undefined behavior, as it is
 * for free(). */
void mojoboost_model_free(MojoBoostModel *model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOJOBOOST_H */
