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
 * Nothing here reimplements mojoboost. Every call forwards to the same Mojo
 * implementation the Python package and the command line tool use, so the
 * trainer, the tree walk, the file format, and the device policy are shared
 * rather than mirrored.
 *
 * Build: capi/build.sh, which produces capi/libmojoboost.{dylib,so}.
 * See capi/README.md for the parameter string and worked examples,
 * docs/C_API.md for the reference, and packaging/native/ for how the
 * header and library are laid out in a release.
 *
 * Copyright the mojoboost authors. Apache-2.0.
 */

#ifndef MOJOBOOST_H
#define MOJOBOOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of this ABI, incremented whenever these declarations change.
 * Check it with mojoboost_abi_version() when loading the library
 * dynamically.
 *
 * Versions are cumulative: every version adds declarations and removes or
 * changes none, so a caller built against version N works unchanged against
 * any library reporting >= N. Test for the version that introduced the
 * newest symbol you call, not for equality. A breaking change, if one ever
 * becomes necessary, would come with a renamed library rather than a
 * silently incompatible version number.
 *
 *   1  train, predict, save, load, accessors, errors.
 *   2  mojoboost_predict_ex (iteration ranges and device selection),
 *      mojoboost_model_num_iterations, mojoboost_gpu_available,
 *      mojoboost_model_dump_json, mojoboost_string_free.
 */
#define MOJOBOOST_ABI_VERSION 2

/* Status codes. Success is 0; every failure is negative. */
#define MOJOBOOST_OK 0
#define MOJOBOOST_ERROR_INVALID_ARGUMENT (-1)
#define MOJOBOOST_ERROR_TRAINING (-2)
#define MOJOBOOST_ERROR_IO (-3)
#define MOJOBOOST_ERROR_UNSUPPORTED (-4)

/* Where a call runs. The same three values the Mojo API and the `device`
 * parameter use, so "cpu", "gpu", and "auto" mean one thing across every
 * front end. CPU is the dependable path. GPU fails with
 * MOJOBOOST_ERROR_UNSUPPORTED, carrying the reason, when no accelerator is
 * present or the workload is outside what the accelerated path covers; it
 * never silently falls back. AUTO picks the accelerator only when it is
 * available, covers the workload, and evidence selects it, and otherwise
 * runs on the CPU. Since ABI version 2. */
#define MOJOBOOST_DEVICE_CPU 0
#define MOJOBOOST_DEVICE_GPU 1
#define MOJOBOOST_DEVICE_AUTO 2

/* Flags for mojoboost_predict_ex. Undefined bits are rejected rather than
 * ignored, so a flag a newer mojoboost understands cannot be silently
 * dropped by an older one. Since ABI version 2. */
#define MOJOBOOST_PREDICT_RESPONSE 0
#define MOJOBOOST_PREDICT_RAW 1

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

/* 1 when MOJOBOOST_DEVICE_GPU can be honored by this build on this machine,
 * 0 otherwise. Accelerator support is a property of the build as well as of
 * the hardware, so a library built without one returns 0 on a machine that
 * has one.
 *
 * A 1 means a GPU request is worth making, not that every GPU request
 * succeeds: whether a particular workload is covered is decided per call,
 * and an uncovered one is MOJOBOOST_ERROR_UNSUPPORTED with the reason in
 * the error object. Since ABI version 2. */
int32_t mojoboost_gpu_available(void);

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

/* Predict with the whole prediction surface exposed. The two calls above
 * are this one with the ensemble, the CPU, and one flag fixed, so they keep
 * behaving exactly as they always have and everything new is opt-in.
 *
 *   start_iteration  first boosting iteration to score with. Clamped to the
 *                    ensemble; negative means 0.
 *   num_iteration    how many iterations from there, or <= 0 for all the
 *                    remaining ones. This is LightGBM's convention.
 *   flags            MOJOBOOST_PREDICT_RESPONSE or MOJOBOOST_PREDICT_RAW.
 *                    Any other value is MOJOBOOST_ERROR_INVALID_ARGUMENT.
 *   device           a MOJOBOOST_DEVICE_* value.
 *
 * The base score belongs to iteration 0, so a range starting at 0 includes
 * it and a later range does not, and [0, k) and [k, n) sum to the full raw
 * score. For a multiclass model an iteration is one tree per class, and the
 * softmax is taken over the sliced scores, so response-scale output from a
 * range is the truncated model's probabilities rather than a slice of the
 * full model's. Since ABI version 2. */
int32_t mojoboost_predict_ex(const MojoBoostModel *model, const double *data,
                             int64_t n_rows, int64_t n_features,
                             int64_t start_iteration, int64_t num_iteration,
                             int32_t flags, int32_t device,
                             double *out_values, int64_t out_len,
                             MojoBoostError *error);

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

/* Number of boosting iterations, which is the tree count for a
 * single-output model and trees / num_class for a multiclass one. This, not
 * the tree count, is the unit mojoboost_predict_ex slices in. Since ABI
 * version 2. */
int32_t mojoboost_model_num_iterations(const MojoBoostModel *model,
                                       int64_t *out_value,
                                       MojoBoostError *error);

/* --------------------------------------------------------------- inspect */

/* The model as JSON in the mojoboost inspection schema: the ensemble's
 * trees, split thresholds, leaf values, and the bin mapper's view of each
 * feature. See docs/MODEL_INSPECTION_SCHEMA.md for the schema, which is
 * versioned independently of this ABI.
 *
 * On success *out_text is a NUL terminated UTF-8 string the caller owns and
 * must release with mojoboost_string_free. On failure *out_text is
 * untouched. A model carries no feature names, so the dump names features
 * Column_0, Column_1, ... as LightGBM does. Since ABI version 2. */
int32_t mojoboost_model_dump_json(const MojoBoostModel *model,
                                  char **out_text, MojoBoostError *error);

/* --------------------------------------------------------------- destroy */

/* Free a string the library allocated, such as one from
 * mojoboost_model_dump_json. NULL is accepted and does nothing. Do not pass
 * it a mojoboost_error_message pointer: that one is owned by the error
 * object. Do not release it with free() either, since the library may not
 * share the caller's allocator. Since ABI version 2. */
void mojoboost_string_free(char *text);

/* Free a model handle. NULL is accepted and does nothing. Using a handle
 * after freeing it, or freeing it twice, is undefined behavior, as it is
 * for free(). */
void mojoboost_model_free(MojoBoostModel *model);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOJOBOOST_H */
