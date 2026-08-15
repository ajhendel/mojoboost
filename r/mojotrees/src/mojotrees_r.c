/*
 * R glue for the mojotrees C ABI.
 *
 * Everything here talks to the public interface in capi/mojotrees.h and
 * nothing else. No Mojo type, layout, or symbol is referenced, so this file
 * only has to change when the C ABI does.
 *
 * Three things it is responsible for that the ABI is not:
 *
 * - Lifetime. A model lives behind an R external pointer with a registered
 *   finalizer, so the garbage collector frees it. The finalizer clears the
 *   pointer's address first, which makes a double free impossible even if
 *   the user also calls mb.free() explicitly.
 * - Errors. The ABI reports failures through an explicit error object
 *   rather than a global last-error, so every call here creates one, and
 *   every failure becomes an R condition through Rf_error carrying its
 *   message. Callers get tryCatch()-able errors rather than status codes.
 * - The error object's own lifetime. Rf_error is a long jump, so it must
 *   never run while an error object is still owned by this frame. Each
 *   entry point copies the message out, frees the object, and only then
 *   raises. mb_fail() below is the single place that sequence lives.
 *
 * R stores matrices column-major, which is exactly the layout
 * mojotrees_train_dense and mojotrees_predict expect, so no transpose is
 * ever done.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mojotrees.h"

/* --- error plumbing --------------------------------------------------- */

/* Longest error message copied out of the ABI before raising. Messages are
 * explanatory sentences, not data, so this is far above what any of them
 * runs to; a longer one is truncated rather than allowed to overflow. */
#define MB_MESSAGE_MAX 1024

/* Free `error` and raise its message as an R condition. Never returns.
 *
 * The copy matters. Rf_error long jumps out of this frame, so freeing after
 * raising would never happen and freeing before raising would leave the
 * message pointer dangling, since the ABI documents it as owned by the
 * error object. Copy, free, then raise. */
static void mb_fail(MojoTreesError *error) {
  char buffer[MB_MESSAGE_MAX];
  const char *message = mojotrees_error_message(error);

  if (message == NULL || message[0] == '\0') {
    mojotrees_error_free(error);
    Rf_error("mojotrees failed without reporting a reason");
  }
  strncpy(buffer, message, sizeof(buffer) - 1);
  buffer[sizeof(buffer) - 1] = '\0';
  mojotrees_error_free(error);
  Rf_error("%s", buffer);
}

/* An error object, or an R condition if one cannot be allocated. */
static MojoTreesError *mb_new_error(void) {
  MojoTreesError *error = mojotrees_error_create();
  if (error == NULL) {
    Rf_error("could not allocate a mojotrees error object");
  }
  return error;
}

/* --- handle plumbing -------------------------------------------------- */

static SEXP mb_booster_tag = NULL;

static void mb_finalize_booster(SEXP ext) {
  MojoTreesModel *handle;
  if (TYPEOF(ext) != EXTPTRSXP) {
    return;
  }
  handle = (MojoTreesModel *)R_ExternalPtrAddr(ext);
  if (handle == NULL) {
    return;
  }
  /* Clear first, then free: if this runs twice (an explicit mb.free() plus
   * the garbage collector, say) the second pass sees NULL and does nothing. */
  R_ClearExternalPtr(ext);
  mojotrees_model_free(handle);
}

static SEXP mb_wrap_handle(MojoTreesModel *handle) {
  SEXP ext = PROTECT(R_MakeExternalPtr(handle, mb_booster_tag, R_NilValue));
  R_RegisterCFinalizerEx(ext, mb_finalize_booster, TRUE);
  UNPROTECT(1);
  return ext;
}

static MojoTreesModel *mb_unwrap_handle(SEXP ext) {
  MojoTreesModel *handle;
  if (TYPEOF(ext) != EXTPTRSXP || R_ExternalPtrTag(ext) != mb_booster_tag) {
    Rf_error("not a mojotrees model handle");
  }
  handle = (MojoTreesModel *)R_ExternalPtrAddr(ext);
  if (handle == NULL) {
    Rf_error(
        "this mojotrees model handle is no longer valid. Model handles do "
        "not survive saveRDS(), save.image(), or a session restart; write "
        "the model with mb.save() and read it back with mb.load().");
  }
  return handle;
}

static const double *mb_optional_reals(SEXP x) {
  if (x == R_NilValue || Rf_length(x) == 0) {
    return NULL;
  }
  return REAL(x);
}

/* Read one int64 accessor, raising on failure. */
static int64_t mb_query(MojoTreesModel *handle,
                        int32_t (*call)(const MojoTreesModel *, int64_t *,
                                        MojoTreesError *)) {
  MojoTreesError *error = mb_new_error();
  int64_t value = 0;
  if (call(handle, &value, error) != MOJOTREES_OK) {
    mb_fail(error);
  }
  mojotrees_error_free(error);
  return value;
}

/* --- entry points ----------------------------------------------------- */

SEXP R_mb_train(SEXP data, SEXP label, SEXP weight, SEXP parameters) {
  MojoTreesModel *handle = NULL;
  MojoTreesError *error;
  SEXP dim;
  int nrow, ncol;

  dim = Rf_getAttrib(data, R_DimSymbol);
  if (TYPEOF(data) != REALSXP || Rf_length(dim) != 2) {
    Rf_error("data must be a numeric matrix");
  }
  nrow = INTEGER(dim)[0];
  ncol = INTEGER(dim)[1];
  if (TYPEOF(label) != REALSXP || Rf_length(label) != nrow) {
    Rf_error("label must be a numeric vector with one value per row (%d)",
             nrow);
  }
  if (weight != R_NilValue && Rf_length(weight) != nrow) {
    Rf_error("weight must be a numeric vector with one value per row (%d)",
             nrow);
  }
  if (TYPEOF(parameters) != STRSXP || Rf_length(parameters) != 1) {
    Rf_error("parameters must be a single string");
  }

  error = mb_new_error();
  if (mojotrees_train_dense(REAL(data), (int64_t)nrow, (int64_t)ncol,
                            REAL(label), mb_optional_reals(weight),
                            CHAR(STRING_ELT(parameters, 0)), &handle,
                            error) != MOJOTREES_OK) {
    mb_fail(error);
  }
  mojotrees_error_free(error);
  return mb_wrap_handle(handle);
}

SEXP R_mb_predict(SEXP model, SEXP data, SEXP predict_type) {
  MojoTreesModel *handle = mb_unwrap_handle(model);
  MojoTreesError *error;
  SEXP dim, result;
  int nrow, ncol, flags;
  int64_t classes;

  dim = Rf_getAttrib(data, R_DimSymbol);
  if (TYPEOF(data) != REALSXP || Rf_length(dim) != 2) {
    Rf_error("newdata must be a numeric matrix");
  }
  nrow = INTEGER(dim)[0];
  ncol = INTEGER(dim)[1];
  if (TYPEOF(predict_type) != INTSXP || Rf_length(predict_type) != 1) {
    Rf_error("predict_type must be a single integer");
  }
  flags = INTEGER(predict_type)[0];
  if (flags != MOJOTREES_PREDICT_RESPONSE && flags != MOJOTREES_PREDICT_RAW) {
    Rf_error("predict_type must be %d (response) or %d (raw)",
             MOJOTREES_PREDICT_RESPONSE, MOJOTREES_PREDICT_RAW);
  }
  classes = mb_query(handle, mojotrees_model_num_classes);

  /* The ABI writes row-major (row * num_class + class). An R matrix is
   * column-major, so a classes x nrow matrix has exactly that memory
   * layout and no transpose is needed here; the R side transposes for
   * presentation. For the common single-column case the two layouts
   * coincide and the R side just drops the dimensions. */
  result = PROTECT(Rf_allocMatrix(REALSXP, (int)classes, nrow));
  error = mb_new_error();
  if (mojotrees_predict_ex(handle, REAL(data), (int64_t)nrow, (int64_t)ncol,
                           0, 0, (int32_t)flags, MOJOTREES_DEVICE_CPU,
                           REAL(result), (int64_t)nrow * classes,
                           error) != MOJOTREES_OK) {
    UNPROTECT(1);
    mb_fail(error);
  }
  mojotrees_error_free(error);
  UNPROTECT(1);
  return result;
}

SEXP R_mb_save(SEXP model, SEXP filename) {
  MojoTreesModel *handle = mb_unwrap_handle(model);
  MojoTreesError *error;
  if (TYPEOF(filename) != STRSXP || Rf_length(filename) != 1) {
    Rf_error("filename must be a single string");
  }
  error = mb_new_error();
  if (mojotrees_save_model(handle, CHAR(STRING_ELT(filename, 0)), error) !=
      MOJOTREES_OK) {
    mb_fail(error);
  }
  mojotrees_error_free(error);
  return R_NilValue;
}

SEXP R_mb_load(SEXP filename) {
  MojoTreesModel *handle = NULL;
  MojoTreesError *error;
  if (TYPEOF(filename) != STRSXP || Rf_length(filename) != 1) {
    Rf_error("filename must be a single string");
  }
  error = mb_new_error();
  if (mojotrees_load_model(CHAR(STRING_ELT(filename, 0)), &handle, error) !=
      MOJOTREES_OK) {
    mb_fail(error);
  }
  mojotrees_error_free(error);
  return mb_wrap_handle(handle);
}

SEXP R_mb_free(SEXP model) {
  /* Explicit release. Deliberately tolerant of an already-released handle,
   * because the finalizer may have run first. */
  if (TYPEOF(model) == EXTPTRSXP && R_ExternalPtrAddr(model) != NULL) {
    mb_finalize_booster(model);
  }
  return R_NilValue;
}

SEXP R_mb_is_valid(SEXP model) {
  int valid = (TYPEOF(model) == EXTPTRSXP &&
               R_ExternalPtrTag(model) == mb_booster_tag &&
               R_ExternalPtrAddr(model) != NULL);
  return Rf_ScalarLogical(valid);
}

/* One shape for the four scalar queries, chosen by `what`, so each does not
 * need its own near-identical entry point. */
SEXP R_mb_info(SEXP model, SEXP what) {
  MojoTreesModel *handle = mb_unwrap_handle(model);
  const char *which;
  int64_t value;

  if (TYPEOF(what) != STRSXP || Rf_length(what) != 1) {
    Rf_error("what must be a single string");
  }
  which = CHAR(STRING_ELT(what, 0));
  if (strcmp(which, "num_class") == 0) {
    value = mb_query(handle, mojotrees_model_num_classes);
  } else if (strcmp(which, "num_feature") == 0) {
    value = mb_query(handle, mojotrees_model_num_features);
  } else if (strcmp(which, "num_iteration") == 0) {
    value = mb_query(handle, mojotrees_model_num_iterations);
  } else if (strcmp(which, "num_trees") == 0) {
    value = mb_query(handle, mojotrees_model_num_trees);
  } else {
    Rf_error("unknown model property '%s'", which);
    return R_NilValue; /* not reached; quiets a maybe-uninitialized warning */
  }
  return Rf_ScalarInteger((int)value);
}

SEXP R_mb_importance(SEXP model, SEXP importance_type) {
  MojoTreesModel *handle = mb_unwrap_handle(model);
  MojoTreesError *error;
  SEXP result;
  int64_t features;
  int kind;

  if (TYPEOF(importance_type) != INTSXP || Rf_length(importance_type) != 1) {
    Rf_error("importance_type must be a single integer");
  }
  kind = INTEGER(importance_type)[0];
  if (kind != MOJOTREES_IMPORTANCE_SPLIT && kind != MOJOTREES_IMPORTANCE_GAIN) {
    Rf_error("importance_type must be %d (split) or %d (gain)",
             MOJOTREES_IMPORTANCE_SPLIT, MOJOTREES_IMPORTANCE_GAIN);
  }
  features = mb_query(handle, mojotrees_model_num_features);

  result = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)features));
  error = mb_new_error();
  if (mojotrees_feature_importance(handle, (int32_t)kind, REAL(result),
                                   features, error) != MOJOTREES_OK) {
    UNPROTECT(1);
    mb_fail(error);
  }
  mojotrees_error_free(error);
  UNPROTECT(1);
  return result;
}

SEXP R_mb_version(void) {
  char text[64];
  int32_t major = 0, minor = 0, patch = 0;
  mojotrees_library_version(&major, &minor, &patch);
  snprintf(text, sizeof(text), "%d.%d.%d", (int)major, (int)minor,
           (int)patch);
  return Rf_mkString(text);
}

SEXP R_mb_parameter_keys(void) {
  MojoTreesError *error = mb_new_error();
  char *keys = NULL;
  SEXP result;

  if (mojotrees_parameter_keys(&keys, error) != MOJOTREES_OK) {
    mb_fail(error);
  }
  mojotrees_error_free(error);
  result = PROTECT(Rf_mkString(keys));
  mojotrees_string_free(keys);
  UNPROTECT(1);
  return result;
}

SEXP R_mb_gpu_available(void) {
  return Rf_ScalarLogical(mojotrees_gpu_available() != 0);
}

/* The ABI version this package was compiled against, so a mismatch with a
 * library loaded at runtime can be reported rather than crashed on. */
SEXP R_mb_abi_version(void) {
  return Rf_ScalarInteger((int)mojotrees_abi_version());
}

/* --- registration ----------------------------------------------------- */

static const R_CallMethodDef call_methods[] = {
    {"R_mb_train", (DL_FUNC)&R_mb_train, 4},
    {"R_mb_predict", (DL_FUNC)&R_mb_predict, 3},
    {"R_mb_save", (DL_FUNC)&R_mb_save, 2},
    {"R_mb_load", (DL_FUNC)&R_mb_load, 1},
    {"R_mb_free", (DL_FUNC)&R_mb_free, 1},
    {"R_mb_is_valid", (DL_FUNC)&R_mb_is_valid, 1},
    {"R_mb_info", (DL_FUNC)&R_mb_info, 2},
    {"R_mb_importance", (DL_FUNC)&R_mb_importance, 2},
    {"R_mb_version", (DL_FUNC)&R_mb_version, 0},
    {"R_mb_parameter_keys", (DL_FUNC)&R_mb_parameter_keys, 0},
    {"R_mb_gpu_available", (DL_FUNC)&R_mb_gpu_available, 0},
    {"R_mb_abi_version", (DL_FUNC)&R_mb_abi_version, 0},
    {NULL, NULL, 0}};

void R_init_mojotrees(DllInfo *dll) {
  mb_booster_tag = Rf_install("mojotrees.Booster");
  R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
