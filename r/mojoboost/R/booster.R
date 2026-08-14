# The mb.Booster object.
#
# A trained model is an external pointer into the mojoboost engine plus the R
# side's own bookkeeping (the parameters it was trained with, the feature
# names, which the model file does not store). The pointer has a finalizer, so
# an unreferenced model is freed at the next garbage collection; mb.free()
# releases one immediately when that matters.

# `objective` is the objective the model was trained with, resolved to the
# engine's default when the caller did not name one. It is NULL only for a
# model from mb.load(), because the model file does not record it. The two
# cases must stay distinguishable: "trained with regression" and "trained with
# something this session never saw" call for different answers.
mb_new_booster <- function(handle, params, params_string, feature_names,
                           objective) {
  structure(
    list(
      handle = handle,
      params = params,
      params_string = params_string,
      feature_names = feature_names,
      objective = objective
    ),
    class = "mb.Booster"
  )
}

# The engine's default objective, applied when `params` does not name one.
MB_DEFAULT_OBJECTIVE <- "regression"

mb_check_booster <- function(object) {
  if (!inherits(object, "mb.Booster")) {
    stop("expected an mb.Booster", call. = FALSE)
  }
  invisible(object)
}

#' Properties of a trained model
#'
#' @param object An `mb.Booster`.
#' @return An integer.
#' @name mb.model.properties
NULL

#' @describeIn mb.model.properties Number of prediction columns: 1 for the
#'   regression objectives and for binary, `num_class` for multiclass. This is
#'   what LightGBM's `LGBM_BoosterGetNumClasses` reports, binary included.
#' @export
mb.num.class <- function(object) {
  mb_check_booster(object)
  .Call(R_mb_info, object$handle, "num_class")
}

#' @describeIn mb.model.properties Number of features the model was trained on.
#' @export
mb.num.feature <- function(object) {
  mb_check_booster(object)
  .Call(R_mb_info, object$handle, "num_feature")
}

#' @describeIn mb.model.properties Number of boosting iterations.
#' @export
mb.num.iteration <- function(object) {
  mb_check_booster(object)
  .Call(R_mb_info, object$handle, "num_iteration")
}

#' @describeIn mb.model.properties Total number of trees, i.e. iterations
#'   times trees per iteration.
#' @export
mb.num.trees <- function(object) {
  mb_check_booster(object)
  .Call(R_mb_info, object$handle, "num_trees")
}

#' Whether a model handle is still usable
#'
#' Model handles live in the engine, not in R, so they do not survive
#' `saveRDS()`, `save.image()`, or a session restart. Use [mb.save()] and
#' [mb.load()] to persist a model.
#'
#' @param object An `mb.Booster`.
#' @return `TRUE` when the model can still be used.
#' @export
mb.is.valid <- function(object) {
  if (!inherits(object, "mb.Booster")) {
    return(FALSE)
  }
  .Call(R_mb_is_valid, object$handle)
}

#' Release a model immediately
#'
#' Optional. An unreferenced model is freed at the next garbage collection
#' anyway; call this when a large model should go now rather than then.
#' Calling it twice, or on an already-collected model, is harmless.
#'
#' @param object An `mb.Booster`.
#' @return `NULL`, invisibly.
#' @export
mb.free <- function(object) {
  mb_check_booster(object)
  .Call(R_mb_free, object$handle)
  invisible(NULL)
}

#' @export
print.mb.Booster <- function(x, ...) {
  if (!mb.is.valid(x)) {
    cat("<mb.Booster: released; reload it with mb.load()>\n")
    return(invisible(x))
  }
  classes <- mb.num.class(x)
  cat(sprintf(
    "<mb.Booster: %d iterations, %d trees, %d features, %s>\n",
    mb.num.iteration(x), mb.num.trees(x), mb.num.feature(x),
    if (classes == 1L) "single output" else sprintf("%d classes", classes)
  ))
  if (nzchar(x$params_string)) {
    cat(sprintf("  params: %s\n", x$params_string))
  }
  invisible(x)
}

MB_PREDICT_NORMAL <- 0L
MB_PREDICT_RAW_SCORE <- 1L

#' Predict with a mojoboost model
#'
#' @param object An `mb.Booster` from [mb.train()] or [mb.load()].
#' @param newdata A numeric matrix or data frame with the same columns, in the
#'   same order, as the training data.
#' @param type One of:
#'   \describe{
#'     \item{`"response"`}{the response scale: probabilities for `binary` and
#'       `multiclass`, expected counts for `poisson`, the predicted value for
#'       the regression objectives. The default.}
#'     \item{`"raw"`}{the ensemble output before any link function, which is
#'       LightGBM's `rawscore`.}
#'     \item{`"class"`}{the predicted class index, `0`-based. Requires a
#'       classification objective.}
#'   }
#' @param ... Unused, present for S3 compatibility.
#' @importFrom stats predict
#' @return For a single-output model, a numeric vector with one value per row.
#'   For multiclass with `type` `"response"` or `"raw"`, a matrix with one row
#'   per observation and one column per class. For `type = "class"`, an
#'   integer vector.
#'
#'   Class indices are `0`-based codes, not the original labels: mojoboost's
#'   model format does not store a label mapping, so translating them back is
#'   the caller's job.
#' @examples
#' X <- matrix(rnorm(200), ncol = 2)
#' y <- as.numeric(X[, 1] > 0)
#' model <- mojoboost(X, y, objective = "binary", nrounds = 20L)
#' probability <- predict(model, X)
#' log_odds <- predict(model, X, type = "raw")
#' @export
predict.mb.Booster <- function(object, newdata,
                               type = c("response", "raw", "class"), ...) {
  mb_check_booster(object)
  # Named before match.arg() so that LightGBM's other prediction types get a
  # straight answer instead of "'arg' should be one of ...".
  if (length(type) == 1L && type %in% c("leaf", "contrib")) {
    stop(sprintf(
      "mojoboost cannot predict %s yet",
      if (type == "leaf") "leaf indices" else "SHAP contributions"
    ), call. = FALSE)
  }
  type <- match.arg(type)
  newdata <- mb_as_matrix(newdata, "newdata")
  expected <- mb.num.feature(object)
  if (ncol(newdata) != expected) {
    stop(sprintf(
      "newdata has %d columns, but the model was trained on %d",
      ncol(newdata), expected
    ), call. = FALSE)
  }
  mb_check_feature_names(object, base::colnames(newdata))

  classes <- mb.num.class(object)
  if (type == "class" && classes == 1L && !mb_is_binary(object)) {
    if (is.null(object$objective)) {
      stop(
        "type = \"class\" is not available for a model read back with ",
        "mb.load(): the file does not record the objective, so a single ",
        "output column could be a probability or a regression value. ",
        "Threshold predict(model, newdata) yourself.",
        call. = FALSE
      )
    }
    stop(
      "type = \"class\" needs a classification objective; this model was ",
      sprintf("trained with objective = \"%s\"", object$objective),
      call. = FALSE
    )
  }
  predict_type <- if (type == "raw") MB_PREDICT_RAW_SCORE else MB_PREDICT_NORMAL

  # The ABI writes row-major, so C hands back a classes x nrow matrix.
  raw <- .Call(R_mb_predict, object$handle, newdata, predict_type)

  if (type == "class") {
    if (classes == 1L) {
      # Binary: one probability column, thresholded at 0.5 as LightGBM does.
      return(as.integer(raw[1L, ] > 0.5))
    }
    return(max.col(t(raw), ties.method = "first") - 1L)
  }
  if (classes == 1L) {
    return(as.numeric(raw[1L, ]))
  }
  out <- t(raw)
  base::colnames(out) <- paste0("Class_", seq_len(classes) - 1L)
  rownames(out) <- base::rownames(newdata)
  out
}

# Whether the model was trained with a binary objective. The ABI does not
# expose the objective, so this reads what the R side was told at training
# time; a model from mb.load() has no record of it and so cannot answer, which
# is why type = "class" is refused for it.
mb_is_binary <- function(object) {
  identical(object$objective, "binary")
}

mb_check_feature_names <- function(object, names) {
  fitted <- object$feature_names
  if (is.null(fitted) || is.null(names)) {
    return(invisible(NULL))
  }
  if (!identical(as.character(fitted), as.character(names))) {
    stop(
      "newdata's column names do not match the training data's:\n",
      "  trained on: ", paste(fitted, collapse = ", "), "\n",
      "  given:      ", paste(names, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(NULL)
}

#' Feature importance
#'
#' The counterpart of `lightgbm::lgb.importance()`.
#'
#' @param model An `mb.Booster`.
#' @param percentage When `TRUE` (the default, as in LightGBM), `Gain` and
#'   `Frequency` are normalized to sum to 1.
#' @return A data frame with columns `Feature`, `Gain`, and `Frequency`,
#'   ordered by `Gain` descending. `Gain` sums the gain each feature's splits
#'   earned; `Frequency` counts those splits, which is LightGBM's `"split"`
#'   importance.
#'
#'   Two differences from LightGBM: there is no `Cover` column, because
#'   mojoboost does not record per-split hessian sums; and split gains are not
#'   part of the model file, so a model from [mb.load()] reports zero `Gain`
#'   and warns.
#' @examples
#' X <- matrix(rnorm(300), ncol = 3)
#' y <- 3 * X[, 1] - 2 * X[, 2] + rnorm(100, sd = 0.1)
#' model <- mojoboost(X, y, nrounds = 20L)
#' mb.importance(model)
#' @export
mb.importance <- function(model, percentage = TRUE) {
  mb_check_booster(model)
  gain <- .Call(R_mb_importance, model$handle, 1L)
  frequency <- .Call(R_mb_importance, model$handle, 0L)

  if (sum(gain) == 0 && sum(frequency) > 0) {
    warning(
      "split gains are not stored in the model file, so Gain is zero for a ",
      "model read back with mb.load(). Compute importance before saving, or ",
      "retrain.",
      call. = FALSE
    )
  }
  if (percentage) {
    if (sum(gain) > 0) gain <- gain / sum(gain)
    if (sum(frequency) > 0) frequency <- frequency / sum(frequency)
  }

  names <- model$feature_names
  if (is.null(names)) {
    names <- paste0("Column_", seq_along(gain) - 1L)
  }
  out <- data.frame(
    Feature = as.character(names),
    Gain = gain,
    Frequency = frequency,
    stringsAsFactors = FALSE
  )
  out <- out[order(-out$Gain, -out$Frequency, out$Feature), , drop = FALSE]
  rownames(out) <- NULL
  out
}
