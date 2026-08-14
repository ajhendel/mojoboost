#' Train a mojoboost model
#'
#' The counterpart of `lightgbm::lgb.train()`. Parameters are given as a named
#' list under LightGBM's own names, so a `params` list written for LightGBM
#' generally works unchanged; see `mb.parameter.keys()` for everything
#' accepted, and the package README for the deliberate differences.
#'
#' @param params A named list of parameters, e.g.
#'   `list(objective = "regression", num_leaves = 31L, learning_rate = 0.1)`.
#'   Unlike LightGBM, an unrecognized name is an error rather than a warning,
#'   so a typo cannot quietly train a different model.
#' @param data An `mb.Dataset`, or a numeric matrix in which case `label` is
#'   required.
#' @param nrounds Number of boosting iterations. This is the R spelling of
#'   `num_iterations`; passing that (or an alias) in `params` as well is an
#'   error.
#' @param label Labels, when `data` is a matrix. Ignored, with an error, when
#'   `data` is already an `mb.Dataset`.
#' @param weight Per-row weights, when `data` is a matrix.
#' @return An object of class `mb.Booster`.
#' @seealso [mojoboost()] for the one-call form, [predict.mb.Booster()],
#'   [mb.save()], [mb.importance()].
#' @examples
#' X <- matrix(rnorm(200), ncol = 2)
#' y <- 3 * X[, 1] - 2 * X[, 2] + rnorm(100, sd = 0.1)
#' model <- mb.train(
#'   params = list(objective = "regression", num_leaves = 15L),
#'   data = mb.Dataset(X, label = y),
#'   nrounds = 20L
#' )
#' head(predict(model, X))
#' @export
mb.train <- function(params = list(), data, nrounds = 100L, label = NULL,
                     weight = NULL) {
  if (missing(data)) {
    stop("data is required", call. = FALSE)
  }
  if (inherits(data, "mb.Dataset")) {
    if (!is.null(label)) {
      stop(
        "label was given twice: data is already an mb.Dataset carrying one. ",
        "Pass either an mb.Dataset or a matrix plus label.",
        call. = FALSE
      )
    }
    if (!is.null(weight)) {
      stop(
        "weight was given twice: set it on the mb.Dataset instead.",
        call. = FALSE
      )
    }
    dataset <- data
  } else {
    dataset <- mb.Dataset(data, label = label, weight = weight)
  }

  params <- mb_merge_nrounds(params, nrounds, missing(nrounds))
  params_string <- mb.params.string(params)

  handle <- .Call(
    R_mb_train,
    dataset$data,
    dataset$label,
    dataset$weight,
    params_string
  )
  objective <- params[["objective"]]
  if (is.null(objective)) {
    objective <- MB_DEFAULT_OBJECTIVE
  }
  mb_new_booster(handle, params, params_string, dataset$colnames, objective)
}

#' Train a mojoboost model in one call
#'
#' The counterpart of `lightgbm::lightgbm()`: the same thing as
#' [mb.train()] without building an [mb.Dataset()] first.
#'
#' @param data A numeric matrix or data frame.
#' @param label Labels, one per row.
#' @param params A named list of parameters.
#' @param nrounds Number of boosting iterations.
#' @param weight Optional per-row weights.
#' @param objective Shorthand for `params$objective`. Giving both is an error.
#' @return An object of class `mb.Booster`.
#' @examples
#' X <- matrix(rnorm(200), ncol = 2)
#' y <- as.numeric(X[, 1] > 0)
#' model <- mojoboost(X, y, objective = "binary", nrounds = 20L)
#' @export
mojoboost <- function(data, label, params = list(), nrounds = 100L,
                      weight = NULL, objective = NULL) {
  if (!is.null(objective)) {
    if (!is.null(params[["objective"]])) {
      stop(
        "objective was given both directly and in params; set only one",
        call. = FALSE
      )
    }
    params[["objective"]] <- objective
  }
  dataset <- mb.Dataset(data, label = label, weight = weight)
  # Forward `nrounds` only when the caller set it, so that
  # `params$num_iterations` alone stays unambiguous rather than colliding
  # with this function's own default.
  if (missing(nrounds)) {
    mb.train(params = params, data = dataset)
  } else {
    mb.train(params = params, data = dataset, nrounds = nrounds)
  }
}

# Fold `nrounds` into the parameter list, refusing to guess when the caller
# has also named it in `params`.
mb_merge_nrounds <- function(params, nrounds, nrounds_missing) {
  if (is.null(params)) {
    params <- list()
  }
  present <- intersect(names(params), MB_NROUNDS_ALIASES)
  if (length(present) > 0L) {
    if (!nrounds_missing) {
      stop(sprintf(
        paste0(
          "the number of boosting rounds was given twice, as nrounds and as ",
          "params$%s. Set only one."
        ),
        present[1L]
      ), call. = FALSE)
    }
    if (length(present) > 1L) {
      stop(sprintf(
        "params names the round count more than once: %s",
        paste(present, collapse = ", ")
      ), call. = FALSE)
    }
    return(params)
  }
  if (length(nrounds) != 1L || !is.numeric(nrounds) || is.na(nrounds) ||
    nrounds < 1 || nrounds != as.integer(nrounds)) {
    stop("nrounds must be a single positive whole number", call. = FALSE)
  }
  params[["num_iterations"]] <- as.integer(nrounds)
  params
}
