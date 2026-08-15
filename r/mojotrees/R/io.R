#' Write a model to a file
#'
#' The counterpart of `lightgbm::lgb.save()`. The file holds the tree
#' ensemble, in mojotrees's versioned text format. It does not hold the
#' training parameters, the feature names, or the split gains, so
#' [mb.importance()] on a reloaded model reports zero `Gain`, and
#' `predict(type = "class")` is unavailable for it.
#'
#' Note that `saveRDS()` on an `mb.Booster` will appear to work and then hand
#' back an unusable object: the engine handle is a pointer, and pointers do
#' not survive serialization. Use this instead.
#'
#' @param model An `mb.Booster`.
#' @param filename Path to write.
#' @return `model`, invisibly, so calls can be chained.
#' @examples
#' X <- matrix(rnorm(200), ncol = 2)
#' y <- X[, 1] + rnorm(100, sd = 0.1)
#' model <- mojotrees(X, y, nrounds = 10L)
#' path <- tempfile(fileext = ".mbst")
#' mb.save(model, path)
#' reloaded <- mb.load(path)
#' stopifnot(all.equal(predict(model, X), predict(reloaded, X)))
#' unlink(path)
#' @export
mb.save <- function(model, filename) {
  mb_check_booster(model)
  filename <- mb_check_filename(filename)
  .Call(R_mb_save, model$handle, filename)
  invisible(model)
}

#' Read a model back from a file
#'
#' The counterpart of `lightgbm::lgb.load()`.
#'
#' What comes back predicts exactly as the saved model did, bit for bit. What
#' does not come back is everything the file never held: the parameters, the
#' feature names, and the split gains. See [mb.save()].
#'
#' @param filename Path to a file written by [mb.save()].
#' @return An object of class `mb.Booster`.
#' @export
mb.load <- function(filename) {
  filename <- mb_check_filename(filename)
  if (!file.exists(filename)) {
    stop(sprintf("no such file: %s", filename), call. = FALSE)
  }
  handle <- .Call(R_mb_load, filename)
  # objective = NULL records that the file does not say; see mb_new_booster.
  mb_new_booster(handle, params = list(), params_string = "",
                 feature_names = NULL, objective = NULL)
}

mb_check_filename <- function(filename) {
  if (!is.character(filename) || length(filename) != 1L || is.na(filename)) {
    stop("filename must be a single string", call. = FALSE)
  }
  if (!nzchar(filename)) {
    stop("filename must not be empty", call. = FALSE)
  }
  path.expand(filename)
}
