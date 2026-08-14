#' Bundle a matrix with its label and weights
#'
#' The counterpart of `lightgbm::lgb.Dataset()`, and the same thing to reach
#' for when you want the training inputs validated once and carried around
#' together.
#'
#' One difference worth knowing: `lgb.Dataset` is a handle to a binned dataset
#' held by the LightGBM engine, and constructing it does real work. An
#' `mb.Dataset` is a plain R object. mojoboost bins inside training, so there
#' is nothing to construct early and nothing engine-side to free. Building one
#' only validates and stores.
#'
#' @param data A numeric matrix, one row per observation. `NA` and `NaN` mark
#'   missing values; infinities are rejected.
#' @param label A numeric vector with one value per row.
#' @param weight Optional per-row weights: finite, nonnegative, not all zero.
#' @param colnames Optional feature names, defaulting to `colnames(data)`.
#' @return An object of class `mb.Dataset`.
#' @examples
#' X <- matrix(rnorm(100), ncol = 2)
#' y <- X[, 1] * 2 + rnorm(50, sd = 0.1)
#' ds <- mb.Dataset(X, label = y)
#' @export
mb.Dataset <- function(data, label, weight = NULL, colnames = NULL) {
  data <- mb_as_matrix(data, "data")
  if (missing(label) || is.null(label)) {
    stop("label is required", call. = FALSE)
  }
  label <- mb_as_label(label, nrow(data))
  weight <- mb_as_weight(weight, nrow(data))
  if (is.null(colnames)) {
    colnames <- base::colnames(data)
  } else if (length(colnames) != ncol(data)) {
    stop(sprintf(
      "colnames has %d entries but data has %d columns",
      length(colnames), ncol(data)
    ), call. = FALSE)
  }
  structure(
    list(
      data = data,
      label = label,
      weight = weight,
      colnames = colnames
    ),
    class = "mb.Dataset"
  )
}

#' @export
print.mb.Dataset <- function(x, ...) {
  cat(sprintf(
    "<mb.Dataset: %d rows, %d columns%s>\n",
    nrow(x$data), ncol(x$data),
    if (is.null(x$weight)) "" else ", weighted"
  ))
  invisible(x)
}

#' @export
dim.mb.Dataset <- function(x) dim(x$data)

# --- validation ----------------------------------------------------------
#
# These run before anything crosses into C. The C ABI checks the same things
# and would report them perfectly well, but an R user is better served by an
# error that names the argument they passed than by one naming a buffer.

mb_as_matrix <- function(data, what) {
  if (inherits(data, c("dgCMatrix", "dgRMatrix", "dgTMatrix", "sparseMatrix"))) {
    stop(sprintf(
      "%s is a sparse matrix. mojoboost has no sparse training path yet; ",
      what
    ), "convert with as.matrix() if the dense form fits in memory.",
      call. = FALSE
    )
  }
  if (is.data.frame(data)) {
    if (!all(vapply(data, is.numeric, logical(1L)))) {
      stop(sprintf("%s has non-numeric columns", what), call. = FALSE)
    }
    data <- as.matrix(data)
  }
  if (!is.matrix(data)) {
    stop(sprintf("%s must be a matrix or a data frame", what), call. = FALSE)
  }
  if (!is.numeric(data)) {
    stop(sprintf("%s must be numeric", what), call. = FALSE)
  }
  if (nrow(data) == 0L || ncol(data) == 0L) {
    stop(sprintf(
      "%s is empty (%d x %d)", what, nrow(data), ncol(data)
    ), call. = FALSE)
  }
  storage.mode(data) <- "double"
  # NA and NaN are both the missing-value marker; only infinities are wrong.
  if (any(is.infinite(data))) {
    stop(sprintf(
      "%s contains infinite values. NA and NaN are accepted as missing, ",
      what
    ), "infinities are not.", call. = FALSE)
  }
  data
}

mb_as_label <- function(label, n) {
  if (is.factor(label)) {
    stop(
      "label is a factor. mojoboost trains on numeric labels: use ",
      "as.integer(label) - 1L for a classification objective, and record ",
      "levels(label) yourself, because the model file does not store them.",
      call. = FALSE
    )
  }
  if (!is.numeric(label) || !is.null(dim(label))) {
    stop("label must be a numeric vector", call. = FALSE)
  }
  if (length(label) != n) {
    stop(sprintf(
      "label has %d values but data has %d rows", length(label), n
    ), call. = FALSE)
  }
  label <- as.double(label)
  if (any(!is.finite(label))) {
    stop("label must be finite", call. = FALSE)
  }
  label
}

mb_as_weight <- function(weight, n) {
  if (is.null(weight)) {
    return(NULL)
  }
  if (!is.numeric(weight) || !is.null(dim(weight))) {
    stop("weight must be a numeric vector", call. = FALSE)
  }
  if (length(weight) != n) {
    stop(sprintf(
      "weight has %d values but data has %d rows", length(weight), n
    ), call. = FALSE)
  }
  weight <- as.double(weight)
  if (any(!is.finite(weight))) {
    stop("weight must be finite", call. = FALSE)
  }
  if (any(weight < 0)) {
    stop("weight must be nonnegative", call. = FALSE)
  }
  if (sum(weight) <= 0) {
    stop("weights sum to zero; at least one must be positive", call. = FALSE)
  }
  weight
}
