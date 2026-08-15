#' Parameter keys mojotrees accepts
#'
#' The primary parameter names, as reported by the C ABI itself, so this can
#' never drift from what the engine parses. Aliases are not listed: the engine
#' accepts LightGBM's aliases but reports one canonical name for each.
#'
#' @return A character vector of parameter names.
#' @examples
#' head(mb.parameter.keys())
#' @export
mb.parameter.keys <- function() {
  keys <- .Call(R_mb_parameter_keys)
  # The ABI returns a human-readable list, which is comma separated. Split on
  # commas and whitespace together rather than on one of them, so the R side
  # does not have to care which the engine happens to use.
  parts <- strsplit(keys, "[,[:space:]]+")[[1L]]
  parts[nzchar(parts)]
}

#' The mojotrees engine version
#'
#' @return The version of the linked mojotrees C ABI, as a string.
#' @export
mb.version <- function() {
  .Call(R_mb_version)
}

#' The C ABI version this package was compiled against
#'
#' The package needs version 3 or newer, for `mb.importance()` and
#' `mb.parameter.keys()`. `configure` refuses to build against an older
#' header, so this is for diagnosing a library swapped in after the fact.
#'
#' @return The ABI version, as an integer.
#' @export
mb.abi.version <- function() {
  .Call(R_mb_abi_version)
}

#' Whether this build can train on an accelerator
#'
#' @return `TRUE` when a GPU backend is usable. Pass `device_type = "gpu"` in
#'   `params` to ask for it; mojotrees raises rather than falling back
#'   silently when it is unavailable.
#' @export
mb.gpu.available <- function() {
  .Call(R_mb_gpu_available)
}

# Format one parameter value the way the C ABI's parser reads it. Kept
# separate from mb.params.string() so the rules are stated once.
mb_format_value <- function(name, value) {
  if (length(value) != 1L) {
    stop(sprintf(
      "parameter '%s' must be a single value, got length %d",
      name, length(value)
    ), call. = FALSE)
  }
  if (is.na(value)) {
    stop(sprintf("parameter '%s' is NA", name), call. = FALSE)
  }
  if (is.logical(value)) {
    # mojotrees has no boolean parameters today; a logical is almost always a
    # user writing TRUE for a 1/0 flag, so translate rather than emit "TRUE".
    return(if (value) "1" else "0")
  }
  if (is.numeric(value)) {
    if (is.infinite(value)) {
      stop(sprintf("parameter '%s' is infinite", name), call. = FALSE)
    }
    # as.character() gives up to 15 significant digits and no scientific
    # notation for the magnitudes parameters take, which is what the parser
    # reads back exactly.
    out <- as.character(value)
    if (grepl("e", out, fixed = TRUE)) {
      out <- format(value, scientific = FALSE, trim = TRUE)
    }
    return(out)
  }
  if (is.character(value)) {
    return(value)
  }
  stop(sprintf(
    "parameter '%s' must be numeric, logical, or character, got %s",
    name, class(value)[1L]
  ), call. = FALSE)
}

#' Render a parameter list as the engine's parameter string
#'
#' mojotrees's C ABI takes LightGBM's `"key=value key=value"` form. This is
#' the conversion `mb.train()` applies, exported because seeing the string is
#' the quickest way to check what was actually asked for.
#'
#' @param params A named list of parameters.
#' @return A single string.
#' @examples
#' mb.params.string(list(objective = "regression", num_leaves = 15L))
#' @export
mb.params.string <- function(params) {
  if (is.null(params) || length(params) == 0L) {
    return("")
  }
  if (!is.list(params) || is.null(names(params)) || any(names(params) == "")) {
    stop("params must be a named list", call. = FALSE)
  }
  names_ <- names(params)
  if (anyDuplicated(names_)) {
    dupes <- unique(names_[duplicated(names_)])
    stop(sprintf(
      "params names each parameter once; repeated: %s",
      paste(dupes, collapse = ", ")
    ), call. = FALSE)
  }
  if (any(grepl("[[:space:]=]", names_))) {
    stop("parameter names must not contain spaces or '='", call. = FALSE)
  }
  keep <- !(names_ %in% MB_IGNORED_PARAMS)
  params <- params[keep]
  names_ <- names_[keep]
  if (length(params) == 0L) {
    return("")
  }
  values <- vapply(
    seq_along(params),
    function(i) mb_format_value(names_[i], params[[i]]),
    character(1L)
  )
  if (any(grepl("[[:space:]]", values))) {
    stop("parameter values must not contain spaces", call. = FALSE)
  }
  paste(paste0(names_, "=", values), collapse = " ")
}

# Parameters accepted and dropped rather than passed on. LightGBM users write
# these reflexively and the engine's parameter string does not take them, so
# forwarding one would turn a habit into an error for no gain.
#
# This is the only place a name is silently ignored. Every other unknown name
# reaches the engine and is refused there, which is the point: a typo must not
# quietly train a different model.
#
#   verbosity, verbose  mojotrees does not log through a parameter.
#   num_threads         thread count comes from MOJOTREES_NUM_WORKERS.
MB_IGNORED_PARAMS <- c("verbosity", "verbose", "num_threads")

# Aliases the engine accepts for the boosting-round count. `nrounds` is the R
# spelling, so passing one of these in `params` as well is ambiguous.
#
# This list has to match what the engine actually parses, in both directions.
# A name here that the engine rejects turns an unknown-parameter error into a
# misleading "given twice" error, and a name the engine accepts but that is
# missing here lets both spellings through to collide downstream. LightGBM's
# `num_trees` is deliberately absent: mojotrees does not accept it.
MB_NROUNDS_ALIASES <- c(
  "num_iterations", "num_iteration", "n_estimators",
  "num_round", "num_rounds", "num_boost_round"
)
