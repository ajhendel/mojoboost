#' mojoboost: gradient boosting with the mojoboost engine
#'
#' R bindings for mojoboost, a gradient boosting decision tree library written
#' in Mojo. The package wraps mojoboost's C ABI and follows the syntax of the
#' `lightgbm` R package, so a script written against that one mostly
#' translates by swapping the prefixes.
#'
#' | lightgbm            | mojoboost         |
#' | ------------------- | ----------------- |
#' | `lgb.Dataset()`     | [mb.Dataset()]    |
#' | `lgb.train()`       | [mb.train()]      |
#' | `lightgbm()`        | [mojoboost()]     |
#' | `predict()`         | [predict.mb.Booster()] |
#' | `lgb.save()`        | [mb.save()]       |
#' | `lgb.load()`        | [mb.load()]       |
#' | `lgb.importance()`  | [mb.importance()] |
#'
#' Parameters keep LightGBM's own names and aliases; [mb.parameter.keys()]
#' lists every one the engine accepts. The differences that are deliberate,
#' rather than merely unimplemented, are collected in the package README:
#' unknown parameter names are errors rather than warnings, only dense
#' matrices are supported, and a model file stores the ensemble but not the
#' training configuration.
#'
#' @section Model lifetime:
#' A trained model lives in the engine behind an external pointer, and R frees
#' it at garbage collection. It therefore does not survive `saveRDS()`,
#' `save.image()`, or a session restart. Use [mb.save()] and [mb.load()], and
#' [mb.is.valid()] to check. [mb.free()] releases one immediately.
#'
#' @keywords internal
#' @useDynLib mojoboost, .registration = TRUE, .fixes = ""
"_PACKAGE"
