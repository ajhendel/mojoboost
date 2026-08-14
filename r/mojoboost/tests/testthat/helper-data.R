# Deterministic fixtures shared by the test files. set.seed keeps failures
# reproducible; the engine itself is deterministic given the same inputs.
make_regression <- function(n = 200L, p = 3L) {
  set.seed(42L)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- paste0("f", seq_len(p))
  y <- 3 * X[, 1] - 2 * X[, 2] + 0.5 * X[, 1] * X[, 2] + rnorm(n, sd = 0.05)
  list(X = X, y = y)
}

make_binary <- function(n = 200L, p = 3L) {
  d <- make_regression(n, p)
  d$y <- as.numeric(d$y > 0)
  d
}

make_multiclass <- function(n = 210L, p = 3L) {
  d <- make_regression(n, p)
  d$y <- as.numeric(cut(d$y, breaks = 3L, labels = FALSE)) - 1
  d
}
