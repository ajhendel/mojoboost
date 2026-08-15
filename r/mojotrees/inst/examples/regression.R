# Minimal regression example.
#
# Build the C ABI first:   capi/build.sh
# Install the package:     R CMD INSTALL r/mojotrees
# Then:                    Rscript r/mojotrees/inst/examples/regression.R

library(mojotrees)

set.seed(1L)
n <- 1000L
X <- matrix(rnorm(n * 4L), nrow = n, ncol = 4L)
colnames(X) <- c("speed", "load", "temp", "noise")
y <- 3 * X[, "speed"] - 2 * X[, "load"] + 0.5 * X[, "temp"]^2 +
  rnorm(n, sd = 0.25)

train <- seq_len(800L)
test <- setdiff(seq_len(n), train)

model <- mb.train(
  params = list(
    objective = "regression",
    num_leaves = 31L,
    learning_rate = 0.05,
    min_data_in_leaf = 20L
  ),
  data = mb.Dataset(X[train, ], label = y[train]),
  nrounds = 200L
)

print(model)

pred <- predict(model, X[test, ])
rmse <- sqrt(mean((y[test] - pred)^2))
r2 <- 1 - sum((y[test] - pred)^2) / sum((y[test] - mean(y[test]))^2)
cat(sprintf("test RMSE %.4f, R^2 %.4f\n", rmse, r2))

# `noise` does not enter the label, so it should rank last.
print(mb.importance(model))

path <- tempfile(fileext = ".mbst")
mb.save(model, path)
reloaded <- mb.load(path)
stopifnot(identical(pred, predict(reloaded, X[test, ])))
cat("save/load round trip is exact\n")
unlink(path)
