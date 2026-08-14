test_that("mb.Dataset carries data, label, weight, and names", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_s3_class(ds, "mb.Dataset")
  expect_equal(dim(ds), dim(d$X))
  expect_equal(ds$colnames, colnames(d$X))
  expect_null(ds$weight)
  expect_output(print(ds), "200 rows, 3 columns")

  weighted <- mb.Dataset(d$X, label = d$y, weight = rep(1, nrow(d$X)))
  expect_output(print(weighted), "weighted")
})

test_that("data is validated with messages naming the argument", {
  d <- make_regression()
  expect_error(mb.Dataset(as.vector(d$X), label = d$y), "data must be")
  expect_error(mb.Dataset(d$X[0, , drop = FALSE], label = numeric(0)),
               "data is empty")
  expect_error(mb.Dataset(matrix("a", 2, 2), label = c(1, 2)),
               "data must be numeric")

  infinite <- d$X
  infinite[1L] <- Inf
  expect_error(mb.Dataset(infinite, label = d$y), "data contains infinite")
})

test_that("labels are validated", {
  d <- make_regression()
  expect_error(mb.Dataset(d$X), "label is required")
  expect_error(mb.Dataset(d$X, label = d$y[-1L]),
               "199 values but data has 200 rows")
  bad <- d$y
  bad[3L] <- NA
  expect_error(mb.Dataset(d$X, label = bad), "label must be finite")
  expect_error(mb.Dataset(d$X, label = matrix(d$y)), "numeric vector")
})

test_that("a factor label is refused with the conversion spelled out", {
  d <- make_regression()
  labels <- factor(rep(c("a", "b"), length.out = nrow(d$X)))
  expect_error(mb.Dataset(d$X, label = labels), "as.integer")
})

test_that("weights are validated", {
  d <- make_regression()
  n <- nrow(d$X)
  expect_error(mb.Dataset(d$X, label = d$y, weight = rep(1, n - 1L)),
               "199 values but data has 200 rows")
  expect_error(mb.Dataset(d$X, label = d$y, weight = rep(-1, n)),
               "nonnegative")
  expect_error(mb.Dataset(d$X, label = d$y, weight = rep(0, n)),
               "sum to zero")
  bad <- rep(1, n)
  bad[2L] <- NA
  expect_error(mb.Dataset(d$X, label = d$y, weight = bad), "finite")
})

test_that("colnames may be supplied and are length checked", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y, colnames = c("x", "y", "z"))
  expect_equal(ds$colnames, c("x", "y", "z"))
  expect_error(mb.Dataset(d$X, label = d$y, colnames = c("x", "y")),
               "2 entries but data has 3 columns")
})

test_that("sparse input is refused with a pointer to the reason", {
  # Constructed without the Matrix package so the test does not depend on it:
  # the check is on the class attribute, which is what a real dgCMatrix has.
  fake <- structure(list(), class = "dgCMatrix")
  expect_error(mb.Dataset(fake, label = 1), "no sparse training path")
})
