test_that("a fresh model is valid and mb.free releases it", {
  d <- make_regression()
  model <- mojoboost(d$X, d$y, nrounds = 5L)
  expect_true(mb.is.valid(model))

  mb.free(model)
  expect_false(mb.is.valid(model))
  expect_error(predict(model, d$X), "no longer valid")
  expect_error(mb.save(model, tempfile()), "no longer valid")
  expect_error(mb.num.class(model), "no longer valid")
})

test_that("mb.free is idempotent, so the finalizer cannot double free", {
  d <- make_regression()
  model <- mojoboost(d$X, d$y, nrounds = 5L)
  expect_silent(mb.free(model))
  expect_silent(mb.free(model))
  # Whatever the garbage collector does now must also be safe.
  expect_silent(gc(verbose = FALSE))
})

test_that("models are independent of one another", {
  d <- make_regression()
  small <- mojoboost(d$X, d$y, nrounds = 5L)
  large <- mojoboost(d$X, d$y, nrounds = 60L)
  before <- predict(small, d$X)
  expect_false(identical(before, predict(large, d$X)))

  mb.free(large)
  # Releasing one model must not disturb another.
  expect_identical(before, predict(small, d$X))
})

test_that("collecting an unreferenced model does not disturb a live one", {
  d <- make_regression()
  keep <- mojoboost(d$X, d$y, nrounds = 10L)
  before <- predict(keep, d$X)
  for (i in seq_len(20L)) {
    transient <- mojoboost(d$X, d$y, nrounds = 3L)
    rm(transient)
  }
  gc(verbose = FALSE)
  expect_identical(before, predict(keep, d$X))
})

test_that("a handle that did not survive serialization says so", {
  d <- make_regression()
  model <- mojoboost(d$X, d$y, nrounds = 5L)
  path <- tempfile(fileext = ".rds")
  saveRDS(model, path)
  restored <- readRDS(path)
  unlink(path)

  # The pointer is gone, and the message must point at the working route
  # rather than leaving the user with a segfault or a silent wrong answer.
  expect_false(mb.is.valid(restored))
  expect_error(predict(restored, d$X), "mb.save")
})

test_that("mb.is.valid tolerates things that are not models", {
  expect_false(mb.is.valid(NULL))
  expect_false(mb.is.valid(42))
  expect_false(mb.is.valid(list()))
})

test_that("the functions reject objects that are not boosters", {
  expect_error(mb.num.class(42), "mb.Booster")
  expect_error(mb.importance("nope"), "mb.Booster")
  expect_error(mb.free(list()), "mb.Booster")
})

test_that("print works before and after release", {
  d <- make_regression()
  model <- mojoboost(d$X, d$y, nrounds = 5L)
  expect_output(print(model), "mb.Booster")
  expect_output(print(model), "5 iterations")
  mb.free(model)
  expect_output(print(model), "released")
})
