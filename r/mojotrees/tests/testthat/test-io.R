test_that("a saved model reloads and predicts identically", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 25L)
  before <- predict(model, d$X)

  path <- tempfile(fileext = ".mbst")
  on.exit(unlink(path), add = TRUE)
  expect_identical(mb.save(model, path), model)
  expect_true(file.exists(path))

  reloaded <- mb.load(path)
  expect_s3_class(reloaded, "mb.Booster")
  expect_equal(mb.num.iteration(reloaded), 25L)
  expect_equal(mb.num.feature(reloaded), ncol(d$X))
  # The format stores raw bit patterns, so this is exact.
  expect_identical(before, predict(reloaded, d$X))
})

test_that("a multiclass model round-trips too", {
  d <- make_multiclass()
  model <- mojotrees(d$X, d$y, objective = "multiclass",
                     params = list(num_class = 3L), nrounds = 10L)
  before <- predict(model, d$X)

  path <- tempfile(fileext = ".mbst")
  on.exit(unlink(path), add = TRUE)
  mb.save(model, path)
  reloaded <- mb.load(path)

  expect_equal(mb.num.class(reloaded), 3L)
  expect_equal(mb.num.trees(reloaded), 30L)
  expect_equal(before, predict(reloaded, d$X))
})

test_that("what the file does not hold is reported honestly", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 15L)
  path <- tempfile(fileext = ".mbst")
  on.exit(unlink(path), add = TRUE)
  mb.save(model, path)
  reloaded <- mb.load(path)

  # Feature names are not in the file, so importance falls back to positions.
  expect_warning(importance <- mb.importance(reloaded), "not stored")
  expect_setequal(importance$Feature,
                  c("Column_0", "Column_1", "Column_2"))
  expect_true(all(importance$Gain == 0))
  # Split counts are recoverable from the trees, so those survive.
  expect_gt(sum(importance$Frequency), 0)

  # The objective is not in the file either, so class prediction is refused
  # rather than guessed.
  expect_error(predict(reloaded, d$X, type = "class"), "mb.load")
})

test_that("io errors are R conditions", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 5L)
  expect_error(mb.load(tempfile()), "no such file")
  expect_error(mb.save(model, ""), "must not be empty")
  expect_error(mb.save(model, c("a", "b")), "single string")
  expect_error(mb.load(42), "single string")
  # An unwritable location is an error. (A merely missing parent directory is
  # not: see the next test.)
  expect_error(mb.save(model, "/m.mbst"), ".")
})

test_that("saving creates missing parent directories", {
  # A documented difference from LightGBM, whose LGBM_BoosterSaveModel fails
  # when the directory does not exist. mojotrees's writer creates the path.
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 5L)
  nested <- file.path(tempdir(), "mb-test-no-such-dir", "m.mbst")
  on.exit(unlink(dirname(nested), recursive = TRUE), add = TRUE)

  expect_false(dir.exists(dirname(nested)))
  expect_silent(mb.save(model, nested))
  expect_true(file.exists(nested))
  expect_identical(predict(model, d$X), predict(mb.load(nested), d$X))
})

test_that("a file that is not a model is rejected, not misread", {
  path <- tempfile(fileext = ".mbst")
  on.exit(unlink(path), add = TRUE)
  writeLines(c("this is not a mojotrees model"), path)
  expect_error(mb.load(path), ".")
})
