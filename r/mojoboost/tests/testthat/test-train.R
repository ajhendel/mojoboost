test_that("mb.train fits a regression model and learns the signal", {
  d <- make_regression()
  model <- mb.train(
    params = list(objective = "regression", num_leaves = 15L),
    data = mb.Dataset(d$X, label = d$y),
    nrounds = 40L
  )
  expect_s3_class(model, "mb.Booster")
  expect_equal(mb.num.iteration(model), 40L)
  expect_equal(mb.num.trees(model), 40L)
  expect_equal(mb.num.feature(model), ncol(d$X))
  expect_equal(mb.num.class(model), 1L)

  pred <- predict(model, d$X)
  expect_type(pred, "double")
  expect_length(pred, nrow(d$X))
  # A tripwire for a scrambled boundary, not a quality claim.
  r2 <- 1 - sum((d$y - pred)^2) / sum((d$y - mean(d$y))^2)
  expect_gt(r2, 0.9)
})

test_that("mojoboost() is mb.train() without the Dataset step", {
  d <- make_regression()
  a <- mb.train(
    params = list(objective = "regression"),
    data = mb.Dataset(d$X, label = d$y),
    nrounds = 10L
  )
  b <- mojoboost(d$X, d$y, params = list(objective = "regression"),
                 nrounds = 10L)
  expect_identical(predict(a, d$X), predict(b, d$X))
})

test_that("the objective shorthand and params agree, and cannot be doubled", {
  d <- make_binary()
  a <- mojoboost(d$X, d$y, objective = "binary", nrounds = 10L)
  b <- mojoboost(d$X, d$y, params = list(objective = "binary"), nrounds = 10L)
  expect_identical(predict(a, d$X), predict(b, d$X))
  expect_error(
    mojoboost(d$X, d$y, objective = "binary",
              params = list(objective = "binary"), nrounds = 10L),
    "only one"
  )
})

test_that("a matrix plus label is accepted directly", {
  d <- make_regression()
  from_matrix <- mb.train(list(objective = "regression"), d$X, 10L,
                          label = d$y)
  from_dataset <- mb.train(list(objective = "regression"),
                           mb.Dataset(d$X, label = d$y), 10L)
  expect_identical(predict(from_matrix, d$X), predict(from_dataset, d$X))
})

test_that("giving the label twice is an error rather than a guess", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_error(mb.train(list(), ds, 10L, label = d$y), "twice")
  expect_error(mb.train(list(), ds, 10L, weight = rep(1, nrow(d$X))), "twice")
})

test_that("nrounds and params$num_iterations cannot both be set", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_error(
    mb.train(list(num_iterations = 10L), ds, nrounds = 20L),
    "given twice"
  )
  # Without an explicit nrounds, params wins and the default does not fight it.
  model <- mb.train(list(num_iterations = 7L), ds)
  expect_equal(mb.num.iteration(model), 7L)
  # An alias counts too.
  expect_error(mb.train(list(n_estimators = 10L), ds, nrounds = 20L),
               "given twice")
  expect_error(mb.train(list(num_iterations = 5L, num_round = 6L), ds),
               "more than once")
})

test_that("nrounds is validated", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_error(mb.train(list(), ds, nrounds = 0L), "positive whole number")
  expect_error(mb.train(list(), ds, nrounds = -1L), "positive whole number")
  expect_error(mb.train(list(), ds, nrounds = 2.5), "positive whole number")
  expect_error(mb.train(list(), ds, nrounds = c(1L, 2L)),
               "positive whole number")
})

test_that("sample weights reach the trainer", {
  d <- make_regression()
  unweighted <- mb.train(list(objective = "regression"),
                         mb.Dataset(d$X, label = d$y), 15L)
  ones <- mb.train(
    list(objective = "regression"),
    mb.Dataset(d$X, label = d$y, weight = rep(1, nrow(d$X))), 15L
  )
  # All-ones weights must reproduce the unweighted model exactly.
  expect_identical(predict(unweighted, d$X), predict(ones, d$X))

  skewed <- rep(c(0.1, 5), each = nrow(d$X) / 2)
  weighted <- mb.train(
    list(objective = "regression"),
    mb.Dataset(d$X, label = d$y, weight = skewed), 15L
  )
  expect_false(identical(predict(unweighted, d$X), predict(weighted, d$X)))
})

test_that("training is deterministic under the sampling parameters", {
  # Row bagging is not reachable from a parameter string (see test-params.R),
  # so the sampling this can vary is over features.
  d <- make_regression()
  params <- list(
    objective = "regression", feature_fraction = 0.8,
    feature_fraction_bynode = 0.9, feature_fraction_seed = 9L,
    extra_trees = TRUE, extra_seed = 4L
  )
  ds <- mb.Dataset(d$X, label = d$y)
  expect_identical(
    predict(mb.train(params, ds, 15L), d$X),
    predict(mb.train(params, ds, 15L), d$X)
  )
})

test_that("every round-count alias is one the engine really accepts", {
  # The alias list exists to detect a collision with nrounds. A name in it
  # that the engine rejects would turn an unknown-parameter error into a
  # misleading "given twice" one, so each is trained with here.
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  for (alias in MB_NROUNDS_ALIASES) {
    params <- list()
    params[[alias]] <- 5L
    expect_no_error(mb.train(params, ds), info = alias)
  }
  # LightGBM's spelling, which mojoboost does not accept. It must fail as an
  # unknown parameter rather than be mistaken for the round count.
  expect_error(mb.train(list(num_trees = 5L), ds), "num_trees")
})

test_that("engine errors surface as R conditions naming the problem", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_error(mb.train(list(num_leves = 31L), ds, 5L), "num_leves")
  expect_error(mb.train(list(objective = "nonesuch"), ds, 5L), "nonesuch")
  expect_error(mb.train(list(num_leaves = 1L), ds, 5L), "num_leaves")
  expect_error(mb.train(list(lambda_l1 = -1), ds, 5L), "lambda_l1")
  expect_error(mb.train(list(device_type = "nonesuch"), ds, 5L), "nonesuch")
})

test_that("binary and multiclass label rules are enforced", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_error(mb.train(list(objective = "binary"), ds, 5L), "binary")

  m <- make_multiclass()
  expect_error(
    mb.train(list(objective = "multiclass", num_class = 2L),
             mb.Dataset(m$X, label = m$y), 5L),
    "class label"
  )
  expect_error(
    mb.train(list(objective = "multiclass"), mb.Dataset(m$X, label = m$y), 5L),
    "num_class"
  )
})

test_that("every documented objective trains", {
  d <- make_regression()
  for (objective in c("regression", "l2", "mse", "huber", "quantile",
                      "regression_l1", "mae")) {
    model <- mojoboost(d$X, d$y, objective = objective, nrounds = 5L)
    expect_equal(mb.num.iteration(model), 5L, info = objective)
  }
  counts <- make_multiclass()
  poisson <- mojoboost(counts$X, counts$y, objective = "poisson", nrounds = 5L)
  expect_true(all(predict(poisson, counts$X) >= 0))
})
