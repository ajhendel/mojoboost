test_that("parameter lists render as the engine's parameter string", {
  expect_equal(mb.params.string(list()), "")
  expect_equal(mb.params.string(NULL), "")
  expect_equal(
    mb.params.string(list(objective = "regression", num_leaves = 15L)),
    "objective=regression num_leaves=15"
  )
  expect_equal(mb.params.string(list(learning_rate = 0.05)),
               "learning_rate=0.05")
  # Logicals become the 1/0 the parser reads, not "TRUE".
  expect_equal(mb.params.string(list(verbose = TRUE)), "verbose=1")
  expect_equal(mb.params.string(list(verbose = FALSE)), "verbose=0")
  # Small values must not go out in scientific notation.
  expect_equal(mb.params.string(list(min_child_weight = 1e-8)),
               "min_child_weight=0.00000001")
})

test_that("malformed parameter lists are rejected before reaching C", {
  expect_error(mb.params.string(list(1L)), "named list")
  expect_error(mb.params.string(list(a = 1L, a = 2L)), "repeated")
  expect_error(mb.params.string(list(`num leaves` = 1L)), "spaces")
  expect_error(mb.params.string(list(num_leaves = c(1L, 2L))), "single value")
  expect_error(mb.params.string(list(num_leaves = NA)), "NA")
  expect_error(mb.params.string(list(learning_rate = Inf)), "infinite")
  expect_error(mb.params.string(list(objective = list())), "single value")
  expect_error(mb.params.string(list(objective = list("a"))), "numeric")
  expect_error(mb.params.string(list(objective = "a b")), "spaces")
})

test_that("the engine reports its own accepted keys", {
  keys <- mb.parameter.keys()
  expect_type(keys, "character")
  expect_true(length(keys) > 20L)
  expect_true(all(c("objective", "num_leaves", "learning_rate", "lambda_l1",
                    "max_depth", "feature_fraction") %in% keys))
  # Primary names only: an alias the engine accepts is not reported here.
  expect_false("eta" %in% keys)
  expect_false(any(duplicated(keys)))
  expect_false(any(keys == ""))
})

test_that("every key the engine reports is accepted by the trainer", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  # One representative value per primary key mb.parameter.keys() reports.
  # Four are excluded because they are conditional on the objective rather
  # than free-standing, and each has its own coverage:
  #   num_class               multiclass only
  #   alpha                   huber and quantile only
  #   fair_c                  the fair objective only
  #   tweedie_variance_power  the tweedie objective only
  # max_conflict_rate is excluded because a value above 0 is deliberately
  # withheld pending a benchmark, so there is no accepted value to pass.
  conditional <- c("num_class", "alpha", "fair_c", "tweedie_variance_power",
                   "max_conflict_rate")
  values <- list(
    objective = "regression", num_iterations = 5L, learning_rate = 0.05,
    num_leaves = 7L, min_data_in_leaf = 5L, min_sum_hessian_in_leaf = 0.01,
    lambda_l1 = 0.1, lambda_l2 = 0.5, max_depth = 3L,
    feature_fraction = 0.9, feature_fraction_bynode = 0.9,
    feature_fraction_bylevel = 0.9, feature_fraction_seed = 13L,
    min_gain_to_split = 0.0, max_delta_step = 0.5, path_smooth = 0.1,
    extra_trees = TRUE, extra_seed = 3L, monotone_penalty = 0.1,
    monotone_constraints_method = "basic", cegb_tradeoff = 0.5,
    cegb_penalty_split = 0.1, enable_bundle = FALSE,
    data_sample_strategy = "bagging", max_bin = 63L, device = "cpu",
    use_missing = TRUE
  )
  # The reported list and the tested list are the same set. This is what
  # catches a key added to the engine and never exercised from R.
  expect_setequal(c(names(values), conditional), mb.parameter.keys())

  for (key in names(values)) {
    params <- list()
    params[[key]] <- values[[key]]
    expect_no_error(mb.train(params, ds, 3L))
  }
})

test_that("the objective-conditional keys are accepted with their objective", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_no_error(mb.train(list(objective = "quantile", alpha = 0.8), ds, 3L))
  expect_no_error(mb.train(list(objective = "huber", alpha = 0.8), ds, 3L))
  # And refused with an objective they do not belong to, rather than ignored.
  expect_error(mb.train(list(objective = "regression", alpha = 0.8), ds, 3L),
               "alpha")
})

test_that("aliases the engine accepts reach the trainer", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  aliases <- list(
    shrinkage_rate = 0.05, eta = 0.05, num_leaf = 7L, min_data = 5L,
    min_child_samples = 5L, min_sum_hessian = 0.01, min_child_weight = 0.01,
    reg_alpha = 0.1, reg_lambda = 0.5, sub_feature = 0.9,
    colsample_bytree = 0.9, colsample_bynode = 0.9, device_type = "cpu"
  )
  for (key in names(aliases)) {
    params <- list()
    params[[key]] <- aliases[[key]]
    expect_no_error(mb.train(params, ds, 3L))
  }
})

test_that("bagging is not reachable through a parameter string", {
  # It exists in the engine, but only through the Mojo API: a parameter
  # string cannot carry it. The engine says so itself, and this pins that
  # the message explains rather than reporting a bare unknown name.
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  for (key in c("bagging_fraction", "bagging_freq", "bagging_seed")) {
    params <- list()
    params[[key]] <- 0.8
    expect_error(mb.train(params, ds, 3L), "Mojo API")
  }
})

test_that("the reflexive LightGBM parameters are dropped, not forwarded", {
  # Documented behavior: these are accepted and ignored, because LightGBM
  # users pass them out of habit and the engine's parameter string has no
  # room for them. Everything else unknown must still reach the engine.
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  expect_identical(mb.params.string(list(verbosity = -1L)), "")
  expect_identical(mb.params.string(list(verbose = 0L)), "")
  expect_identical(mb.params.string(list(num_threads = 2L)), "")
  expect_identical(
    mb.params.string(list(num_threads = 2L, num_leaves = 7L)),
    "num_leaves=7"
  )
  expect_no_error(mb.train(list(verbosity = -1L, num_threads = 2L), ds, 3L))
  # Not a blanket amnesty for unknown names.
  expect_error(mb.train(list(no_such_parameter = 1L), ds, 3L))
})

test_that("the LightGBM aliases train the same model as their canonical name", {
  d <- make_regression()
  ds <- mb.Dataset(d$X, label = d$y)
  # Only aliases the engine's parameter string actually accepts. Bagging is
  # absent on purpose: it is reachable from the Mojo API alone, which
  # test-params.R pins separately.
  pairs <- list(
    c("min_data_in_leaf", "min_child_samples"),
    c("min_sum_hessian_in_leaf", "min_child_weight"),
    c("lambda_l1", "reg_alpha"),
    c("lambda_l2", "reg_lambda"),
    c("feature_fraction", "colsample_bytree"),
    c("feature_fraction_bynode", "colsample_bynode"),
    c("learning_rate", "eta"),
    c("learning_rate", "shrinkage_rate"),
    c("num_leaves", "num_leaf")
  )
  numbers <- list(min_data_in_leaf = 9L, min_sum_hessian_in_leaf = 0.02,
                  lambda_l1 = 0.3, lambda_l2 = 0.7, feature_fraction = 0.7,
                  feature_fraction_bynode = 0.7, learning_rate = 0.07,
                  num_leaves = 9L)
  for (pair in pairs) {
    value <- numbers[[pair[1L]]]
    a <- list(); a[[pair[1L]]] <- value
    b <- list(); b[[pair[2L]]] <- value
    expect_identical(
      predict(mb.train(a, ds, 10L), d$X),
      predict(mb.train(b, ds, 10L), d$X),
      info = paste(pair, collapse = " vs ")
    )
  }
})

test_that("the engine version and GPU flag are reportable", {
  expect_match(mb.version(), "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  expect_type(mb.gpu.available(), "logical")
  expect_length(mb.gpu.available(), 1L)
})

test_that("the ABI version is at least what this package needs", {
  # configure refuses an older header, so this catches a library swapped in
  # underneath an already-installed package rather than a build mistake.
  expect_type(mb.abi.version(), "integer")
  expect_gte(mb.abi.version(), 3L)
})
