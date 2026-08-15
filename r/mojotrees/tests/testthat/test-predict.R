test_that("regression predictions are a plain vector and raw equals response", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 20L)
  response <- predict(model, d$X)
  raw <- predict(model, d$X, type = "raw")
  expect_null(dim(response))
  expect_length(response, nrow(d$X))
  # Squared error has an identity link, so the two are the same numbers.
  expect_identical(response, raw)
})

test_that("binary predictions are probabilities and raw is their logit", {
  d <- make_binary()
  model <- mojotrees(d$X, d$y, objective = "binary", nrounds = 30L)
  # LightGBM reports 1 class for binary; so does mojotrees.
  expect_equal(mb.num.class(model), 1L)

  probability <- predict(model, d$X)
  raw <- predict(model, d$X, type = "raw")
  expect_true(all(probability > 0 & probability < 1))
  expect_equal(probability, 1 / (1 + exp(-raw)), tolerance = 1e-9)

  classes <- predict(model, d$X, type = "class")
  expect_type(classes, "integer")
  expect_true(all(classes %in% c(0L, 1L)))
  expect_identical(classes, as.integer(probability > 0.5))
  expect_gt(mean(classes == d$y), 0.9)
})

test_that("multiclass predictions are a row-normalized matrix", {
  d <- make_multiclass()
  model <- mojotrees(d$X, d$y, objective = "multiclass",
                     params = list(num_class = 3L), nrounds = 15L)
  expect_equal(mb.num.class(model), 3L)
  # One tree per class per iteration, as in LightGBM.
  expect_equal(mb.num.trees(model), 45L)

  proba <- predict(model, d$X)
  expect_true(is.matrix(proba))
  expect_equal(dim(proba), c(nrow(d$X), 3L))
  expect_equal(colnames(proba), c("Class_0", "Class_1", "Class_2"))
  expect_equal(rowSums(proba), rep(1, nrow(d$X)), tolerance = 1e-9)
  expect_true(all(proba >= 0 & proba <= 1))

  raw <- predict(model, d$X, type = "raw")
  expect_equal(dim(raw), c(nrow(d$X), 3L))

  classes <- predict(model, d$X, type = "class")
  expect_type(classes, "integer")
  expect_true(all(classes %in% 0:2))
  # class is the argmax of the probabilities, so the two cannot disagree.
  expect_identical(classes, max.col(proba, ties.method = "first") - 1L)
})

test_that("prediction rejects the wrong shape rather than reading past it", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 5L)
  expect_error(predict(model, d$X[, 1:2, drop = FALSE]), "2 columns")
  expect_error(predict(model, as.vector(d$X)), "matrix")
  expect_error(predict(model, d$X[0, , drop = FALSE]), "empty")
})

test_that("column names must match the ones trained on", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 5L)
  renamed <- d$X
  colnames(renamed) <- c("a", "b", "c")
  expect_error(predict(model, renamed), "do not match")

  # Unnamed newdata is accepted: there is nothing to contradict.
  unnamed <- d$X
  colnames(unnamed) <- NULL
  expect_length(predict(model, unnamed), nrow(d$X))
})

test_that("LightGBM's other prediction types get a clear refusal", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 5L)
  expect_error(predict(model, d$X, type = "leaf"), "leaf indices")
  expect_error(predict(model, d$X, type = "contrib"), "SHAP")
  expect_error(predict(model, d$X, type = "nonesuch"), "should be one of")
})

test_that("type = class needs a classification objective", {
  d <- make_regression()
  # Trained without naming an objective: the engine's default is regression,
  # and the refusal must say so rather than claim the objective is unknown.
  implicit <- mojotrees(d$X, d$y, nrounds = 5L)
  expect_error(predict(implicit, d$X, type = "class"), "classification")
  expect_error(predict(implicit, d$X, type = "class"), "regression")

  explicit <- mojotrees(d$X, d$y, objective = "huber", nrounds = 5L)
  expect_error(predict(explicit, d$X, type = "class"), "huber")
})

test_that("NA and NaN are missing values, infinities are rejected", {
  d <- make_regression()
  X <- d$X
  X[seq(1, nrow(X), by = 10), 3] <- NA
  model <- mojotrees(X, d$y, nrounds = 10L)
  pred <- predict(model, X)
  expect_false(any(is.na(pred)))

  X[1, 1] <- Inf
  expect_error(mojotrees(X, d$y, nrounds = 5L), "infinite")
  expect_error(predict(model, X), "infinite")
})

test_that("a data frame of numeric columns works like a matrix", {
  d <- make_regression()
  df <- as.data.frame(d$X)
  model <- mojotrees(df, d$y, nrounds = 10L)
  expect_identical(predict(model, df), predict(model, d$X))

  df$chr <- letters[seq_len(nrow(df))]
  expect_error(mojotrees(df, d$y, nrounds = 5L), "non-numeric")
})

test_that("importance ranks the informative features first", {
  d <- make_regression()
  model <- mojotrees(d$X, d$y, nrounds = 25L)
  importance <- mb.importance(model)
  expect_s3_class(importance, "data.frame")
  expect_equal(names(importance), c("Feature", "Gain", "Frequency"))
  expect_equal(nrow(importance), ncol(d$X))
  expect_setequal(importance$Feature, colnames(d$X))
  # Normalized by default, as in LightGBM.
  expect_equal(sum(importance$Gain), 1, tolerance = 1e-9)
  expect_equal(sum(importance$Frequency), 1, tolerance = 1e-9)
  expect_true(!is.unsorted(rev(importance$Gain)))
  # y depends on f1 and f2 only.
  expect_true(importance$Feature[1L] %in% c("f1", "f2"))
  expect_equal(importance$Feature[3L], "f3")

  counts <- mb.importance(model, percentage = FALSE)
  expect_gt(sum(counts$Frequency), 1)
  expect_equal(sum(counts$Frequency), sum(counts$Frequency) %/% 1)
})

test_that("importance falls back to positional names without colnames", {
  d <- make_regression()
  X <- d$X
  colnames(X) <- NULL
  model <- mojotrees(X, d$y, nrounds = 10L)
  expect_setequal(mb.importance(model)$Feature,
                  c("Column_0", "Column_1", "Column_2"))
})
