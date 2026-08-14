# Minimal classification example: binary, then multiclass.
#
# Build the C ABI first:   capi/build.sh
# Install the package:     R CMD INSTALL r/mojoboost
# Then:                    Rscript r/mojoboost/inst/examples/classification.R

library(mojoboost)

set.seed(2L)
n <- 1000L
X <- matrix(rnorm(n * 3L), nrow = n, ncol = 3L)
colnames(X) <- c("a", "b", "c")
score <- 2 * X[, "a"] - X[, "b"]

train <- seq_len(800L)
test <- setdiff(seq_len(n), train)

# --- binary ---------------------------------------------------------------
# Labels are 0/1, as LightGBM's `binary` objective expects.
y <- as.numeric(score + rnorm(n, sd = 0.5) > 0)

binary <- mojoboost(
  X[train, ], y[train],
  objective = "binary",
  params = list(num_leaves = 15L, learning_rate = 0.1),
  nrounds = 150L
)
print(binary)

probability <- predict(binary, X[test, ])          # response scale
log_odds <- predict(binary, X[test, ], type = "raw")
label <- predict(binary, X[test, ], type = "class")

cat(sprintf("binary accuracy %.4f\n", mean(label == y[test])))
cat(sprintf("probability is the sigmoid of the raw score: %s\n",
            isTRUE(all.equal(probability, 1 / (1 + exp(-log_odds))))))

# --- multiclass -----------------------------------------------------------
# Class labels must be the integers 0 .. num_class - 1. mojoboost's model
# format does not store a label mapping, so keep your own if the classes are
# not already coded that way.
levels_ <- c("low", "mid", "high")
k <- as.integer(cut(score, breaks = 3L, labels = FALSE)) - 1L

multi <- mojoboost(
  X[train, ], k[train],
  objective = "multiclass",
  params = list(num_class = 3L, num_leaves = 15L),
  nrounds = 100L
)
print(multi)

proba <- predict(multi, X[test, ])                 # nrow x 3, rows sum to 1
predicted <- predict(multi, X[test, ], type = "class")

cat(sprintf("multiclass accuracy %.4f\n", mean(predicted == k[test])))
cat(sprintf("row probabilities sum to 1: %s\n",
            isTRUE(all.equal(unname(rowSums(proba)), rep(1, length(test))))))
print(head(data.frame(
  predicted = levels_[predicted + 1L],
  actual = levels_[k[test] + 1L],
  round(proba, 3L)
)))
