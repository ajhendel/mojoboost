/* Tests for the mojotrees C ABI, written as a C caller would use it.
 *
 * Covers the three things a C boundary gets wrong that a Mojo test cannot
 * see: handle lifecycle, invalid input from a language with no checks, and
 * ownership over many create/destroy cycles.
 *
 * Build and run with capi/run_c_tests.sh. Under a leak checker:
 *
 *     leaks --atExit -- ./capi/test_capi          # macOS
 *     valgrind --leak-check=full ./capi/test_capi # Linux
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mojotrees.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond, what)                                                 \
    do {                                                                  \
        checks++;                                                         \
        if (!(cond)) {                                                    \
            failures++;                                                   \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, (what));       \
        }                                                                 \
    } while (0)

#define CHECK_OK(rc, err, what)                                           \
    do {                                                                  \
        checks++;                                                         \
        if ((rc) != MOJOTREES_OK) {                                       \
            failures++;                                                   \
            printf("FAIL %s:%d: %s: rc=%d %s\n", __FILE__, __LINE__,      \
                   (what), (int)(rc), mojotrees_error_message(err));      \
        }                                                                 \
    } while (0)

/* Deterministic data so a failure is reproducible: splitmix64, the same
 * generator the benchmarks use. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static double next_uniform(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return (double)(z >> 11) / 9007199254740992.0;
}

#define N_ROWS 200
#define N_FEATURES 4

static double x[N_ROWS * N_FEATURES];
static double y_reg[N_ROWS];
static double y_bin[N_ROWS];
static double y_multi[N_ROWS];

static void make_data(void) {
    for (int f = 0; f < N_FEATURES; f++) {
        for (int r = 0; r < N_ROWS; r++) {
            x[f * N_ROWS + r] = next_uniform();
        }
    }
    for (int r = 0; r < N_ROWS; r++) {
        double signal = 2.0 * x[0 * N_ROWS + r] - x[1 * N_ROWS + r];
        y_reg[r] = signal + 0.01 * next_uniform();
        y_bin[r] = signal > 0.5 ? 1.0 : 0.0;
        y_multi[r] = (double)((int)(3.0 * x[0 * N_ROWS + r]) % 3);
    }
}

/* ------------------------------------------------------------- lifecycle */

static void test_version(void) {
    CHECK(mojotrees_abi_version() == MOJOTREES_ABI_VERSION,
          "abi version matches the header");

    int32_t major = -1, minor = -1, patch = -1;
    mojotrees_library_version(&major, &minor, &patch);
    CHECK(major >= 0 && minor >= 0 && patch >= 0,
          "library version is populated");
    /* Every out pointer is optional. */
    mojotrees_library_version(NULL, NULL, NULL);
    mojotrees_library_version(&major, NULL, NULL);
}

static void test_error_lifecycle(void) {
    MojoTreesError *err = mojotrees_error_create();
    CHECK(err != NULL, "error object is created");
    CHECK(mojotrees_error_message(err) != NULL, "fresh message is not NULL");
    CHECK(strlen(mojotrees_error_message(err)) == 0, "fresh message is empty");

    /* A failure fills the message in. */
    MojoTreesModel *model = NULL;
    int32_t rc = mojotrees_train_dense(NULL, N_ROWS, N_FEATURES, y_reg, NULL,
                                       NULL, &model, err);
    CHECK(rc == MOJOTREES_ERROR_INVALID_ARGUMENT, "NULL data is rejected");
    CHECK(strlen(mojotrees_error_message(err)) > 0, "message is set");

    /* A later success clears it again. */
    rc = mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                               "num_iterations=2", &model, err);
    CHECK_OK(rc, err, "train after a failure");
    CHECK(strlen(mojotrees_error_message(err)) == 0,
          "a successful call clears the message");
    mojotrees_model_free(model);

    /* NULL is accepted everywhere it can be. */
    CHECK(mojotrees_error_message(NULL) == NULL, "message of NULL is NULL");
    mojotrees_error_free(NULL);
    mojotrees_model_free(NULL);
    mojotrees_error_free(err);
}

static void test_error_is_optional(void) {
    /* Passing no error object must work on both paths. */
    MojoTreesModel *model = NULL;
    int32_t rc = mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                       "num_iterations=2", &model, NULL);
    CHECK(rc == MOJOTREES_OK, "train with a NULL error object");
    mojotrees_model_free(model);

    model = NULL;
    rc = mojotrees_train_dense(x, 0, N_FEATURES, y_reg, NULL, NULL, &model,
                               NULL);
    CHECK(rc == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "failure with a NULL error object");
    CHECK(model == NULL, "out_model is untouched on failure");
}

/* ------------------------------------------------------------- happy path */

static void test_train_predict_regression(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    int32_t rc = mojotrees_train_dense(
        x, N_ROWS, N_FEATURES, y_reg, NULL,
        "objective=regression num_iterations=20 learning_rate=0.2"
        " num_leaves=7 min_data_in_leaf=5",
        &model, err);
    CHECK_OK(rc, err, "train regression");

    int64_t n_features = 0, n_classes = 0, n_trees = 0;
    CHECK_OK(mojotrees_model_num_features(model, &n_features, err), err,
             "num_features");
    CHECK_OK(mojotrees_model_num_classes(model, &n_classes, err), err,
             "num_classes");
    CHECK_OK(mojotrees_model_num_trees(model, &n_trees, err), err,
             "num_trees");
    CHECK(n_features == N_FEATURES, "feature count round-trips");
    CHECK(n_classes == 1, "a single-output model predicts one value");
    CHECK(n_trees == 20, "one tree per iteration");

    double pred[N_ROWS];
    CHECK_OK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, pred, N_ROWS,
                               err),
             err, "predict");

    /* The model has to have learned the signal, not just returned. */
    double sse = 0.0, sst = 0.0, mean = 0.0;
    for (int r = 0; r < N_ROWS; r++) mean += y_reg[r] / N_ROWS;
    for (int r = 0; r < N_ROWS; r++) {
        sse += (pred[r] - y_reg[r]) * (pred[r] - y_reg[r]);
        sst += (y_reg[r] - mean) * (y_reg[r] - mean);
    }
    CHECK(sse < 0.5 * sst, "predictions beat the mean");

    /* For squared error the response scale and the raw score are the
     * same, which is the cheapest check that both entry points run. */
    double raw[N_ROWS];
    CHECK_OK(mojotrees_predict_raw(model, x, N_ROWS, N_FEATURES, raw, N_ROWS,
                                   err),
             err, "predict_raw");
    int identical = 1;
    for (int r = 0; r < N_ROWS; r++) {
        if (raw[r] != pred[r]) identical = 0;
    }
    CHECK(identical, "regression raw scores equal response-scale values");

    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

static void test_train_predict_binary(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_bin, NULL,
                                   "objective=binary num_iterations=20"
                                   " num_leaves=7 min_data_in_leaf=5",
                                   &model, err),
             err, "train binary");

    double proba[N_ROWS], raw[N_ROWS];
    CHECK_OK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, proba, N_ROWS,
                               err),
             err, "predict binary");
    CHECK_OK(mojotrees_predict_raw(model, x, N_ROWS, N_FEATURES, raw, N_ROWS,
                                   err),
             err, "predict_raw binary");

    int in_range = 1, matches_sigmoid = 1, correct = 0;
    for (int r = 0; r < N_ROWS; r++) {
        if (!(proba[r] > 0.0 && proba[r] < 1.0)) in_range = 0;
        double expected = 1.0 / (1.0 + exp(-raw[r]));
        if (fabs(expected - proba[r]) > 1e-9) matches_sigmoid = 0;
        if ((proba[r] > 0.5) == (y_bin[r] > 0.5)) correct++;
    }
    CHECK(in_range, "binary predictions are probabilities");
    CHECK(matches_sigmoid, "response scale is the sigmoid of the raw score");
    CHECK(correct > (int)(0.9 * N_ROWS), "binary model fits the labels");

    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

static void test_train_predict_multiclass(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_multi, NULL,
                                   "objective=multiclass num_class=3"
                                   " num_iterations=10 num_leaves=7"
                                   " min_data_in_leaf=5",
                                   &model, err),
             err, "train multiclass");

    int64_t n_classes = 0, n_trees = 0;
    CHECK_OK(mojotrees_model_num_classes(model, &n_classes, err), err,
             "multiclass num_classes");
    CHECK_OK(mojotrees_model_num_trees(model, &n_trees, err), err,
             "multiclass num_trees");
    CHECK(n_classes == 3, "num_classes is the class count");
    CHECK(n_trees == 30, "one tree per class per iteration");

    double proba[N_ROWS * 3];
    CHECK_OK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, proba,
                               N_ROWS * 3, err),
             err, "predict multiclass");
    int normalized = 1, correct = 0;
    for (int r = 0; r < N_ROWS; r++) {
        double total = 0.0;
        int best = 0;
        for (int k = 0; k < 3; k++) {
            total += proba[r * 3 + k];
            if (proba[r * 3 + k] > proba[r * 3 + best]) best = k;
        }
        if (fabs(total - 1.0) > 1e-9) normalized = 0;
        if (best == (int)y_multi[r]) correct++;
    }
    CHECK(normalized, "class probabilities sum to one");
    CHECK(correct > (int)(0.9 * N_ROWS), "multiclass model fits the labels");

    /* An output buffer sized for one value per row is too small here. */
    double narrow[N_ROWS];
    CHECK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, narrow, N_ROWS,
                            err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "multiclass needs n_rows * num_classes of output space");

    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

/* ------------------------------------------------------------- save/load */

static void round_trip(const char *params, const char *path, int width) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    const double *labels = width == 1 ? y_reg : y_multi;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, labels, NULL,
                                   params, &model, err),
             err, "train for round trip");

    double *before = malloc(sizeof(double) * N_ROWS * width);
    double *after = malloc(sizeof(double) * N_ROWS * width);
    CHECK_OK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, before,
                               N_ROWS * width, err),
             err, "predict before save");
    CHECK_OK(mojotrees_save_model(model, path, err), err, "save");
    mojotrees_model_free(model);

    MojoTreesModel *loaded = NULL;
    CHECK_OK(mojotrees_load_model(path, &loaded, err), err, "load");
    int64_t n_classes = 0;
    CHECK_OK(mojotrees_model_num_classes(loaded, &n_classes, err), err,
             "loaded num_classes");
    CHECK(n_classes == width, "the file says which kind of model it holds");
    CHECK_OK(mojotrees_predict(loaded, x, N_ROWS, N_FEATURES, after,
                               N_ROWS * width, err),
             err, "predict after load");

    int identical = 1;
    for (int i = 0; i < N_ROWS * width; i++) {
        if (before[i] != after[i]) identical = 0;
    }
    CHECK(identical, "a reloaded model predicts bit-identically");

    free(before);
    free(after);
    mojotrees_model_free(loaded);
    mojotrees_error_free(err);
    remove(path);
}

static void test_save_load(void) {
    round_trip("objective=regression num_iterations=10 num_leaves=7",
               "capi_test_single.mbst", 1);
    round_trip("objective=multiclass num_class=3 num_iterations=5"
               " num_leaves=7",
               "capi_test_multi.mbst", 3);
}

/* --------------------------------------------------------- invalid input */

static void test_invalid_training_input(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    MojoTreesModel *sentinel = (MojoTreesModel *)0x1;

    CHECK(mojotrees_train_dense(x, 0, N_FEATURES, y_reg, NULL, NULL, &model,
                                err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "n_rows must be positive");
    CHECK(mojotrees_train_dense(x, N_ROWS, 0, y_reg, NULL, NULL, &model,
                                err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "n_features must be positive");
    CHECK(mojotrees_train_dense(x, -1, N_FEATURES, y_reg, NULL, NULL, &model,
                                err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "negative n_rows is rejected");
    CHECK(mojotrees_train_dense(NULL, N_ROWS, N_FEATURES, y_reg, NULL, NULL,
                                &model, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL data is rejected");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, NULL, NULL, NULL,
                                &model, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL labels are rejected");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL, NULL,
                                NULL, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL out_model is rejected");

    /* Nothing above may have written a handle anywhere. */
    CHECK(model == NULL, "out_model stays NULL across every failure");

    /* Parameter strings. */
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "num_leaves", &sentinel, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "a parameter needs a value");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "not_a_parameter=1", &sentinel, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "unknown parameters are rejected");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "num_leaves=1", &sentinel, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "num_leaves is range checked");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "objective=nonesuch", &sentinel, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "unknown objectives are rejected");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "objective=multiclass", &sentinel, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "multiclass requires num_class");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "bagging_fraction=0.5", &sentinel, err) ==
              MOJOTREES_ERROR_UNSUPPORTED,
          "a Mojo-API-only parameter reports unsupported, not unknown");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "objective=lambdarank", &sentinel, err) ==
              MOJOTREES_ERROR_UNSUPPORTED,
          "ranking needs query groups a parameter string cannot carry");
    CHECK(sentinel == (MojoTreesModel *)0x1,
          "a rejected parameter string never writes out_model");

    /* Labels the objective cannot accept are a training error, not an
     * argument error: the arguments were well formed. */
    double negative[N_ROWS];
    for (int r = 0; r < N_ROWS; r++) negative[r] = -1.0;
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, negative, NULL,
                                "objective=poisson", &sentinel, err) ==
              MOJOTREES_ERROR_TRAINING,
          "poisson rejects negative labels");
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                "objective=multiclass num_class=3",
                                &sentinel, err) == MOJOTREES_ERROR_TRAINING,
          "multiclass rejects non-integer labels");
    CHECK(sentinel == (MojoTreesModel *)0x1,
          "a failed fit never writes out_model");

    mojotrees_error_free(err);
}

static void test_invalid_predict_input(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                   "num_iterations=5", &model, err),
             err, "train for predict validation");

    double out[N_ROWS];
    CHECK(mojotrees_predict(NULL, x, N_ROWS, N_FEATURES, out, N_ROWS, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL model is rejected");
    CHECK(mojotrees_predict(model, NULL, N_ROWS, N_FEATURES, out, N_ROWS,
                            err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL data is rejected");
    CHECK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, NULL, N_ROWS,
                            err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL output buffer is rejected");
    CHECK(mojotrees_predict(model, x, N_ROWS, N_FEATURES - 1, out, N_ROWS,
                            err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "the feature count must match the model");
    CHECK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, out, N_ROWS - 1,
                            err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "a short output buffer is rejected");
    CHECK(mojotrees_predict_raw(NULL, x, N_ROWS, N_FEATURES, out, N_ROWS,
                                err) == MOJOTREES_ERROR_INVALID_ARGUMENT,
          "predict_raw validates too");

    /* A rejected prediction must not have written into the buffer. */
    for (int r = 0; r < N_ROWS; r++) out[r] = -12345.0;
    (void)mojotrees_predict(model, x, N_ROWS, N_FEATURES, out, N_ROWS - 1,
                            err);
    CHECK(out[0] == -12345.0, "a rejected prediction writes nothing");

    /* Accessors. */
    int64_t value = 0;
    CHECK(mojotrees_model_num_features(NULL, &value, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "accessors reject a NULL model");
    CHECK(mojotrees_model_num_classes(model, NULL, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "accessors reject a NULL out pointer");

    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

static void test_invalid_io(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    MojoTreesModel *sentinel = (MojoTreesModel *)0x1;

    CHECK(mojotrees_load_model("capi_test_no_such_model.mbst", &model,
                               err) == MOJOTREES_ERROR_IO,
          "loading a missing file is an I/O error");
    CHECK(model == NULL, "a failed load writes no handle");
    CHECK(mojotrees_load_model(NULL, &model, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL path is rejected");
    CHECK(mojotrees_load_model("", &model, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "empty path is rejected");
    CHECK(mojotrees_load_model("x.mbst", NULL, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL out_model is rejected");

    /* A file that exists but is not a model. */
    FILE *f = fopen("capi_test_garbage.txt", "w");
    if (f) {
        fputs("this is not a model file\n", f);
        fclose(f);
        CHECK(mojotrees_load_model("capi_test_garbage.txt", &sentinel, err) ==
                  MOJOTREES_ERROR_IO,
              "a file that is not a model is rejected");
        CHECK(sentinel == (MojoTreesModel *)0x1,
              "a rejected load leaves out_model alone");
        remove("capi_test_garbage.txt");
    }

    CHECK(mojotrees_save_model(NULL, "x.mbst", err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "saving a NULL model is rejected");

    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                   "num_iterations=2", &model, err),
             err, "train for save validation");
    CHECK(mojotrees_save_model(model, NULL, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "NULL path is rejected");
    CHECK(mojotrees_save_model(model, "", err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "empty path is rejected");
    /* Saving creates missing parent directories, so an unwritable path
     * has to be one that cannot be a directory at all. */
    CHECK(mojotrees_save_model(model, "/dev/null/model.mbst", err) ==
              MOJOTREES_ERROR_IO,
          "an unwritable path is an I/O error");
    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

/* ---------------------------------------------------------------- leaks */

static void test_handle_churn(void) {
    /* Every handle this creates is freed. Run the binary under leaks or
     * valgrind (see the header comment) to turn this into a leak
     * assertion; on its own it catches use-after-free and double-free
     * crashes, and allocator growth that would abort the process. */
    for (int i = 0; i < 2000; i++) {
        MojoTreesError *err = mojotrees_error_create();
        /* Write a message on half of them so the message buffer is
         * reallocated and freed, not just the empty initial one. */
        if (i % 2 == 0) {
            (void)mojotrees_predict(NULL, x, N_ROWS, N_FEATURES, NULL, 0,
                                    err);
        }
        mojotrees_error_free(err);
    }

    for (int i = 0; i < 25; i++) {
        MojoTreesError *err = mojotrees_error_create();
        MojoTreesModel *model = NULL;
        CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                       "num_iterations=3 num_leaves=7",
                                       &model, err),
                 err, "churn: train");
        double out[N_ROWS];
        CHECK_OK(mojotrees_predict(model, x, N_ROWS, N_FEATURES, out, N_ROWS,
                                   err),
                 err, "churn: predict");
        CHECK_OK(mojotrees_save_model(model, "capi_test_churn.mbst", err),
                 err, "churn: save");
        MojoTreesModel *loaded = NULL;
        CHECK_OK(mojotrees_load_model("capi_test_churn.mbst", &loaded, err),
                 err, "churn: load");
        mojotrees_model_free(loaded);
        mojotrees_model_free(model);
        mojotrees_error_free(err);
    }
    remove("capi_test_churn.mbst");
}

static void test_weights(void) {
    MojoTreesError *err = mojotrees_error_create();
    double weights[N_ROWS];
    for (int r = 0; r < N_ROWS; r++) weights[r] = r < N_ROWS / 2 ? 1.0 : 0.0;

    MojoTreesModel *weighted = NULL;
    MojoTreesModel *unweighted = NULL;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, weights,
                                   "num_iterations=5 num_leaves=7"
                                   " min_data_in_leaf=2",
                                   &weighted, err),
             err, "train weighted");
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                   "num_iterations=5 num_leaves=7"
                                   " min_data_in_leaf=2",
                                   &unweighted, err),
             err, "train unweighted");

    double a[N_ROWS], b[N_ROWS];
    CHECK_OK(mojotrees_predict(weighted, x, N_ROWS, N_FEATURES, a, N_ROWS,
                               err),
             err, "predict weighted");
    CHECK_OK(mojotrees_predict(unweighted, x, N_ROWS, N_FEATURES, b, N_ROWS,
                               err),
             err, "predict unweighted");
    int differs = 0;
    for (int r = 0; r < N_ROWS; r++) {
        if (a[r] != b[r]) differs = 1;
    }
    CHECK(differs, "sample weights reach the trainer");

    /* A weight vector of all zeros leaves nothing to fit. */
    double zeros[N_ROWS];
    for (int r = 0; r < N_ROWS; r++) zeros[r] = 0.0;
    MojoTreesModel *sentinel = (MojoTreesModel *)0x1;
    CHECK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, zeros, NULL,
                                &sentinel, err) == MOJOTREES_ERROR_TRAINING,
          "all-zero weights are a training error");

    mojotrees_model_free(weighted);
    mojotrees_model_free(unweighted);
    mojotrees_error_free(err);
}

/* Importance and the parameter key list, the two calls ABI version 3 added
 * for the R bindings. Both are what a binding needs to avoid keeping its own
 * copy of something the engine already knows. */
static void test_feature_importance(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_reg, NULL,
                                   "num_iterations=12 num_leaves=7", &model,
                                   err),
             err, "train for importance");

    double split[N_FEATURES], gain[N_FEATURES];
    CHECK_OK(mojotrees_feature_importance(model, MOJOTREES_IMPORTANCE_SPLIT,
                                          split, N_FEATURES, err),
             err, "split importance");
    CHECK_OK(mojotrees_feature_importance(model, MOJOTREES_IMPORTANCE_GAIN,
                                          gain, N_FEATURES, err),
             err, "gain importance");

    int splits = 0;
    double total_gain = 0.0;
    for (int f = 0; f < N_FEATURES; f++) {
        CHECK(split[f] >= 0.0, "a split count is not negative");
        CHECK(split[f] == (double)(long)split[f], "split counts are whole");
        CHECK(gain[f] >= 0.0, "a gain is not negative");
        splits += (int)split[f];
        total_gain += gain[f];
    }
    CHECK(splits > 0, "a fitted model splits on something");
    CHECK(total_gain > 0.0, "a fitted model records gain");

    /* A feature that is never split on has no gain either, so the two types
     * agree about which features the model actually used. */
    for (int f = 0; f < N_FEATURES; f++) {
        if (split[f] == 0.0) {
            CHECK(gain[f] == 0.0, "an unused feature has zero gain");
        }
    }

    /* Rejections. Nothing is written when the call fails. */
    double guard[N_FEATURES];
    for (int f = 0; f < N_FEATURES; f++) guard[f] = -7.0;
    CHECK(mojotrees_feature_importance(model, 2, guard, N_FEATURES, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "an unknown importance type is refused");
    CHECK(guard[0] == -7.0, "a refused type writes nothing");
    CHECK(strlen(mojotrees_error_message(err)) > 0,
          "the refusal explains itself");
    CHECK(mojotrees_feature_importance(model, MOJOTREES_IMPORTANCE_SPLIT,
                                       guard, N_FEATURES - 1, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "too small a buffer is refused");
    CHECK(guard[0] == -7.0, "a refused length writes nothing");
    CHECK(mojotrees_feature_importance(NULL, MOJOTREES_IMPORTANCE_SPLIT, guard,
                                       N_FEATURES, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "a NULL model is refused");
    CHECK(mojotrees_feature_importance(model, MOJOTREES_IMPORTANCE_SPLIT, NULL,
                                       N_FEATURES, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "a NULL buffer is refused");

    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

static void test_multiclass_importance_sums_over_classes(void) {
    MojoTreesError *err = mojotrees_error_create();
    MojoTreesModel *model = NULL;
    CHECK_OK(mojotrees_train_dense(x, N_ROWS, N_FEATURES, y_multi, NULL,
                                   "objective=multiclass num_class=3"
                                   " num_iterations=6",
                                   &model, err),
             err, "train multiclass for importance");

    int64_t trees = 0, iterations = 0;
    CHECK_OK(mojotrees_model_num_trees(model, &trees, err), err, "trees");
    CHECK_OK(mojotrees_model_num_iterations(model, &iterations, err), err,
             "iterations");
    CHECK(trees == iterations * 3, "one tree per class per iteration");

    double split[N_FEATURES];
    CHECK_OK(mojotrees_feature_importance(model, MOJOTREES_IMPORTANCE_SPLIT,
                                          split, N_FEATURES, err),
             err, "multiclass split importance");
    int total = 0;
    for (int f = 0; f < N_FEATURES; f++) total += (int)split[f];
    /* Summed over every tree, so over classes as well as iterations. A
     * per-class count could not exceed the single-class tree count. */
    CHECK(total > 0, "a multiclass model splits on something");

    mojotrees_model_free(model);
    mojotrees_error_free(err);
}

static void test_parameter_keys(void) {
    MojoTreesError *err = mojotrees_error_create();
    char *keys = NULL;
    CHECK_OK(mojotrees_parameter_keys(&keys, err), err, "parameter keys");
    CHECK(keys != NULL, "keys are returned");
    CHECK(strlen(keys) > 0, "keys are not empty");
    /* Spot check a few the parser really does accept, since the point of
     * this call is that a binding does not keep its own copy. */
    CHECK(strstr(keys, "objective") != NULL, "objective is listed");
    CHECK(strstr(keys, "num_iterations") != NULL, "num_iterations is listed");
    CHECK(strstr(keys, "learning_rate") != NULL, "learning_rate is listed");
    mojotrees_string_free(keys);

    CHECK(mojotrees_parameter_keys(NULL, err) ==
              MOJOTREES_ERROR_INVALID_ARGUMENT,
          "a NULL out pointer is refused");
    mojotrees_error_free(err);
}

int main(void) {
    make_data();

    test_version();
    test_error_lifecycle();
    test_error_is_optional();
    test_train_predict_regression();
    test_train_predict_binary();
    test_train_predict_multiclass();
    test_save_load();
    test_invalid_training_input();
    test_invalid_predict_input();
    test_invalid_io();
    test_weights();
    test_handle_churn();
    test_feature_importance();
    test_multiclass_importance_sums_over_classes();
    test_parameter_keys();

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
