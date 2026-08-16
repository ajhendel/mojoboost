"""C3: LightGBM's row-wise builder against its column-wise builder, interleaved.

Answers the question the CPU round's C3 rule was rewritten around. Our
comparator pins `force_row_wise=True` in `scenarios.LIGHTGBM_ALIGNMENT`, so
every LightGBM figure this project has recorded was taken on the row-wise
builder. This measures how much of LightGBM's lead at 1,000,000 x 50 is a
builder shape mojotrees does not have at all.

Both arms are pinned, so neither pays LightGBM's auto-selection cost, and the
difference between them is the builder alone.

Interleaved in one process, alternating arm by arm, per C-ops: both arms are
sampled repeatedly inside the same window so drift hits them alike. Samples are
reported in the order they ran so a monotone rise reads as a trend rather than
as a spread, per C8.

Reads bench/bench_lightgbm.py by path and reuses its data generator, its
parameter alignment and its timed region verbatim. Nothing here re-implements
the comparator.
"""

import importlib.util
import os
import statistics
import sys

REPO = "/Users/andrewhendel/CascadeProjects/mojotrees"
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
FEATURES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
ROUNDS = 100


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = load("bench_lightgbm", os.path.join(REPO, "bench", "bench_lightgbm.py"))

print(f"C3 builder A/B: {ROWS} rows x {FEATURES} features, "
      f"{THREADS} threads, {REPEATS} repeats, {ROUNDS} rounds")

X, y = bl.make_data(ROWS, FEATURES, "reg", 0)

ARMS = ["force_row_wise", "force_col_wise"]
datasets = {}
for arm in ARMS:
    params = bl.lgbm_params("reg", THREADS, ROWS)
    # The alignment pins row-wise. Clear both, then set exactly one, so the
    # two arms differ in the builder and in nothing else.
    params.pop("force_row_wise", None)
    params.pop("force_col_wise", None)
    params[arm] = True
    dataset, binning_s = bl.build_dataset(X, y, params)
    datasets[arm] = (params, dataset)
    print(f"{arm}_binning_s: {binning_s:.6f}")
    print(f"{arm}_params: {bl.params_summary(params)}")

samples = {arm: [] for arm in ARMS}
for rep in range(REPEATS):
    for arm in ARMS:
        params, dataset = datasets[arm]
        _, seconds = bl.train_once(params, dataset, ROUNDS)
        samples[arm].append(seconds)
        print(f"run {rep + 1} {arm} train_s: {seconds:.6f}", flush=True)

print()
medians = {}
spreads = {}
for arm in ARMS:
    lo, med, hi, spread = bl.reduce_samples(samples[arm])
    medians[arm] = med
    spreads[arm] = spread * 100.0
    print(f"{arm}_train_s_samples: " + " ".join(f"{v:.6f}" for v in samples[arm]))
    print(f"{arm}_train_s_median: {med:.6f}")
    print(f"{arm}_train_s_min: {lo:.6f}")
    print(f"{arm}_train_s_max: {hi:.6f}")
    print(f"{arm}_spread_pct: {spread * 100.0:.1f}")

row, col = medians["force_row_wise"], medians["force_col_wise"]
delta_pct = abs(col - row) / min(col, row) * 100.0
floor = max(spreads.values())
# M0: resolved when the medians differ by more than the wider arm's own spread.
verdict = "resolved" if delta_pct > floor else "indistinguishable"
faster = "force_row_wise" if row < col else "force_col_wise"
print()
print(f"builder_delta_pct: {delta_pct:.1f}")
print(f"builder_noise_floor_pct: {floor:.1f}")
print(f"builder_faster: {faster}")
print(f"builder_verdict: {verdict}")
