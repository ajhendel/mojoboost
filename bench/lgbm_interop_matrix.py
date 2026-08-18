"""Widen the go/no-go: which REAL LightGBM files can our importer read?

One binary fixture passing is not coverage. This sweeps the axes a real
model varies on, so the caveat that gets rewritten afterwards is rewritten
to what was actually shown and not to what one case suggested.
"""
import sys, os, warnings
import numpy as np, lightgbm as lgb
sys.path.insert(0, '/Users/andrewhendel/CascadeProjects/mojotrees/python')
warnings.filterwarnings("ignore")
from mojotrees import lgbm_model_io as io

rng = np.random.default_rng(0)
n, f = 3000, 24
Xc = rng.normal(size=(n, f))
lin = Xc[:, 0] + 0.5 * Xc[:, 3] - 0.25 * Xc[:, 7]

def run(name, X, params, y, rounds=80, ds_kw=None):
    ds = lgb.Dataset(X, label=y, **(ds_kw or {}))
    p = dict(verbosity=-1, seed=7, num_leaves=31); p.update(params)
    bst = lgb.train(p, ds, num_boost_round=rounds)
    path = "/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/49bb4333-37b1-40aa-ad0a-20dacd241517/scratchpad/m_%s.txt" % name
    bst.save_model(path)
    why = io.unsupported_reason(path)
    if why:
        print("%-22s REFUSED  %s" % (name, why[:70])); return
    m = io.load_lightgbm_model(path)
    ref = np.asarray(bst.predict(X, raw_score=True)).reshape(-1)
    got = np.asarray(m.predict(X, raw_score=True)).reshape(-1)
    if ref.shape != got.shape:
        print("%-22s SHAPE    ref%s got%s" % (name, ref.shape, got.shape)); return
    d = np.abs(ref - got)
    tag = "BIT-IDENTICAL" if d.max() == 0.0 else "max|d|=%.2e" % d.max()
    print("%-22s ok  %s  (%d/%d exact, %d bytes)"
          % (name, tag, int((ref == got).sum()), len(ref), os.path.getsize(path)))

yb = (lin + rng.normal(scale=.3, size=n) > 0).astype(int)
run("binary", Xc, {"objective": "binary"}, yb)
run("regression", Xc, {"objective": "regression"}, lin + rng.normal(scale=.3, size=n))
run("regression_l1", Xc, {"objective": "regression_l1"}, lin)
run("poisson", Xc, {"objective": "poisson"}, rng.poisson(np.exp(lin / 3)))
run("multiclass_k5", Xc, {"objective": "multiclass", "num_class": 5},
    rng.integers(0, 5, size=n))
run("deep_d12", Xc, {"objective": "binary", "num_leaves": 512, "max_depth": 12}, yb)
run("stumps_d1", Xc, {"objective": "binary", "num_leaves": 2, "max_depth": 1}, yb)
run("many_bins_1024", Xc, {"objective": "binary", "max_bin": 1023}, yb)
run("few_bins_15", Xc, {"objective": "binary", "max_bin": 15}, yb)
run("l1l2_reg", Xc, {"objective": "binary", "lambda_l1": .5, "lambda_l2": 3.}, yb)

Xm = Xc.copy(); Xm[rng.random(Xm.shape) < 0.15] = np.nan
run("missing_15pct", Xm, {"objective": "binary"}, yb)

Xk = Xc.copy(); Xk[:, 1] = rng.integers(0, 12, size=n); Xk[:, 2] = rng.integers(0, 40, size=n)
run("categorical", Xk, {"objective": "binary"}, yb,
    ds_kw={"categorical_feature": [1, 2]})

Xs = Xc.copy(); Xs[rng.random(Xs.shape) < 0.9] = 0.0
run("sparse_90pct_zero", Xs, {"objective": "binary"}, yb)
run("single_tree", Xc, {"objective": "binary"}, yb, rounds=1)
