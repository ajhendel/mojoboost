# mojoboost

Gradient boosted decision trees in [Mojo](https://www.modular.com/mojo),
with a scikit-learn style Python API backed by a CPython extension module
built from the same Mojo code.

mojoboost is a from-scratch GBDT library in the LightGBM family. It uses
histogram-based split finding, leaf-wise (best-first) tree growth, and
defaults matched to LightGBM. Objectives include squared error, binary
logistic, Poisson, and multiclass softmax, with sample weights and
bit-exact model save/load.

## Usage

```python
from mojoboost import MojoBoostRegressor, MojoBoostClassifier

model = MojoBoostRegressor(num_leaves=31, n_estimators=100).fit(X, y)
pred = model.predict(X)          # numpy in/out when numpy is available
model.save("model.mbst")
model = MojoBoostRegressor.load("model.mbst")

clf = MojoBoostClassifier().fit(X, labels)   # binary or multiclass by labels
proba = clf.predict_proba(X)
```

`fit` accepts `sample_weight`. numpy is optional; plain Python sequences
work without it. Install with the `numpy` extra to pull it in.

## Device selection

```python
from mojoboost import MojoBoostRegressor, gpu_available

model = MojoBoostRegressor(device="auto").fit(X, y)
model.device_          # the backend that actually ran: "cpu" or "gpu"
```

`device="cpu"` is the default and the dependable backend. `device="gpu"`
raises when no accelerator is available or when the GPU path does not
cover the workload (multiclass is CPU-only), rather than falling back
silently. `device="auto"` picks for you and currently always picks the
CPU: no benchmark has established a workload size where end-to-end GPU
training wins, so no crossover threshold ships enabled.
`gpu_available()` reports whether this build can train on an accelerator.

The device is a training choice rather than part of the model, so saved
models are identical either way and a loaded estimator carries no
`device_`.

## Platform support

Wheels currently target macOS on Apple silicon. The wheel bundles the Mojo
runtime libraries it needs, so no Mojo or MAX installation is required at
runtime. On other platforms, build from source with
[pixi](https://pixi.sh) using the instructions in the repository.

## Links

Source, benchmarks against LightGBM, and the native Mojo API:
[github.com/ajhendel/mojoboost](https://github.com/ajhendel/mojoboost)

## License

Apache-2.0
