"""Deterministic data generators shared by the test suite.

Every test file needs the same three things: a seeded integer stream, a
uniform in [0, 1) drawn from it, and a column-major feature matrix filled
from that uniform.  Before this module each file carried its own copy, and
`_splitmix64` and `_uniform` were duplicated twenty-six and twenty-five
times respectively, byte for byte.  Twenty-six copies of a generator is
twenty-six definitions of what "deterministic" means, and they only stay in
agreement by accident.

Reach these with `-I tests`, which `tools/run_tests.sh` passes on every
invocation:

```mojo
from support import _make_features, _splitmix64, _uniform
```

Helpers that are genuinely per-file stay per-file.  `_params`, `_target`,
and `_row` look duplicated by name but differ in signature and in what they
model from one test to the next, and collapsing them here would hide that.
"""


def _splitmix64(state: UInt64) -> UInt64:
    """One round of SplitMix64.  Fixed constants, so a seed pins a stream."""
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    """A Float64 in [0, 1) from the top 53 bits of `_splitmix64(counter)`."""
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_features(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major deterministic features in [0, 1)."""
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    return features^
