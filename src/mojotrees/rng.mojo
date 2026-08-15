"""The one splitmix64 authority.

Every sampler in mojotrees draws from a counter-based splitmix64 stream:
bagging (`bagging.mojo`), feature sampling (`sampling.mojo`), GOSS
(`goss.mojo`), DART drop selection (`boosting_dart.mojo`), stochastic
gradient rounding (`quantized_gradient.mojo`), and extra_trees thresholds
(`tree_parameters_extra.mojo`). Each of them used to carry its own copy of
the mixing function and its constants "so each sampler owns its stream";
the streams stay independent because each module derives its own stream
start, not because the mixer is duplicated. This module holds the mixer
once. Every function here is bit-for-bit the copy the samplers carried, so
seeds keep reproducing the bags, feature sets, and trees they always did.

    splitmix64(state)   the finalizer: a bijection with full avalanche, so
                        consecutive counter values look independent
    uniform(counter)    splitmix64(counter) >> 11, scaled by 2^-53, in [0, 1)
    GOLDEN              the golden-ratio increment 0x9E3779B97F4A7C15, used
                        both inside the mixer and by callers to spread stream
                        indices apart before mixing
    TWO_POW_NEG_53      2^-53, scaling a 53-bit integer into [0, 1)

Stream derivation (how a seed, round, tree, node, or feature becomes the
start of a counter stream) stays with each sampler: the derivations differ
on purpose and are part of each sampler's reproducibility contract.
"""

# The golden-ratio increment splitmix64 advances by; also the multiplier
# callers use to spread stream indices apart before mixing.
comptime GOLDEN = UInt64(0x9E3779B97F4A7C15)

# 2^-53, scaling a 53-bit integer into [0, 1).
comptime TWO_POW_NEG_53 = 1.0 / 9007199254740992.0


def splitmix64(state: UInt64) -> UInt64:
    """splitmix64's mixing function: a bijection with full avalanche, so
    consecutive counter values produce independent-looking draws."""
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def uniform(counter: UInt64) -> Float64:
    """Uniform in [0, 1) with 53 significant bits, from a counter value."""
    return Float64(splitmix64(counter) >> 11) * TWO_POW_NEG_53
