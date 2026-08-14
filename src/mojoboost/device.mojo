"""Training device selection.

One vocabulary for choosing where training runs, shared by the Mojo API,
the CPython bindings, and the Python estimators:

- `cpu`: the dependable path. Float64 throughout, every objective, every
  entry point.
- `gpu`: device-resident tree growth (`train_gpu` in train_gpu.mojo). It
  raises when no accelerator is present, or when the workload is outside
  what the GPU path covers. It never silently falls back to the CPU.
- `auto`: the GPU when the complete GPU path covers the workload and a
  conservative size heuristic selects it, the CPU otherwise.

What the GPU path covers today is single-output training: squared error,
binary logistic, poisson, huber, quantile, and L1. Multiclass grows one
tree per class per round on the CPU only, so `gpu` raises for it and
`auto` chooses the CPU.

`auto`'s size heuristic is disabled by default, so `auto` currently always
resolves to the CPU. The only end-to-end GPU training measurement we have
(Apple M4, bench/bench_train_gpu.mojo) is slower than the CPU trainer, and
no benchmark on any device has established a workload size where the GPU
wins, so shipping a crossover threshold would be a performance claim
without evidence behind it. `MOJOBOOST_AUTO_MIN_CELLS` enables the
heuristic: an integer number of cells (n_rows * n_features) at or above
which `auto` chooses the GPU, with `0` meaning "whenever the GPU path
covers the workload". It is the knob for running the crossover benchmark
that would justify a default, and it is deliberately device independent:
no per vendor or per chip special cases live here.

`MOJOBOOST_DISABLE_GPU=1` makes this module report that no accelerator is
present: `gpu` raises and `auto` chooses the CPU on a machine that does
have one. It exists to exercise the unavailable-GPU path in tests and to
pin a mixed fleet to the CPU backend.

LightGBM difference: LightGBM spells this `device_type` and takes `cpu`,
`gpu`, or `cuda`, with no `auto`. mojoboost has a single portable GPU
backend rather than separate OpenCL and CUDA ones, so the value is `gpu`
for every accelerator, and `auto` is an addition.
"""

from std.os import getenv
from std.sys import has_accelerator

comptime CPU_DEVICE = 0
comptime GPU_DEVICE = 1
comptime AUTO_DEVICE = 2

# Cells (n_rows * n_features) at or above which `auto` chooses the GPU.
# Negative disables the heuristic, which is the default: see the module
# docstring for why there is no measured crossover to ship.
comptime AUTO_MIN_CELLS = -1


def gpu_available() -> Bool:
    """True when training can run on an accelerator: one was present when
    this build was compiled and `MOJOBOOST_DISABLE_GPU=1` is not set."""
    comptime if not has_accelerator():
        return False
    else:
        return getenv("MOJOBOOST_DISABLE_GPU") != "1"


def env_auto_min_cells() -> Int:
    """The `auto` size threshold in cells. Unset, negative, or unparsable
    means disabled, in which case `auto` always chooses the CPU."""
    var s = getenv("MOJOBOOST_AUTO_MIN_CELLS")
    if s.byte_length() == 0:
        return AUTO_MIN_CELLS
    try:
        return Int(s)
    except:
        return AUTO_MIN_CELLS


def parse_device(name: String) raises -> Int:
    """Device code for a public device name ("cpu", "gpu", or "auto")."""
    if name == "cpu":
        return CPU_DEVICE
    if name == "gpu":
        return GPU_DEVICE
    if name == "auto":
        return AUTO_DEVICE
    raise Error(
        "unknown device '", name, "'; expected 'cpu', 'gpu', or 'auto'"
    )


def device_name(device: Int) raises -> String:
    """Public device name for a device code."""
    if device == CPU_DEVICE:
        return "cpu"
    if device == GPU_DEVICE:
        return "gpu"
    if device == AUTO_DEVICE:
        return "auto"
    raise Error("unknown device code ", device)


def gpu_supports(n_outputs: Int) -> Bool:
    """Whether the complete GPU training path covers this workload.
    `n_outputs` is 1 for single-output training and the class count for
    multiclass."""
    return n_outputs == 1


def resolve_device(
    device: Int, n_rows: Int, n_features: Int, n_outputs: Int = 1
) raises -> Int:
    """Resolve a requested device to the backend that will actually run:
    CPU_DEVICE or GPU_DEVICE, never AUTO_DEVICE.

    CPU_DEVICE resolves to itself. GPU_DEVICE raises when no accelerator is
    available or when the GPU path does not cover the workload, rather than
    falling back. AUTO_DEVICE chooses the GPU only when it is available,
    covers the workload, and the size heuristic selects it.
    """
    if device == CPU_DEVICE:
        return CPU_DEVICE

    if device == GPU_DEVICE:
        if not gpu_available():
            raise Error(
                "device 'gpu' requested but no accelerator is available;"
                " use device 'cpu' or 'auto'"
            )
        if not gpu_supports(n_outputs):
            raise Error(
                "device 'gpu' does not support multiclass training;"
                " use device 'cpu' or 'auto'"
            )
        return GPU_DEVICE

    if device == AUTO_DEVICE:
        if not gpu_available() or not gpu_supports(n_outputs):
            return CPU_DEVICE
        var min_cells = env_auto_min_cells()
        if min_cells < 0:
            return CPU_DEVICE
        if n_rows * n_features < min_cells:
            return CPU_DEVICE
        return GPU_DEVICE

    raise Error("unknown device code ", device)
