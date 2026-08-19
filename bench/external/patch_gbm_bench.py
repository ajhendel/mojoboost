#!/usr/bin/env python3
"""Make a gbm-bench checkout runnable on a machine with no CUDA, and
register the two mojotrees arms.

gbm-bench is NVIDIA's harness and assumes NVIDIA hardware: `algorithms.py`
imports `dask_cuda` and `xgboost` at module scope, so on an Apple silicon Mac
it fails at import before any benchmark runs. This script makes exactly those
imports optional and adds two names to the algorithm factory. It changes no
timing code, no metric, no dataset, and no parameter belonging to another
library, which is the property that makes a number from this harness worth
more than a number from ours.

Every edit is anchored to an exact upstream string and asserted. If upstream
moves, this fails loudly rather than silently patching the wrong thing. Run
it as many times as you like; it detects its own work and stops.
"""

import io
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "# --- mojotrees arm (bench/external/patch_gbm_bench.py) ---"


def _read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


def _write(path, text):
    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _replace_once(text, old, new, what):
    if old not in text:
        raise SystemExit(
            "gbm-bench has moved: could not find the anchor for " + what +
            ".\nExpected to find:\n" + old +
            "\nRe-read algorithms.py and update patch_gbm_bench.py rather "
            "than forcing this through."
        )
    if text.count(old) != 1:
        raise SystemExit("anchor for " + what + " is not unique; refusing")
    return text.replace(old, new, 1)


def patch_algorithms(root):
    path = os.path.join(root, "algorithms.py")
    text = _read(path)
    if MARKER in text:
        print("algorithms.py already patched")
        return

    # 1. CUDA-only imports become optional so the module loads off-NVIDIA.
    text = _replace_once(
        text,
        "import dask.dataframe as dd\n"
        "import dask.array as da\n"
        "from dask.distributed import Client\n"
        "from dask_cuda import LocalCUDACluster\n"
        "import xgboost as xgb\n",
        MARKER + "\n"
        "# These four are CUDA-only or CUDA-adjacent and are not installable\n"
        "# on every machine this harness now runs on. Made optional so the\n"
        "# module imports; every algorithm that needs one still fails at\n"
        "# construction time if it is missing.\n"
        "try:\n"
        "    import dask.dataframe as dd\n"
        "    import dask.array as da\n"
        "    from dask.distributed import Client\n"
        "except ImportError:\n"
        "    dd = da = Client = None\n"
        "try:\n"
        "    from dask_cuda import LocalCUDACluster\n"
        "except ImportError:\n"
        "    LocalCUDACluster = None\n"
        "try:\n"
        "    import xgboost as xgb\n"
        "except ImportError:\n"
        "    xgb = None\n",
        "the CUDA-only import block",
    )

    # 2. Register the two mojotrees arms in the factory.
    text = _replace_once(
        text,
        "        raise ValueError(\"Unknown algorithm: \" + name)",
        "        " + MARKER + "\n"
        "        if name == 'mojotrees-cpu':\n"
        "            return MojoTreesCPUAlgorithm()\n"
        "        if name == 'mojotrees-gpu':\n"
        "            return MojoTreesGPUAlgorithm()\n"
        "        if name == 'mojotrees-gpu-compact':\n"
        "            return MojoTreesGPUCompactAlgorithm()\n"
        "        if name == 'lgbm-cpu-det':\n"
        "            return LgbmCPUDeterministicAlgorithm()\n"
        "        raise ValueError(\"Unknown algorithm: \" + name)",
        "the algorithm factory",
    )

    # 3. Import the adapters. They subclass Algorithm, so this goes at the
    #    end of the file, after the base class and shared_params exist.
    text = text.rstrip("\n") + "\n\n\n" + MARKER + "\n" + \
        "from mojotrees_algorithm import (  # noqa: E402\n" \
        "    MojoTreesCPUAlgorithm,\n" \
        "    MojoTreesGPUAlgorithm,\n" \
        "    MojoTreesGPUCompactAlgorithm,\n" \
        "    LgbmCPUDeterministicAlgorithm,\n" \
        ")\n"

    _write(path, text)
    print("patched algorithms.py")


def install_adapter(root):
    src = os.path.join(HERE, "gbm_bench", "mojotrees_algorithm.py")
    dst = os.path.join(root, "mojotrees_algorithm.py")
    shutil.copyfile(src, dst)
    print("installed mojotrees_algorithm.py")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_gbm_bench.py <path-to-gbm-bench>")
    root = os.path.abspath(sys.argv[1])
    if not os.path.isfile(os.path.join(root, "algorithms.py")):
        raise SystemExit("not a gbm-bench checkout: " + root)
    install_adapter(root)
    patch_algorithms(root)
    print("\nready. mojotrees-cpu, mojotrees-gpu and mojotrees-gpu-compact "
          "are now valid -algorithm values.")


if __name__ == "__main__":
    main()
