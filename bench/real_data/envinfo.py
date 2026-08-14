"""Environment metadata for a result record.

A benchmark number without the machine it ran on is a rumour. Everything
here is collected once per run and copied into every record that run
produces, so a results file can be read a year later by somebody who was
not there.

The load average and the thermal state are collected because a laptop is
the likely machine. A run started while something else was compiling is
not wrong, it is just not comparable, and the record should say so rather
than leaving the reader to wonder.

Nothing here fails a run. Every probe that cannot answer records null and
the reason.
"""

import json
import os
import platform
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

#: Environment variables that change what is being measured.
WATCHED_ENV = (
    "MOJOBOOST_NUM_WORKERS",
    "MOJOBOOST_PARALLEL_MIN_OPS",
    "MOJOBOOST_DISABLE_GPU",
    "MOJOBOOST_BENCH_DATA",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _run(cmd, timeout=15):
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu():
    system = platform.system()
    info = {
        "arch": platform.machine(),
        "system": system,
        "release": platform.release(),
        "model": None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "memory_bytes": None,
        "performance_cores": None,
        "efficiency_cores": None,
    }
    if system == "Darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        for key, field in (
            ("hw.physicalcpu", "physical_cores"),
            ("hw.memsize", "memory_bytes"),
            ("hw.perflevel0.logicalcpu", "performance_cores"),
            ("hw.perflevel1.logicalcpu", "efficiency_cores"),
        ):
            value = _run(["sysctl", "-n", key])
            info[field] = int(value) if value and value.isdigit() else None
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        info["model"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemTotal"):
                        info["memory_bytes"] = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    return info


def _versions():
    out = {
        "python": sys.version.split()[0],
        "numpy": None,
        "scipy": None,
        "lightgbm": None,
        "mojo": None,
        "max": None,
    }
    for name in ("numpy", "scipy", "lightgbm"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = None
    version = _run(["mojo", "--version"])
    out["mojo"] = version
    # The MAX runtime version, which is what the accelerator path actually
    # links against. `mojo --version` does not always carry it.
    out["max"] = _run(["python", "-c", "import max; print(max.__version__)"])
    return out


def _git():
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
        "describe": _run(["git", "describe", "--always", "--dirty"]),
    }


def _accelerator():
    """What mojoboost thinks it can train on, asked of the built extension
    rather than guessed from the platform."""
    out = {"gpu_available": None, "probe_error": None, "backend": None}
    try:
        import mojoboost

        out["gpu_available"] = bool(mojoboost.gpu_available())
    except Exception as exc:
        out["probe_error"] = f"{type(exc).__name__}: {exc}"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        out["backend"] = "metal"
    elif out["gpu_available"]:
        out["backend"] = "unknown"
    return out


def _machine_state():
    """Conditions that make timings incomparable without making them
    wrong. Recorded, never acted on."""
    state = {"load_average_1m": None, "on_battery": None, "thermal_pressure": None}
    try:
        state["load_average_1m"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    if platform.system() == "Darwin":
        power = _run(["pmset", "-g", "batt"])
        if power:
            state["on_battery"] = "Battery Power" in power
        thermal = _run(["pmset", "-g", "therm"])
        if thermal:
            state["thermal_pressure"] = thermal.splitlines()[-1].strip()
    return state


def collect(extra=None):
    """The full environment block for a run."""
    info = {
        "collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": _cpu(),
        "versions": _versions(),
        "git": _git(),
        "accelerator": _accelerator(),
        "machine_state": _machine_state(),
        "env": {name: os.environ.get(name) for name in WATCHED_ENV},
        "argv": list(sys.argv),
    }
    if extra:
        info.update(extra)
    return info


def comparable_key(info):
    """The fields two runs must agree on before their timings may be put in
    the same table. report.py refuses to compare across differing keys."""
    cpu = info.get("cpu", {})
    versions = info.get("versions", {})
    return {
        "hostname": info.get("hostname"),
        "arch": cpu.get("arch"),
        "cpu_model": cpu.get("model"),
        "mojo": versions.get("mojo"),
        "lightgbm": versions.get("lightgbm"),
        "git_commit": (info.get("git") or {}).get("commit"),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
