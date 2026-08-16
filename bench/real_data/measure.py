"""Timing, memory, model size, and digests.

The measurement rules this module encodes:

- Wall time and CPU time are both recorded for every phase. Their ratio is
  the only honest way to tell a four-thread run from a one-thread run that
  was told it had four, and it catches a thread setting that did not take.
- The first call of anything is measured separately and reported as
  warmup, never averaged into the steady state. A Mojo extension pays a
  shared-library load on first import and an accelerator pays kernel
  compilation on first launch. Both are real costs and both belong in the
  record, but neither is a per-iteration cost.
- Peak resident set is read from the operating system for the whole
  process, which is why the runner gives each measured run its own
  process. A peak measured in a process that already trained something
  else is the other model's peak.
- Repeats produce a list of samples, never a mean. Aggregation happens in
  report.py, where the spread can be shown next to the middle.

Nothing here decides whether a number is good. That is verify.py's job for
correctness and nobody's job for speed, which is the point.
"""

import hashlib
import os
import platform
import resource
import time

import numpy as np


class Phase:
    """Wall and CPU time for one measured phase.

    Use as a context manager. `elapsed_s` is wall time from
    `time.perf_counter`, which is monotonic and unaffected by clock
    adjustments, and `cpu_s` is `time.process_time`, which counts every
    thread of this process.
    """

    def __init__(self, name):
        self.name = name
        self.elapsed_s = None
        self.cpu_s = None

    def __enter__(self):
        self._wall0 = time.perf_counter()
        self._cpu0 = time.process_time()
        return self

    def __exit__(self, *exc):
        self.elapsed_s = time.perf_counter() - self._wall0
        self.cpu_s = time.process_time() - self._cpu0
        return False

    def as_dict(self):
        return {
            "elapsed_s": self.elapsed_s,
            "cpu_s": self.cpu_s,
            "parallel_efficiency": (
                None
                if not self.elapsed_s
                else self.cpu_s / self.elapsed_s
            ),
        }


def peak_rss_bytes():
    """Peak resident set of this process, in bytes.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux. Getting this
    wrong by a factor of 1024 is the classic way to publish a memory
    comparison that is nonsense, so it is handled here once.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if platform.system() == "Darwin" else int(raw) * 1024


def digest(array):
    """A sha256 over an array's exact bytes, as float64 and C-contiguous.

    This is the determinism check. Two runs that produce the same digest
    produced bit-identical predictions; two that do not, did not, however
    close their metrics look.
    """
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


#: The pieces a feature matrix is hashed over, in the order they are fed to
#: the hash. Dense matrices are one contiguous block; a CSC matrix is its
#: three vectors, because the dense form of a 50,000-column matrix is not a
#: thing this harness is going to materialize to take a digest of.
def matrix_pieces(x):
    if hasattr(x, "tocsc"):
        return [x.data, x.indices, x.indptr]
    return [np.ascontiguousarray(x)]


def matrix_digest(x):
    """sha256 over a feature matrix's exact bytes, in its own dtype.

    Unlike `digest` above, this does NOT cast to float64. It hashes what it
    is given, which is the point: the question it answers is "are these the
    same bytes", and a cast would answer a weaker one.
    """
    h = hashlib.sha256()
    for piece in matrix_pieces(x):
        h.update(np.ascontiguousarray(piece).tobytes())
    return h.hexdigest()


def canonical_digest(x, y, group=None):
    """The digest of a scenario's canonical form: matrix, then label, then
    group, fed to one running sha256.

    Lives here rather than in `worker` because it has TWO callers and the
    whole value of it is that they are the same function. `worker.data_
    digest` computes it from the canonical data before any engine sees it.
    `engines._catboost_categorical_frame` computes it again from the
    canonical matrix reconstructed out of the mixed frame CatBoost was
    handed, which is how "the re-encoding did not change a value" becomes a
    measurement instead of a claim. A proof that hashes a different way,
    or in a different order, or with a different cast, proves nothing.

    The byte stream is unchanged from the inline version in `worker` that
    wrote every record in bench/results. Moving it must never change the
    value: a digest whose definition drifts silently makes old records and
    new ones incomparable, which is a worse failure than the one the digest
    was added to catch.
    """
    pieces = list(matrix_pieces(x))
    pieces.append(np.asarray(y, dtype=np.float64))
    if group is not None:
        pieces.append(np.asarray(group, dtype=np.int64))
    h = hashlib.sha256()
    for piece in pieces:
        h.update(np.ascontiguousarray(piece).tobytes())
    return h.hexdigest()


def file_digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def timed(fn, *args, **kwargs):
    """Run `fn` once and return (result, Phase)."""
    phase = Phase(getattr(fn, "__name__", "call"))
    with phase:
        result = fn(*args, **kwargs)
    return result, phase


def repeat(fn, times, warmup=1):
    """Call `fn` `warmup + times` times and return (last_result, samples).

    `samples` holds the warmup calls and the measured calls separately, so
    a report can show both and a reader can see how large the warmup was
    rather than being told it was excluded.
    """
    warmup_samples, measured = [], []
    result = None
    for i in range(warmup + times):
        result, phase = timed(fn)
        (warmup_samples if i < warmup else measured).append(phase.as_dict())
    return result, {"warmup": warmup_samples, "measured": measured}


def model_size(booster, scratch_dir=None):
    """Model size in bytes, both as a serialised string and as a file.

    The two can differ: a text format writes a trailing newline, a binary
    format does not, and one engine's file may carry metadata the other's
    string does not. Both are recorded rather than picking a favourite.
    """
    out = {"string_bytes": None, "file_bytes": None, "error": None}
    try:
        out["string_bytes"] = len(booster.model_to_string().encode("utf-8"))
    except Exception as exc:  # pragma: no cover - engine-dependent
        out["error"] = f"model_to_string: {type(exc).__name__}: {exc}"
    if scratch_dir is not None:
        path = os.path.join(scratch_dir, "model.txt")
        try:
            booster.save_model(path)
            out["file_bytes"] = os.path.getsize(path)
            out["file_sha256"] = file_digest(path)
        except Exception as exc:  # pragma: no cover - engine-dependent
            out["error"] = (out["error"] or "") + (
                f" save_model: {type(exc).__name__}: {exc}"
            )
        finally:
            if os.path.exists(path):
                os.remove(path)
    return out


def unavailable(reason):
    """A measurement that could not be taken, recorded as null plus a
    reason.

    Every field in a result record is either a number that was measured or
    this. There is no third state where a plausible value stands in for a
    measurement nobody made.
    """
    return {"value": None, "unavailable_reason": reason}


def measured(value, unit):
    return {"value": value, "unit": unit}
