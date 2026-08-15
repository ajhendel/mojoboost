#!/usr/bin/env python3
"""Static reachability audit: what in this repository is actually connected.

mojotrees grew as many parallel lanes, and a lane can finish a capability
without the capability ever becoming reachable. A module compiles, its tests
pass, its handoff is written, and no production entry point ever names it.
This script finds that state and nothing else. It answers one question, in
several shapes: **starting from the entry points a user can actually reach,
what code does the repository get to, and what does it not?**

The entry points, which are the roots of every graph below:

- `python/mojotrees/__init__.py` - the public Python package.
- `bindings/_mojotrees.mojo` - the CPython extension, whose `def_function`
  table is the entire Python-to-native surface.
- `src/mojotrees/__init__.mojo` - the native package's own re-export block,
  which is the public Mojo API.
- `capi/mojotrees_capi.mojo` - the C ABI, whose `@export`ed functions are the
  entire C surface.
- `cli/mojotrees_cli.mojo` - the `mojotrees` command line tool.

What it reports:

1.  **Orphan native modules.** A `.mojo` file under `src/mojotrees/` that no
    root reaches through any chain of imports. Annotated with whether a test
    or a benchmark reaches it, which is the difference between "dead" and
    "written, exercised, and never wired in".
2.  **Imported but never called.** A module that a reachable file imports and
    then never mentions again, and individual imported symbols that are never
    used in the importing file.
3.  **Duplicate registries and policies.** The same public name defined by
    more than one native module - two objective tables, two device
    vocabularies, two capability structs - which is how a second registry
    starts.
4.  **Public parameters with no downstream consumer.** A keyword argument on a
    Python estimator that is stored on `self` and never read again, and a
    native parameter field that nothing outside its own defining module ever
    sets.
5.  **Binding functions with no Python caller**, and, before that, **binding
    modules the extension entry point never imports** - a sibling under
    `bindings/` that `_mojotrees.mojo` does not import is compiled by nothing
    and reachable from nothing, however many functions it defines.
6.  **Python APIs with no native call.** The mirror: a native function name
    that Python reaches for - `_mojotrees.foo`, `getattr(_mojotrees, "foo")` -
    that the binding table does not export. These are the degraded paths,
    where Python reimplements in Python what Mojo already computes.
7.  **Serialization fields written but not read, or read but not written.**
    Section keywords emitted by `save_model`/`save_multiclass_model` against
    the keywords `load_model`/`load_multiclass_model` consume, plus fields of
    the serialized structs that neither side ever touches.
8.  **Referenced paths that do not exist.** Every repository-relative path
    named by a document or by a pixi task, checked for existence. Parity
    evidence lives here as a special case.
9.  **C ABI drift.** `mojotrees.h` declarations against `@export`ed
    definitions, in both directions.

What it deliberately does not do:

- **It does not re-check the LightGBM parity contract.** `check_parity.py`
  owns `docs/LIGHTGBM_PARITY.md` and `docs/CAPABILITY_LEVELS.md`: the level
  definitions, the evidence columns, the claim/row schema. This script checks
  only that paths named as evidence exist as files, and defers everything
  about what the evidence means. Two parity checkers would be exactly the
  duplication this script exists to find.
- **It does not import mojotrees, build anything, or run Mojo.** It reads
  text. A finding here is a statement about the source, never about a running
  program: this script cannot tell you that a connected path is *correct*,
  only that it exists.
- **It does not parse Mojo.** It matches import statements and top-level
  declarations with regular expressions tuned to this repository's style
  (`from .x import (...)`, `from mojotrees.x import ...`, `def name(`,
  `struct Name(`, `comptime NAME =`). A file written in some other style is
  under-reported, not mis-reported: unknown text contributes no edges.

Every finding is one of three kinds, and the distinction is the point:

- `DEAD` - nothing reaches it and nothing is expected to. Remove it.
- `EXPERIMENTAL` - deliberately not wired, kept for a later decision. Leave
  it, but say so in its own docstring so the next audit does not re-find it.
- `PENDING` - implemented and meant to be reachable, blocked on a named edit
  in a file its own lane does not own. This is the interesting one: every
  `PENDING` finding should appear in the cross-lane patch queue in
  `handoffs/connect_22_audit.md` with an owner.

The classification table `CLASSIFICATION` below is the only place that
judgment lives; everything above it is mechanical. An unclassified finding
defaults to `PENDING`, which is the conservative answer: it shows up in the
queue and a human decides.

Usage:

    python3 tools/connectivity_audit.py                 # full report
    python3 tools/connectivity_audit.py --section orphans
    python3 tools/connectivity_audit.py --json          # machine readable
    python3 tools/connectivity_audit.py --fail-on PENDING

Exit status is 0 when no finding at or above `--fail-on` is present, 1 when
one is, and 2 when the repository layout is not what this script expects
(a missing root, an unreadable file). The default `--fail-on` is `none`, so
the plain invocation reports and succeeds: this is a map, not a gate, until
someone decides to gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Layout. Everything this script knows about where things live.
# --------------------------------------------------------------------------

NATIVE_PKG = os.path.join("src", "mojotrees")
PY_PKG = os.path.join("python", "mojotrees")
BINDINGS_DIR = "bindings"
CAPI_DIR = "capi"
CLI_DIR = "cli"

#: The native package's own re-export block. Reaching a module from here
#: means the module is part of the public Mojo API.
NATIVE_ROOT = os.path.join(NATIVE_PKG, "__init__.mojo")

#: Roots for native reachability, in the order the report lists them. Each is
#: (path, label). A native module is an orphan when no root reaches it.
NATIVE_ROOTS = [
    (NATIVE_ROOT, "mojo-api"),
    (os.path.join(BINDINGS_DIR, "_mojotrees.mojo"), "bindings"),
    (os.path.join(CAPI_DIR, "mojotrees_capi.mojo"), "c-abi"),
    (os.path.join(CLI_DIR, "mojotrees_cli.mojo"), "cli"),
]

#: Directories whose files reach native modules but are not entry points.
#: A module reached only from here is exercised, not shipped.
NON_SHIPPING_ROOTS = [("tests", "tests"), ("bench", "bench")]

#: The Python package root.
PY_ROOT = os.path.join(PY_PKG, "__init__.py")

#: Build outputs and caches that are copies of real source. Walking them
#: doubles every finding.
SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pixi",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}

#: Where documents that name repository paths live. Order is the report order.
DOC_GLOBS = [
    "README.md",
    os.path.join("python", "README.md"),
    os.path.join("docs", "*.md"),
    os.path.join("docs", "design", "*.md"),
    os.path.join("docs", "tutorials", "*.md"),
]

#: `tools/check_parity.py` owns the parity contract. This script only asks
#: whether the files it names exist.
PARITY_CONTRACT = os.path.join("docs", "LIGHTGBM_PARITY.md")
PARITY_CHECKER = os.path.join("tools", "check_parity.py")


# --------------------------------------------------------------------------
# Judgment. The one table in this file that is an opinion.
# --------------------------------------------------------------------------

DEAD = "DEAD"
EXPERIMENTAL = "EXPERIMENTAL"
PENDING = "PENDING"
CONNECTED = "CONNECTED"

#: Severity order, low to high, for `--fail-on`.
KIND_ORDER = [CONNECTED, EXPERIMENTAL, PENDING, DEAD]

#: How to read a finding about a specific module or name. Keys are either a
#: native module name (`gpu_leaf_batching`), a Python dotted name
#: (`mojotrees.inspection`), or a binding function name. The value is
#: (kind, owning lane, one-line reason).
#:
#: Rules for editing this table:
#:
#: - Add a row only when a handoff justifies it. The reason field should be
#:   readable by someone who has not read that handoff.
#: - `EXPERIMENTAL` is a commitment that the module is not meant to be
#:   reachable yet. It silences the finding, so it needs a reason that says
#:   what would change the answer.
#: - Never mark something `CONNECTED` here to silence a finding. If the graph
#:   says it is unreachable and it is not, the parser is wrong; fix the
#:   parser.
#: Written against the tree at commit dc21f03. Rows go stale as lanes land
#: work: `unified_memory_policy`, `inspection`, `objective_registry`,
#: `initialization`, `raw_data`, `distributed_transport`, `gpu_fused_round`,
#: `gpu_leaf_batching`, and `apple_histogram_policy` were all orphans earlier
#: the same day and are not any more. Re-run the script before trusting a
#: row; a `CONNECTED` classification on something the graph still calls an
#: orphan means the row is stale, not that the graph is wrong.
#:
#: A row whose finding has been fixed is deleted rather than flipped to
#: `CONNECTED`, for the reason above: a row that says `CONNECTED` while the
#: graph still reports an orphan hides a regression, and a deleted row costs
#: nothing, since an unclassified finding defaults to `PENDING` and lands
#: back in the queue. Deleted in the connect_22 fix pass, all three now
#: reachable and all three verifiable by reading the imports named here:
#: `gpu_portability` (imported by `src/mojotrees/histogram_gpu.mojo`, which
#: also calls `require_bins_supported` and `require_histogram_launchable`),
#: `gpu_backend_policy` (reached through it), and `mojotrees._compat`
#: (called from `python/mojotrees/__init__.py` before the extension import).
CLASSIFICATION = {
    # -- native modules no entry point reaches -----------------------------
    "gpu_binned_layout": (
        PENDING,
        "connect_02",
        "Packed-bin layout planner; train_gpu never asks for a plan.",
    ),
    "gpu_bin_packing": (
        PENDING,
        "connect_02",
        "Reached only from gpu_binned_layout, itself unreachable.",
    ),
    "gpu_multiclass_batch": (
        PENDING,
        "connect_04",
        "Class-batched GPU rounds; multiclass GPU training is per-class.",
    ),
    "hybrid_leaf_scheduler": (
        PENDING,
        "connect_04",
        "CPU/GPU per-leaf placement; no trainer consults it.",
    ),
    "histogram_cache_policy": (
        PENDING,
        "connect_04",
        "Reached only from hybrid_leaf_scheduler, itself unreachable.",
    ),
    "backend": (
        EXPERIMENTAL,
        "connect_01",
        "A one-function dispatch shim kept as the reference the CPU/GPU "
        "equivalence test compares against. Test-only by design.",
    ),
    "unified_memory_policy": (
        EXPERIMENTAL,
        "connect_05",
        "Route evidence ledger, now reached from device_policy and "
        "histogram_gpu. The routes it scores are still not implemented in "
        "any trainer, so the decision it returns has one live outcome.",
    ),
    "cegb": (
        PENDING,
        "consolidation_K10",
        "LightGBM cegb_* controls, complete and self-contained. Parked until "
        "a trainer accepts the cegb params and the boosting loop hooks it.",
    ),
    "gpu_categorical": (
        PENDING,
        "consolidation_K10",
        "GPU category statistics; the GPU trainer refuses categoricals. "
        "Parked until train_gpu accepts categorical specs.",
    ),
    "gpu_sparse": (
        PENDING,
        "consolidation_K10",
        "Reached only from gpu_categorical; same unblocker.",
    ),
    "gpu_sparse_layout": (
        PENDING,
        "consolidation_K10",
        "Reached only from gpu_sparse; same unblocker.",
    ),
    "gpu_vendor_policy": (
        EXPERIMENTAL,
        "consolidation_K2",
        "CUDA and HIP occupancy policy, merged from the gpu_cuda_policy / "
        "gpu_amd_policy twins (f23bd1b). Reached only from its test until a "
        "discrete-GPU trainer consults it; that is the same status the twins "
        "had. handoffs/migration_20_device_policy.md.",
    ),
    # -- Python modules ----------------------------------------------------
    "mojotrees.lgbm_model_io": (
        EXPERIMENTAL,
        "integration_C0",
        "Lazy submodule (_LAZY_SUBMODULES) like dask: reachable as "
        "mojotrees.lgbm_model_io after a plain import, exporting nothing at "
        "top level so a LightGBM model-file conversion is asked for by name. "
        "Its four entry points are bound in bindings/lgbm_bindings.mojo and "
        "covered by python/tests/test_lgbm_interop.py.",
    ),
    "mojotrees.dask": (
        EXPERIMENTAL,
        "consolidation_K7",
        "Intentional PEP 562 lazy submodule (_LAZY_SUBMODULES): importing "
        "the package must not import dask, and it exports nothing at top "
        "level by decision (_public_api_plan.NOT_EXPORTED).",
    ),
    "mojotrees._dask_runtime": (
        EXPERIMENTAL,
        "consolidation_K7",
        "Backend contract reached only from dask.py inside functions, by "
        "design.",
    ),
    "mojotrees._public_api_plan": (
        EXPERIMENTAL,
        "connect_07",
        "A plan expressed as data. Its own docstring states that nothing in "
        "the package imports it; importing it would be the bug.",
    ),
    # -- binding table entries with no Python caller ------------------------
    #
    # The ask-before-a-fit family first. These five answer a question about a
    # configuration rather than doing anything to one, and no Python code
    # asks: the estimators send the same keys into a fit, where the same
    # native code checks them, so nothing they cover goes unchecked. What
    # has no Python route is asking *early* -- before the data is read, or
    # without a dataset at all. EXPERIMENTAL rather than PENDING because
    # nothing is blocked: a Python caller is a decision about the public API
    # (`docs/COMPATIBILITY_POLICY.md`), not an edit some other lane owes.
    "dataset_num_data": (
        PENDING,
        "connect_07",
        "Dataset.num_data() reads a cached Python int instead.",
    ),
    "dataset_num_feature": (
        PENDING,
        "connect_07",
        "Dataset.num_feature() reads a cached Python int instead.",
    ),
    "num_trees": (
        PENDING,
        "connect_07",
        "Python asks num_iterations everywhere; the tree count is never "
        "read, so a DART or RF model's tree count has no Python route.",
    ),
    "predict_raw": (
        PENDING,
        "connect_07",
        "Python reaches raw scores through predict_range instead.",
    ),
    "predict": (
        PENDING,
        "connect_07",
        "Superseded in Python by predict_range without being removed.",
    ),
    "predict_proba": (
        PENDING,
        "connect_07",
        "Superseded in Python by predict_proba_range without being removed.",
    ),
    "predict_batch": (
        PENDING,
        "connect_07",
        "Batched prediction entry; no Python estimator routes to it.",
    ),
    "predict_proba_batch": (
        PENDING,
        "connect_07",
        "Batched probability entry; no Python estimator routes to it.",
    ),
    "predict_leaf_batch": (
        PENDING,
        "connect_07",
        "Batched leaf indices; no Python estimator routes to it.",
    ),
    "predict_leaf_multiclass_batch": (
        PENDING,
        "connect_07",
        "Batched multiclass leaf indices; no Python estimator routes to it.",
    ),
    # -- native names Python reaches for that no table exports --------------
    "dump_model": (
        PENDING,
        "connect_06",
        "Implemented in bindings/inspection_bindings.mojo, which "
        "bindings/_mojotrees.mojo neither imports nor registers.",
    ),
    "objective_code": (
        PENDING,
        "connect_06",
        "Implemented in bindings/inspection_bindings.mojo, which "
        "bindings/_mojotrees.mojo neither imports nor registers.",
    ),
    "registry_metrics": (
        PENDING,
        "connect_06",
        "Implemented in bindings/objective_bindings.mojo, which "
        "bindings/_mojotrees.mojo neither imports nor registers.",
    ),
    "decide_device": (
        PENDING,
        "connect_06",
        "device_policy.decide_device exists natively and no binding module "
        "wraps it, so device_selection.py runs in its degraded mode.",
    ),
    # -- binding exports Python does not call yet (consolidation K6) ---------
    # The K6 forward surface got its Python readers in the integration round
    # (W8: Dataset field/bin readers, registry lookups in _fit_args and
    # _eval, model_to_json / file kinds, build_info startup and format
    # versions, dask machine-list and status texts). One remains:
}


def classify(name):
    """(kind, owner, reason) for a finding key, defaulting to PENDING.

    An unclassified finding is PENDING on purpose: it lands in the patch
    queue and a human decides whether it is dead, experimental, or blocked.
    """
    if name in CLASSIFICATION:
        return CLASSIFICATION[name]
    return (PENDING, "unassigned", "Not yet classified; see the patch queue.")


# --------------------------------------------------------------------------
# Reading the tree.
# --------------------------------------------------------------------------


class AuditError(Exception):
    """The repository is not shaped the way this script expects."""


def read(path):
    """Text of a repository-relative path, or None when it is not there."""
    full = os.path.join(ROOT, path)
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except (OSError, IOError):
        return None


def must_read(path):
    text = read(path)
    if text is None:
        raise AuditError("required file is missing: " + path)
    return text


def walk(rel_dir, suffix):
    """Repository-relative paths under `rel_dir` ending in `suffix`, sorted,
    skipping build outputs and caches."""
    out = []
    base = os.path.join(ROOT, rel_dir)
    if not os.path.isdir(base):
        return out
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
        for name in sorted(filenames):
            if name.endswith(suffix):
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, ROOT))
    return sorted(out)


def expand_globs(patterns):
    """Repository-relative files matching simple `dir/*.ext` patterns. Only
    a trailing `*` in the basename is supported, which is all DOC_GLOBS uses.
    """
    out = []
    for pattern in patterns:
        head, tail = os.path.split(pattern)
        if "*" not in tail:
            if os.path.exists(os.path.join(ROOT, pattern)):
                out.append(pattern)
            continue
        prefix, _, ext = tail.partition("*")
        directory = os.path.join(ROOT, head)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.startswith(prefix) and name.endswith(ext):
                out.append(os.path.join(head, name))
    return out


# --------------------------------------------------------------------------
# Mojo: imports, declarations, exports.
# --------------------------------------------------------------------------

#: `from .module import ...` and `from mojotrees.module import ...`, with the
#: import list either on one line or parenthesized across many.
MOJO_IMPORT = re.compile(
    r"^from\s+(?:\.|mojotrees\.)([A-Za-z_][A-Za-z_0-9]*)\s+import\s+(.*)$",
    re.MULTILINE,
)

#: A top-level declaration whose name becomes part of the module's surface.
MOJO_DECL = re.compile(
    r"^(?:def|fn|struct|trait)\s+([A-Za-z_][A-Za-z_0-9]*)", re.MULTILINE
)

#: A module-level constant. Mojo spells this `comptime` here; `alias` is the
#: older spelling and still appears in a few files.
MOJO_CONST = re.compile(
    r"^(?:comptime|alias)\s+([A-Za-z_][A-Za-z_0-9]*)\s*=", re.MULTILINE
)

#: The CPython binding table: `m.def_function[symbol]("public_name")`, with
#: the string on the same line or on the next one.
DEF_FUNCTION = re.compile(
    r"def_function\[\s*([A-Za-z_][A-Za-z_0-9]*)\s*\]\s*\(\s*\n?\s*"
    r"\"([A-Za-z_][A-Za-z_0-9]*)\""
)

#: A C ABI definition: `@export` then `def mojotrees_name(` ... `abi("C")`.
CAPI_EXPORT = re.compile(
    r"@export\s*\n\s*(?:def|fn)\s+(mojotrees_[A-Za-z_0-9]*)\s*\(", re.MULTILINE
)

#: A declaration in the C header, which is what a C caller can link against.
CAPI_HEADER_DECL = re.compile(r"\b(mojotrees_[a-z_0-9]+)\s*\(")

#: A CLI command, dispatched by string comparison in `run`.
CLI_COMMAND = re.compile(r"command\s*==\s*\"([a-z][a-z_-]*)\"")


def strip_mojo_comments(text):
    """Text with `#` line comments and triple-quoted docstrings removed.

    Both carry prose that names modules and functions, and prose is not an
    edge. Docstrings in this repository name other modules constantly - that
    is the house style - so leaving them in would make every module reachable
    from every other one.
    """
    out = []
    i = 0
    n = len(text)
    in_doc = False
    while i < n:
        if in_doc:
            end = text.find('"""', i)
            if end == -1:
                break
            i = end + 3
            in_doc = False
            continue
        triple = text.find('"""', i)
        hashmark = text.find("#", i)
        newline_of_hash = -1
        if hashmark != -1:
            newline_of_hash = text.find("\n", hashmark)
            if newline_of_hash == -1:
                newline_of_hash = n
        if triple != -1 and (hashmark == -1 or triple < hashmark):
            out.append(text[i:triple])
            i = triple + 3
            in_doc = True
            continue
        if hashmark != -1:
            out.append(text[i:hashmark])
            i = newline_of_hash
            continue
        out.append(text[i:])
        break
    return "".join(out)


def read_import_list(tail):
    """(names, characters consumed) for an import list, given every character
    from the start of the list to the end of the file.

    Handles the two forms both languages use here: a bare list that ends at
    the newline, and a parenthesized list that ends at its closing paren,
    however many lines later. `X as _Y` contributes `X`, because the question
    the graph asks is which symbol of the *source* module was taken.

    The consumed count lets a caller cut the whole statement out of the file,
    which is how "imported and never used" avoids counting the import itself
    as a use.
    """
    lead = len(tail) - len(tail.lstrip())
    stripped = tail.lstrip()
    if not stripped.startswith("("):
        body = stripped.split("\n", 1)[0]
        consumed = lead + len(body)
    else:
        depth = 1
        collected = []
        for char in stripped[1:]:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            collected.append(char)
        body = "".join(collected)
        consumed = lead + len(body) + 2
    names = []
    for piece in body.replace("\n", " ").split(","):
        piece = piece.strip()
        if not piece or piece == "*":
            continue
        head = piece.split()[0]
        if re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", head):
            names.append(head)
    return names, consumed


def parse_import_list(tail):
    """Just the names. See `read_import_list`."""
    return read_import_list(tail)[0]


def strip_mojo_import_block(body):
    """`body` with every `from .x import (...)` statement cut out entirely,
    parenthesized continuations included."""
    spans = []
    for match in MOJO_IMPORT.finditer(body):
        _names, consumed = read_import_list(body[match.start(2) :])
        spans.append((match.start(), match.start(2) + consumed))
    out = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        out.append(body[cursor:start])
        cursor = end
    out.append(body[cursor:])
    return "".join(out)


def mojo_imports(path, text=None):
    """{module: [imported names]} for one Mojo file, comments removed."""
    if text is None:
        text = read(path)
        if text is None:
            return {}
    body = strip_mojo_comments(text)
    found = defaultdict(list)
    for match in MOJO_IMPORT.finditer(body):
        module = match.group(1)
        # `.` does not cross a newline under re.MULTILINE, so group(2) is only
        # the first line of a parenthesized list. Slice from where that group
        # starts instead, and let parse_import_list find the closing paren.
        found[module].extend(parse_import_list(body[match.start(2) :]))
    return dict(found)


def native_modules():
    """Every module name under src/mojotrees, `__init__` excluded."""
    names = []
    for path in walk(NATIVE_PKG, ".mojo"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem != "__init__":
            names.append(stem)
    return sorted(names)


def native_import_graph():
    """{module: set(modules it imports)} over src/mojotrees only."""
    graph = {}
    for name in native_modules():
        path = os.path.join(NATIVE_PKG, name + ".mojo")
        graph[name] = set(mojo_imports(path).keys())
    return graph


def reachable_from(seed_modules, graph):
    """Breadth-first closure of `seed_modules` over `graph`."""
    seen = set()
    queue = list(seed_modules)
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for nxt in sorted(graph.get(current, ())):
            if nxt not in seen:
                queue.append(nxt)
    return seen


# --------------------------------------------------------------------------
# Python: imports and attribute reads.
# --------------------------------------------------------------------------

PY_FROM = re.compile(
    r"^\s*from\s+\.([A-Za-z_][A-Za-z_0-9]*)?\s+import\s+(.*)$", re.MULTILINE
)
PY_IMPORT_PKG = re.compile(
    r"^\s*(?:from|import)\s+mojotrees\.([A-Za-z_][A-Za-z_0-9]*)", re.MULTILINE
)

#: `_mojotrees.name` and `getattr(_mojotrees, "name")`, the only two ways
#: Python in this package reaches the extension module.
#: `_mojotrees.name` or `ext.name`; `ext` is the conventional local the
#: package binds the extension module to (`_sequence.stream_dataset`,
#: `device_selection`).
NATIVE_ATTR = re.compile(r"\b(?:_mojotrees|ext)\.(?!mojo\b)([A-Za-z_][A-Za-z_0-9]*)")
NATIVE_GETATTR = re.compile(
    r"getattr\(\s*_mojotrees\s*,\s*\"([A-Za-z_][A-Za-z_0-9]*)\""
)

#: The keyword arguments of a Python estimator's `__init__`, taken from the
#: `self.x = x` block that follows it. Matching the assignment rather than
#: the signature keeps generated defaults and `**kwargs` out.
PY_SELF_ASSIGN = re.compile(
    r"^\s{8}self\.([a-z_][a-z_0-9]*)\s*=\s*\1\s*$", re.MULTILINE
)


def python_modules():
    """Every module under python/mojotrees, build outputs excluded."""
    names = []
    for path in walk(PY_PKG, ".py"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem != "__init__":
            names.append(stem)
    return sorted(names)


def python_import_graph():
    """{module: set(sibling modules it imports)} over python/mojotrees.

    Both the eager `from . import x` at the top of a file and the lazy
    `from . import x` inside a function body count as edges: the second is
    this package's deliberate way of keeping `import mojotrees` cheap, not a
    weaker kind of dependency.
    """
    graph = {}
    modules = set(python_modules()) | {"__init__"}
    for name in sorted(modules):
        path = os.path.join(PY_PKG, name + ".py")
        text = read(path)
        if text is None:
            continue
        edges = set()
        for match in PY_FROM.finditer(text):
            module = match.group(1)
            if module:
                edges.add(module)
            else:
                # `from . import x`: x is a submodule when one is named that,
                # and otherwise an attribute of the package (the package uses
                # both forms, and only the first is an edge).
                for symbol in parse_import_list(text[match.start(2) :]):
                    if symbol in modules:
                        edges.add(symbol)
        for match in PY_IMPORT_PKG.finditer(text):
            edges.add(match.group(1))
        graph[name] = {e for e in edges if e in modules}
    return graph


# --------------------------------------------------------------------------
# Findings.
# --------------------------------------------------------------------------


class Finding(object):
    """One statement about connectivity, with its judgment attached."""

    def __init__(self, section, subject, detail, kind=None, owner=None):
        judged_kind, judged_owner, reason = classify(subject)
        self.section = section
        self.subject = subject
        self.detail = detail
        self.kind = kind or judged_kind
        self.owner = owner or judged_owner
        self.reason = reason

    def as_dict(self):
        return {
            "section": self.section,
            "subject": self.subject,
            "detail": self.detail,
            "kind": self.kind,
            "owner": self.owner,
            "reason": self.reason,
        }

    def line(self):
        return "  [%s] %-34s %s\n      owner=%s  %s" % (
            self.kind,
            self.subject,
            self.detail,
            self.owner,
            self.reason,
        )


# -- 1. orphan native modules ----------------------------------------------


#: `from <name> import ...` where `<name>` is a top-level module, as the
#: bindings entry point imports its capability modules.
TOP_LEVEL_IMPORT = re.compile(
    r"^from\s+([A-Za-z_][A-Za-z_0-9]*)\s+import\b", re.MULTILINE
)


def sibling_modules(path, text):
    """Files next to `path` that it imports as top-level modules."""
    directory = os.path.dirname(path)
    out = []
    for name in TOP_LEVEL_IMPORT.findall(strip_mojo_comments(text)):
        sibling = os.path.join(directory, name + ".mojo")
        if os.path.exists(os.path.join(ROOT, sibling)):
            out.append(sibling)
    return out


def audit_orphans():
    """Native modules no entry point reaches, annotated with what does."""
    findings = []
    graph = native_import_graph()

    reached_by = defaultdict(set)
    for path, label in NATIVE_ROOTS:
        text = read(path)
        if text is None:
            raise AuditError("missing entry point: " + path)
        seeds = set(mojo_imports(path, text).keys()) & set(graph)
        # The extension entry point is split across bindings/*.mojo: a
        # capability module it imports (`from x_bindings import ...`) is
        # part of the same shared library, so what that module imports from
        # the package is reached by the bindings root too. Which capability
        # modules the entry point forgets to import is the binding-modules
        # section's finding, not this one's.
        for sibling in sibling_modules(path, text):
            seeds |= set(mojo_imports(sibling).keys()) & set(graph)
        for module in reachable_from(seeds, graph):
            reached_by[module].add(label)

    for directory, label in NON_SHIPPING_ROOTS:
        seeds = set()
        for path in walk(directory, ".mojo"):
            seeds |= set(mojo_imports(path).keys()) & set(graph)
        for module in reachable_from(seeds, graph):
            reached_by[module].add(label)

    shipping = {label for _, label in NATIVE_ROOTS}
    for module in sorted(graph):
        labels = reached_by.get(module, set())
        if labels & shipping:
            continue
        if labels:
            detail = "no entry point reaches it; reached only from %s" % (
                ", ".join(sorted(labels)),
            )
        else:
            detail = "nothing in the repository imports it"
        findings.append(Finding("orphans", module, detail))
    return findings


# -- 2. imported but never called ------------------------------------------


#: Modules whose import block IS their surface. Each one says so in its own
#: docstring: device.mojo "defines nothing" and re-exports device_policy's
#: vocabulary unchanged; inspection.mojo re-exports model_dump's schema
#: primitives "so a caller that reaches for inspection finds them where the
#: schema doc says they are"; apple_gpu_policy.mojo imports gpu_tiling's
#: geometry constants "and re-exported under the same names" where earlier
#: revisions carried copies. Names they import and never mention are the
#: point, not a fake connection, so the unused-import check skips them. Any
#: other file that wants the same exemption has to earn it with a docstring
#: and a row here.
REEXPORT_FACADES = frozenset(
    [
        os.path.join(NATIVE_PKG, "apple_gpu_policy.mojo"),
        os.path.join(NATIVE_PKG, "device.mojo"),
        os.path.join(NATIVE_PKG, "inspection.mojo"),
    ]
)


def audit_unused_imports():
    """Symbols a file imports and never mentions again.

    An import that is never used is the cheapest possible fake connection:
    the graph says the module is reached, and no behavior depends on it. This
    is why `orphans` alone is not enough.
    """
    findings = []
    paths = walk(NATIVE_PKG, ".mojo")
    paths += walk(BINDINGS_DIR, ".mojo")
    for path in paths:
        if os.path.basename(path) == "__init__.mojo":
            # The package root imports in order to re-export; every name in
            # its import block is the public surface, not an unused import.
            # Whether a re-exported module is reached is the orphan section's
            # question, not this one's.
            continue
        if path in REEXPORT_FACADES:
            continue
        text = read(path)
        if text is None:
            continue
        body = strip_mojo_comments(text)
        # Everything after the import block, so an import does not count as
        # its own use.
        imports = mojo_imports(path, text)
        without_imports = strip_mojo_import_block(body)
        # `X as _Y` binds the local name `_Y`; a use of `_Y` is a use of `X`.
        aliases = dict(
            re.findall(
                r"\b([A-Za-z_][A-Za-z_0-9]*)\s+as\s+([A-Za-z_][A-Za-z_0-9]*)",
                body,
            )
        )
        for module, names in sorted(imports.items()):
            unused = []
            for name in names:
                local = aliases.get(name, name)
                if not re.search(r"\b%s\b" % re.escape(local), without_imports):
                    unused.append(name)
            if unused and len(unused) == len(names) and names:
                findings.append(
                    Finding(
                        "unused-imports",
                        module,
                        "%s imports %s and uses none of them"
                        % (path, ", ".join(sorted(unused))),
                    )
                )
            elif unused:
                findings.append(
                    Finding(
                        "unused-imports",
                        module,
                        "%s imports unused %s"
                        % (path, ", ".join(sorted(unused))),
                    )
                )
    return findings


# -- 3. duplicate registries and policies ----------------------------------

#: Names that mean "a table of the things mojotrees supports". When one of
#: these is defined by two modules, there are two tables, and they will drift.
REGISTRY_SHAPED = re.compile(
    r"^(?:"
    r"objective_|metric_|parse_objective|objective$|"
    r"parse_device|device_name|gpu_supports|resolve_device|decide_device|"
    r"parse_metric|metric_name|task_from_name|task_name|"
    r"is_builtin|register_|lookup_"
    r")"
)


def audit_duplicate_registries():
    """Public names defined by more than one native module."""
    findings = []
    where = defaultdict(list)
    for name in native_modules():
        path = os.path.join(NATIVE_PKG, name + ".mojo")
        text = read(path)
        if text is None:
            continue
        body = strip_mojo_comments(text)
        imported = set()
        for names in mojo_imports(path, text).values():
            imported.update(names)
        # `from .x import Y as _Y` binds `_Y` locally; `read_import_list`
        # reports the source name `Y`, so collect the local alias too.
        for match in MOJO_IMPORT.finditer(body):
            _names, consumed = read_import_list(body[match.start(2) :])
            for piece in body[match.start(2) : match.start(2) + consumed].split(","):
                words = piece.replace("(", " ").replace(")", " ").split()
                if len(words) == 3 and words[1] == "as":
                    imported.add(words[2])
        for match in MOJO_DECL.finditer(body):
            symbol = match.group(1)
            if not symbol.startswith("_"):
                where[symbol].append(name)
        for match in MOJO_CONST.finditer(body):
            # `comptime NAME = _NAME` where `_NAME` was imported is a
            # re-export of another module's constant, not a second
            # definition; the value is bound once, in the source module.
            rhs = body[match.end() :].split("\n", 1)[0].strip()
            if re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", rhs) and rhs in imported:
                continue
            where[match.group(1)].append(name)

    for symbol in sorted(where):
        modules = sorted(set(where[symbol]))
        if len(modules) < 2:
            continue
        registryish = bool(REGISTRY_SHAPED.match(symbol))
        detail = "%s is defined in %s" % (symbol, ", ".join(modules))
        if registryish:
            detail += " - two tables of the same fact"
        findings.append(
            Finding(
                "duplicate-registries",
                symbol,
                detail,
                kind=PENDING if registryish else EXPERIMENTAL,
            )
        )
    return findings


# -- 4. public parameters with no downstream consumer ----------------------


def audit_dead_parameters():
    """Estimator keyword arguments that are stored and never read again.

    The pattern this catches: a parameter is added to `__init__`, assigned to
    `self`, documented, and never threaded into the call that reaches native
    code. It validates, it round-trips through `get_params`, and it changes
    nothing about the model.
    """
    findings = []
    text = read(PY_ROOT)
    if text is None:
        raise AuditError("missing entry point: " + PY_ROOT)

    package_text = []
    for name in sorted(set(python_modules()) | {"__init__"}):
        body = read(os.path.join(PY_PKG, name + ".py"))
        if body is not None:
            package_text.append(body)
    everything = "\n".join(package_text)

    for match in PY_SELF_ASSIGN.finditer(text):
        param = match.group(1)
        # Uses that are not the storing assignment itself.
        pattern = r"\b%s\b" % re.escape(param)
        hits = len(re.findall(pattern, everything))
        # Each parameter costs at least: the signature, the assignment (twice
        # on one line), and usually a docstring mention. Anything at or below
        # that floor never reaches a call.
        if hits <= 4:
            findings.append(
                Finding(
                    "dead-parameters",
                    param,
                    "estimator parameter mentioned %d times in the whole "
                    "package; check that it reaches a native call" % hits,
                )
            )
    return findings


# -- 5. binding functions with no Python caller ----------------------------


def binding_exports():
    """{public name: mojo symbol} from the `def_function` table."""
    path = os.path.join(BINDINGS_DIR, "_mojotrees.mojo")
    text = must_read(path)
    out = {}
    for match in DEF_FUNCTION.finditer(text):
        out[match.group(2)] = match.group(1)
    if not out:
        raise AuditError("no def_function table found in " + path)
    return out


#: A quoted identifier. Python also reaches the extension by string
#: dispatch (`_eval._REGISTRY_HOOKS`, `inspection._hook(booster, name)`,
#: `sklearn._predict_batch(entry, legacy)`, `_dask_runtime` provider probes,
#: `device_selection`'s `getattr(ext, name)`), so a quoted string that spells
#: an exported binding name counts as a read. `inspection._hook` appends
#: `_multiclass` to its argument, hence the second form.
QUOTED_NAME = re.compile(r"[\"']([A-Za-z_][A-Za-z_0-9]*)[\"']")


def python_native_reads(exports=None):
    """Native names Python reaches for, whether or not they exist.

    Attribute and getattr reads count unconditionally. A quoted string counts
    only when `exports` (the binding table) is given and the string, or the
    string plus `_multiclass`, names an export; a plain scan of every string
    literal would otherwise invent reads.
    """
    names = set()
    quoted = set()
    for name in sorted(set(python_modules()) | {"__init__"}):
        text = read(os.path.join(PY_PKG, name + ".py"))
        if text is None:
            continue
        names |= set(NATIVE_ATTR.findall(text))
        names |= set(NATIVE_GETATTR.findall(text))
        quoted |= set(QUOTED_NAME.findall(text))
    if exports:
        for q in quoted:
            if q in exports:
                names.add(q)
            if q + "_multiclass" in exports:
                names.add(q + "_multiclass")
    return names


def audit_unused_bindings():
    """Exported binding functions no Python module mentions."""
    findings = []
    exports = binding_exports()
    used = python_native_reads(exports)
    for name in sorted(exports):
        if name not in used:
            findings.append(
                Finding(
                    "unused-bindings",
                    name,
                    "exported by bindings/_mojotrees.mojo; no module under "
                    "python/mojotrees calls it",
                )
            )
    return findings


def audit_binding_modules():
    """Binding modules the extension entry point never imports.

    `bindings/_mojotrees.mojo` is the whole extension: the
    `PythonModuleBuilder` block in it is the only thing that becomes an
    attribute of
    `mojotrees._mojotrees`. A sibling module under `bindings/` that it does
    not import is compiled by nothing and callable from nothing, however many
    `def`s it holds - and `bindings/build.sh` compiles only the entry point,
    so a sibling is not even on the include path.

    This is the cheapest way to write a whole binding surface that does not
    exist at runtime, which is why it gets its own section rather than being
    inferred from the Python side.
    """
    findings = []
    entry = os.path.join(BINDINGS_DIR, "_mojotrees.mojo")
    text = must_read(entry)
    body = strip_mojo_comments(text)
    build = read(os.path.join(BINDINGS_DIR, "build.sh")) or ""

    # Reachability is transitive: a sibling the entry point imports may
    # itself import a helper module (binding_support is reached that way by
    # every capability module), and build.sh's `-I bindings` puts the whole
    # directory on the include path.
    reached = set()
    frontier = ["_mojotrees"]
    while frontier:
        stem = frontier.pop()
        if stem in reached:
            continue
        reached.add(stem)
        src = read(os.path.join(BINDINGS_DIR, stem + ".mojo"))
        if src is None:
            continue
        for dep in re.findall(
            r"^from\s+([A-Za-z_][A-Za-z_0-9]*)\s+import",
            strip_mojo_comments(src),
            re.MULTILINE,
        ):
            frontier.append(dep)

    for path in walk(BINDINGS_DIR, ".mojo"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == "_mojotrees":
            continue
        if stem in reached:
            continue
        defined = 0
        sibling = read(path)
        if sibling is not None:
            defined = len(
                re.findall(
                    r"^def\s+[a-z][A-Za-z_0-9]*\(",
                    strip_mojo_comments(sibling),
                    re.MULTILINE,
                )
            )
        detail = (
            "%s defines %d functions and %s never imports it, so none of "
            "them reach Python" % (path, defined, entry)
        )
        findings.append(
            Finding(
                "binding-modules",
                stem,
                detail,
                kind=PENDING,
                owner="connect_06",
            )
        )

    if "-I bindings" not in build:
        siblings = [
            p
            for p in walk(BINDINGS_DIR, ".mojo")
            if not p.endswith("_mojotrees.mojo")
        ]
        if siblings:
            findings.append(
                Finding(
                    "binding-modules",
                    "bindings/build.sh",
                    "compiles only the entry point with -I src; a sibling "
                    "binding module importing another sibling needs "
                    "-I bindings on the same command",
                    kind=PENDING,
                    owner="unassigned",
                )
            )
    return findings


# -- 6. Python APIs with no native call ------------------------------------


def audit_missing_bindings():
    """Native names Python reaches for that the binding table does not export.

    Each of these is a degraded path: Python either falls back to a slower
    reimplementation or announces a reduced mode. They are the highest-value
    findings in this script, because each one is a specific function that a
    specific lane can add.
    """
    findings = []
    exports = set(binding_exports())
    # Names that are attributes of the extension module for other reasons.
    not_functions = {"so", "__file__", "__name__", "__doc__"}
    for name in sorted(python_native_reads()):
        if name in exports or name in not_functions:
            continue
        findings.append(
            Finding(
                "missing-bindings",
                name,
                "python/mojotrees reaches for _mojotrees.%s; the binding "
                "table does not export it" % name,
                kind=PENDING,
                owner="connect_14",
            )
        )
    return findings


# -- 7. serialization ------------------------------------------------------

#: A section keyword the writer emits, as a literal at the start of a written
#: token run: `out += "monotone " + ...`.
#: `out += "kw ..."` on one line, or a `"kw "` string continuation on the
#: next line of a multi-line concatenation.
WRITE_KEYWORD = re.compile(
    r"out\s*\+=\s*\(?\s*(?:\"[^\"\n]*\"[^\n]*\n\s*\+?\s*)*\"([a-z_]+) "
    r"|out\s*\+=\s*\"([a-z_]+)[\" ]"
)

#: A keyword the reader consumes or peeks at, directly or through
#: `_read_kind`, which returns the token after the version.
READ_KEYWORD = re.compile(
    r"(?:r\.(?:next|peek)\(\)|_read_kind\(r\))\s*[!=]=\s*\"([a-z_]+)\""
)


def audit_serialization():
    """Fields written but not read, read but not written, and struct fields
    neither side touches.

    The third check is the one that matters: a field of `Tree`, `Booster`, or
    `BinMapper` that training fills in and serialization drops is state a
    saved model silently loses. Prediction may not need it; importance,
    inspection, and contribution do.
    """
    findings = []
    text = read(os.path.join(NATIVE_PKG, "serialize.mojo"))
    if text is None:
        raise AuditError("missing src/mojotrees/serialize.mojo")

    written = set()
    for match in WRITE_KEYWORD.finditer(text):
        written.add(match.group(1) or match.group(2))
    read_back = set(READ_KEYWORD.findall(text))
    for keyword in sorted(written - read_back):
        findings.append(
            Finding(
                "serialization",
                keyword,
                "save writes a %r section that load never looks for" % keyword,
                kind=PENDING,
                owner="connect_11",
            )
        )
    for keyword in sorted(read_back - written):
        findings.append(
            Finding(
                "serialization",
                keyword,
                "load accepts a %r section that save never writes" % keyword,
                kind=PENDING,
                owner="connect_11",
            )
        )

    for struct_name, module in (
        ("Tree", "tree"),
        ("Booster", "boosting"),
        ("BinMapper", "binning"),
    ):
        for field in struct_fields(module, struct_name):
            if re.search(r"\.%s\b" % re.escape(field), text):
                continue
            findings.append(
                Finding(
                    "serialization",
                    "%s.%s" % (struct_name, field),
                    "field of the serialized model that serialize.mojo never "
                    "writes or reads; a saved model loses it",
                    kind=PENDING,
                    owner="connect_11",
                )
            )
    return findings


def struct_fields(module, struct_name):
    """`var` field names declared directly inside one struct."""
    text = read(os.path.join(NATIVE_PKG, module + ".mojo"))
    if text is None:
        return []
    body = strip_mojo_comments(text)
    start = re.search(
        r"^struct\s+%s\b" % re.escape(struct_name), body, re.MULTILINE
    )
    if start is None:
        return []
    fields = []
    for line in body[start.end() :].splitlines():
        stripped = line.strip()
        if line and not line[0].isspace() and stripped:
            break
        match = re.match(r"^var\s+([a-z_][a-z_0-9]*)\s*:", stripped)
        if match and line.startswith("    var "):
            fields.append(match.group(1))
    return fields


# -- 8. referenced paths that do not exist ---------------------------------

#: A repository path as prose cites it. The lookbehind keeps a longer path
#: from being read as one of ours (`r/mojotrees/src/mojotrees_r.c` is not
#: `src/mojotrees_r.c`); the lookahead keeps `.h` from matching the front of
#: LightGBM's `.hpp` files.
REPO_PATH = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:docs|tests|bench|tools|src|python|bindings|"
    r"capi|cli|packaging|examples|hardware|launch)/[A-Za-z0-9_./-]+"
    r"\.(?:md|py|mojo|toml|sh|yml|yaml|h|c|json|txt))(?![A-Za-z0-9])"
)


def audit_missing_paths():
    """Repository paths named by a document or a pixi task that do not exist.

    Parity evidence is the case that matters, and it is checked here only for
    existence: `tools/check_parity.py` owns everything about what the parity
    contract may claim and which evidence a level requires. If that script is
    missing, say so rather than growing a second contract checker here.
    """
    findings = []
    if not os.path.exists(os.path.join(ROOT, PARITY_CHECKER)):
        findings.append(
            Finding(
                "missing-paths",
                PARITY_CHECKER,
                "the parity contract checker is gone; this script "
                "deliberately does not replace it",
                kind=DEAD,
                owner="connect_19",
            )
        )

    sources = expand_globs(DOC_GLOBS) + ["pixi.toml"]
    seen = {}
    for source in sources:
        text = read(source)
        if text is None:
            continue
        for path in REPO_PATH.findall(text):
            clean = path.rstrip(".,);:`")
            if os.path.exists(os.path.join(ROOT, clean)):
                continue
            seen.setdefault(clean, []).append(source)

    for path in sorted(seen):
        where = ", ".join(sorted(set(seen[path])))
        owner = "connect_19" if path.endswith(".md") else "unassigned"
        if source_is_parity(seen[path]):
            detail = "named as parity evidence in %s; not there" % where
        else:
            detail = "named in %s; the file is not there" % where
        findings.append(
            Finding("missing-paths", path, detail, kind=PENDING, owner=owner)
        )
    return findings


def source_is_parity(sources):
    return PARITY_CONTRACT in sources


# -- 9. C ABI drift --------------------------------------------------------


def audit_c_abi():
    """Header declarations against `@export`ed definitions, both ways."""
    findings = []
    mojo = read(os.path.join(CAPI_DIR, "mojotrees_capi.mojo"))
    header = read(os.path.join(CAPI_DIR, "mojotrees.h"))
    if mojo is None or header is None:
        raise AuditError("capi/ is incomplete")

    defined = set(CAPI_EXPORT.findall(mojo))
    declared = set(CAPI_HEADER_DECL.findall(header))
    for name in sorted(declared - defined):
        findings.append(
            Finding(
                "c-abi",
                name,
                "declared in capi/mojotrees.h with no @export definition; "
                "a C caller linking against it fails at load",
                kind=DEAD,
                owner="connect_21",
            )
        )
    for name in sorted(defined - declared):
        findings.append(
            Finding(
                "c-abi",
                name,
                "exported from the shared library and absent from the "
                "header, so no C caller can reach it",
                kind=PENDING,
                owner="connect_21",
            )
        )
    return findings


# -- 10. Python package reachability ---------------------------------------


def audit_python_orphans():
    """Modules under python/mojotrees that the package root never reaches."""
    findings = []
    graph = python_import_graph()
    reached = reachable_from(["__init__"], graph)
    for module in sorted(set(python_modules())):
        if module in reached:
            continue
        findings.append(
            Finding(
                "python-orphans",
                "mojotrees." + module,
                "no import chain from python/mojotrees/__init__.py reaches "
                "it; only an explicit submodule import does",
            )
        )
    return findings


# -- 11. CLI commands ------------------------------------------------------


def audit_cli():
    """CLI commands the tool dispatches, against what its docs promise."""
    findings = []
    text = read(os.path.join(CLI_DIR, "mojotrees_cli.mojo"))
    if text is None:
        raise AuditError("missing cli/mojotrees_cli.mojo")
    commands = set(CLI_COMMAND.findall(text))
    docs = read(os.path.join("docs", "CLI.md")) or ""
    readme = read(os.path.join(CLI_DIR, "README.md")) or ""
    prose = docs + "\n" + readme
    if not prose.strip():
        findings.append(
            Finding(
                "cli",
                "docs/CLI.md",
                "the CLI dispatches %s and no document describes any of them"
                % ", ".join(sorted(commands)),
                kind=PENDING,
                owner="connect_21",
            )
        )
        return findings
    for command in sorted(commands):
        if not re.search(r"\b%s\b" % re.escape(command), prose):
            findings.append(
                Finding(
                    "cli",
                    command,
                    "dispatched by cli/mojotrees_cli.mojo and named by no "
                    "document",
                    kind=PENDING,
                    owner="connect_21",
                )
            )
    return findings


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------

SECTIONS = [
    ("orphans", "Native modules no entry point reaches", audit_orphans),
    (
        "python-orphans",
        "Python modules the package root never reaches",
        audit_python_orphans,
    ),
    (
        "unused-imports",
        "Imports whose symbols the importing file never uses",
        audit_unused_imports,
    ),
    (
        "duplicate-registries",
        "Public names defined by more than one native module",
        audit_duplicate_registries,
    ),
    (
        "dead-parameters",
        "Estimator parameters with no visible downstream consumer",
        audit_dead_parameters,
    ),
    (
        "binding-modules",
        "Binding modules the extension entry point never imports",
        audit_binding_modules,
    ),
    (
        "unused-bindings",
        "Binding functions no Python module calls",
        audit_unused_bindings,
    ),
    (
        "missing-bindings",
        "Native functions Python reaches for that do not exist",
        audit_missing_bindings,
    ),
    (
        "serialization",
        "Model state that save and load disagree about",
        audit_serialization,
    ),
    (
        "missing-paths",
        "Paths named by documents and tasks that are not there",
        audit_missing_paths,
    ),
    ("c-abi", "C header and shared library disagreements", audit_c_abi),
    ("cli", "CLI commands and their documentation", audit_cli),
]


def run(selected):
    """{section: [Finding]} for the requested sections."""
    results = {}
    for name, _title, function in SECTIONS:
        if selected and name not in selected:
            continue
        results[name] = function()
    return results


def report(results, stream):
    total = 0
    counts = defaultdict(int)
    for name, title, _function in SECTIONS:
        if name not in results:
            continue
        findings = results[name]
        stream.write("\n%s\n%s\n" % (title, "-" * len(title)))
        if not findings:
            stream.write("  nothing to report\n")
            continue
        for finding in findings:
            stream.write(finding.line() + "\n")
            counts[finding.kind] += 1
            total += 1
    stream.write("\n%s\n%s\n" % ("Summary", "-------"))
    stream.write("  %d findings\n" % total)
    for kind in KIND_ORDER:
        stream.write("  %-14s %d\n" % (kind, counts[kind]))
    stream.write(
        "\n  DEAD        remove it\n"
        "  EXPERIMENTAL  deliberately unwired; the reason says what would "
        "change that\n"
        "  PENDING     implemented and blocked on a named cross-lane edit\n"
    )
    stream.write(
        "\n  Parity claims themselves are checked by %s, not here.\n"
        % PARITY_CHECKER
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Static reachability audit of the mojotrees repository."
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        choices=[name for name, _t, _f in SECTIONS],
        help="run only this section; repeatable",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit findings as JSON"
    )
    parser.add_argument(
        "--fail-on",
        default="none",
        choices=["none"] + KIND_ORDER,
        help="exit 1 when a finding of this kind or worse is present",
    )
    args = parser.parse_args(argv)

    try:
        results = run(set(args.section))
    except AuditError as error:
        sys.stderr.write("connectivity_audit: %s\n" % error)
        return 2

    if args.json:
        payload = {
            name: [f.as_dict() for f in findings]
            for name, findings in results.items()
        }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        report(results, sys.stdout)

    if args.fail_on == "none":
        return 0
    threshold = KIND_ORDER.index(args.fail_on)
    for findings in results.values():
        for finding in findings:
            if KIND_ORDER.index(finding.kind) >= threshold:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
