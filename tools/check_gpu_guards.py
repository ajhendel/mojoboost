#!/usr/bin/env python3
"""Fail when a GPU API can be elaborated by a build that has no accelerator.

WHY THIS EXISTS.  This project's central claim is one source across CPUs and
three GPU backends, and CI proves the CPU half on x86-64 and ARM64 Linux
runners that have no accelerator at all.  The development machine is an Apple
M4, which HAS one, so it can never reproduce the failure this file is about:

    constraint failed: Unknown GPU architecture detected

That is a COMPILE error, not a runtime one.  `ctx.enqueue_function[k]` asks the
compiler to build `k` for the accelerator's architecture, and on a build with
no accelerator there is no architecture to name.  Any `enqueue_function` the
compiler can reach kills the build, whatever the kernel does.  The compiler
also reports ONE error stack and stops, so a round of guards never proves the
set was complete; it only reveals the next site.  On 2026-08-17 one symptom
cost four CI cycles and three separate fixes for exactly that reason.

So this check reads the tree as text and answers the question by enumeration
instead.  It never invokes `mojo`, never builds, and runs in well under a
second, which is what lets it sit in front of a push on a machine that cannot
otherwise see the problem.  It is a static approximation and says so: see
LIMITS at the bottom of this docstring.

WHAT IT CHECKS.

  R1  Every `enqueue_function` in a MODULE-LEVEL function (that is, not a
      method of a struct) must be guarded.  A module-level launcher is one
      `from mojotrees.gpu_x import launcher` away from a CPU-set test, so
      this rule has no exemptions and no baseline.

  R2  Every other `enqueue_function` and bare `DeviceContext(` in the library
      must be either guarded or named in `tools/gpu_guard_baseline.txt`.  The
      baseline is the set of sites that were unguarded when this check was
      written; the rule is that the set may shrink and may not grow.  A NEW
      unguarded launcher fails the check on the machine that wrote it.

  R3  In a test module that CPU-only CI compiles, a function that names a
      GPU-carrying symbol must itself be guarded.  This is the mechanism that
      is easiest to get wrong, because it is not about reachability at all:
      `TestSuite.discover_tests[__functions_in_module()]()` enumerates every
      function in the module, so every function is INSTANTIATED whether or not
      a live call reaches it.  `tests/test_const_hessian.mojo` had every
      caller of `_gpu_leaf_matches` guarded and still failed to compile,
      because the helper itself built a `GpuHistogramBuilder`.  In a test
      module the guard belongs on the HELPER, not only on the tests that call
      it.

THE GUARD.  Two spellings, both of which prune at compile time:

    comptime if not has_accelerator():
        raise Error("...")
    else:
        <the body that touches the GPU>

    comptime if has_accelerator():
        <the body that touches the GPU>
    else:
        <the CPU answer>

An early `return` does NOT prune and does not work.  The `comptime if` with an
`else` is what removes the branch, and it prunes wherever it sits, so a narrow
wrap around one launch is as valid as a whole-body wrap.  The whole-body form
exists only because an early return does not prune.

LIMITS, stated because a checker that claims more than it does would recreate
the problem it exists to end.

  * It matches text, not types.  R3 keys on names imported into the test file
    from a `mojotrees.gpu_*`, `histogram_gpu` or `train_gpu*` module, which is
    precise for imports and blind to a symbol reached some other way.
  * It knows nothing about reachability in `src/`.  R2 is a ratchet over a
    hand-checked baseline, not a proof that the baseline is safe.
  * It cannot see a hazard that is neither `enqueue_function` nor
    `DeviceContext(`.  `_accelerator_arch()` is checked too; anything else new
    has to be added to `HAZARDS` by hand.
  * A green run does not prove the build is green.  Only CPU-only CI does.

Usage:
    python3 tools/check_gpu_guards.py            # check
    python3 tools/check_gpu_guards.py --list     # print every hazard site
    python3 tools/check_gpu_guards.py --write-baseline
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "tools", "gpu_guard_baseline.txt")

# Constructs that force the compiler to name an accelerator architecture, or
# that are close enough to one that an unguarded appearance is worth a look.
HAZARDS = {
    "launch": re.compile(r"\.enqueue_function\b"),
    "devctx": re.compile(r"(?<![\w.])DeviceContext\("),
    "arch": re.compile(r"(?<![\w.])_accelerator_arch\("),
}

DEF = re.compile(r"^(\s*)(?:def|fn) (\w+)")
STRUCT = re.compile(r"^(\s*)struct (\w+)")
GUARD_NEG = re.compile(r"^(\s*)comptime if not has_accelerator\(\)\s*:")
GUARD_POS = re.compile(r"^(\s*)comptime if has_accelerator\(\)\s*:")
ELSE = re.compile(r"^(\s*)else\s*:")

# The list `tools/run_tests.sh` keeps of test files that need an accelerator.
# Duplicated here rather than parsed out of the shell, and checked against it
# by `_check_gpu_only_list_is_current` so the copy cannot drift silently.
GPU_ONLY = """
test_apple_gpu_policy test_backend_equivalence test_device test_gpu_active_rows
test_gpu_fma_consistency test_gpu_kernel_family test_gpu_objectives
test_gpu_objectives_native test_gpu_portability test_gpu_predict
test_gpu_ranking_device test_gpu_row_compaction test_gpu_random_score_noise
test_gpu_runtime test_gpu_scale_refresh test_gpu_scan_primitives
test_gpu_sparse test_gpu_sparse_skip test_gpu_speculation_build
test_gpu_split_scan test_gpu_split_search test_gpu_strategies test_gpu_tiling
test_gpu_training test_gpu_vendor_policy test_host_replica
""".split()

CPU_SAFE_MARKER = re.compile(r"^\s*#\s*run_tests: cpu-safe", re.M)

GPU_MODULE = re.compile(r"^from mojotrees\.(gpu_\w+|histogram_gpu|train_gpu\w*)")


# --- Reading a Mojo file ---------------------------------------------------
#
# Docstrings are blanked so that prose ABOUT a launch is not read as one; this
# tree's docstrings quote `enqueue_function` and `comptime if not
# has_accelerator()` constantly, and counting those would make the check
# useless in both directions.  Comments are stripped for the same reason.


def read_code(path: str) -> list[str]:
    out: list[str] = []
    in_doc = False
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().split("\n")
    for line in raw:
        stripped = line.strip()
        if in_doc:
            if '"""' in stripped:
                in_doc = False
            out.append("")
            continue
        if stripped.startswith('"""') or stripped.startswith('r"""'):
            rest = stripped[4:] if stripped.startswith('r"""') else stripped[3:]
            if '"""' not in rest:
                in_doc = True
            out.append("")
            continue
        out.append(re.sub(r"#.*$", "", line))
    return out


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def enclosing_def(code: list[str], i: int) -> tuple[int, str, str] | None:
    """(line index, function name, owning struct or "") for the def that owns
    line `i`, found by scanning back for the nearest `def`/`fn` at a smaller
    indent.  Robust against multi-line signatures, which an indent stack is
    not: `def f[\\n    T: X, //\\n](` returns to the def's own column before
    the body starts."""
    want = indent_of(code[i])
    for j in range(i - 1, -1, -1):
        m = DEF.match(code[j])
        if m and len(m.group(1)) < want:
            fn_indent = len(m.group(1))
            owner = ""
            for k in range(j - 1, -1, -1):
                s = STRUCT.match(code[k])
                if s and len(s.group(1)) < fn_indent:
                    owner = s.group(2)
                    break
            return (j, m.group(2), owner)
    return None


def is_guarded(code: list[str], i: int, start: int) -> bool:
    """Whether line `i` sits in the branch of a `comptime if ... has_accelerator()`
    that a CPU-only build discards.  `start` bounds the search at the top of
    the enclosing function."""
    want = indent_of(code[i])
    for j in range(start + 1, i):
        neg = GUARD_NEG.match(code[j])
        pos = GUARD_POS.match(code[j])
        if not (neg or pos):
            continue
        g = len((neg or pos).group(1))
        if g >= want:
            continue
        # Where does this comptime block end, and where is its `else`?
        else_at = -1
        end = len(code)
        for k in range(j + 1, len(code)):
            if not code[k].strip():
                continue
            if indent_of(code[k]) > g:
                continue
            if ELSE.match(code[k]) and indent_of(code[k]) == g and else_at < 0:
                else_at = k
                continue
            end = k
            break
        if else_at < 0:
            # No `else`, so nothing is pruned.  A bare `comptime if not
            # has_accelerator(): raise` followed by the body is the shape that
            # looks right and is not.
            continue
        if neg and else_at < i < end:
            return True
        if pos and j < i < else_at:
            return True
    return False


def hazard_sites(path: str) -> list[dict]:
    code = read_code(path)
    sites: list[dict] = []
    for i, line in enumerate(code):
        kinds = [k for k, rx in HAZARDS.items() if rx.search(line)]
        if not kinds:
            continue
        owner = enclosing_def(code, i)
        if owner is None:
            start, fn, struct = -1, "<module>", ""
        else:
            start, fn, struct = owner
        sites.append(
            {
                "file": os.path.relpath(path, ROOT),
                "line": i + 1,
                "kind": ",".join(kinds),
                "fn": fn,
                "struct": struct,
                "module_level": struct == "",
                "guarded": is_guarded(code, i, start) if start >= 0 else False,
            }
        )
    return sites


def key(site: dict) -> str:
    """A baseline key that survives edits above it: file, owner, function.
    Deliberately not the line number."""
    owner = site["struct"] + "." if site["struct"] else ""
    return f"{site['file']}::{owner}{site['fn']}::{site['kind']}"


# --- Test-module rules -----------------------------------------------------


def library_files() -> list[str]:
    out = []
    for sub in ("src/mojotrees", "bindings", "capi", "cli"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".mojo"):
                out.append(os.path.join(d, name))
    return out


def test_files() -> list[str]:
    d = os.path.join(ROOT, "tests")
    return [
        os.path.join(d, n)
        for n in sorted(os.listdir(d))
        if n.startswith("test_") and n.endswith(".mojo")
    ]


def is_cpu_test(path: str) -> bool:
    """The classification `tools/run_tests.sh` applies: named in GPU_ONLY, or
    named `test_gpu_*` without the `# run_tests: cpu-safe` marker."""
    name = os.path.basename(path)[: -len(".mojo")]
    if name in GPU_ONLY:
        return False
    if name.startswith("test_gpu_"):
        with open(path, encoding="utf-8") as fh:
            return bool(CPU_SAFE_MARKER.search(fh.read()))
    return True


def gpu_carrying_symbols() -> set[str]:
    """Names that carry a GPU API with them: any struct with a method that
    launches or opens a context, and any module-level function that does.

    Direct containment, not a call-graph closure.  A closure over names alone
    is hopeless in this tree -- `route`, `pack`, `estimate`, `compute`,
    `build`, `train` all collide between the host and the device planes, and a
    closure taints every one of them.  Direct containment is exactly right for
    what R3 needs, because a private helper that only forwards to a GUARDED
    launcher is pruned along with it."""
    out: set[str] = {"DeviceContext"}
    for path in library_files():
        for site in hazard_sites(path):
            if site["struct"]:
                out.add(site["struct"])
            elif not site["fn"].startswith("_"):
                out.add(site["fn"])
    return out


def test_module_findings(symbols: set[str]) -> list[str]:
    problems = []
    for path in test_files():
        if not is_cpu_test(path):
            continue
        code = read_code(path)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        imported = set()
        for block in re.findall(
            r"^from mojotrees\.(?:gpu_\w+|histogram_gpu|train_gpu\w*)"
            r"\s+import\s*(\([^)]*\)|[^\n]*)",
            raw,
            re.M,
        ):
            for name in re.findall(r"\w+", block):
                imported.add(name)
        watch = (imported & symbols) | {"DeviceContext"}
        if not watch:
            continue
        # Split the file into top-level functions and test each one.
        starts = [i for i, l in enumerate(code) if DEF.match(l or "") and not indent_of(l)]
        starts.append(len(code))
        for a, b in zip(starts, starts[1:]):
            body = code[a:b]
            name = DEF.match(code[a]).group(2)
            hit = sorted(
                w
                for w in watch
                if any(
                    re.search(r"(?<![\w.])" + re.escape(w) + r"\s*[\(\[]", l)
                    for l in body[1:]
                )
            )
            if not hit:
                continue
            if any(
                GUARD_NEG.match(l) or GUARD_POS.match(l) for l in body
            ):
                continue
            problems.append(
                f"{os.path.relpath(path, ROOT)}:{a + 1}: {name} names "
                f"{', '.join(hit)} and is not guarded. "
                "TestSuite instantiates every function in the module, so a "
                "guard on the callers does not prune this one."
            )
    return problems


def _check_gpu_only_list_is_current() -> list[str]:
    """The GPU_ONLY copy above against the one in tools/run_tests.sh."""
    sh = os.path.join(ROOT, "tools", "run_tests.sh")
    if not os.path.exists(sh):
        return []
    with open(sh, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r'GPU_ONLY="\n(.*?)\n"', text, re.S)
    if not m:
        return ["tools/run_tests.sh: could not find the GPU_ONLY list"]
    theirs = set(m.group(1).split())
    mine = set(GPU_ONLY)
    if theirs == mine:
        return []
    return [
        "tools/check_gpu_guards.py: GPU_ONLY has drifted from "
        "tools/run_tests.sh; missing here "
        f"{sorted(theirs - mine)}, extra here {sorted(mine - theirs)}"
    ]


# --- Driver ----------------------------------------------------------------


def load_baseline() -> set[str]:
    if not os.path.exists(BASELINE):
        return set()
    with open(BASELINE, encoding="utf-8") as fh:
        return {
            l.strip()
            for l in fh
            if l.strip() and not l.lstrip().startswith("#")
        }


def main(argv: list[str]) -> int:
    all_sites: list[dict] = []
    for path in library_files():
        all_sites.extend(hazard_sites(path))

    unguarded = [s for s in all_sites if not s["guarded"]]

    if "--list" in argv:
        for s in all_sites:
            flag = "guarded" if s["guarded"] else "UNGUARDED"
            scope = "module" if s["module_level"] else "method"
            print(
                f"{s['file']}:{s['line']}\t{s['kind']}\t{scope}\t{flag}\t"
                f"{s['struct'] + '.' if s['struct'] else ''}{s['fn']}"
            )
        print(
            f"\n{len(all_sites)} hazard sites, "
            f"{len(unguarded)} unguarded, "
            f"{sum(1 for s in unguarded if s['module_level'])} of those "
            "module-level"
        )
        return 0

    if "--write-baseline" in argv:
        keys = sorted({key(s) for s in unguarded if not s["module_level"]})
        with open(BASELINE, "w", encoding="utf-8") as fh:
            fh.write(
                "# Hazard sites that are unguarded today, one per\n"
                "# file::owner.function::kind.  This set may SHRINK and may\n"
                "# not GROW: tools/check_gpu_guards.py fails on anything new.\n"
                "# Regenerate with --write-baseline only when REMOVING lines.\n"
                "#\n"
                "# WHY THESE ARE HERE RATHER THAN GUARDED.  Every one is a\n"
                "# method of a device-owning struct, and no test in the\n"
                "# CPU-only set constructs any of those structs outside a\n"
                "# guard, so nothing elaborates them on a CPU-only build\n"
                "# today.  That was checked by hand on 2026-08-17 and it is\n"
                "# NOT a property this file can prove; it is a property that\n"
                "# holds until somebody writes the test that breaks it, which\n"
                "# is exactly what rule R3 in the checker watches for.\n"
                "# docs/design/CPU_ONLY_BUILD_AUDIT.md carries the table and\n"
                "# the reasoning.\n"
                "#\n"
                "# Guarding an entry above one of these prunes it with the\n"
                "# entry, so the honest way to shrink this list is to guard\n"
                "# the public method a caller reaches, then regenerate.\n"
            )
            for k in keys:
                fh.write(k + "\n")
        print(f"wrote {len(keys)} baseline entries to {BASELINE}")
        return 0

    failures: list[str] = []
    failures.extend(_check_gpu_only_list_is_current())

    # R1: module-level launchers, no exemptions.
    for s in unguarded:
        if s["module_level"] and s["kind"] != "devctx":
            failures.append(
                f"{s['file']}:{s['line']}: module-level {s['fn']} has an "
                f"unguarded {s['kind']}. A module-level launcher is one "
                "import away from a CPU-set test; wrap it in "
                "`comptime if not has_accelerator(): raise` with an `else:`."
            )

    # R2: everything else ratchets against the baseline.
    baseline = load_baseline()
    for s in unguarded:
        if s["module_level"]:
            continue
        k = key(s)
        if k not in baseline:
            failures.append(
                f"{s['file']}:{s['line']}: new unguarded {s['kind']} in "
                f"{s['struct']}.{s['fn']}. Guard it, or add `{k}` to "
                "tools/gpu_guard_baseline.txt with a reason."
            )

    # R3: CPU-set test modules.
    failures.extend(test_module_findings(gpu_carrying_symbols()))

    if failures:
        print("GPU guard check FAILED\n")
        for f in failures:
            print("  " + f + "\n")
        print(
            "See docs/design/CPU_ONLY_BUILD_AUDIT.md. These are COMPILE "
            "errors on a CPU-only runner and cannot be reproduced on a "
            "machine that has an accelerator."
        )
        return 1

    print(
        f"GPU guard check passed: {len(all_sites)} hazard sites, "
        f"{len(unguarded)} unguarded and all of them in the baseline, "
        f"{sum(1 for p in test_files() if is_cpu_test(p))} CPU-set test "
        "modules clean."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
