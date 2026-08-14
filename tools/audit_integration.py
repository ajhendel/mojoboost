#!/usr/bin/env python3
"""Check `docs/INTEGRATION_INVENTORY.md` against the tree it describes.

The inventory claims things about reachability: these modules are orphans,
these binding modules are not registered, these native names Python reaches
for do not exist, this policy exists twice. Every one of those claims can go
stale in one direction that matters, which is a claim of *disconnection*
that a lane has since connected. A parity contract that says a capability is
unreachable when a user can reach it is worse than one that says nothing.

This script does not compute reachability. `tools/connectivity_audit.py`
does, from the four shipping roots, and this script imports it and reads its
answers. Two import graphs would be exactly the duplication both scripts
exist to find, so there is one graph engine, one judgment table
(`connectivity_audit.CLASSIFICATION`), and this file is the gate that keeps
the written inventory agreeing with them.

Division of labor across the three checkers:

- `tools/connectivity_audit.py` owns the import graph and the judgment about
  each disconnection. Authoritative.
- `tools/audit_integration.py` (this file) owns
  `docs/INTEGRATION_INVENTORY.md`, and only asks whether it still describes
  the tree.
- `tools/check_parity.py` owns `docs/LIGHTGBM_PARITY.md` and
  `docs/CAPABILITY_LEVELS.md`, which are claims about LightGBM rather than
  about reachability.

Two severities, because they are not the same kind of wrong:

    ERROR   The inventory states something false: it calls a module an
            orphan that a root now reaches, says a binding module is
            unregistered when `_mojoboost.mojo` imports it, or says a native
            name is unbound when the table exports it. Always exits 1.
    GAP     The tree has something the inventory has not caught up with: a
            new orphan, a new unregistered binding module, a new unbound
            hook. Exits 1 only under `--strict`, because on a tree with
            several lanes in flight a gap is a few minutes old and is a
            to-do rather than a false claim.

Nothing here imports the mojoboost package, builds anything, or runs Mojo.
It reads text.

    python3 tools/audit_integration.py             # report; ERROR exits 1
    python3 tools/audit_integration.py --strict    # GAP exits 1 too
    python3 tools/audit_integration.py --table     # print a corrected
                                                   # orphan table to paste
"""

from __future__ import annotations

import argparse
import os
import re
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import connectivity_audit as ca  # noqa: E402 - after the path setup above

INVENTORY = os.path.join("docs", "INTEGRATION_INVENTORY.md")

ERROR = "ERROR"
GAP = "GAP"

#: Headings whose tables this script reads, and the column count each is
#: expected to have. A heading that moves or a column that is added fails
#: here rather than being silently skipped, because a table this script
#: cannot find is a table it is not checking.
TABLES = {
    "Orphan native modules": 4,
    "Binding modules the extension does not register": 3,
    "Native names Python reaches for that no binding registers": 3,
    "Policy that exists twice": 3,
    "Reachable, but with no default effect": 4,
}

BACKTICKED = re.compile(r"`([^`]+)`")
REPO_PATH = re.compile(
    r"`((?:src|python|bindings|capi|cli|tests|bench|docs|tools|packaging)"
    r"/[A-Za-z0-9_./-]+\.(?:mojo|py|md|sh|toml|yml|h))`"
)


class Problem(object):
    """One finding, with the edit that would resolve it."""

    def __init__(self, severity, subject, detail):
        self.severity = severity
        self.subject = subject
        self.detail = detail

    def line(self):
        return "  %-5s %-34s %s" % (self.severity, self.subject, self.detail)


# --------------------------------------------------------------------------
# Reading the inventory.
# --------------------------------------------------------------------------


def inventory_text():
    text = ca.read(INVENTORY)
    if text is None:
        raise ca.AuditError("required file is missing: " + INVENTORY)
    return text


def is_separator(cells):
    """True for the `|---|---|` row under a table's header."""
    return all(cell and set(cell) <= set("-: ") for cell in cells)


def tables(text):
    """{heading: [row cells]} for every table under a `##` heading.

    Only the first table under a heading is read, and the header row is
    dropped. A section that grows a second table is a section this script
    would be reading half of, so the first one is the contract and whatever
    follows it is prose.
    """
    found = {}
    heading = None
    state = "idle"  # idle -> body -> done, reset at each heading
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            state = "idle"
            continue
        if heading is None:
            continue
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            if state == "body":
                state = "done"
            continue
        if state == "done":
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if is_separator(cells):
            continue
        if state == "idle":
            state = "body"
            found[heading] = []
            continue  # the header row itself
        found[heading].append(cells)
    return found


def read_tables(problems):
    """The inventory's tables, with the shape of each one checked."""
    found = tables(inventory_text())
    for heading, width in sorted(TABLES.items()):
        rows = found.get(heading)
        if rows is None:
            problems.append(
                Problem(
                    ERROR,
                    heading,
                    "%s has no `## %s` section with a table. Restore the "
                    "heading or update TABLES in this script; a section "
                    "this script cannot find is one it is not checking."
                    % (INVENTORY, heading),
                )
            )
            continue
        for row in rows:
            if len(row) != width:
                problems.append(
                    Problem(
                        ERROR,
                        heading,
                        "a row has %d cells, expected %d: %s"
                        % (len(row), width, " | ".join(row)),
                    )
                )
    return found


def names_in(cell):
    """Every backticked name in a table cell, in order."""
    return BACKTICKED.findall(cell)


def first_name(cell):
    """The one backticked name a first column is expected to carry."""
    found = names_in(cell)
    return found[0] if found else cell.strip()


# --------------------------------------------------------------------------
# What the tree says.
# --------------------------------------------------------------------------


def computed_orphans():
    """{module: labels that reach it} for every native orphan.

    Same definition `connectivity_audit.audit_orphans` uses, read from that
    function's findings rather than recomputed, so the two can never differ.
    """
    return {f.subject: f.detail for f in ca.audit_orphans()}


def registered_binding_modules():
    """Modules under bindings/ that `_mojoboost.mojo` imports."""
    path = os.path.join(ca.BINDINGS_DIR, "_mojoboost.mojo")
    text = ca.must_read(path)
    body = ca.strip_mojo_comments(text)
    names = set()
    for match in re.finditer(r"^\s*from\s+\.?(\w+)\s+import\b", body, re.M):
        names.add(match.group(1))
    for match in re.finditer(r"^\s*import\s+(\w+)\s*$", body, re.M):
        names.add(match.group(1))
    return names


def binding_module_files():
    """Every bindings/*.mojo file except the extension module itself."""
    out = []
    for path in ca.walk(ca.BINDINGS_DIR, ".mojo"):
        if os.path.basename(path) != "_mojoboost.mojo":
            out.append(path)
    return sorted(out)


def base_name(name):
    """A native hook name without its multiclass suffix.

    `python/mojoboost/inspection.py` composes `dump_model_multiclass` at
    call time from a literal `dump_model`, so the regexes in
    `connectivity_audit` see only the base. The inventory lists both, and
    comparing on the base is what lets the two agree.
    """
    if name.endswith("_multiclass"):
        return name[: -len("_multiclass")]
    return name


# --------------------------------------------------------------------------
# The checks.
# --------------------------------------------------------------------------


def check_orphans(rows, problems):
    """The orphan table against the graph and against CLASSIFICATION."""
    computed = computed_orphans()
    listed = {}
    for row in rows:
        module = first_name(row[0])
        listed[module] = (row[1].strip(), row[2].strip())

    for module, (kind, owner) in sorted(listed.items()):
        if module not in computed:
            problems.append(
                Problem(
                    ERROR,
                    module,
                    "%s lists it as an orphan, but a shipping root now "
                    "reaches it. Drop the row and re-audit whatever parity "
                    "row cited it as unreachable." % INVENTORY,
                )
            )
            continue
        want_kind, want_owner, _reason = ca.classify(module)
        if kind != want_kind or owner != want_owner:
            problems.append(
                Problem(
                    ERROR,
                    module,
                    "the inventory says %s/%s; connectivity_audit's "
                    "CLASSIFICATION says %s/%s. The table there is the "
                    "judgment and this one is its rendering, so change the "
                    "rendering or change the judgment, not both silently."
                    % (kind, owner, want_kind, want_owner),
                )
            )

    for module in sorted(computed):
        if module not in listed:
            kind, owner, reason = ca.classify(module)
            problems.append(
                Problem(
                    GAP,
                    module,
                    "no root reaches it and %s does not list it. Add a row: "
                    "| `%s` | %s | %s | %s |"
                    % (INVENTORY, module, kind, owner, reason.rstrip(".")),
                )
            )


def check_binding_modules(rows, problems):
    """The unregistered-binding-module table against the import block."""
    registered = registered_binding_modules()
    listed = {}
    for row in rows:
        path = first_name(row[0])
        listed[path] = row

    for path in sorted(listed):
        stem = os.path.splitext(os.path.basename(path))[0]
        if ca.read(path) is None:
            problems.append(
                Problem(
                    ERROR,
                    path,
                    "%s names it and the file does not exist" % INVENTORY,
                )
            )
            continue
        if stem in registered:
            problems.append(
                Problem(
                    ERROR,
                    path,
                    "%s says the extension does not register it, but "
                    "bindings/_mojoboost.mojo imports it. Drop the row and "
                    "re-audit the capability it said was blocked."
                    % INVENTORY,
                )
            )

    for path in binding_module_files():
        if path in listed:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in registered:
            continue
        problems.append(
            Problem(
                GAP,
                path,
                "bindings/_mojoboost.mojo does not import it and %s does "
                "not list it" % INVENTORY,
            )
        )


def check_unbound_hooks(rows, problems):
    """The unbound-native-name table against the def_function table."""
    exports = set(ca.binding_exports())
    listed = set()
    for row in rows:
        for name in names_in(row[0]):
            listed.add(name)
            if name in exports:
                problems.append(
                    Problem(
                        ERROR,
                        name,
                        "%s says no binding registers it, but the "
                        "def_function table in bindings/_mojoboost.mojo "
                        "exports it. Drop it from the row, and delete the "
                        "Python fallback that row describes." % INVENTORY,
                    )
                )

    covered = {base_name(name) for name in listed}
    for finding in ca.audit_missing_bindings():
        if base_name(finding.subject) in covered:
            continue
        problems.append(
            Problem(
                GAP,
                finding.subject,
                "python/mojoboost reaches for _mojoboost.%s and no binding "
                "registers it; %s does not list it"
                % (finding.subject, INVENTORY),
            )
        )


def check_reachable_rows(rows, problems):
    """Modules in the no-default-effect table must actually be reachable."""
    computed = computed_orphans()
    for row in rows:
        # The first backticked name, so a cell that reads
        # "`device_policy` crossover table" still names its module.
        module = first_name(row[0])
        if module in computed:
            problems.append(
                Problem(
                    ERROR,
                    module,
                    "%s lists it under a reachable heading, but no shipping "
                    "root reaches it. Move the row to the orphan table."
                    % INVENTORY,
                )
            )


def check_cited_paths(problems):
    """Every repository path the inventory names must exist."""
    for path in sorted(set(REPO_PATH.findall(inventory_text()))):
        if ca.read(path) is None:
            problems.append(
                Problem(ERROR, path, "%s cites it; it does not exist" % INVENTORY)
            )


def corrected_table():
    """The orphan table the tree would write today, ready to paste."""
    lines = [
        "| Module | Kind | Owner | Why it is not reached |",
        "|---|---|---|---|",
    ]
    for module in sorted(computed_orphans()):
        kind, owner, reason = ca.classify(module)
        lines.append(
            "| `%s` | %s | %s | %s |"
            % (module, kind, owner, reason.rstrip("."))
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check docs/INTEGRATION_INVENTORY.md against the tree."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on a GAP as well as on an ERROR",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="print the orphan table the tree would write today and exit",
    )
    args = parser.parse_args(argv)

    try:
        if args.table:
            sys.stdout.write(corrected_table() + "\n")
            return 0

        problems = []
        found = read_tables(problems)
        rows = lambda heading: found.get(heading) or []  # noqa: E731
        check_orphans(rows("Orphan native modules"), problems)
        check_binding_modules(
            rows("Binding modules the extension does not register"), problems
        )
        check_unbound_hooks(
            rows("Native names Python reaches for that no binding registers"),
            problems,
        )
        check_reachable_rows(
            rows("Reachable, but with no default effect"), problems
        )
        check_cited_paths(problems)
    except ca.AuditError as error:
        sys.stderr.write("audit_integration: %s\n" % error)
        return 2

    errors = [p for p in problems if p.severity == ERROR]
    gaps = [p for p in problems if p.severity == GAP]

    if not problems:
        sys.stdout.write(
            "%s agrees with the tree: no stale claim, no unrecorded "
            "disconnection.\n" % INVENTORY
        )
        return 0

    if errors:
        sys.stdout.write("\nFalse claims in %s\n%s\n" % (INVENTORY, "-" * 40))
        for problem in errors:
            sys.stdout.write(problem.line() + "\n")
    if gaps:
        sys.stdout.write(
            "\nDisconnections %s has not caught up with\n%s\n"
            % (INVENTORY, "-" * 40)
        )
        for problem in gaps:
            sys.stdout.write(problem.line() + "\n")

    sys.stdout.write(
        "\n  %d ERROR, %d GAP. ERROR means the inventory says something "
        "false and always fails;\n  GAP means a lane landed work the "
        "inventory has not recorded and fails under --strict.\n"
        "  `--table` prints the corrected orphan table.\n"
        % (len(errors), len(gaps))
    )
    if errors or (gaps and args.strict):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
