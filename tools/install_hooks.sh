#!/usr/bin/env bash
#
# Install (or remove) this repository's git hooks.
#
#   tools/install_hooks.sh              install
#   tools/install_hooks.sh --uninstall  remove
#   tools/install_hooks.sh --status     say what is installed
#
# The installed hook is a two-line shim that execs `tools/hooks/pre-commit`,
# so the hook's behavior is versioned in the repository and an update to it
# takes effect without anybody reinstalling. The shim is what lives in
# .git/hooks, which git does not track.
#
# Works in a `git worktree` as well as a normal checkout, because the hooks
# directory is resolved with `git rev-parse --git-path` rather than assumed
# to be .git/hooks.
#
# An existing hook that is not ours is moved aside to `pre-commit.local` and
# named in the output rather than overwritten.

set -euo pipefail

MARKER="# mojotrees-hook-shim"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if custom="$(git config --get core.hooksPath 2>/dev/null)" && [ -n "$custom" ]; then
  case "$custom" in
    /*) HOOKS="$custom" ;;
    *)  HOOKS="$ROOT/$custom" ;;
  esac
  echo "note: core.hooksPath is set, installing into $HOOKS"
else
  HOOKS="$ROOT/$(git rev-parse --git-path hooks)"
fi
TARGET="$HOOKS/pre-commit"

case "${1:---install}" in
  --status)
    if [ -f "$TARGET" ] && grep -q "$MARKER" "$TARGET" 2>/dev/null; then
      echo "installed: $TARGET (mojotrees shim)"
    elif [ -f "$TARGET" ]; then
      echo "present but not ours: $TARGET"
    else
      echo "not installed: $TARGET"
    fi
    exit 0
    ;;
  --uninstall)
    if [ -f "$TARGET" ] && grep -q "$MARKER" "$TARGET" 2>/dev/null; then
      rm -f "$TARGET"
      echo "removed $TARGET"
      if [ -f "$HOOKS/pre-commit.local" ]; then
        echo "note: $HOOKS/pre-commit.local is the hook that was moved aside;"
        echo "      rename it back if you still want it."
      fi
    elif [ -f "$TARGET" ]; then
      echo "left alone: $TARGET was not installed by this script"
    else
      echo "nothing to remove: $TARGET does not exist"
    fi
    exit 0
    ;;
  --install|"")
    ;;
  *)
    echo "usage: tools/install_hooks.sh [--install|--uninstall|--status]" >&2
    exit 2
    ;;
esac

mkdir -p "$HOOKS"

if [ -f "$TARGET" ] && ! grep -q "$MARKER" "$TARGET" 2>/dev/null; then
  mv "$TARGET" "$HOOKS/pre-commit.local"
  echo "moved the existing pre-commit hook to $HOOKS/pre-commit.local"
fi

cat > "$TARGET" <<SHIM
#!/usr/bin/env bash
$MARKER
exec "\$(git rev-parse --show-toplevel)/tools/hooks/pre-commit" "\$@"
SHIM
chmod +x "$TARGET"

echo "installed $TARGET"
echo
echo "It runs a gate only when that gate's artifact is staged:"
echo "  compatibility/api_snapshot.json  ->  tools/api_snapshot.py --check"
echo "  docs/LIGHTGBM_PARITY.md          ->  tools/check_parity.py"
echo "  pixi.toml                        ->  tools/check_pixi_tasks.py"
echo "  docs/INTEGRATION_INVENTORY.md    ->  tools/audit_integration.py"
echo "  tools/connectivity_audit.py      ->  tools/audit_integration.py"
echo "  tools/default_refusal_audit.py   ->  tools/default_refusal_audit.py"
echo "  tests/test_*.mojo                ->  tools/audit_test_structure.py"
echo
echo "The last is a shape rather than a generated artifact: a test file must"
echo "define a def main() that reaches the suite and import from std.testing."
echo "It is a text check and does NOT prove the file compiles."
echo
echo "Commits touching none of those are unaffected."
echo "Bypass once with MOJOTREES_SKIP_GATES=1; remove with --uninstall."
