#!/bin/sh
# Keep the benchmark datasets from being deleted by anything running on this
# machine, including this repository's own tooling and any other Claude Code
# session sharing the checkout.
#
# WHY THIS EXISTS, stated plainly so it does not get "cleaned up" later.
# On 2026-08-18 ~/gbm-datasets/covtype was emptied between a benchmark run and
# the next command.  YearPredictionMSD.txt.zip in the same directory survived,
# and the ONLY difference between them was that the zip carried the macOS
# `uchg` (user immutable) flag and covtype did not.  That is the whole design
# of this script: the flag is what saved the file, so the flag goes on
# everything, every time, rather than whenever somebody remembers.
#
# WHAT `uchg` ACTUALLY STOPS, and what it does not.
#   STOPS: rm, rm -rf, git clean -xfd, mv over the file, truncation by `>`,
#          find -delete, and every accidental deletion this threat model is
#          about.  `unlink` fails with EPERM before the filesystem is touched.
#   DOES NOT STOP: a deliberate `chflags nouchg` followed by a delete.  Any
#          process running as this user can do that.  This defends against
#          mistakes, not against intent, and the distinction is real: every
#          loss so far has been a mistake.
#
# THE VAULT is the second half, and it is the half that matters.  Locking the
# working copy protects files that already exist; it does nothing for a file
# downloaded five minutes from now into an unlocked directory.  So a locked
# COPY lives at $VAULT with the flag on the directory as well as on the files,
# which makes the directory refuse creation and deletion of entries outright.
# `restore` copies back out of it.  A loss then costs a copy rather than a
# re-download over a slow link.
#
# The vault is on the same disk, so it is not a backup: it survives `rm -rf`,
# it does not survive disk failure.  Time Machine or an external copy is the
# answer to that and this script is not it.

set -eu

DATA="${GBM_DATASETS:-$HOME/gbm-datasets}"
VAULT="${GBM_DATASETS_VAULT:-$HOME/gbm-datasets-vault}"

# THE SECOND DATASET DIRECTORY, AND IT IS INSIDE THE GIT REPOSITORY.
#
# Added 2026-08-18. `bench/real_data/loaders.py::cache_dir` defaults to
# `bench/real_data/data` when MOJOTREES_BENCH_DATA is unset, and on this
# machine that directory held 227 MB of real datasets, unprotected, in a
# SHARED CHECKOUT: adult, bank_marketing, covertype, rcv1_train_binary and
# year_prediction_msd. Covertype and year are the two datasets every
# performance number of 2026-08-18 came from.
#
# `git clean -xfd` deletes gitignored files. So the default location for the
# benchmark corpus was the one place a routine command run by any of three
# concurrent sessions could vaporize it, and it had been that way the whole
# time we were worrying about `rm -rf`.
#
# It is covered here rather than moved, because moving it would orphan a cache
# every existing result record was measured against. Point
# MOJOTREES_BENCH_DATA outside the repository for new work and this directory
# stops growing.
REPO_DATA="${MOJOTREES_BENCH_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." \
    && pwd)/bench/real_data/data}"

usage() {
    cat <<USAGE
usage: protect_datasets.sh <command>

  lock      Flag every file under $DATA immutable, copy anything new into the
            vault, and relock the vault.  RUN THIS AFTER EVERY DOWNLOAD.
  unlock    Clear the flag on $DATA only, so a downloader can write there.
            The vault stays locked.  Follow with 'lock'.
  status    Show what is protected and what is not.  Exits 1 if any file
            under $DATA is unprotected, so CI or a hook can gate on it.
  verify    Prove the protection by trying to delete a decoy, rather than
            asserting it.  Changes nothing real.
  restore   Copy anything present in the vault and missing from $DATA back.
USAGE
}

flagged() { ls -lO "$1" 2>/dev/null | awk '{print $5}' | grep -q uchg; }

cmd_lock() {
    [ -d "$DATA" ] || { echo "no dataset directory at $DATA"; exit 1; }
    find "$DATA" -type f -exec chflags uchg {} + 2>/dev/null || true
    # The in-repo cache is flagged but NOT vaulted. Flagging stops rm and
    # git clean, which is the whole exposure; mirroring it would double 227 MB
    # of files that are re-downloadable from a pinned checksum, which the
    # manual datasets in $DATA are not.
    if [ -d "$REPO_DATA" ]; then
        find "$REPO_DATA" -type f -exec chflags uchg {} + 2>/dev/null || true
    fi
    mkdir -p "$VAULT"
    chflags nouchg "$VAULT" 2>/dev/null || true
    # -n never overwrites, so a corrupted working copy cannot poison the vault.
    ( cd "$DATA" && find . -type f -print ) | while read -r rel; do
        mkdir -p "$VAULT/$(dirname "$rel")"
        chflags nouchg "$VAULT/$(dirname "$rel")" 2>/dev/null || true
        cp -n "$DATA/$rel" "$VAULT/$rel" 2>/dev/null || true
    done
    find "$VAULT" -type f -exec chflags uchg {} + 2>/dev/null || true
    find "$VAULT" -type d -exec chflags uchg {} + 2>/dev/null || true
    echo "locked: $DATA and $VAULT"
}

cmd_unlock() {
    find "$DATA" -type f -exec chflags nouchg {} + 2>/dev/null || true
    if [ -d "$REPO_DATA" ]; then
        find "$REPO_DATA" -type f -exec chflags nouchg {} + 2>/dev/null || true
    fi
    find "$DATA" -type d -exec chflags nouchg {} + 2>/dev/null || true
    echo "unlocked $DATA (vault untouched).  Run 'lock' when the download finishes."
}

cmd_status() {
    rc=0
    echo "working copy: $DATA"
    if [ -d "$DATA" ]; then
        find "$DATA" -type f | while read -r f; do
            if flagged "$f"; then echo "  PROTECTED   $f"
            else echo "  UNPROTECTED $f"; fi
        done
        find "$DATA" -type f | while read -r f; do
            flagged "$f" || exit 1
        done || rc=1
    else
        echo "  (missing)"; rc=1
    fi
    echo "in-repo cache: $REPO_DATA"
    if [ -d "$REPO_DATA" ]; then
        find "$REPO_DATA" -type f | while read -r f; do
            if flagged "$f"; then echo "  PROTECTED   $f"
            else echo "  UNPROTECTED $f"; rc=1; fi
        done
    else
        echo "  (none, MOJOTREES_BENCH_DATA points elsewhere or nothing cached)"
    fi
    echo "vault: $VAULT"
    if [ -d "$VAULT" ]; then
        flagged "$VAULT" && echo "  directory itself is immutable" \
                         || echo "  WARNING directory is NOT immutable"
        find "$VAULT" -type f -exec echo "  vaulted     {}" \;
    else
        echo "  (missing)"; rc=1
    fi
    return $rc
}

cmd_verify() {
    # A decoy rather than a real file: proving the mechanism must not risk the
    # thing the mechanism protects.
    d=$(mktemp -d)
    printf 'decoy\n' > "$d/decoy"
    chflags uchg "$d/decoy"
    if rm -f "$d/decoy" 2>/dev/null; then
        echo "FAIL: uchg did not stop rm on this filesystem."
        chflags nouchg "$d/decoy" 2>/dev/null || true; rm -rf "$d"; exit 1
    fi
    [ -f "$d/decoy" ] || { echo "FAIL: decoy vanished."; rm -rf "$d"; exit 1; }
    if rm -rf "$d" 2>/dev/null && [ ! -d "$d" ]; then
        echo "FAIL: rm -rf removed a tree holding an immutable file."
        exit 1
    fi
    chflags nouchg "$d/decoy"; rm -rf "$d"
    echo "PASS: uchg blocks rm and rm -rf on this filesystem."
}

cmd_restore() {
    [ -d "$VAULT" ] || { echo "no vault at $VAULT"; exit 1; }
    mkdir -p "$DATA"
    ( cd "$VAULT" && find . -type f -print ) | while read -r rel; do
        if [ ! -f "$DATA/$rel" ]; then
            mkdir -p "$DATA/$(dirname "$rel")"
            cp "$VAULT/$rel" "$DATA/$rel"
            chflags uchg "$DATA/$rel"
            echo "restored $rel"
        fi
    done
}

case "${1:-}" in
    lock) cmd_lock ;;
    unlock) cmd_unlock ;;
    status) cmd_status ;;
    verify) cmd_verify ;;
    restore) cmd_restore ;;
    *) usage; exit 2 ;;
esac
