# Corrective record: work swept into 142e32f by the rename session

Status: closed, no action required from the affected lanes. This exists so
that `git log` does not mislead anyone about who wrote what.

## What happened

On 2026-08-15 the rename session (mojoboost -> mojotrees) ran `git add -A`
instead of committing by explicit path, in a checkout shared with several
live sessions. Two other lanes had uncommitted work in the tree at that
moment, and it was committed and pushed under a commit message that
describes none of it.

The offending commit is **142e32f**, whose message is
"Say --pre in every install instruction, because pip ignores pre-releases".
That message accurately describes 8 of the 100 files in it. It does not
describe the other 92.

Both 142e32f and its parent cc4d847 were already pushed to `main` before
this was noticed, so history was NOT rewritten. Rewriting a pushed `main`
in a checkout with several live sessions would cost more than the wrong
commit message does.

## Exactly what was swept, by lane

**GPU-scheduling lane (connect_04).** Documentation only, no code change:

- `src/mojotrees/train_gpu.mojo` (module docstring, division-of-labor note
  pointing at `docs/ARCHITECTURE.md` seam 4)
- `src/mojotrees/hybrid_leaf_scheduler.mojo` (a "what this module does not
  do" paragraph on why placement is substitution rather than a host/device
  race, citing HYBRID_TRAINING.md section 9 E5)
- `docs/design/HYBRID_TRAINING.md`
- `docs/ARCHITECTURE.md`

**Unified-memory bench lane.**

- `bench/apple/unified_memory.mojo`
- `bench/results/apple_m4_unified_memory_2026-08-15/` (86 files: `.out`,
  `.time`, `.vm_before.txt`, `.vm_after.txt` across the resident and
  rewrite arms at 256 and 1024)

**Actually belonging to 142e32f's stated purpose** (8 files): `README.md`,
`python/README.md`, `docs/INSTALLATION.md`, and the five `launch/` drafts.

## Consequences worth knowing

Nothing was lost. Both lanes' work is intact on `main`.

`main` is not broken by it. The `src/` changes are comments only, which is
why CI was never at risk from the sweep.

**A clean `git status` is no longer proof that a lane has not landed.** The
consolidation plan originally gated wave-2 lanes on "files clean in git
status" as the landed signal, and this sweep falsified that signal for
connect_04, whose files went clean without the lane having finished. That
plan has been re-gated on handoff notes marking a lane landed, with clean
status only as a secondary check. Any other process keyed to clean status
should be re-checked against the same failure.

## The rule this violated

Commit by explicit path in this checkout, always. Never `git add -A`, never
`git commit -a`. Several sessions share one working tree, so a broad add
stages whatever anyone else happens to have in flight. When a peer has
edits in a file you also touched, split the diff into hunks, stage only
your own with `git apply --cached`, and commit with no paths so the commit
is exactly the index.
