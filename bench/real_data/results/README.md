# Results

Empty, and not by accident. This harness has never been run.

Results land here as `results/<run_id>/`, holding the manifest, one job
file and one record per run, the predictions, and the concatenated
`records.json` and `records.csv`. None of it is committed: `.gitignore` in
the parent directory keeps everything under this directory out of the
repository except this file.

## Why results are not committed

A benchmark result is a measurement of a machine on a day. Committed to a
repository it becomes a claim about a library, quoted later by someone who
never saw the manifest that said the laptop was on battery. The record
format carries the machine, the versions, the git commit, the thread count,
the thermal state, and the dataset digests precisely so that a result can be
read in context, and a file in git is exactly where that context gets lost.

When a number is worth publishing, a person publishes it: reads the
distribution, names the conditions, writes the sentence, and puts it
somewhere it can be dated and corrected. `README.md` and
`docs/LIGHTGBM_PARITY.md` are that somewhere.

## Before anything from here is quoted

- `verify.py` returned zero on the run. A red verdict means the two engines
  were not compared on the same problem, whatever the timings say.
- The manifest's environment block is quoted with the number: machine,
  thread count, device, mojo and lightgbm versions, and the commit.
- The record says `data_kind: real` and `pinned: true`, if the number is
  described as a real-data result.
- The cell is not marked `!` in the report, or the mark is quoted too.
- At least three repeats. `report.py` will not print a ratio from fewer,
  and a ratio it refused to print should not be reconstructed by hand.

A number that fails any of these is not a result yet. It is an observation,
and it belongs in a message to whoever is working on that lane, not in a
README.
