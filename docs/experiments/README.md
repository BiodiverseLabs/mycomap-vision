# Experiment registry

Every MycoMap Vision experiment gets one file here, so it can be reviewed later: what was
asked, everything that was tried, what came out, and what was decided. The site shows them
under Research > Experiments to signed-in mycomap.org members (they hold development
numbers and unpublished results). `tests/test_experiments.py` checks every file.

## File

`docs/experiments/YYYY-MM-DD-<slug>.md`, the date the experiment started (or was first
recorded), the slug in lower case with hyphens.

```markdown
---
title: Plain title
slug: the-slug                # must match the file name
date: 2026-10-09              # must match the file name
status: done                  # planned | running | done | adopted | dropped
reproducibility: exploratory-pre-freeze   # or reproduced-on-v1 (a dataset release id)
question: One sentence.
branch: exp/the-branch        # where the work is (or "none")
commits: [abc1234]
benchmark: heldout-2026-10-08 # or the comparison id, or "8,000-record sample"
split: dev                    # dev | test | temporal | sample | n/a
model: bioclip-2-ft-20261007-165400
methods: [nearest, nearest+mean]
headline: One line with the key numbers and the records they are on.
verdict: What it showed, in one sentence.
decision: Steve's decision, with the date, or "pending".
related: [other-slug]
# Required once reproducibility is reproduced-on-<release>:
dataset_release: v1
reference_hash: 4ef7b0035bc9...   # the release's reference index (records + labels)
code_commit: abc1234
reproduce_command: mv compare --release v1 --backbones ... --methods ...
---

## Question
## Why it matters
## Setup
## What we tried
## Results
## Verdict
## Decision
## Next
```

## Statuses

- **planned**: designed and ready, not run yet.
- **running**: under way, or built and waiting to be tuned or scored.
- **done**: measured; no change to the identifier depends on it (or the decision is pending).
- **adopted**: measured, and the result is now part of the identifier or the scoring.
- **dropped**: measured and not taken up. Say whether it may come back.

## Reproducibility

Until dataset release v1 is frozen (docs/PLAN.md, "a reproducible dataset release"), every entry
is `exploratory-pre-freeze`: it read the live manifest, whose records, labels and photos kept
changing, so its numbers cannot be rebuilt exactly and every decision taken from it is
provisional. After the freeze, an entry is `reproduced-on-<release>` and names the release, its
reference hash, the code commit and the one command that rebuilds every number; the tests refuse
an entry dated on or after the freeze without them.

## Rules

1. **Record every variant tried**, not only the winner, with its numbers. Choosing the best of
   many settings on one set inflates that set's number; the reader needs to see how many were
   tried.
2. **Tune on development only.** Settings are chosen on the development split (or the
   comparison they were fitted on, said plainly); a test split confirms, it does not choose.
3. **The answer key is the record's observation name** on MycoMap after a refresh from the
   legacy database, never the name of a linked sequence.
4. **Results in the standard format** where the experiment produces them: top 1 / 3 / 5 / 10
   by species strict, *s.l.* and complex, genus strict and *s.l.*; species by the true
   species' reference records (0, 1-4, 5-19, 20-99, 100+); iNaturalist's computer vision on
   the same records, or "not run". State n for every table.
5. **Say what the numbers are**: which records, how many, development or test, and whether
   the set was sealed. A development benchmark is never called the paper's test set.
6. **No data in these files**: no record or observation ids, names of people, coordinates,
   or file paths to photos. Aggregates only.
7. **Small sets are direction, not results**: say so when n is under a few hundred.
8. Link the code (branch, commits) and any longer write-up (docs/PLAN.md sections, the
   replication README) rather than copying them.
9. **After the freeze, research reads a release, never the live manifest**, and every analysis
   ships as one re-runnable command (`--release` / `--manifest`).
