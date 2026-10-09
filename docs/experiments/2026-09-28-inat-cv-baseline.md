---
title: iNaturalist computer vision on the same records
slug: inat-cv-baseline
date: '2026-09-28'
status: running
reproducibility: exploratory-pre-freeze
question: How does iNaturalist's computer vision do on DNA-verified records, scored exactly like Vision?
branch: main (inat_cv.py, mv inat-baseline; S3-photo fix 3efa072)
commits:
- 3efa072
benchmark: sample comparisons (68, 74, 77, 91, 307 records); heldout-2026-10-08 2,000-record subsample
  (running); 4ef7b0 (to rerun)
split: sample; temporal; dev/test subsample
model: iNat CV (score_image)
methods:
- vision-max (photo only)
- combined-max (with location)
headline: 'Sample comparisons: iNat 26-30% species, 66-71% genus with location; Vision ties at species
  and trails by up to 8 points at genus on the small sample reference set. Held-out subsample and full
  comparison still running.'
verdict: 'Not yet: the fair comparison is on the held-out subsample and the 1,152 records, both still
  running.'
decision: pending
related:
- genus-gap
- heldout-benchmark
- published-danish-models
---

## Question
Where does Vision stand against the identifier most community scientists use?

## Why it matters
No peer-reviewed study has tested iNat's computer vision on fungi against DNA-verified names.

## Setup
Every photo of a record sent to iNat's score_image (with and without the record's place); the
best score across photos per taxon; judged by iNat taxon id. A 24-hour token per run; iNat
throttles to about 2 records per minute.

## What we tried
Photo only (vision-max) and with location (combined-max) on every sample comparison; the 4ef7b0
run; a 2,000-record subsample of the held-out benchmark.

## Results
Species / genus / family top-1, combined-max (photo only):

| Comparison | Records | iNat CV | Vision on the same records |
|---|---|---|---|
| 60274d (sample) | 68 | 29.4 / 67.7 / 70.6 (27.9 / 64.7 / 67.7) | BioCLIP 2 25.0 / 61.8 / 71.4 |
| 85e107 (sample, large photos) | 77 | 27.3 / 66.2 / 70.1 (26.0 / 66.2 / 70.1) | BioCLIP 2 26.0 / 58.4 / 68.2 |
| 60e782 (sample, new labels) | 74 | 29.2 / 68.9 / 74.3 | 27.8 / 61.6 / 72.6 |
| b9ca61 (sample references) | 307 | 26.4 / 71.0 / 76.2 (24.8 / 71.3 / 77.2) | fine-tuned on sample 22.4 / 62.6 / 71.4 |

All against the small sample reference set (8,830 records or fewer). The first 4ef7b0 run sent
no photos (the test photos were only in S3) and scored 1.2%: those rows are void, never
published, and will be replaced by the rerun. A partial rerun (349 older records) is not
comparable and not cited.

Caveat for every result: iNat's model may have trained on some test observations with labels
informed by our DNA results; paper test records will have iNat's identifications snapshotted
first, and the model version and date recorded.

## Verdict
Pending the full comparisons.

## Decision
pending

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
Finish the 2,000-record held-out subsample (about 18:00 UTC 2026-10-09), then rerun 4ef7b0 with a
new token; apply the fair-comparison protocol (A13).
