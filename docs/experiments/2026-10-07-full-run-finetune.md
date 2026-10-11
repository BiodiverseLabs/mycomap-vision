---
title: Full-data fine-tune on the AWS GPU
slug: full-run-finetune
date: '2026-10-07'
status: adopted
reproducibility: exploratory-pre-freeze
question: How much does fine-tuning BioCLIP 2 on all DNA-verified records improve identification, and
  which model should the site serve?
branch: main (finetune.py, trainer); run 20261007-165400
commits:
- 5c2e617
benchmark: comparison 20261008-012435-4ef7b0 (1,152 test / 152,915 reference records)
split: temporal
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- species-mean
- nearest+mean
- hybrid
- linear
headline: Fine-tuned nearest 34.5 / 71.6 / 79.7 species / genus / family vs frozen 32.6 / 68.6 / 76.3
  on 1,152 records; served since 2026-10-08.
verdict: Fine-tuning the last 4 blocks for 2 epochs adds 2-3 points at every rank for about US$8.
decision: 'Steve 2026-10-07: serve whichever model wins on the mean of the three ranks with nearest: the
  fine-tuned model, live 2026-10-08.'
related:
- backbone-screen
- trained-heads
- depth-bias
- heldout-benchmark
---

## Question
Does fine-tuning pay off at full scale?

## Why it matters
The sample fine-tune (91 records) pointed the right way by only 3 records.

## Setup
One NVIDIA L4 (AWS g6.xlarge, 4 vCPU). Embed all 592,926 large photos with frozen BioCLIP 2
(1 h 31 min), fine-tune the last 4 of 24 blocks on the reference records (validated through
2026-09-07) with cosine heads at species, genus and family initialized from class means,
photos sampled by 1/sqrt(species photos), no hue or saturation augmentation; 2 epochs, 18,328
steps, batch 64, loss 5.7 -> 2.7; then embed with the fine-tuned model. About 9 h and US$8.
Data loading was CPU-bound (59 photos/s training). The classifier heads were not saved.

## What we tried
Frozen and fine-tuned, each with nearest, species average, hybrid and linear heads; nearest +
species average added on 2026-10-09 (depth-bias).

## Results
Comparison 4ef7b0, all photos of a find. Species top 1 / 3 / 5 / 10 (n 1,121 at species) and
genus / family top-1:

| Model | Method | Species top 1 / 3 / 5 / 10 | Genus | Family |
|---|---|---|---|---|
| fine-tuned | nearest + species average | 38.4 / 55.7 / 61.2 / 68.6 | 74.4 | 81.8 |
| fine-tuned | nearest (served) | 34.5 / 50.4 / 56.8 / 65.0 | 71.6 | 79.7 |
| fine-tuned | species average | 34.3 / 52.2 / 61.8 / 68.9 | 72.7 | 81.2 |
| frozen | nearest + species average | 35.3 / 51.0 / 56.2 / 62.4 | 72.1 | 78.6 |
| frozen | nearest | 32.5 / 47.4 / 52.7 / 60.3 | 68.6 | 76.3 |
| frozen | species average | 30.7 / 47.9 / 54.7 / 61.6 | 69.7 | 78.2 |
| fine-tuned | hybrid head | 26.0 | 66.5 | 74.6 |
| fine-tuned | linear head | 17.7 | 55.8 | 66.4 |

(Frozen nearest was 32.6 in the trainer's own run and 32.5 when rescored on the laptop: two
records, from vectors pulled minutes apart.) Species top-1 by the true species' reference
records, fine-tuned nearest: 0 (141 finds): 0; 1-4 (215): 12.6; 5-19 (296): 31.8; 20-99 (364):
54.1; 100+ (105): 65.7. iNat CV on these records: not yet (inat-cv-baseline).

## Verdict
Fine-tuning helps at every rank; the fine-tuned model is served.

## Decision
Adopted (Steve's rule, 2026-10-07); live on the site 2026-10-08.

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Known data fault
A label audit (2026-10-09) found 8,868 of 154,067 reference and training records (5.8%) whose
photos belong to unrelated iNaturalist observations (non-iNat record ids taken for iNat ids).
These results were measured with that reference set; the answer keys are unaffected. The rebuilt
reference set and retrain will be the "after", measured on the new experiment dataset (the
13,145 held-out records join v1's training).

## Next
Relabel and retrain after the legacy-name refresh and full sync; save the heads; try more
blocks, more epochs, a metric-learning loss.
