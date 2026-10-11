---
title: Calibrated lists of likely names
slug: likely-sets
date: '2026-10-08'
status: adopted
reproducibility: exploratory-pre-freeze
question: Can the identifier show a short list of names at each rank with a stated chance that the right
  one is on it?
branch: feat/likely-sets (merged 5a34f2c); fix/likely-sets-memory (merged)
commits:
- 5a34f2c
- 6dc33ca
benchmark: comparison 20261008-012435-4ef7b0 (held-back halves); heldout-2026-10-08 test
split: temporal; test
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
headline: '1,152 records: family 90% coverage with 1.8 names, genus 90% with 3.8, species 55% with 3.4
  (62% when the species has references); held-out test species 70% (nearest) and 77% (blend).'
verdict: Genus and family lists are short and reliable; species lists are capped by species with no reference
  record (31% unlistable on the 1,152).
decision: 'Steve 2026-10-08: adopted; live on the site since 2026-10-09.'
related:
- full-run-finetune
- heldout-benchmark
---

## Question
Instead of one answer, show the smallest list that holds the right name about 9 times in 10?

## Why it matters
Species is often a coin toss between look-alikes; a calibrated list is more honest and more
useful than one name.

## Setup
Split-conformal prediction (least ambiguous set-valued classifier). Target coverage 90%,
stepped down until the mean list size on held-back records is at most five; lists cut at 15
names; a species absent from the references counts as a miss.

## What we tried
Adaptive prediction sets (APS): rejected, 10-12 genera needed for 90%. LAC with the stepping
rule: kept.

## Results
Comparison 4ef7b0 (held-back halves, nearest): family 90% with 1.77 names; genus 90% with 3.8;
species 55% with 3.44 (62.3% when the true species has a reference; 31% of species truths
unlistable). With nearest + species average: species 60% (69%) with 4.1.

Held-out test: family 94.8% with 1.67 (nearest) or 95.1% with 1.43 (blend); genus 94.2% with 3.48
or 94.1% with 2.49; species 70.2% with 3.43 or 76.6% with 3.98.

## Verdict
Adopt; coverage generalizes (held-out coverage is at or above the fitted target).

## Decision
Adopted; live 2026-10-09.

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
Coverage by reference band; refit on the development split; class-conditional lists for rare
species.
