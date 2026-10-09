---
title: Place and date prior on top of nearest + species average, tuned on development
slug: prior-tuning
date: '2026-10-09'
status: running
question: Does a place-and-date prior (our DNA records, iNat occurrences, or both, with a season
  weight) add to nearest + species average once its settings are tuned honestly on development?
branch: exp/prior-tuning
commits: []
benchmark: heldout-2026-10-08 development split (tuning, 5-fold cross-validation by observer); test
  (confirmation only, once)
split: dev
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
- nearest+prior@org
- nearest+mean+prior
- nearest+mean+occ
- nearest+mean+prior+occ
headline: 'Grid declared before any result: 1,824 settings over three prior families; not run yet.'
verdict: Pending.
decision: pending (Steve 2026-10-09 approved the experiment, CPU only, development only).
related:
- occurrence-prior
- dna-range-prior
- depth-bias
- heldout-benchmark
---

## Question
nearest + species average is the photos-only default. Does adding where and when a fungus was
found improve it, and which prior: our own DNA records, iNaturalist's open occurrence data, or
both?

## Why it matters
The DNA-record prior added about 3 points at species on nearest, untuned. Our DNA records are too
sparse to draw ranges for sparse species; iNat's occurrence data is denser but was never tuned.

## Setup
Development split of heldout-2026-10-08 (records with photos), scored against the reference
index of the fine-tuned BioCLIP 2 as it stands today. Photo scores are on the log-probability
scale at temperature 0.02; the prior is added to them. Place is the record's own place on
mycomap.org; each record's own iNat observation is taken back out of the occurrence counts.

Grid, declared in `priortune.py` before any result was seen (every setting scored and kept):
- DNA prior (288): place kernel 75 / 150 / 300 km; season kernel 10 / 20 / 40 days; place and
  season weights 0 / 0.25 / 0.5 / 1 each; cap log 5 or log 20 per term.
- iNat occurrence prior (1,152): wide berth 1,000 / 1,500 / 2,000 km; out-of-range penalty
  0 / 2 / 6 / 50 (exclusion) nats; genus rule on or off; density kernel 75 / 150 / 300 km;
  density and season weights 0 / 0.25 / 0.5 / 1 each.
- Combined (384): berth 1,500 km with the genus rule; penalty 0 / 2 / 6 / 50; iNat density
  0 / 0.25 / 0.5; DNA place 0 / 0.25 / 0.5 / 1; season from DNA or iNat, weight
  0 / 0.25 / 0.5 / 1.
- Reference, never chosen: the DNA prior as it ships (nearest+prior's).

Selection: species top-1, then genus top-1, then the gentler setting. Reported number: 5-fold
cross-validation with folds grouped by observer, each fold scored with the setting chosen on the
other four. Test is scored once, with the setting chosen on all of development.

## What we tried
Pending.

## Results
Pending.

## Verdict
Pending.

## Decision
pending

## Next
Run the grid; report in the standard format; confirm once on test.
