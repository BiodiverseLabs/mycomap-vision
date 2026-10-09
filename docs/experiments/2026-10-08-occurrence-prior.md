---
title: Range and season from iNat occurrence data
slug: occurrence-prior
date: '2026-10-08'
status: running
question: Does a range-and-season score from iNaturalist's open occurrence data, with only a wide berth
  for out of range, improve identification where our DNA records are too sparse?
branch: feat/occurrence-prior (merged 88cc672)
commits:
- '3972727'
- a531bac
- 88cc672
benchmark: heldout-2026-10-08 development split (tuning), test (check)
split: dev
model: bioclip-2-ft-20261007-165400
methods:
- nearest+occ
- species-mean+occ
headline: 'Built: 9.07 million North American fungal observations counted as observer-days per 0.5 degree
  cell; not tuned or scored yet.'
verdict: Pending the development-split tuning.
decision: 'pending (Steve 2026-10-08: use iNat occurrence counts; out of range only as a wide berth, about
  1,500 km).'
related:
- dna-range-prior
- heldout-benchmark
---

## Question
Our own DNA records are too sparse to draw ranges (dna-range-prior); does iNat's open data do
better?

## Why it matters
iNat's location model adds about 1.5 points for iNat; the DNA-record prior added about 3 at
species on the held-out set but nothing at genus.

## Setup
`mv build-occurrence` on the 2026-09-27 iNaturalist open-data export: kingdom Fungi by ancestry,
research and needs-ID grades, inside North America plus 5 degrees, accuracy no worse than 25 km.
283 million rows read, 9.07 million fungal observations kept, 21,710 taxa. Counts are
observer-days (ten photos on one walk are one find), per 0.5 degree cell, per species and genus,
with season by week within 10 degree latitude bands. A scored record's own observation is taken
back out of the counts.

Scores: out of range = a strong penalty only when a species with at least 20 occurrences has none
within 1,500 km and no DNA record nearby; density near the place (150 km kernel, shrunk to genus,
capped); season (capped). MycoMap Atlas range maps are a second, pluggable source (stub).

## What we tried
Built and unit-tested; the grid search (`mv tune-occurrence`: radius, penalty including outright
exclusion, weights, temperature) on the development split has not run.

## Results
None yet.

## Verdict
Pending.

## Decision
pending

## Next
Tune on development, check once on test; count how often the true species is penalized; try it
on top of nearest + species average.
