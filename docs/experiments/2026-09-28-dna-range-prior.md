---
title: Range and season prior from DNA-verified records
slug: dna-range-prior
date: '2026-09-28'
status: done
reproducibility: exploratory-pre-freeze
question: Does a range-and-season score built from our own DNA-verified records improve identification?
branch: main (prior.py, methods '+prior')
commits: []
benchmark: 8,000-record sample; 100-record held-out pilot; heldout-2026-10-08 dev and test
split: sample; dev; test
model: bioclip-2 frozen; bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+prior@org
headline: 'Held-out test: species 48.3 -> 51.2% (744 fixed, 454 broken), genus 80.1 -> 80.1%, family 86.8
  -> 86.5%; it loses to nearest + species average (54.2%).'
verdict: Too sparse on the small sample; on the full held-out set it adds about 3 points at species, mostly
  for species with 5-19 references, and nothing at genus.
decision: Kept as an experimental method, not the default. Range work continues with the iNat occurrence
  prior.
related:
- occurrence-prior
- heldout-benchmark
- depth-bias
---

## Question
Should the species score be weighted by how plausible the species is at that place and date,
judged from our own DNA-verified records?

## Why it matters
iNaturalist's location model adds about 1.5 points for iNat. Our references are sparse, so a
prior from them may penalize the right species.

## Setup
A capped prior multiplied into the photo score, place from the record (mycomap.org's
coordinates, `@org`) and date.

## What we tried
Sample, prior weight 0, 0.1, 0.25, 0.5 and 1 on nearest. 100-record pilot. Full held-out
benchmark, development and test splits.

## Results
Sample (82 of 91 test records have a place), species / genus top-1 by weight 0 / 0.1 / 0.25 /
0.5 / 1: species 24.2 / 23.1 / 23.1 / 22.0 / 23.1; genus 59.3 / 59.3 / 56.0 / 51.6 / 47.2.

Pilot: 45 to 47 of 95 right; in 22 of 95 the true species had no DNA record within 300 km.

Held-out test (10,143 records), top-1 %: species 48.3 -> 51.2, genus 80.1 -> 80.1, family 86.8 ->
86.5. Development: species 48.3 -> 51.3 (209 fixed, 123 broken, McNemar P < 0.001); genus 79.0 ->
79.5 (P = 0.37). By the true species' reference records (test, species top-1): 0: 0.3 -> 0.3;
1-4: 14.5 -> 17.6; 5-19: 32.0 -> 39.4; 20-99: 59.4 -> 62.0; 100+: 71.3 -> 69.7.
Full standard tables: heldout-benchmark.

The prior's stated confidence saturated near 100% (it fell back to the cosine temperature);
fixed on feat/occurrence-prior.

## Verdict
Helps species on large data, not genus or family, and less than the species-average blend.

## Decision
Not the default; kept as `+prior`.

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
iNat occurrence prior (occurrence-prior), then whether either prior adds to nearest + species
average.
