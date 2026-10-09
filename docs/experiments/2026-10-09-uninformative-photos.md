---
title: 'Uninformative photos: drop or down-weight?'
slug: uninformative-photos
date: '2026-10-09'
status: dropped
reproducibility: exploratory-pre-freeze
question: Should photos dominated by a voucher slip, a basket of fungi or habitat be dropped or weighted
  down when a find's photos are combined?
branch: exp/uninformative-photos (merged 4a3265d, docs only)
commits:
- ed93b64
- 4a3265d
benchmark: comparison 20261008-012435-4ef7b0 (1,152); 100-record held-out audit
split: temporal; dev (audit)
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- similarity floor
- confidence / margin weights
- odd-one-out
- few-shot slip detector
- oracle hand labels
headline: No rule moves more than a few records either way; even perfect hand labels of all 369 audit
  photos change species by +2 / -2.
verdict: Low confidence already marks these records honestly; no photo-filtering rule earns a change.
decision: 'Steve 2026-10-09: no change, and don''t propose photo filtering again.'
related:
- photo-views
- nearest-mix
- depth-bias
---

## Question
After the 100-record held-out audit, do slip, basket and habitat photos drag answers down?

## Why it matters
A third of audit records had such a photo; a simple filter would be cheap to ship.

## Setup
Each test photo's best similarity to every species stored once; a rule only changes how much each
photo counts in the mean. Settings chosen on 4ef7b0 by net records fixed, then scored on the
audit. Sign tests; calibration at each rule's own temperature.

## What we tried
Base on 4ef7b0: 34.3 / 71.6 / 79.5 species / genus / family. Fixed / broken vs base:

| Rule (best setting) | Species | Genus | Family | Audit |
|---|---|---|---|---|
| drop photo with best match < 0.52 | 34.3 (+2/-3) | 71.4 (+1/-3) | 79.7 (+3/-1) | no change |
| same, floor 0.65 | 33.5 (+13/-22) | 70.7 | 78.9 | - |
| drop photo > 0.15 below the record's best | 34.3 (+11/-11) | 71.3 | 79.3 | -1 / 0 / 0 |
| weight by own confidence^0.25 | 34.7 (+6/-2) | 71.3 (+5/-8) | 79.6 | +1 / +1 / +1 |
| weight by margin^0.25 | 34.7 (+9/-5) | 71.6 | 79.9 | 0 / -2 / -1 |
| weight exp(best / 0.1) | 34.0 (+18/-22) | 70.9 | 79.8 | -2 / -1 / 0 |
| 3+ photos: drop genus odd-one-out | 33.5 (+11/-20) | 71.0 | 79.0 | 0 / 0 / 0 |
| few-shot slip detector: drop slips | 34.5 (+7/-5) | 71.7 | 79.8 | (trained on audit) |
| drop slips + habitat | 34.6 (+10/-7) | 71.6 | 79.8 | - |

Stronger versions of every rule lose, often significantly (margin^2: 32.3, +26/-49).

## Results
Hand labels of the 369 audit photos: 28 slip-dominated, 2 basket, 19 habitat, 4 microscope.
Dropping exactly those: species 40.4 (+2/-2), genus 76.8 (+2/-3), family 89.6 (+1/-2). The
"every photo points elsewhere" records are mostly ordinary specimen photos of look-alikes; their
mean confidence (0.18 on 4ef7b0) already says "unsure". A perfect photo picker could add about 5
points (some single photo is right in 40.9% of records against 34.3% combined), but no signal
tried finds that photo.

## Verdict
No change to nearest.

## Decision
Dropped; not to be proposed again (Steve, 2026-10-09).

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
None. Better levers: underside and stem photo hints, species-complex scoring, more references.
