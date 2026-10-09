---
title: Per-photo votes instead of mean similarity (nearest-mix)
slug: nearest-mix
date: '2026-09-29'
status: dropped
question: Does turning each photo's scores into probabilities and averaging those beat the plain mean
  of best matches?
branch: feat/nearest-mix (bfbb0ac, not merged)
commits:
- bfbb0ac
benchmark: 5,704 multi-photo sample records (leave-own-record-out); comparison 20261007-061934-b9ca61
  (307 test)
split: sample; temporal
model: bioclip-2 frozen; bioclip-2-ft-sample
methods:
- nearest
- nearest-mix (T 0.02, 0.05, 0.1)
headline: '307 records: fine-tuned 21.8 / 60.3 / 69.4 (mix) vs 22.4 / 62.6 / 71.4 (nearest) species /
  genus / family; sharper votes lose (T 0.02: 21.1 vs 22.2%).'
verdict: No gain at any temperature; combining photos is not where the errors are.
decision: 'Steve 2026-09-29: hold until a full-set rerun; it showed no gain there either, so dropped.'
related:
- depth-bias
- uninformative-photos
---

## Question
Is the way photos are combined holding the identifier back?

## Why it matters
A find has 3.2 photos on average; the combination rule decides which photo wins.

## Setup
Per-photo softmax over species, then average the probabilities (averaging log-probabilities
would rank like the plain mean).

## What we tried
Temperatures 0.02, 0.05 and 0.1 on 5,704 multi-photo sample records (each record's own photos
masked out of the references); then T 0.05 on the 307-record comparison, frozen and sample
fine-tune.

## Results
Leave-own-record-out, species top-1: T 0.02 21.1% vs nearest 22.2% (141 vs 203 records right
only one way); T 0.05 and 0.1 tie.

307 records (comparison 20261007-061934-b9ca61), species / genus / family top-1 %:

| Model | nearest | nearest-mix |
|---|---|---|
| fine-tuned on sample | 22.4 / 62.6 / 71.4 | 21.8 / 60.3 / 69.4 |
| frozen | 21.8 / 60.7 / 70.7 | 20.8 / 61.3 / 70.1 |

## Verdict
No gain. Later work (depth-bias) found the gain is in steadier evidence per species, not in
combining photos.

## Decision
Dropped (held, then no gain on retest).

## Next
None.
