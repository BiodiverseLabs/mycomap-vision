---
title: 'Held-out benchmark: the ''before'' number'
slug: heldout-benchmark
date: '2026-10-08'
status: done
question: How does the served model do on DNA-verified records that no model has seen?
branch: feat/heldout-benchmark (merged e4a6d38); heldout-summary-format (merged)
commits:
- e4a6d38
- 3b8162f
benchmark: 'heldout-2026-10-08 (13,145 records: 3,000 development, 10,145 test; not sealed)'
split: dev; test
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
- nearest+prior@org
headline: 'Test (10,143 records): species top-1 48.3% nearest, 54.2% nearest + species average, 51.2%
  with the DNA-record prior; genus 80.1 / 83.5 / 80.1.'
verdict: On never-seen records accuracy tracks reference depth as on the newest weeks; the blend wins
  on records it was not chosen on.
decision: 'Steve 2026-10-08: a development benchmark, not the paper''s test set (a fresh ~1,000 will be
  sealed later).'
related:
- depth-bias
- dna-range-prior
- name-equivalence
- published-danish-models
- likely-sets
---

## Question
The number for the current model on records it never saw, before any relabel or retrain.

## Why it matters
It is the fairest single measure available now, and the baseline the retrain is judged against.

## Setup
13,145 North American records validated on the legacy database that never reached Vision (a
synchronization fault), frozen 2026-10-08. Answer key: the observation name. Split 3,000
development / 10,145 test. Photos fetched at 1024 px into the benchmark's own store; every
answer stored ten deep against the served reference index. A never-train list keeps the records
out of every reference and training path.

## What we tried
Nearest; nearest + species average; nearest with the DNA-record prior (place from mycomap.org).

## Results
Test split, % right within the top 1 / 3 / 5 / 10 (species n 9,867; genus n 10,002):

| Method | Row | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|---|
| nearest | species strict | 48.3 | 67.0 | 73.6 | 80.5 |
| | species s.l. | 48.3 | 67.0 | 73.6 | 80.6 |
| | species complex | 51.3 | 69.2 | 75.4 | 81.8 |
| | genus strict / s.l. | 80.1 / 80.8 | 91.0 / 91.2 | 93.7 / 93.8 | 95.8 / 95.9 |
| nearest + species average | species strict | 54.2 | 72.0 | 77.4 | 83.4 |
| | species complex | 57.2 | 73.9 | 79.0 | 84.4 |
| | genus strict / s.l. | 83.5 / 84.1 | 92.8 / 92.9 | 94.7 / 94.7 | 96.3 / 96.3 |
| nearest + DNA-record prior | species strict | 51.2 | 69.3 | 75.8 | 82.1 |
| | genus strict | 80.1 | 90.9 | 93.6 | 95.8 |

Species top-1 (top-5) by the true species' reference records, test:

| References (finds) | nearest | + species average | + DNA-record prior |
|---|---|---|---|
| 0 (344) | 0.3 (0.3) | 0.3 (0.3) | 0.3 (0.3) |
| 1-4 (1,075) | 14.5 (33.5) | 19.2 (39.6) | 17.6 (39.4) |
| 5-19 (2,241) | 32.0 (64.9) | 40.9 (73.7) | 39.4 (69.2) |
| 20-99 (4,512) | 59.4 (86.2) | 66.2 (88.8) | 62.0 (87.3) |
| 100+ (1,695) | 71.3 (92.0) | 73.2 (91.6) | 69.7 (92.4) |

Species top-1 with observer-clustered 95% intervals: 48.3 (45.7-50.6), 54.2 (51.6-56.7), 51.2
(48.6-53.6). Family top-1: 86.8 / 89.5 / 86.5. Development split: species 48.3 / 52.8 / 51.3,
genus 79.0 / 81.7 / 79.5. iNat CV on the same records: not run yet.

Higher than the newest-weeks comparison (34.5%) mainly through the depth mix: 3.5% of finds here
have no reference species, against 12% there.

## Verdict
The "before" number is recorded; nearest + species average is confirmed (depth-bias).

## Decision
Development benchmark only (Steve, 2026-10-08).

## Next
The same records after the relabel and retrain ("after"); iNat on the subsample; seal a fresh
paper test set.
