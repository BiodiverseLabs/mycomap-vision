---
title: Is nearest biased toward well-sampled species? (nearest + species average)
slug: depth-bias
date: '2026-10-09'
status: adopted
question: Does scoring a species by its single best-matching photo favour species with many reference
  photos, and what scoring fixes it?
branch: exp/depth-bias (merged fb396f0)
commits:
- 3d31dca
- 319e86b
- fb396f0
benchmark: comparison 20261008-012435-4ef7b0 (choice); heldout-2026-10-08 dev and test (confirmation)
split: temporal (chosen); dev, test (confirmed)
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- depth penalties
- hubness
- per-depth offsets
- top-k means
- species-mean
- nearest+mean
headline: 'Nearest + species average (0.6 x mean of top-2 matches + 0.4 x species average): 34.5 -> 38.4%
  species on 1,152 records; confirmed on held-out test 48.3 -> 54.2% (898 fixed, 308 broken).'
verdict: Taking depth out of the score loses; blending each species' best matches with its average photo
  wins in every band with references.
decision: 'Steve 2026-10-09: nearest + species average is the default photos-only method (MV_DEFAULT_METHOD,
  feat/nearest-mean-default); live once deployed and its calibration is on the box.'
related:
- full-run-finetune
- heldout-benchmark
- nearest-mix
- trained-heads
---

## Question
Species accuracy rises steeply with the true species' reference records (a 99-record audit: 12%
under 20 references, 63% at 20 or more), and in 37 of 55 misses the winner had more references.
Is nearest's maximum over photos the cause?

## Why it matters
Half the names have one record; if scoring were biased against them, fixing it would help most
species.

## Setup
Every variant re-scores the same stored similarities of comparison 4ef7b0 (no retraining).
Settings chosen there; confirmed on the held-out development and test splits.

## What we tried
Species top-1 on 4ef7b0 by the true species' reference records (finds 141 / 215 / 296 / 364 / 105):

| Scoring | All | 1-4 | 5-19 | 20-99 | 100+ | fixed / broken vs nearest |
|---|---|---|---|---|---|---|
| nearest | 34.5 | 12.6 | 31.8 | 54.1 | 65.7 | |
| best - 0.005 ln N | 33.9 | 13.0 | 33.4 | 51.9 | 61.0 | +10 / -17 |
| best - 0.01 ln N | 33.7 | 14.4 | 34.5 | 51.4 | 55.2 | +24 / -33 |
| best - 0.02 ln N | 29.1 | 16.7 | 33.8 | 41.2 | 38.1 | +39 / -100 |
| best + 0.005 ln N | 35.0 | 12.6 | 29.4 | 56.0 | 70.5 | +17 / -12 |
| best - expected best (x0.25) | 33.8 | 14.9 | 34.5 | 51.4 | 55.2 | +25 / -33 |
| per-species z-score | 12.1 | 18.6 | 16.9 | 11.8 | 2.9 | +35 / -286 |
| hubness (top-50 x0.25) | 34.3 | 15.3 | 33.8 | 51.6 | 60.0 | +30 / -33 |
| per-depth offsets (2-fold by observer) | 33.9 | 13.5 | 34.8 | 49.2 | 65.7 | +18 / -25 |
| mean of top-2 photos | 36.0 | 10.7 | 32.8 | 56.6 | 73.3 | +36 / -20 |
| mean of top-5 photos | 35.3 | 8.8 | 29.7 | 57.1 | 77.1 | +53 / -44 |
| top-2 over distinct records | 35.6 | 9.3 | 31.8 | 57.1 | 73.3 | +43 / -31 |
| species average | 34.3 | 13.5 | 39.2 | 50.8 | 52.4 | +87 / -89 |
| 0.5 nearest + 0.5 species average | 37.9 | 14.0 | 40.9 | 57.4 | 61.9 | +70 / -32 |
| **0.6 top-2 + 0.4 species average** | **38.4** | 13.0 | 39.9 | 58.5 | 67.6 | **+76 / -33** |

k = 2 and weight 0.6 sit on a plateau (0.5-0.65 with top-2: 38.2-38.4%). Fourteen variants
were tried on this one comparison, which is why the confirmation below matters.

## Results
On 4ef7b0: species / genus / family top-1 34.5 / 71.6 / 79.7 -> 38.4 / 74.4 / 81.8; species top-5
56.8 -> 61.2; first photo only 28.9 -> 35.0; species pairs +76 / -33 (sign test P < 0.0001),
genus +56 / -25; provisional names +4.7 points.

Held-out development (choice confirmed, not re-tuned), species top-1 (top-5) by band, nearest ->
blend: 0 (99): 1.0 -> 1.0; 1-4 (307): 12.4 -> 16.0 (34.5 -> 40.4); 5-19 (682): 34.2 -> 40.9 (65.8 ->
73.6); 20-99 (1,344): 59.3 -> 65.6 (86.1 -> 87.5); 100+ (483): 70.2 -> 67.9 (91.5 -> 91.3), the only
band that lost (339 -> 328 right). Overall 48.3 -> 52.8 species (235 fixed, 104 broken), genus 79.0
-> 81.7.

Held-out test (confirmation only): species 48.3 -> 54.2 (898 / 308), genus 80.1 -> 83.5; 100+
band 71.3 -> 73.2. Full tables: heldout-benchmark.

## Verdict
Depth penalties trade a few sparse-species records for more deep ones: test records arrive at
real frequencies, so a deep species really is more likely. Most sparse-species misses are far
misses (truth below rank 20 for 46% of 1-4-reference truths). What helps is steadier evidence
per species: the average photo is good for species with a few records, the best match for
species with many, and the blend keeps both.

## Decision
Adopted (Steve, 2026-10-09): the default photos-only method on the site. Identification shares
the photo similarities with the specimens it shows, so the blend no longer computes them twice.

## Next
A `+prior` / `+occ` version of the blend; share the photo similarities in identify.py (computed
twice now); time it on the 4 GB box before switching.
