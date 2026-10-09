---
title: Observation sets (set-to-set matching and a trained attention set head)
slug: observation-sets
date: '2026-10-09'
status: done
question: Does scoring a record's photos as a set, against reference observations as sets, identify
  species better than nearest+mean's photo-by-photo rule?
branch: exp/observation-sets (not merged)
commits:
- 0494b90
benchmark: heldout-2026-10-08 (choice); comparison 20261008-012435-4ef7b0 layout as a time-slice check
split: dev (chosen); temporal (checked); test not run
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
- nearest+prior@org
- obs-forward
- obs-chamfer
- obs-chamfer-top2
- blends with nearest+mean
- set-mean (untrained pooling)
- set-head (gated attention MIL, proxy loss)
- set-head, attention only
headline: 'Dev 2,915 species-named records, species top-1 vs nearest+mean 52.9%: set methods alone
  45.4-53.1% (none better); best blends 54.5% (set matching) and 55.1% (trained head), but on the
  1,109-record time slice the same blends are level (38.6% and 38.9% vs 38.5%).'
verdict: Nothing beats nearest+mean robustly; blends gain on dev only in species with 20+ references,
  and the overall gain does not carry over to an independent time slice.
decision: pending
related:
- depth-bias
- uninformative-photos
- photo-views
- heldout-benchmark
- name-equivalence
- trained-heads
- nearest-mix
---

## Question
Vision scores photos one at a time and combines them by fixed rules: `nearest` averages each photo's
best match within a species; `nearest+mean` blends each photo's two best matches with the species'
average vector. An observation is a set of complementary views (cap, gills, stipe, habitat, slip),
and so is every reference record. Does comparing, or learning, at set level do better?

## Why it matters
If a record's views were being wasted by scoring them separately, set-level matching would be a
cheap gain with no retraining (Step 1), or a small head on the frozen model (Step 2). It also
decides whether end-to-end set fine-tuning (Step 3) is worth a place in the retrain after the
labels are finalised. A caution from earlier audits: in a 99-record audit only 8 of 55 misses had
any single photo right, so look-alikes and sparse species dominate the errors, and combining
photos may not be where the errors are.

## Setup
- Records: the held-out benchmark's dev split, 2,986 records with photos (2,915 with a species
  answer, 2,963 with a genus answer). Development benchmark, not sealed; not the paper's test set.
  The test split was **not** scored: Steve (2026-10-09) asked to wait until the labels are
  finalised, so a confirmation can sit beside the other tests.
- Answer key: the record's observation name on MycoMap (the legacy record title), never a sequence
  name. Scored by the held-out report's own rules (strict, s.l., complex beta from name
  equivalence; temporary codes count as names).
- Model: frozen BioCLIP 2 fine-tune `bioclip-2-ft-20261007-165400`, stored photo vectors only.
  The manifest was opened read-only; nothing was written to it.
- Reference: every record Vision serves. Two versions, both reported:
  **clean** (the main tables) leaves out the label audit's 8,868 records whose numeric ids were
  not iNat ids, so their photos show unrelated iNat observations (145,199 records, 578,197
  photos); **full** keeps them (154,067 records, 590,677 photos). Depth bands count the reference
  in use.
- Step 2 training data: reference records only, split by time. Records validated up to
  2026-09-07 (the fine-tune's own cut) train the head; the newer ones (1,140 clean, 1,152 full)
  are the validation slice for early stopping. All 13,145 benchmark ids are removed before
  training, and the code refuses a split that holds one.
- Selection: settings chosen on dev (rule 2). Each blend was also checked on a **time slice**:
  the reference's records validated after 2026-09-07 scored against the older ones, the layout
  of comparison 4ef7b0 (1,109 species-named records clean; nearest 34.8%, nearest+mean 38.5%).
  Nothing was tuned there for Step 1. For Step 2 the head's epoch was chosen on it, so that
  check is only partly independent.
- Baselines: nearest and nearest+mean recomputed here (identical answers to the stored dev run
  on the full reference, 0 records differ for nearest+mean). nearest+prior@org is the stored
  dev answer, which exists only against the full reference.
- iNat CV on the same records: not run.

## What we tried
Step 1, no training (every variant scored on dev):
- (a) query coverage (each query photo's best match among the species' photos, averaged) is
  exactly what `nearest` computes, so (a) is nearest.
- (b) per reference observation: **obs-forward** (each query photo's best match within one
  reference record, averaged; species = its best record); **obs-chamfer** (forward and backward
  averaged: also each reference photo's best match among the query photos; species = best
  record); **obs-chamfer-top2** (species = mean of its two best records).
- (c) blends w x nearest+mean + (1 - w) x each of the three, w = 0.25, 0.5, 0.75.

Step 2, a small trained head on the frozen vectors: gated attention MIL pooling (Ilse et al.
2018) plus a learned projection that starts at zero, so the untrained head is the plain mean
of the photos (`set-mean`, reported as a control). Proxy loss (normalised softmax over one proxy
per reference group, temperature 0.05) at observation level, photo dropout (each photo kept with
0.6, a quarter of records cut to one photo), AdamW 1e-3, early stopping on the validation slice
(patience 4). Answers by cosine to every reference observation's head vector, species = best
observation (`-max`) or mean of its two best (`-top2`), and blends with nearest+mean at 0.25, 0.5
and 0.75. Variants: with the projection, and attention only (projection fixed at zero).

Every variant, clean reference, dev species-named records (n = 2,915; genus n = 2,963). Paired
species top-1 vs nearest+mean:

| variant | species top-1 | species top-5 | genus top-1 | vs nearest+mean |
|---|---|---|---|---|
| nearest | 48.4 | 73.9 | 78.9 | +103 / -236 |
| nearest+mean | 52.9 | 77.0 | 81.7 | |
| obs-forward | 45.4 | 72.9 | 78.2 | +129 / -349 |
| obs-chamfer | 45.5 | 73.4 | 78.9 | +176 / -392 |
| obs-chamfer-top2 | 50.4 | 74.2 | 80.3 | +183 / -256 |
| blend obs-forward 0.25 / 0.5 / 0.75 | 49.5 / 52.1 / 52.7 | 75.3 / 76.3 / 76.7 | 80.4 / 81.5 / 81.6 | +118/-219, +94/-119, +42/-49 |
| blend obs-chamfer 0.25 / 0.5 / 0.75 | 50.6 / 53.6 / 53.7 | 75.6 / 76.5 / 76.7 | 81.1 / 82.3 / 82.2 | +165/-233, +138/-119, +76/-53 |
| blend obs-chamfer-top2 0.25 / 0.5 / **0.75** | 53.1 / 54.1 / **54.5** | 75.3 / 76.3 / 76.9 | 81.6 / 82.2 / 82.3 | +167/-161, +128/-94, **+79/-34** |
| set-mean-max (untrained) | 46.9 | 74.0 | 79.3 | +157 / -333 |
| set-mean-top2 (untrained) | 50.8 | 74.7 | 80.9 | +161 / -224 |
| set-head-max | 49.5 | 76.0 | 80.8 | +191 / -290 |
| set-head-top2 | 53.1 | 76.7 | 82.2 | +205 / -199 |
| blend set-head-max 0.25 / 0.5 / 0.75 | 52.2 / 54.1 / 54.2 | 77.3 / 77.4 / 77.4 | 81.8 / 82.5 / 82.5 | +178/-200, +148/-115, +96/-58 |
| blend set-head-top2 0.25 / **0.5** / 0.75 | 54.3 / **55.1** / 54.4 | 77.4 / 77.4 / 77.3 | 83.1 / 83.0 / 82.5 | +182/-141, **+151/-89**, +89/-45 |
| set-head-max, attention only | 48.7 | 75.9 | 80.4 | +165 / -289 |
| set-head-top2, attention only | 52.8 | 76.4 | 82.1 | +192 / -196 |
| blend set-head-max 0.25 / 0.5 / 0.75, attention only | 51.3 / 53.5 / 53.8 | 76.9 / 77.6 / 77.4 | 81.9 / 82.2 / 82.2 | +156/-204, +132/-115, +77/-51 |
| blend set-head-top2 0.25 / 0.5 / 0.75, attention only | 54.3 / 54.5 / 54.0 | 76.7 / 77.0 / 77.4 | 82.3 / 82.5 / 82.3 | +169/-128, +132/-86, +75/-43 |

That is 30 scored variants on one dev split (12 in Step 1, 18 in Step 2), plus the time-slice
checks below. The best dev numbers carry that selection.

Head training (clean): validation group top-1 36.1% for the plain mean, best 40.6% at epoch 9 of
13 with the projection (attention only: 38.7% at epoch 2). On the full reference the head peaked
after one epoch (39.5% vs 35.3%): it overfits fast, likely because the fine-tune had already
trained on those same records.

## Results
**Standard tables, clean reference, dev** (n = 2,986 records; species rows n = 2,915, genus rows
n = 2,963). nearest+prior@org exists only against the full reference (below).

nearest

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 48.4 | 67.2 | 73.9 | 80.9 | 2,915 |
| species s.l. | 48.4 | 67.2 | 73.9 | 80.9 | 2,915 |
| species complex (beta) | 51.1 | 69.2 | 75.6 | 82.2 | 2,915 |
| genus strict | 78.9 | 90.5 | 93.5 | 95.8 | 2,963 |
| genus s.l. | 79.6 | 90.7 | 93.5 | 95.8 | 2,963 |

nearest+mean

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 52.9 | 70.7 | 77.0 | 83.2 | 2,915 |
| species s.l. | 52.9 | 70.7 | 77.0 | 83.2 | 2,915 |
| species complex (beta) | 55.7 | 72.6 | 78.6 | 84.4 | 2,915 |
| genus strict | 81.7 | 92.1 | 94.7 | 96.6 | 2,963 |
| genus s.l. | 82.4 | 92.3 | 94.7 | 96.6 | 2,963 |

Best Step 1: blend 0.75 x nearest+mean + 0.25 x obs-chamfer-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 54.5 | 71.1 | 76.9 | 83.4 | 2,915 |
| species s.l. | 54.5 | 71.1 | 76.9 | 83.4 | 2,915 |
| species complex (beta) | 57.0 | 73.0 | 78.5 | 84.5 | 2,915 |
| genus strict | 82.3 | 92.5 | 94.7 | 96.6 | 2,963 |
| genus s.l. | 83.1 | 92.6 | 94.7 | 96.6 | 2,963 |

Best Step 2 alone: set-head-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 53.1 | 71.0 | 76.7 | 83.6 | 2,915 |
| species s.l. | 53.1 | 71.0 | 76.7 | 83.6 | 2,915 |
| species complex (beta) | 55.7 | 72.7 | 78.1 | 84.5 | 2,915 |
| genus strict | 82.2 | 92.2 | 94.7 | 96.6 | 2,963 |
| genus s.l. | 83.0 | 92.2 | 94.8 | 96.7 | 2,963 |

Best Step 2 blend: 0.5 x nearest+mean + 0.5 x set-head-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 55.1 | 72.0 | 77.4 | 84.0 | 2,915 |
| species s.l. | 55.1 | 72.0 | 77.4 | 84.0 | 2,915 |
| species complex (beta) | 57.8 | 73.8 | 78.9 | 85.0 | 2,915 |
| genus strict | 83.0 | 92.7 | 95.2 | 96.9 | 2,963 |
| genus s.l. | 83.6 | 92.8 | 95.3 | 96.9 | 2,963 |

Species top-1 / top-5 by the true species' reference records, clean (n 105 / 327 / 692 / 1,338 /
453):

| method | 0 | 1-4 | 5-19 | 20-99 | 100+ |
|---|---|---|---|---|---|
| nearest | 0.9 / 0.9 | 11.9 / 35.2 | 35.0 / 67.5 | 60.1 / 86.4 | 71.5 / 91.6 |
| nearest+mean | 0.9 / 0.9 | 16.8 / 41.9 | 41.9 / 75.0 | 66.3 / 87.8 | 68.4 / 91.4 |
| obs-chamfer-top2 | 0.9 / 0.9 | 13.8 / 32.4 | 35.4 / 68.1 | 63.5 / 86.9 | 72.6 / 92.9 |
| blend obs-chamfer-top2 0.75 | 0.9 / 0.9 | 15.9 / 39.8 | 41.9 / 74.4 | 68.3 / 88.0 | 73.1 / 92.3 |
| set-head-top2 | 0.9 / 0.9 | 13.8 / 38.2 | 39.0 / 71.8 | 66.5 / 88.9 | 75.7 / 93.4 |
| blend set-head-top2 0.5 | 0.9 / 0.9 | 16.2 / 40.7 | 41.8 / 74.1 | 68.5 / 88.5 | 76.2 / 93.6 |

Paired top-1 vs nearest+mean, clean (McNemar p; species by band):

| method | species | p | genus | p | 1-4 | 5-19 | 20-99 | 100+ |
|---|---|---|---|---|---|---|---|---|
| nearest | +103 / -236 | <0.0001 | +73 / -155 | <0.0001 | +2 / -18 | +21 / -69 | +44 / -127 | +36 / -22 |
| obs-chamfer-top2 | +183 / -256 | 0.0006 | +107 / -149 | 0.01 | +10 / -20 | +42 / -87 | +83 / -120 | +48 / -29 |
| blend obs-chamfer-top2 0.75 | +79 / -34 | <0.0001 | +44 / -24 | 0.02 | +1 / -4 | +17 / -17 | +38 / -11 | +23 / -2 |
| set-head-top2 | +205 / -199 | 0.80 | +131 / -115 | 0.34 | +7 / -17 | +45 / -65 | +95 / -92 | +58 / -25 |
| blend set-head-top2 0.5 | +151 / -89 | <0.0001 | +95 / -56 | 0.002 | +6 / -8 | +32 / -33 | +71 / -41 | +42 / -7 |

**Full reference** (the 8,868 non-iNat records kept; dev n as above; depth bands n 99 / 307 /
682 / 1,344 / 483). Species top-1 / top-5 overall, then by band (top-1 / top-5), then paired
species vs nearest+mean:

| method | species | 0 | 1-4 | 5-19 | 20-99 | 100+ | vs nearest+mean |
|---|---|---|---|---|---|---|---|
| nearest | 48.3 / 73.9 | 1.0 / 1.0 | 12.4 / 34.5 | 34.2 / 65.7 | 59.3 / 86.1 | 70.4 / 91.5 | +105 / -235 |
| nearest+mean | 52.8 / 77.0 | 1.0 / 1.0 | 16.0 / 40.4 | 40.9 / 73.6 | 65.6 / 87.5 | 67.9 / 91.3 | |
| nearest+prior@org | 51.2 / 75.8 | 1.0 / 1.0 | 15.0 / 43.0 | 39.0 / 69.8 | 63.2 / 86.3 | 68.7 / 91.1 | +180 / -225 (p 0.03) |
| blend obs-chamfer-top2 0.75 | 54.5 / 76.7 | 1.0 / 1.0 | 15.3 / 37.5 | 41.2 / 72.4 | 67.9 / 87.9 | 71.6 / 92.1 | +80 / -31 (p <0.0001) |
| set-head-top2 | 53.0 / 76.1 | 1.0 / 1.0 | 13.7 / 33.9 | 39.1 / 68.9 | 64.8 / 88.8 | 75.2 / 93.2 | +195 / -190 (p 0.84) |
| blend set-head-top2 0.75 | 54.5 / 77.3 | 1.0 / 1.0 | 14.7 / 39.4 | 42.2 / 72.3 | 67.5 / 88.5 | 72.3 / 92.8 | +87 / -36 (p <0.0001) |

Genus top-1 on the full reference: nearest 79.0, nearest+mean 81.7, nearest+prior@org 79.5,
best Step 1 blend 82.5, best Step 2 blend 82.7. Leaving out the non-iNat records moves dev by at
most 0.1 point for every untrained method (the heads were retrained, so theirs move more).

**Time slice** (species top-1, strict, 1,109 clean records validated after 2026-09-07 against the
older ones; band n 152 / 209 / 297 / 363 / 88; nothing tuned here for Step 1):

| method | species | 1-4 | 5-19 | 20-99 | 100+ | vs nearest+mean |
|---|---|---|---|---|---|---|
| nearest | 34.8 | 13.9 | 32.7 | 55.4 | 67.0 | +32 / -73 |
| nearest+mean | 38.5 | 14.8 | 40.1 | 59.5 | 69.3 | |
| obs-chamfer-top2 | 34.6 | 14.3 | 32.3 | 54.0 | 70.5 | +49 / -92 |
| blend obs-chamfer-top2 0.5 | 37.4 | 14.8 | 36.0 | 59.0 | 71.6 | +31 / -43 |
| blend obs-chamfer-top2 0.75 (the dev choice) | 38.6 | 14.8 | 39.1 | 59.8 | 72.7 | +23 / -22 (p 1.0) |
| blend obs-chamfer-top2 0.9 | 39.0 | 15.3 | 40.4 | 60.1 | 70.5 | +9 / -4 (p 0.27) |
| set-head-top2 | 37.7 | | | | | +59 / -68 |
| blend set-head-top2 0.5 (the dev choice) | 38.9 | | | | | +40 / -36 (p 0.73) |
| blend set-head-top2 0.75 | 38.6 | | | | | +21 / -20 |
| blend set-head-top2 0.9 | 39.3 | | | | | +12 / -3 (p 0.04, best of three) |

The full-reference time slice (1,121 records, the exact 4ef7b0 set: nearest 34.4, nearest+mean
38.4, matching the published 34.5 / 38.4) gives the same picture: Step 1 blend 0.75 38.1 (+22 / -25).

**What the attention learned** (set head on the full reference, all 2,680 dev records with 2+
photos, 11,890 photos). Weights stay close to even (normalised entropy 0.98; 10th-90th
percentile of weight x photos 0.69-1.28), mildly favour photos that match some reference photo
well (Spearman 0.20) and the first photo (1.09x). Against the uninformative-photos audit's 369
hand labels (all matched to dev photos by their best reference cosine), mean weight x photos
(1 = an even share), with no photo-type labels in training:

| photo kind | photos | mean weight | below an even share |
|---|---|---|---|
| specimen | 316 | 1.04 | 40% |
| slip, specimen on or beside it | 24 | 0.81 | 88% |
| habitat | 19 | 0.77 | 89% |
| slip only | 4 | 0.65 | 100% |
| spore print / microscope | 4 | 0.54 | 100% |
| basket of several fungi | 2 | 1.21 | 0% |

The kinds under 30 photos are direction only.

Cost: Step 1 scoring of 2,986 records 4-5 minutes on the laptop GPU (all variants at once);
head training 1-3.5 minutes per run. GPU time for the whole experiment about 51 minutes (head
training 8 minutes of it); wall time about 2 hours.

## Verdict
Nothing beats nearest+mean robustly. Scored alone, no set method beats it: per-observation
matching loses 2.5-7.5 points (requiring one reference record to hold all of a query's views is
the wrong constraint for records photographed differently), and the trained head only matches it
(53.1 vs 52.9, +205 / -199). Blends gain 1.6-2.2 points on dev, but all of that comes from species
with 20+ references (20-99: +38 / -11, 100+: +23 / -2 for the Step 1 blend), with small losses at
1-4. On the time slice, where nothing was tuned and only 41% of records are in the 20+ bands (dev:
61%), the overall gain is gone (+23 / -22). The deep-band gain shows on both sets (100+: +3.4
points on the slice, +4.7 on dev), but on 88 and 453 records, so it is a direction, not a result.
The attention head finds the uninformative photos on its own (slips, habitat and microscope get
0.5-0.8 of an even share), confirming the earlier audits: down-weighting them is real but small,
and is not where the errors are. **Recommend: do not adopt; study further only cheaply** (below).
**Step 3** (end-to-end set fine-tuning): not worth doing in the retrain on this evidence. A trained
set head on frozen features found nothing that nearest+mean misses, and the errors are look-alikes
and sparse species, which set pooling does not address.

## Decision
Pending (Steve). Recommended here: keep nearest+mean; leave this branch unmerged as a record.

## Next
- When the labels are final and the retrain is done, re-score the two dev-chosen blends once
  on the new dev and the test split (`mv obsets step1 --split test`, `mv obsets step2 --split
  test`), labelled confirmation only. Do that only if the deep-band gain is still of interest.
- If deep species ever matter more (the 20+ bands grow every month), the 0.75 Step 1 blend is the
  one to retest: no training, and gains concentrated at 100+ on both sets.
- No Step 3. Better levers for the misses: references for sparse species, the species-complex
  score for look-alikes, and the Identify page's hint to add underside and stem photos.
- Code: `src/mycomap_vision/obsets.py`, `obsets_head.py`, `mv obsets`; reports
  `obsets-step{1,2}-dev[-clean]` in the benchmark's reports folder (private data).
