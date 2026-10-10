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
- 2babdd6
- c606f9d
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
  45.4-52.9% (none better); best blends 54.5% (set matching) and 55.2% (trained head), but on the
  1,107-record time slice the same blends are level (38.7% and 38.4% vs 38.6%). Exploratory,
  pre-freeze.'
verdict: 'Provisional, confirm on v1: nothing beats nearest+mean robustly; blends gain on dev only in
  species with 20+ references, and the overall gain does not carry over to an independent time slice.'
decision: pending
reproducibility: exploratory-pre-freeze
dataset_release: none (pre-freeze manifest snapshot manifest-obsets-20261009T1517.sqlite)
reference_hash: 0b853385cc50
code_commit: c606f9d
reproduce_command: mv obsets all --manifest <manifest or release manifest> --exclude <non-iNat
  reference list> [--split dev|test]
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
- **Reproducibility: exploratory, pre-freeze** (Steve's standing rule, 2026-10-09). Every number
  below is from one run of `mv obsets all` (code c606f9d) on a stated manifest snapshot,
  `manifest-obsets-20261009T1517.sqlite` (taken read-only with SQLite's backup API at
  2026-10-09 15:18:52 UTC, sha256 b01daa47...7446), and will be re-run unchanged on dataset
  release v1. The run writes `obsets-all-dev-clean.json` with the manifest and exclusion-list
  hashes, the reference hash, the grid and the seeds.
- Records: the held-out benchmark's dev split, 2,986 records with photos (2,915 with a species
  answer, 2,963 with a genus answer). Development benchmark, not sealed; not the paper's test set.
  The test split was **not** scored: Steve (2026-10-09) asked to wait until the labels are
  finalised, so a confirmation can sit beside the other tests.
- Answer key: the record's observation name on MycoMap (the legacy record title), never a sequence
  name. Scored by the held-out report's own rules (strict, s.l., complex beta from name
  equivalence; temporary codes count as names).
- Model: frozen BioCLIP 2 fine-tune `bioclip-2-ft-20261007-165400`, stored photo vectors only.
  Manifest opened read-only; nothing was written to it.
- Reference: every record Vision serves on the snapshot, less the 9,266 records the label audit,
  fix and scan lanes agreed to leave out (`non-inat-reference-records-2026-10-09.tsv`, sha256
  069e417e...b650: ids that were not iNat ids, so their photos show another iNat observation,
  plus mislinked records whose photos show a different fungus). 144,801 records, 577,633 photos,
  16,703 groups; reference hash 0b853385cc50 (heldout.reference_summary's rule). Depth bands
  count this reference.
- Baselines: nearest, nearest+mean and nearest+prior@org all scored here on the same reference
  (nearest+prior = nearest at temperature 0.02 as log-probabilities plus the range-and-season
  prior, with iNat-hidden true coordinates, as the stored `@org` runs). The stored dev answers
  (reference 5dbfdb1d24a5) are not compared: the reference has drifted since.
- Step 2 training data: reference records only, split by time. Records validated up to
  2026-09-07 (the fine-tune's own cut, 143,663) train the head; the newer ones (1,138) are the
  validation slice for early stopping. All 13,145 benchmark ids are removed before training, and
  the code refuses a split that holds one. Seeds fixed (0).
- Selection: settings chosen on dev (rule 2). Each blend was also checked on a **time slice**:
  the reference's records validated after 2026-09-07 scored against the older ones, the layout
  of comparison 4ef7b0 (1,107 species-named records; nearest 34.9%, nearest+mean 38.6%).
  Nothing was tuned there for Step 1. For Step 2 the head's epoch was chosen on it, so that
  check is only partly independent.
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

Every variant, dev species-named records (n = 2,915; genus n = 2,963). Paired species top-1 vs
nearest+mean:

| variant | species top-1 | species top-5 | genus top-1 | vs nearest+mean |
|---|---|---|---|---|
| nearest | 48.4 | 73.9 | 78.9 | +103 / -236 |
| nearest+mean | 52.9 | 77.0 | 81.7 | |
| nearest+prior@org | 51.2 | 75.7 | 79.3 | +180 / -231 |
| obs-forward | 45.4 | 72.9 | 78.2 | +129 / -349 |
| obs-chamfer | 45.6 | 73.5 | 79.1 | +176 / -389 |
| obs-chamfer-top2 | 50.4 | 74.2 | 80.3 | +183 / -256 |
| blend obs-forward 0.25 / 0.5 / 0.75 | 49.5 / 52.1 / 52.7 | 75.3 / 76.3 / 76.7 | 80.4 / 81.5 / 81.6 | +118/-218, +94/-119, +42/-49 |
| blend obs-chamfer 0.25 / 0.5 / 0.75 | 50.6 / 53.6 / 53.8 | 75.5 / 76.5 / 76.7 | 81.1 / 82.3 / 82.2 | +165/-233, +138/-119, +78/-53 |
| blend obs-chamfer-top2 0.25 / 0.5 / **0.75** | 53.1 / 54.1 / **54.5** | 75.3 / 76.3 / 76.9 | 81.7 / 82.2 / 82.3 | +167/-161, +128/-94, **+79/-34** |
| set-mean-max (untrained) | 46.9 | 74.0 | 79.3 | +157 / -333 |
| set-mean-top2 (untrained) | 50.8 | 74.7 | 80.9 | +161 / -224 |
| set-head-max | 49.5 | 76.0 | 80.5 | +191 / -290 |
| set-head-top2 | 52.9 | 76.8 | 82.2 | +206 / -206 |
| blend set-head-max 0.25 / 0.5 / 0.75 | 52.2 / 54.1 / 54.3 | 77.0 / 77.8 / 77.6 | 81.9 / 82.5 / 82.6 | +179/-201, +150/-115, +89/-50 |
| blend set-head-top2 0.25 / **0.5** / 0.75 | 54.5 / **55.2** / 54.5 | 77.4 / 77.1 / 77.3 | 83.0 / 83.0 / 82.5 | +185/-140, **+147/-81**, +86/-41 |
| set-head-max, attention only | 49.1 | 75.9 | 81.0 | +182 / -295 |
| set-head-top2, attention only | 52.6 | 76.4 | 82.1 | +192 / -203 |
| blend set-head-max 0.25 / 0.5 / 0.75, attention only | 51.8 / 53.6 / 53.9 | 77.0 / 77.5 / 77.4 | 82.1 / 82.5 / 82.3 | +167/-201, +137/-117, +80/-52 |
| blend set-head-top2 0.25 / 0.5 / 0.75, attention only | 54.6 / 54.8 / 54.3 | 76.9 / 77.2 / 77.4 | 82.6 / 82.7 / 82.3 | +173/-124, +136/-82, +80/-39 |

That is 30 scored variants on one dev split (12 in Step 1, 18 in Step 2), plus the time-slice
checks below. The best dev numbers carry that selection.

Head training: validation group top-1 36.2% for the plain mean, best 40.2% at epoch 6 of 10 with
the projection (attention only: 38.6% at epoch 4). It overfits fast, likely because the
fine-tune had already trained on those same records.

Earlier exploratory runs (same code before the snapshot existed, on the live manifest, so with no
recorded reference hash): with the 8,868-id exclusion list and with no exclusion at all. Every
variant there is within 1 point of the same variant below, and every conclusion is the same; with no
exclusion, nearest 48.3 / nearest+mean 52.8 / best Step 1 blend 54.5 / best Step 2 blend 54.5.

## Results
**Standard tables, dev** (n = 2,986 records; species rows n = 2,915, genus rows n = 2,963).

nearest

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 48.4 | 67.2 | 73.9 | 80.9 | 2,915 |
| species s.l. | 48.4 | 67.2 | 73.9 | 80.9 | 2,915 |
| species complex (beta) | 51.1 | 69.2 | 75.6 | 82.2 | 2,915 |
| genus strict | 78.9 | 90.5 | 93.5 | 95.8 | 2,963 |
| genus s.l. | 79.6 | 90.7 | 93.6 | 95.8 | 2,963 |

nearest+mean

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 52.9 | 70.7 | 77.0 | 83.2 | 2,915 |
| species s.l. | 52.9 | 70.7 | 77.0 | 83.2 | 2,915 |
| species complex (beta) | 55.7 | 72.6 | 78.7 | 84.4 | 2,915 |
| genus strict | 81.7 | 92.2 | 94.7 | 96.6 | 2,963 |
| genus s.l. | 82.4 | 92.3 | 94.7 | 96.6 | 2,963 |

nearest+prior@org

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 51.2 | 70.2 | 75.7 | 82.0 | 2,915 |
| species s.l. | 51.2 | 70.2 | 75.7 | 82.0 | 2,915 |
| species complex (beta) | 53.5 | 71.9 | 77.0 | 83.1 | 2,915 |
| genus strict | 79.3 | 91.0 | 93.6 | 95.8 | 2,963 |
| genus s.l. | 80.1 | 91.1 | 93.6 | 95.9 | 2,963 |

Best Step 1 alone: obs-chamfer-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 50.4 | 68.3 | 74.2 | 81.8 | 2,915 |
| species s.l. | 50.4 | 68.3 | 74.2 | 81.8 | 2,915 |
| species complex (beta) | 53.2 | 70.6 | 75.8 | 83.0 | 2,915 |
| genus strict | 80.3 | 91.8 | 94.2 | 96.4 | 2,963 |
| genus s.l. | 81.3 | 91.9 | 94.2 | 96.4 | 2,963 |

Best Step 1 blend: 0.75 x nearest+mean + 0.25 x obs-chamfer-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 54.5 | 71.1 | 76.9 | 83.4 | 2,915 |
| species s.l. | 54.5 | 71.1 | 76.9 | 83.4 | 2,915 |
| species complex (beta) | 57.0 | 73.0 | 78.6 | 84.6 | 2,915 |
| genus strict | 82.3 | 92.6 | 94.7 | 96.6 | 2,963 |
| genus s.l. | 83.1 | 92.7 | 94.7 | 96.6 | 2,963 |

Best Step 2 alone: set-head-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 52.9 | 70.9 | 76.8 | 83.7 | 2,915 |
| species s.l. | 52.9 | 70.9 | 76.8 | 83.7 | 2,915 |
| species complex (beta) | 55.7 | 72.7 | 78.3 | 84.7 | 2,915 |
| genus strict | 82.2 | 92.1 | 94.8 | 96.7 | 2,963 |
| genus s.l. | 82.9 | 92.2 | 94.8 | 96.7 | 2,963 |

Best Step 2 blend: 0.5 x nearest+mean + 0.5 x set-head-top2

| | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| species strict | 55.2 | 72.0 | 77.1 | 84.0 | 2,915 |
| species s.l. | 55.2 | 72.0 | 77.1 | 84.0 | 2,915 |
| species complex (beta) | 58.0 | 74.0 | 78.7 | 85.1 | 2,915 |
| genus strict | 83.0 | 92.6 | 95.3 | 96.9 | 2,963 |
| genus s.l. | 83.5 | 92.6 | 95.3 | 96.9 | 2,963 |

Species top-1 / top-5 by the true species' reference records (n 106 / 326 / 692 / 1,338 / 453):

| method | 0 | 1-4 | 5-19 | 20-99 | 100+ |
|---|---|---|---|---|---|
| nearest | 0.9 / 0.9 | 12.0 / 35.3 | 35.0 / 67.5 | 60.1 / 86.4 | 71.5 / 91.6 |
| nearest+mean | 0.9 / 0.9 | 16.9 / 42.0 | 41.9 / 75.0 | 66.3 / 87.8 | 68.4 / 91.4 |
| nearest+prior@org | 0.9 / 0.9 | 15.6 / 45.1 | 39.2 / 70.7 | 63.9 / 86.6 | 69.3 / 91.0 |
| obs-chamfer-top2 | 0.9 / 0.9 | 13.8 / 32.5 | 35.4 / 68.1 | 63.5 / 86.9 | 72.6 / 92.9 |
| blend obs-chamfer-top2 0.75 | 0.9 / 0.9 | 16.0 / 39.9 | 41.9 / 74.4 | 68.3 / 88.0 | 73.1 / 92.3 |
| set-head-top2 | 0.0 / 0.9 | 12.9 / 39.0 | 40.0 / 72.2 | 66.2 / 88.9 | 74.6 / 93.4 |
| blend set-head-top2 0.5 | 0.9 / 0.9 | 15.0 / 39.6 | 42.0 / 73.6 | 68.8 / 88.6 | 76.8 / 93.2 |

Paired top-1 vs nearest+mean (McNemar p; species by band):

| method | species | p | genus | p | 1-4 | 5-19 | 20-99 | 100+ |
|---|---|---|---|---|---|---|---|---|
| nearest | +103 / -236 | <0.0001 | +73 / -155 | <0.0001 | +2 / -18 | +21 / -69 | +44 / -127 | +36 / -22 |
| nearest+prior@org | +180 / -231 | 0.01 | +108 / -178 | <0.0001 | +11 / -15 | +46 / -65 | +81 / -113 | +42 / -38 |
| obs-chamfer-top2 | +183 / -256 | 0.0006 | +108 / -149 | 0.01 | +10 / -20 | +42 / -87 | +83 / -120 | +48 / -29 |
| blend obs-chamfer-top2 0.75 | +79 / -34 | <0.0001 | +44 / -24 | 0.02 | +1 / -4 | +17 / -17 | +38 / -11 | +23 / -2 |
| set-head-top2 | +206 / -206 | 1.0 | +134 / -118 | 0.34 | +7 / -20 | +47 / -60 | +95 / -96 | +57 / -29 |
| blend set-head-top2 0.5 | +147 / -81 | <0.0001 | +93 / -54 | 0.002 | +5 / -11 | +30 / -29 | +67 / -34 | +45 / -7 |

**Time slice** (species top-1, strict, 1,107 records validated after 2026-09-07 against the older
ones; band n 151 / 210 / 296 / 363 / 87; nothing tuned here for Step 1):

| method | species | 1-4 | 5-19 | 20-99 | 100+ | vs nearest+mean |
|---|---|---|---|---|---|---|
| nearest | 34.9 | 13.8 | 32.8 | 55.4 | 67.8 | +32 / -73 |
| nearest+mean | 38.6 | 14.8 | 40.2 | 59.5 | 70.1 | |
| obs-chamfer-top2 | 34.7 | 14.3 | 32.4 | 54.0 | 71.3 | +49 / -92 |
| blend obs-chamfer-top2 0.5 | 37.5 | 14.8 | 36.1 | 59.0 | 72.4 | +31 / -43 |
| blend obs-chamfer-top2 0.75 (the dev choice) | 38.7 | 14.8 | 39.2 | 59.8 | 73.6 | +23 / -22 (p 1.0) |
| blend obs-chamfer-top2 0.9 | 39.0 | 15.2 | 40.5 | 60.1 | 71.3 | +9 / -4 (p 0.27) |
| set-head-top2 | 38.1 | | | | | +61 / -66 |
| blend set-head-top2 0.5 (the dev choice) | 38.4 | | | | | +34 / -36 (p 0.90) |
| blend set-head-top2 0.75 | 38.5 | | | | | +22 / -23 |
| blend set-head-top2 0.9 | 39.4 | | | | | +13 / -4 (p 0.05, best of three) |

**What the attention learned** (set head of an earlier exploratory run, no reference exclusion,
all 2,680 dev records with 2+ photos, 11,890 photos). Weights stay close to even (normalised
entropy 0.98; 10th-90th percentile of weight x photos 0.69-1.28), mildly favour photos that match
some reference photo well (Spearman 0.20) and the first photo (1.09x). Against the
uninformative-photos audit's 369 hand labels (all matched to dev photos by their best reference
cosine), mean weight x photos (1 = an even share), with no photo-type labels in training:

| photo kind | photos | mean weight | below an even share |
|---|---|---|---|
| specimen | 316 | 1.04 | 40% |
| slip, specimen on or beside it | 24 | 0.81 | 88% |
| habitat | 19 | 0.77 | 89% |
| slip only | 4 | 0.65 | 100% |
| spore print / microscope | 4 | 0.54 | 100% |
| basket of several fungi | 2 | 1.21 | 0% |

The kinds under 30 photos are direction only. The hand labels point at private audit photos and
are not in the repo, so this table is not part of the re-runnable command.

Cost: on an idle laptop GPU, Step 1 takes about 4 minutes for 2,986 records (all variants at once)
and a head trains in 1-3.5 minutes. The snapshot run took 7.2 hours of wall time because other
jobs were sharing the CPU and GPU; about 51 GPU-minutes in the earlier uncontended runs.

## Verdict
**Provisional, confirm on v1.** Nothing beats nearest+mean robustly. Scored alone, no set method
beats it: per-observation matching loses 2.5-7.5 points (requiring one reference record to hold
all of a query's views is the wrong constraint for records photographed differently), and the
trained head only matches it (52.9 vs 52.9, +206 / -206). Blends gain 1.6-2.3 points on dev, but
nearly all of that comes from species with 20+ references (Step 1 blend: 20-99 +38 / -11, 100+
+23 / -2), with small losses at 1-4. On the time slice, where nothing was tuned and only 41% of
records are in the 20+ bands (dev: 61%), the overall gain is gone (+23 / -22). The deep-band gain
shows on both sets (100+: +3.5 points on the slice, +4.7 on dev), but on 87 and 453 records, so it
is a direction, not a result. The attention head finds the uninformative photos on its own
(slips, habitat and microscope get 0.5-0.8 of an even share), confirming the earlier audits:
down-weighting them is real but small, and is not where the errors are. **Recommend: do not
adopt; study further only cheaply** (below). **Step 3** (end-to-end set fine-tuning): not worth
doing in the retrain on this evidence. A trained set head on frozen features found nothing that
nearest+mean misses, and the errors are look-alikes and sparse species, which set pooling does
not address.

## Decision
Pending (Steve). Recommended here: keep nearest+mean; leave this branch unmerged as a record.
Like every pre-freeze result, this stays provisional until re-run on dataset release v1.

## Next
- On dataset release v1, re-run unchanged: `mv obsets all --manifest <v1 manifest> --exclude
  <v1's exclusion list, if any>`; then, once the labels are final, the same with `--split test`
  as the confirmation (Steve, 2026-10-09).
- If deep species ever matter more (the 20+ bands grow every month), the 0.75 Step 1 blend is the
  one to retest: no training, and gains concentrated at 100+ on both sets.
- No Step 3. Better levers for the misses: references for sparse species, the species-complex
  score for look-alikes, and the Identify page's hint to add underside and stem photos.
- Code: `src/mycomap_vision/obsets.py`, `obsets_head.py`, `mv obsets`; reports
  `obsets-all-dev-clean.json` and `obsets-step{1,2}[-attnonly]-dev-clean` in the benchmark's
  reports folder (private data).
