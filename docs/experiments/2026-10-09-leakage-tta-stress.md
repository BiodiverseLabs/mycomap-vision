---
title: Leakage, test-time augmentation and stress slices on the held-out benchmark
slug: leakage-tta-stress
date: '2026-10-09'
status: done
question: How much of Vision's held-out score comes from the query's own observer or own collecting
  trip being in the reference, does averaging a few flipped and cropped views of each photo help,
  and where does Vision break?
branch: exp/leakage-tta-stress (pushed, not merged)
commits:
- 318cf04
- 1765827
- cc0fb56
benchmark: heldout-2026-10-08 (development benchmark, not sealed), reference 5dbfdb1d24a5
split: dev (all choices); test (leakage confirmation only)
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
- nearest+prior@org
- test-time augmentation (2, 4 or 8 views; embedding or score average)
headline: 'Removing the query''s own collecting trip changes species top-1 by 0.1 point (test,
  nearest+mean 54.3 -> 54.2, 10,107 records); removing everything by its own observer costs 1.1,
  of which 0.9 is observer-specific (species-matched control 54.1). Four-view TTA: nearest+mean
  52.8 -> 54.2 on dev (2,986 records, +87 / -46), at about 4x the image-tower time.'
verdict: 'Leakage is small: the paper can quote the plain numbers and state about 1 point of
  observer shortcut (2-3 points for species with under 20 references). TTA gives about 1.4 points
  to nearest+mean at about 4x the CPU cost per photo: provisional, not confirmed on test. The real
  weak spots are finds far from every reference of their species, off-season finds and
  long-ago validations; the DNA place prior makes the first two worse.'
decision: pending (exploratory-pre-freeze; re-run on Dataset release v1 before any decision)
related:
- heldout-benchmark
- depth-bias
- occurrence-prior
- dna-range-prior
- uninformative-photos
- photo-views
---

Reproducibility: **exploratory-pre-freeze**. Everything here ran on a copy of the manifest taken
2026-10-09 14:37 UTC (reference hash 5dbfdb1d24a5: 154,067 records, 590,677 photos, the same
reference the 06:13 UTC held-out re-score used) with code at 318cf04 / 1765827. Per Steve's rule
(2026-10-09), every result is provisional until it is re-run on the frozen Dataset release v1.
Commands: `scripts/leakage_tta_stress/` (`prepare.py --manifest`, `score.py`, `tta_embed.py`,
`report.py`, `pairs.py`, `true_species_leak.py`, `cpu_cost.py`; README in each docstring).

## Question
Three questions the paper has to answer. (A) Leakage: is part of the score Vision recognising a
photographer's style or site, or the very same collection, rather than the species? (B) Does
test-time augmentation (TTA: embed each photo under a few flips and crops and average) help, and
what does it cost to serve? (C) Where does Vision break, not just how it does on average?

## Why it matters
Most held-out queries come from prolific contributors whose other records are in the reference
(97.8% of test queries share an observer with the reference, median 494 of that observer's
records), and over half were collected on a trip that also produced reference records. A
reviewer will ask whether that inflates the numbers. TTA is a cheap-looking gain that would
change the serving cost on the CPU-only box. The slices say what to fix next.

## Setup
- Benchmark heldout-2026-10-08: dev 3,000 records (2,986 answerable: photos, not a reference
  photo), test 10,145 (10,107 answerable). Development benchmark, not sealed. Answer key = the
  observation name (heldout.answer_key). Standard scoring (heldout_summary: strict, *s.l.*,
  complex beta; depth bands by the true species' reference records).
- Model bioclip-2-ft-20261007-165400 with nearest, nearest+mean (the default) and
  nearest+prior@org (true place and date).
- One numpy scorer (`mycomap_vision.stress`) scores each query against the whole reference and
  masks reference records per condition; a masked record leaves the nearest match, its species'
  average vector and the place prior's counts, exactly as if it had never been in the reference
  (unit tests compare against a full recompute and a refitted prior). Base condition reproduces
  the stored re-score on the same reference: species top-1 identical for 99.8% (nearest), 100%
  (nearest+mean) and 99.9% (+prior) of the 2,986 dev records (the rest are float16 near-ties); a
  single-process and a 4-shard run gave identical top-10 lists.
- Conditions: **obs** (every reference record by the query's iNat observer removed); **obs
  control** (per species, the same number of records removed but drawn at random from *other*
  observers, so it takes the same depth from every species); **coll** (the same collection: the
  same observer within 1 km and 1 day, or any reference record with a photo at cosine > 0.98 to a
  query photo); **coll wide** (5 km, 3 days, cosine > 0.95); **clean** (the 9,259 references whose
  .org source is MO, MyCoPortal, .com Sequences or GenBank removed: ids read as iNat ids, wrong
  photos; 112 references with no source row kept), alone and with obs / coll.
- Perceptual hashing was not run: the reference photos are not all on this machine. Embedding
  cosine (fine-tuned space) stands in for it; the held-out report's own sha256 check found 9 dev
  records with a photo file identical to a reference copy.
- TTA: views made before the model's own resize and centre crop: orig, horizontal flip, centre 80%
  crop, the four 80% corner crops, flipped centre crop. Colour never changed. 12,196 dev photos x 8
  views on the laptop GPU (16 images per forward, 49 min on a shared GPU); the orig view
  reproduced every stored dev vector (min cosine 0.9994, float16). Combined by averaging the
  embeddings (then nearest / nearest+mean as usual) or by averaging the scores of each view.
- Slices from the base run; "depth-adjusted" re-weights a slice's per-band top-1 to the whole dev
  set's band mix, so a slice is not called weak just for holding rarer species. Regions are the
  state or province of the nearest reference record; no coordinate leaves the scorer.

## What we tried
Leakage conditions (7 on dev, 4 confirmed on test), each with all three methods. TTA: 5 variants
on dev: flip (embedding average), flip + centre crops (4 views), all 8 views, flip by score
average; the 8-view score average was started but not finished (CPU time) and is not reported.
Picking the best of these on dev inflates its dev gain somewhat; the 4- and 8-view averages tie.

## Results
**A. Leakage, test (10,107 records; species rows 9,869, genus rows 10,004).** Standard format,
top 1 / 3 / 5 / 10:

| condition / method | species strict | species s.l. | species complex (beta) | genus strict | genus s.l. |
|---|---|---|---|---|---|
| base: nearest | 48.2 / 67.0 / 73.6 / 80.5 | 48.3 / 67.0 / 73.6 / 80.5 | 51.3 / 69.2 / 75.4 / 81.8 | 80.1 / 91.0 / 93.7 / 95.8 | 80.8 / 91.2 / 93.7 / 95.9 |
| base: nearest+mean | 54.3 / 72.0 / 77.4 / 83.4 | 54.3 / 72.0 / 77.4 / 83.4 | 57.2 / 73.9 / 79.0 / 84.3 | 83.5 / 92.8 / 94.6 / 96.3 | 84.1 / 92.9 / 94.7 / 96.3 |
| base: nearest+prior@org | 51.2 / 69.3 / 75.8 / 82.1 | 51.2 / 69.3 / 75.8 / 82.1 | 53.7 / 71.2 / 77.4 / 83.3 | 80.1 / 90.9 / 93.6 / 95.8 | 80.7 / 91.1 / 93.7 / 95.8 |
| coll: nearest | 48.1 / 66.8 / 73.4 / 80.4 | 48.1 / 66.8 / 73.4 / 80.4 | 51.1 / 68.9 / 75.2 / 81.6 | 80.0 / 90.9 / 93.5 / 95.8 | 80.8 / 91.0 / 93.6 / 95.8 |
| coll: nearest+mean | 54.2 / 71.8 / 77.2 / 83.3 | 54.2 / 71.8 / 77.2 / 83.3 | 57.1 / 73.8 / 78.8 / 84.2 | 83.4 / 92.7 / 94.6 / 96.2 | 84.1 / 92.8 / 94.6 / 96.2 |
| coll: nearest+prior@org | 51.0 / 69.1 / 75.6 / 82.0 | 51.0 / 69.1 / 75.6 / 82.0 | 53.5 / 71.0 / 77.2 / 83.2 | 80.0 / 90.8 / 93.5 / 95.7 | 80.7 / 91.0 / 93.6 / 95.7 |
| obs: nearest | 47.0 / 65.8 / 72.4 / 79.3 | 47.0 / 65.8 / 72.4 / 79.3 | 50.1 / 68.1 / 74.2 / 80.6 | 79.6 / 90.8 / 93.5 / 95.7 | 80.4 / 91.0 / 93.5 / 95.8 |
| obs: nearest+mean | 53.2 / 70.7 / 76.2 / 82.2 | 53.2 / 70.7 / 76.2 / 82.2 | 56.1 / 72.8 / 77.8 / 83.2 | 83.1 / 92.6 / 94.6 / 96.2 | 83.8 / 92.8 / 94.7 / 96.3 |
| obs: nearest+prior@org | 49.9 / 68.1 / 74.2 / 80.8 | 50.0 / 68.1 / 74.2 / 80.9 | 52.5 / 69.9 / 76.0 / 82.0 | 79.6 / 90.6 / 93.3 / 95.7 | 80.3 / 90.8 / 93.5 / 95.7 |
| obs control: nearest | 48.1 / 66.8 / 73.4 / 80.3 | 48.1 / 66.8 / 73.4 / 80.3 | 51.1 / 69.0 / 75.2 / 81.5 | 80.0 / 90.9 / 93.6 / 95.8 | 80.7 / 91.1 / 93.7 / 95.9 |
| obs control: nearest+mean | 54.1 / 71.9 / 77.2 / 83.3 | 54.1 / 71.9 / 77.2 / 83.3 | 57.0 / 73.9 / 78.8 / 84.2 | 83.4 / 92.8 / 94.6 / 96.3 | 84.0 / 92.9 / 94.7 / 96.3 |
| obs control: nearest+prior@org | 50.9 / 69.2 / 75.6 / 82.0 | 50.9 / 69.2 / 75.6 / 82.0 | 53.5 / 71.1 / 77.2 / 83.2 | 80.1 / 90.8 / 93.6 / 95.8 | 80.7 / 91.0 / 93.7 / 95.9 |

Species top 1 / top 5 by the true species' reference records (test; bands from the base
reference):

| condition / method | 0 (346) | 1-4 (1,075) | 5-19 (2,241) | 20-99 (4,512) | 100+ (1,695) |
|---|---|---|---|---|---|
| base: nearest | 0.3 / 0.3 | 14.5 / 33.5 | 32.1 / 65.0 | 59.4 / 86.2 | 71.3 / 92.0 |
| base: nearest+mean | 0.3 / 0.3 | 19.2 / 39.6 | 41.0 / 73.7 | 66.2 / 88.8 | 73.2 / 91.6 |
| base: nearest+prior@org | 0.3 / 0.3 | 17.6 / 39.4 | 39.4 / 69.2 | 62.0 / 87.3 | 69.7 / 92.4 |
| coll: nearest+mean | 0.3 / 0.3 | 19.0 / 39.4 | 40.7 / 73.4 | 66.3 / 88.7 | 73.0 / 91.5 |
| obs: nearest | 0.3 / 0.3 | 12.1 / 27.9 | 30.1 / 62.5 | 58.4 / 86.0 | 70.4 / 91.9 |
| obs: nearest+mean | 0.3 / 0.3 | 15.6 / 33.2 | 38.3 / 71.7 | 66.2 / 88.8 | 72.8 / 91.3 |
| obs: nearest+prior@org | 0.3 / 0.3 | 14.8 / 32.3 | 36.9 / 66.4 | 61.3 / 87.3 | 69.4 / 91.6 |
| obs control: nearest+mean | 0.3 / 0.3 | 18.2 / 39.2 | 40.6 / 73.5 | 66.4 / 88.6 | 73.1 / 91.6 |

Paired species top-1 (test). Same trip removed vs base: nearest +17 / -34 (p 0.02), nearest+mean
+11 / -20 (p 0.15), +prior +30 / -44 (p 0.13). Own observer removed vs its species-matched
control (the observer-specific part): nearest +239 / -352 (-1.1 points, p < 0.0001), nearest+mean
+189 / -280 (-0.9, p < 0.0001), +prior +336 / -432 (-1.0, p 0.0006); for nearest+mean by band
1-4 -28 records (-2.6 points), 5-19 -51 (-2.3), 20-99 -7, 100+ -5. Dev agrees: same trip 52.8 ->
52.9, own observer 51.9, control 52.4 (nearest 48.4 / 48.3 / 46.8 / 48.0; +prior 51.3 / 51.2 /
50.2 / 51.5).

How much there is to leak (test): 56.0% of queries share a trip (<= 1 km, +-1 day) with reference
records (median 6), 64.9% share an observer-day anywhere, 0.4% have a reference photo at cosine >
0.98. On dev the trip holds the **true species** for only 97 of 2,915 queries (3.3%); there,
removing it costs nearest 56.7 -> 46.4 and +prior 63.9 -> 51.6 but nearest+mean only 56.7 -> 54.6.
The observer holds the true species for 1,110 (38%); there nearest+mean goes 63.8 -> 59.5.

Wrong-photo references (dev): removing the 9,259 non-iNat references changes nothing that
matters: nearest 48.4 -> 48.4 (+1 / -0), nearest+mean 52.8 -> 52.9 (+5 / -1), +prior 51.3 -> 51.2
(+14 / -16); with the trip also removed 48.3 / 53.0 / 51.1.

**B. TTA (dev, 2,986 records; species rows 2,915, genus rows 2,963).** Top 1 / 3 / 5 / 10:

| variant / method | species strict | species complex (beta) | genus strict | genus s.l. | species top-1 vs base |
|---|---|---|---|---|---|
| base: nearest | 48.4 / 67.1 / 73.9 / 80.9 | 51.1 / 69.1 / 75.7 / 82.1 | 79.0 / 90.5 / 93.3 / 95.7 | 79.7 / 90.7 / 93.5 / 95.7 | |
| flip, embedding avg: nearest | 48.7 / 66.6 / 73.9 / 80.7 | 51.1 / 68.7 / 75.6 / 81.9 | 79.2 / 90.5 / 93.3 / 95.8 | 79.9 / 90.6 / 93.4 / 95.8 | +58 / -48 |
| 4 views: nearest | 49.2 / 67.5 / 74.3 / 80.5 | 51.6 / 69.5 / 75.9 / 81.9 | 79.8 / 91.2 / 93.5 / 95.7 | 80.4 / 91.4 / 93.6 / 95.7 | +104 / -80 (p 0.09) |
| 8 views: nearest | 49.0 / 68.1 / 74.6 / 81.1 | 51.6 / 69.9 / 76.0 / 82.3 | 79.8 / 91.3 / 93.6 / 95.9 | 80.4 / 91.5 / 93.6 / 96.0 | +117 / -99 |
| flip, score avg: nearest | 48.7 / 66.7 / 74.1 / 80.8 | 51.1 / 68.8 / 75.8 / 82.1 | 79.3 / 90.6 / 93.4 / 95.7 | 80.0 / 90.8 / 93.5 / 95.7 | +52 / -43 |
| base: nearest+mean | 52.8 / 70.7 / 77.0 / 83.2 | 55.6 / 72.7 / 78.6 / 84.4 | 81.7 / 92.1 / 94.7 / 96.6 | 82.5 / 92.3 / 94.7 / 96.6 | |
| flip, embedding avg: nearest+mean | 53.4 / 71.2 / 76.5 / 83.1 | 56.0 / 73.2 / 78.1 / 84.4 | 81.9 / 92.2 / 94.6 / 96.6 | 82.7 / 92.3 / 94.6 / 96.6 | +45 / -28 (p 0.06) |
| **4 views: nearest+mean** | **54.2** / 71.7 / 77.1 / 83.4 | **56.8** / 73.7 / 78.9 / 84.5 | **82.5** / 92.7 / 95.1 / 96.6 | 83.2 / 92.8 / 95.1 / 96.6 | **+87 / -46 (p 0.0005)** |
| 8 views: nearest+mean | 54.3 / 71.7 / 77.5 / 83.3 | 56.9 / 73.6 / 79.2 / 84.5 | 82.6 / 93.0 / 95.0 / 96.6 | 83.3 / 93.1 / 95.0 / 96.6 | +99 / -55 (p 0.0005) |
| flip, score avg: nearest+mean | 53.6 / 71.2 / 76.6 / 83.2 | 56.2 / 73.3 / 78.1 / 84.4 | 82.1 / 92.2 / 94.6 / 96.5 | 82.9 / 92.3 / 94.6 / 96.6 | +50 / -26 (p 0.008) |
| base: nearest+prior@org | 51.3 / 70.0 / 75.8 / 82.0 | 53.7 / 71.7 / 77.0 / 83.1 | 79.5 / 91.1 / 93.7 / 95.8 | 80.3 / 91.2 / 93.7 / 95.9 | |
| 4 views: nearest+prior@org | 51.9 / 70.3 / 76.1 / 82.2 | 54.2 / 72.0 / 77.4 / 83.3 | 80.3 / 91.2 / 93.9 / 95.8 | 81.0 / 91.5 / 93.9 / 95.8 | +80 / -62 |

(Species *s.l.* equals strict in every row.) By band, nearest+mean base -> 4 views, top-1: 1-4
16.0 -> 16.0, 5-19 40.9 -> 43.3, 20-99 65.6 -> 66.3, 100+ 67.9 -> 71.2; top-5 changes by under 1
point. Genus with 4 views +63 / -39 (p 0.02). Cost on the laptop CPU (4 threads, under heavy
background load, medians of 3 rounds of 8 photos): image tower 2.2 s a photo for 1 view, 4.0 for 2,
8.5 for 4, 13.6 for 8 (close to linear); the similarity pass against the reference is the same for
an embedding average (one vector) and grows with the views for a score average.

**C. Stress slices (dev, nearest+mean; species top-1, its 95% interval, and depth-adjusted).**

| slice | n | species top-1 | depth-adjusted |
|---|---|---|---|
| nearest true-species reference <= 10 km | 836 | 65.4 (62.1-68.6) | 60.7 |
| 50-200 km | 706 | 54.7 (51.0-58.3) | 53.8 |
| 200-1,000 km | 520 | 38.1 (34.0-42.3) | 45.6 |
| > 1,000 km | 119 | 14.3 (9.1-21.7) | 20.3 |
| season: 50%+ of true-species refs within 30 days | 1,286 | 56.2 | 58.1 |
| off season: none within 30 days | 165 | 20.0 (14.6-26.8) | 28.5 |
| validated in the newest 4 weeks | 819 | 56.4 | 57.2 |
| validated more than 8 weeks before the freeze | 1,167 | 47.0 (44.1-49.8) | 50.7 |
| observer with 0 reference records | 77 | 42.9 (32.4-54.0) | 43.8 |
| observer with 100+ | 2,221 | 53.3 | 55.0 |
| true species already found by this observer: no / yes | 1,805 / 1,110 | 46.0 / 63.8 | 52.2 / 59.7 |
| 1 photo / 2-3 / 4-6 / 7+ | 303 / 1,274 / 930 / 408 | 48.5 / 50.9 / 55.9 / 54.9 | 51.1 / 52.2 / 57.2 / 59.3 |
| records with 2+ photos: first photo only vs all | 2,612 | 44.2 vs 53.3 | |
| < 100 reference records of any species within 100 km | 134 | 43.3 | 47.3 |
| Canada / US | 471 / 2,408 | 46.5 / 54.6 | 51.1 / 55.2 |
| weakest regions with 38+ records (depth-adjusted) | | Minnesota 37.2, New York 38.6, Alberta 41.3, Pennsylvania 44.8 | |
| strongest | | West Virginia 65.0, Missouri 63.1, Washington 62.0 | |

The 31 Puerto Rico records score 9.7% (22.3 depth-adjusted). The DNA place prior (nearest+prior
@org) makes the far and off-season slices worse, not better: true species 200-1,000 km from its
references 38.1 -> 24.8, > 1,000 km 14.3 -> 8.4, off season 20.0 -> 14.5 (nearest+mean ->
+prior); it helps near home (<= 10 km 65.4 -> 67.7). Test was used only for the leakage
confirmation; the slices and TTA are dev only.

## Verdict
- **(A) Leakage is small.** Same-trip and near-duplicate references are common but almost never the
  true species, and removing them moves species top-1 by about 0.1 point (test). The paper can
  quote the base numbers (test nearest+mean 54.3 / genus 83.5; trip removed 54.2 / 83.4), and must
  state the observer effect: a query whose photographer has no other records in the reference
  scores about 1 point lower (test 53.2 vs 54.1 for the control that removes the same depth), about
  2-3 points for species with under 20 references. Wrong-photo references don't move dev.
- **(B) TTA: provisional "worth it for nearest+mean if the box can pay".** Four views (orig, flip,
  centre crop, flipped centre crop) by embedding average: +1.4 species, +0.8 genus on dev, chosen
  from 5 variants, not confirmed on test. It costs about 4x the image-tower time per photo on the
  CPU box (about 2 s -> 8 s a photo on the laptop under load); a flip alone gives about +0.6 for 2x.
  It does nothing significant for nearest or +prior. Re-run on the frozen release, with a test
  score, before deciding; if the box can't afford 4x, the flip alone is the cheap option.
- **(C) Where Vision breaks**, after depth is taken out: (1) finds far from every reference of their
  species (> 1,000 km: 20% adjusted; 200-1,000 km: 46%) and (2) off-season finds (29%): the
  photos of those species we hold come from elsewhere and other seasons, and the DNA place prior
  penalises exactly these; (3) records validated long ago (51% vs 57%), most likely names never
  refreshed; (4) first-time contributors (44%, n 77) and single-photo records; (5) a few regions
  (Minnesota, New York, Alberta, Pennsylvania; Puerto Rico).

## Decision
Pending. Exploratory-pre-freeze: every number here is re-run on Dataset release v1 before a
decision. No change to the identifier was made.

## Next
- Paper: report the leakage table (trip removed, own observer removed, species-matched control) on
  the frozen release's test set; this branch's commands take `--manifest`.
- Retrain / labels: refresh the long-ago-validated records' names before the retrain (slice 3);
  the wrong-photo references can go as planned (no effect on dev either way).
- Place prior: give far-from-range and off-season finds a wide berth instead of a penalty (the
  occurrence-prior rule); re-check these two slices on any prior change.
- References: aim new DNA sampling at species' range edges and off-season stages; these are the
  slices where depth alone doesn't explain the misses.
- TTA: decide after a box timing and a frozen-release test score; finish the 8-view score average
  only if the 4-view embedding average is in question.
