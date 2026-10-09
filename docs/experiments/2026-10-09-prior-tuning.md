---
title: Place and date prior on top of nearest + species average, tuned on development
slug: prior-tuning
date: '2026-10-09'
status: done
question: Does a place-and-date prior (our DNA records, iNat occurrences, or both, with a season
  weight) add to nearest + species average once its settings are tuned honestly on development?
branch: exp/prior-tuning
commits:
- 89b53e9
- 31f6770
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
headline: 'Development, cross-validated: nearest + species average 52.8 -> 55.1% species top-1 with the
  tuned DNA-record prior (149 fixed, 83 broken), genus 81.3 -> 82.1%; test (confirmation only) 54.3 -> 56.2% (496 fixed, 304 broken), genus 83.0 -> 83.7%.'
verdict: The DNA-record prior at half weight adds about 2 points at species and 0.7 at genus to nearest
  + species average on development and test, but costs records whose species is known only far away;
  the iNat occurrence prior adds less and nothing on top of it.
decision: pending
related:
- occurrence-prior
- dna-range-prior
- depth-bias
- heldout-benchmark
---

## Question
nearest + species average is the photos-only default. Does adding where and when a fungus was
found improve it, and which prior: our own DNA records, iNaturalist's open occurrence data, or
both? Tuned on development only, with an honest (cross-validated) number.

## Why it matters
The DNA-record prior added about 3 points at species to nearest, untuned. Our DNA records are too
sparse to draw ranges for sparse species, so the iNat occurrence prior (denser, never tuned) was
the hope for them. A prior must not bury true finds far from known records.

## Setup
Development split of heldout-2026-10-08: 2,986 records with photos (2,915 with a species answer,
2,979 with a genus), scored against the fine-tuned BioCLIP 2 reference index as it stood on
2026-10-09 (154,135 records; the benchmark's own runs used 154,067, and nearest, nearest + species
average and nearest+prior reproduce their published development numbers to 0.1 point). Photo
scores are put on the log-probability scale at temperature 0.02 and the prior is added. Place is
the record's own place on mycomap.org (never shown); each record's own iNat observation is taken
back out of the occurrence counts. CPU only, on a copy of the manifest.

Grid, declared in `priortune.py` and committed (89b53e9) before any result was seen; every
setting scored, its counts in `2026-10-09-prior-tuning-settings.csv`:
- DNA prior (288): place kernel 75 / 150 / 300 km; season kernel 10 / 20 / 40 days; place and
  season weights 0 / 0.25 / 0.5 / 1 each; cap log 5 or log 20 per term.
- iNat occurrence prior (1,152): wide berth 1,000 / 1,500 / 2,000 km; out-of-range penalty
  0 / 2 / 6 / 50 (exclusion) nats; genus rule on or off; density kernel 75 / 150 / 300 km;
  density and season weights 0 / 0.25 / 0.5 / 1 each.
- Combined (384): berth 1,500 km with the genus rule; penalty 0 / 2 / 6 / 50; iNat density
  0 / 0.25 / 0.5; DNA place 0 / 0.25 / 0.5 / 1; season from DNA or iNat, weight
  0 / 0.25 / 0.5 / 1.
- Reference, never chosen: the DNA prior as it ships (place 150 km + season 20 days, summed,
  capped at log 20, weight 1).

Selection: species top-1, then genus top-1, then the gentler setting. Reported number: 5-fold
cross-validation, folds grouped by observer (597-598 records each), each fold scored with the
setting chosen on the other four; confidence temperatures fitted the same way. The search counts a
species right only for the group its answer key maps to; the report's judge (used for every table
below) also accepts a group of the same name written another way, so the search's counts run
about 0.5 point lower, alike for every setting.

## What we tried
All 1,824 settings on the development split (species right of 2,915; nearest + species average
alone: 1,526 by the search's count):

| family | settings | best | worst | within 10 records of best |
|---|---|---|---|---|
| DNA prior | 288 | 1,624 | 1,526 | 21 |
| iNat occurrence prior | 1,152 | 1,569 | 1,526 | 228 |
| combined | 384 | 1,620 | 1,526 | 48 |

Best of each family on all of development: DNA place 0.5 at 150 km and season 0.5 at 10 days, cap
log 20 (1,624); iNat berth 1,000 km, penalty 2, density 1 at 150 km, season 0.25 (1,569; the
penalty and genus rule hardly matter: 228 settings within 10 records); combined: berth 1,500 km
with penalty 0 or 2, iNat density 0.25, DNA place 0.5, DNA season 0.5 (1,620). The folds' own
choices moved around a plateau (DNA place weight always 0.5; kernel 75-300 km and season
10-40 days varied), which is why the cross-validated number is the one to quote.

Cross-validated species top-1 by the search's count, and in-sample (all of development) for
the setting each family picks: DNA 54.6 (in-sample 55.7); iNat 53.6 (53.8); combined 55.0 (55.6);
whole grid 54.6 (55.7, the DNA pick).

## Results
Development, n = 2,986 records (species rows n = 2,915, genus rows n = 2,979). Top 1 / 3 / 5 / 10.
The tuned rows are cross-validated (each record scored with a setting chosen without it).

| method | species strict | species s.l. | species complex (beta) | genus strict | genus s.l. |
|---|---|---|---|---|---|
| nearest | 48.4 / 67.1 / 73.9 / 80.9 | 48.4 / 67.1 / 73.9 / 80.9 | 51.1 / 69.1 / 75.7 / 82.1 | 78.6 / 90.0 / 92.8 / 95.2 | 79.3 / 90.2 / 93.0 / 95.2 |
| nearest + species average | 52.8 / 70.7 / 77.0 / 83.2 | 52.8 / 70.7 / 77.0 / 83.2 | 55.6 / 72.7 / 78.6 / 84.4 | 81.3 / 91.6 / 94.2 / 96.1 | 82.0 / 91.8 / 94.2 / 96.1 |
| nearest + DNA prior as shipped (reference) | 51.3 / 70.0 / 75.8 / 82.0 | 51.3 / 70.0 / 75.8 / 82.0 | 53.7 / 71.7 / 77.0 / 83.1 | 79.0 / 90.6 / 93.2 / 95.3 | 79.9 / 90.7 / 93.2 / 95.4 |
| nearest + species average + DNA prior as shipped | 55.5 / 72.2 / 78.3 / 84.5 | 55.5 / 72.2 / 78.3 / 84.5 | 58.0 / 74.0 / 79.6 / 85.4 | 82.5 / 91.4 / 94.1 / 96.1 | 83.1 / 91.5 / 94.2 / 96.2 |
| + DNA prior, tuned (CV; also the whole-grid pick) | 55.1 / 72.5 / 78.4 / 84.3 | 55.1 / 72.5 / 78.4 / 84.3 | 57.8 / 74.2 / 79.7 / 85.4 | 82.1 / 91.8 / 94.4 / 96.3 | 82.8 / 91.8 / 94.4 / 96.4 |
| + iNat occurrence prior, tuned (CV) | 54.0 / 71.2 / 77.1 / 83.8 | 54.0 / 71.2 / 77.1 / 83.8 | 56.6 / 73.4 / 78.8 / 85.1 | 82.1 / 91.7 / 94.4 / 96.1 | 82.8 / 91.8 / 94.4 / 96.1 |
| + both, tuned (CV) | 55.4 / 72.4 / 78.3 / 84.5 | 55.4 / 72.4 / 78.3 / 84.5 | 58.3 / 74.3 / 79.7 / 85.6 | 82.3 / 92.0 / 94.4 / 96.4 | 82.9 / 92.0 / 94.4 / 96.4 |

iNaturalist's computer vision: not run in this experiment (its subsample is in heldout-benchmark).

Species top-1 (top-5) by the true species' reference records (n 98 / 326 / 684 / 1,327 / 480):

| method | 0 | 1-4 | 5-19 | 20-99 | 100+ |
|---|---|---|---|---|---|
| nearest | 0.0 (0.0) | 14.4 (37.1) | 34.1 (65.8) | 59.8 (86.3) | 70.2 (91.5) |
| nearest + species average | 0.0 (0.0) | 18.4 (42.9) | 40.8 (73.5) | 65.9 (87.7) | 67.7 (91.2) |
| nearest + DNA prior as shipped | 0.0 (0.0) | 16.9 (45.1) | 38.9 (69.7) | 63.6 (86.5) | 68.8 (91.0) |
| nearest + species average + DNA prior as shipped | 0.0 (0.0) | 20.5 (48.2) | 46.1 (75.1) | 68.5 (88.6) | 68.1 (91.0) |
| + DNA prior, tuned (CV) | 0.0 (0.0) | 18.7 (45.1) | 45.8 (75.3) | 68.2 (89.1) | 67.9 (91.7) |
| + iNat occurrence prior, tuned (CV) | 0.0 (0.0) | 17.2 (41.7) | 42.0 (73.5) | 67.8 (88.2) | 68.8 (91.0) |
| + both, tuned (CV) | 0.0 (0.0) | 20.2 (45.4) | 45.3 (75.3) | 68.3 (88.9) | 69.6 (91.7) |

Paired against nearest + species average, top-1 fixed / broken (McNemar P):

| method | species (n 2,915) | genus (n 2,979) |
|---|---|---|
| nearest | 105 / 234 (< 0.001) | 73 / 153 (< 0.001) |
| nearest + DNA prior as shipped | 181 / 225 (0.033) | 106 / 173 (< 0.001) |
| + DNA prior as shipped | 192 / 113 (< 0.001) | 109 / 72 (0.007) |
| + DNA prior, tuned (CV) | 149 / 83 (< 0.001) | 73 / 48 (0.029) |
| + iNat occurrence prior, tuned (CV) | 113 / 79 (0.017) | 79 / 55 (0.047) |
| + both, tuned (CV) | 143 / 66 (< 0.001) | 74 / 43 (0.005) |

Range edges. Species top-1 right by how far the true species' nearest known find is (iNat
occurrence or DNA record, the record's own observation left out), species-answer records:

| nearest known find | n | nearest + species average | + DNA as shipped | + DNA tuned (CV) | + iNat tuned (CV) | + both tuned (CV) |
|---|---|---|---|---|---|---|
| < 100 km | 2,161 | 1,331 | 1,419 | 1,389 | 1,370 | 1,407 |
| 100-300 km | 366 | 133 | 143 | 143 | 132 | 144 |
| 300-1,000 km | 208 | 50 | 33 | 50 | 49 | 41 |
| 1,000-1,500 km | 41 | 9 | 8 | 8 | 8 | 8 |
| >= 1,500 km | 41 | 3 | 2 | 2 | 2 | 2 |
| no known find | 98 | 0 | 0 | 0 | 0 | 0 |

The wide berth (no find within 1,500 km, enough finds elsewhere) flagged the true species in 6
records (14 at 1,000 km, 4 at 2,000 km); nearest + species average had 1 of the 6 right and every
prior variant 0, the tuned DNA prior included, which has no berth: those species are far from
every DNA record of theirs too.

Confidence (species top-1, temperature fitted on the other folds; 10-bin ECE, NLL per record,
share of answers stated at 99% or more): nearest + species average ECE 0.066, NLL 2.14, 1.2%;
+ DNA as shipped 0.044, 2.02, 1.2%; + DNA tuned 0.046, 2.03, 1.2%; + iNat tuned 0.047, 2.09,
1.4%; + both 0.045, 2.02, 1.4%. Mean stated confidence 0.50 against 0.55 right (tuned DNA), so
slightly modest. At the old fallback temperature (the cosine 0.02 on log-probability scores)
95-96% of every method's answers would read 99% or more: that was the saturation bug, and it
stays fixed only while a prior method carries its own temperature (1.52 fitted for the tuned
DNA prior on all of development; the shipped default for +prior is 1.9).

Test confirmation (confirmation only, not used for any choice): scored once, after the choice was frozen (commit 31f6770), with
the one setting the procedure chose on all of development (DNA place 0.5 at 150 km, season 0.5 at
10 days, cap log 20, confidence temperature 1.52) next to the fixed references. Test split, n =
10,105 records (2 guests left out; species rows n = 9,867, genus rows n = 10,066). The references
reproduce the benchmark's published test numbers (nearest 48.3, nearest + species average 54.2 ->
54.3 here, nearest + DNA prior 51.2).

| method | species strict | species s.l. | species complex (beta) | genus strict | genus s.l. |
|---|---|---|---|---|---|
| nearest | 48.3 / 67.0 / 73.6 / 80.5 | 48.3 / 67.0 / 73.6 / 80.6 | 51.3 / 69.2 / 75.4 / 81.8 | 79.6 / 90.5 / 93.1 / 95.2 | 80.3 / 90.6 / 93.1 / 95.2 |
| nearest + species average | 54.3 / 72.0 / 77.4 / 83.4 | 54.3 / 72.0 / 77.4 / 83.4 | 57.2 / 73.9 / 79.0 / 84.4 | 83.0 / 92.2 / 94.1 / 95.7 | 83.6 / 92.3 / 94.2 / 95.7 |
| nearest + DNA prior as shipped (reference) | 51.2 / 69.3 / 75.8 / 82.1 | 51.2 / 69.3 / 75.8 / 82.1 | 53.7 / 71.2 / 77.4 / 83.3 | 79.6 / 90.4 / 93.0 / 95.1 | 80.2 / 90.5 / 93.1 / 95.2 |
| **nearest + species average + DNA prior, tuned (chosen)** | **56.2 / 73.5 / 78.9 / 84.5** | 56.2 / 73.6 / 78.9 / 84.5 | 58.9 / 75.3 / 80.3 / 85.5 | **83.7 / 92.4 / 94.1 / 95.9** | 84.2 / 92.6 / 94.2 / 95.9 |

Species top-1 (top-5) by the true species' reference records (n 343 / 1,134 / 2,250 / 4,450 /
1,690): nearest 0.0 (0.0) / 16.5 (36.3) / 32.1 (65.0) / 59.5 (86.2) / 71.2 (92.0); nearest +
species average 0.0 (0.0) / 21.4 (42.2) / 41.1 (73.8) / 66.3 (88.8) / 73.1 (91.5); nearest + DNA
prior as shipped 0.0 (0.0) / 19.8 (42.1) / 39.4 (69.2) / 62.1 (87.3) / 69.6 (92.4); chosen 0.0
(0.0) / 22.9 (44.8) / 45.3 (76.4) / 68.3 (89.8) / 72.6 (92.4).

Paired against nearest + species average: chosen species 496 fixed / 304 broken, genus 237 / 170
(McNemar P < 0.001 and 0.001); nearest 307 / 900 and 200 / 543; nearest + DNA prior as shipped
590 / 896 and 279 / 622.

Range edges on test (species right by the true species' nearest known find; n 7,265 / 1,220 / 789
/ 116 / 134 / 343): nearest + species average 4,581 / 494 / 199 / 17 / 14 / 0; chosen 4,789 / 527
/ 160 / 13 / 7 / 0; nearest + DNA prior as shipped 4,445 / 442 / 107 / 8 / 2 / 0. Beyond 300 km
(1,039 records) the chosen prior gets 180 right against 230 without it (17.3% vs 22.1%), a cost
the development split was too small to show (60 vs 62 of 290); within 300 km it gains 241. The
wide berth would have flagged the true species in 21 test records at 1,500 km: 6 right without a
prior, 4 with the chosen one, 1 with the shipped one.

Confidence on test at the development temperature: ECE 0.051, mean stated 0.51 against 0.56
right, 1.5% of answers at 99% or more (at the old fallback temperature: 96%).

## Verdict
A place-and-date prior helps nearest + species average: on development, cross-validated, the
DNA-record prior adds 2.3 points at species and 0.8 at genus; test confirms 1.9 and 0.7, with
gains in every band of reference depth except 100+ (73.1 -> 72.6 on test). The iNat occurrence
prior, which was meant to cover sparse species, adds less (1.2 points on development) and nothing
once the DNA prior is there: its density term overlaps the DNA one, and its wide berth touches
under 1% of records. Tuning mattered less than expected: 21 DNA settings sit within 10 records of
the best, and the shipped settings score as well overall (55.5% on development). What tuning
bought is gentleness: half weight keeps more of the records whose species is known only far away
(the shipped settings keep 117 of 1,039 such test records with nearest, the tuned 180 with
nearest + species average), but even the tuned prior gives up about 50 of them for about 240 gains
nearer home. Confidence stays honest when the method carries its own temperature.

## Decision
pending

## Next
- Steve: whether `nearest+mean+prior` becomes the method when a place and date are given
  (photos only stays `nearest+mean`).
- Range edges: try a prior that boosts but penalises less (an asymmetric cap), tuned on
  development, to win back the records far from known finds.
- MycoMap Atlas's range maps as the place term (the pluggable source in `atlasrange.py`) once
  its rebuilt release exists; compare on the same records and range-edge bands.
- iNat's public (possibly obscured) place instead of mycomap.org's for the same records.
