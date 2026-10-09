---
title: 'Non-fungus photos in the references: a scan, and the mislinked records it found'
slug: non-fungus-photo-scan
date: '2026-10-09'
status: done
question: Which reference and training photos show something other than a fungus (animals, people,
  plants alone, screenshots), and does leaving them out change identification?
branch: exp/non-fungus-photo-scan (not merged)
commits:
- f80a0a1
- ea1370b
benchmark: heldout-2026-10-08
split: dev
model: bioclip-2-ft-20261007-165400 (re-score); base BioCLIP-2 (zero-shot scan)
methods:
- zero-shot text prompts on stored vectors
- linear probe on mislinked-record photos
- nearest
- nearest+mean
headline: 17,673 of 590,971 reference photos flagged; 11,851 are in 9,265 mislinked non-iNat records,
  and on genuine iNat records only about 14-23% of flags are truly non-fungus. Dev species top-1
  moves by at most 0.2 points either way (n = 2,918).
verdict: The non-fungus photos are mostly a record-source bug (9,265 Mushroom Observer, MyCoPortal and
  Sequences records fetched an unrelated iNat observation); on genuine iNat records no automatic
  photo filter is precise enough to use.
decision: pending
related:
- uninformative-photos
- photo-views
- heldout-benchmark
---

## Question
A test photo of a duck found, as its nearest specimens, real duck photos attached to two
DNA-validated records. How many reference photos show another organism or an unrelated thing
(animals, people, plants alone, screenshots, maps, documents), and does leaving them out of the
reference index change answers? Slip, basket, habitat and microscope photos were out of scope:
Steve decided on 2026-09-29 and 2026-10-09 not to filter them (see uninformative-photos), and
they were kept out of the "junk" classes here.

Reproducibility: exploratory-pre-freeze. Every number here is to be re-run on the Dataset
release v1 freeze; decisions taken on it are provisional until then. Inputs: a copy of the
manifest taken 2026-10-09 (reference hash a19ddac8464a), a read-only export of .org record
sources the same day, code at ea1370b; one command re-runs it all
(experiments/non-fungus-photo-scan/run_all.py --manifest --org-sources [--data-dir]).

## Why it matters
A non-fungus photo under a fungus label can be any query's nearest specimen, is trained on
by the fine-tune, counts toward a species' reference depth, and can be shown to a user as
"a specimen like yours".

## Setup
- Reference index: every record `evaluate.load_records` serves, read from a copy of the
  manifest taken 2026-10-09 (154,135 records, 590,971 photos, 18,482 species groups;
  reference hash a19ddac8464a). The official dev predict of 2026-10-09 04:16 UTC ran on an
  earlier state (154,067 records), so the "current" rows below are re-scored here on the
  same copy as every variant, not taken from that report.
- Scan: base BioCLIP-2's stored image vectors (all 592,926 photos) against BioCLIP-2 text
  prompts for 11 classes: fungus; bird, mammal, insect/arthropod, other animal, person,
  plant only, document/screen/map (the "junk" classes); slip/label, microscope, habitat
  (kept). A photo is flagged when its top class is a junk class. The fine-tuned model's
  vectors no longer line up with the text tower (it calls 160,000 photos "slip") and were
  not used for the scan. CPU only.
- Record sources: a read-only export of each .org observation's `source`.
- Hand check: thumbnails from Vision's own photo store, judged as non-fungus subject, fungus
  visible, kept class (habitat, slip, microscope, gear), host organism of a fungus (an insect
  in a vial for a Laboulbeniales record), or can't tell.
- Dev re-score: the 3,000 held-out dev records (2,989 with photo vectors; 2,918 with a species
  answer, 2,982 with a genus answer), answered with nearest and nearest+mean against four
  indexes built in memory from the same copy. Depth bands use the full current index for every
  variant, so each band holds the same records. The batched scorer was checked against the
  served code: nearest 8 of 8 records with an identical top 10 at every rank and similarities
  within 1e-7; nearest+mean 40 of 40 identical (checked before the run moved to CPU), with the
  tie rule checked against the method's own sort and the species-mean term against the served
  scorer. Nothing was written to the manifest or the benchmark's tables.

## What we tried
1. Zero-shot scan of every reference photo (above).
2. Cross-checking the flagged photos against each record's .org source and its iNat
   observation's own taxon and date. This found the cause of most flags: Vision's export marks
   any numeric .org observation id as an iNat id, so green Mushroom Observer, MycoPortal /
   MyCoPortal and .com Sequences records fetched whatever iNat observation has the same
   number. The .org date matches the iNat date for 0.000 of them (0.998 for real iNat
   records), and only 3.5% of those iNat observations are fungi at all.
3. A shared list of these records, agreed with the label audit and the record-sources fix:
   a record is kept only if one of its .org rows has source iNaturalist (plus 7 records no
   longer on .org whose iNat look-alike has another date). 9,265 records on this copy.
4. A linear probe (fungus / animal / plant) on base BioCLIP-2, trained on the mislinked
   records' photos (their iNat taxon is known) against genuine iNat fungus photos, to find
   non-fungus photos on genuine records better than the text prompts do.
5. Four dev indexes: current; all zero-shot-flagged photos out; the non-iNat records out;
   the non-iNat records out and the flagged photos on genuine iNat records out.

## Results
**Scan (590,971 reference photos).** 17,673 flagged (3.0%) in 12,370 records: plant only
9,054, other animal 4,304, insect/arthropod 2,487, mammal 1,117, bird 711. Person, document and
slip never win as top class (BioCLIP-2's text side is weak there; phylogenetic-tree
screenshots land in "plant only"). By the label's reference depth (photos): 1-4 records 6,014,
5-19 4,083, 20-99 5,982, 100+ 1,594. 6,617 are a record's only photo; 8,644 records would lose
all their photos.

**Mislinked records.** 9,265 records (6.0% of the reference), 13,043 photos (2.2%): Mushroom
Observer 5,529 (+2 mixed), MycoPortal 2,969, MyCoPortal 680, Sequences 78, no longer on .org 7.
Their iNat look-alikes are plants 3,031, birds 2,076, insects 2,010, other animals 1,329,
mammals 437, fungi 323 (a wrong fungus under the label), other 49. 1,773 species groups have
no reference record except mislinked ones. 11,851 of the 17,673 flagged photos are in these
records (8,396 records would lose every photo). Hand check: 30 of 30 flagged photos there show
no fungus; of 30 unflagged ones, 9 are fungi or lichens (of the wrong record), about 9 are
birds, insects, moths and plants the scan missed, and the rest trees, landscapes and a
micrograph. The fix is upstream (the export), not a photo filter; 399 of these records had been missed by a list built from .com, because their number
also exists on .com as an iNat fungus.

**Genuine iNat records.** 5,822 flagged photos in 3,816 records (248 would lose all photos;
72 are a record's only photo). Hand check of a stratified random 124 (30 per class, 4 birds):
17 show no fungus (plants alone, phylogenetic-tree screenshots, a person, dogs, equipment):
14% raw, about 23% weighted by class size (plant only 11/30, other animal 3/30,
insect 1/30, mammal 2/30, bird 0/4). The rest: kept classes 65 (habitat, slips, microscope,
gear), fungus visible 23 (rusts and leaf spots on plants, entomopathogens on insects),
insect hosts 5, can't tell 14. No score band does better (score above 0.99: 2 of 13). The
probe's top hits (score above 0.99, 30 checked): 8 plants alone, 8 insects in vials (hosts of
Laboulbeniales records, the real subject), 10 habitat, 3 fungi or diseased plants, 1 can't
tell. Thumbnail judgements; n is small, so these are direction, not results.

**Dev re-score (heldout-2026-10-08 dev, not sealed; n = 2,918 records with a species answer,
2,982 with a genus answer; 2,989 answered).** Top 1 / 3 / 5 / 10. Species *s.l.* equals strict on
this set. The current nearest row matches the official dev report of 2026-10-09 within 0.1 point (48.3 /
67.2 / 73.9 / 80.9 on 2,915, an earlier reference state).

| nearest | current | flagged photos out | non-iNat records out | non-iNat out + flagged out |
|---|---|---|---|---|
| species strict | 48.3 / 67.1 / 73.9 / 80.8 | 48.4 / 67.1 / 73.8 / 80.8 | 48.4 / 67.1 / 73.9 / 80.8 | 48.4 / 67.1 / 73.8 / 80.8 |
| species complex (beta) | 51.0 / 69.0 / 75.6 / 82.1 | 51.1 / 69.1 / 75.6 / 82.1 | 51.1 / 69.0 / 75.6 / 82.1 | 51.1 / 69.1 / 75.6 / 82.1 |
| genus strict | 78.6 / 90.0 / 92.8 / 95.2 | 78.5 / 90.0 / 92.9 / 95.2 | 78.5 / 90.0 / 92.9 / 95.3 | 78.5 / 90.0 / 93.0 / 95.3 |
| genus *s.l.* | 79.2 / 90.1 / 92.9 / 95.2 | 79.2 / 90.2 / 93.0 / 95.3 | 79.2 / 90.2 / 93.0 / 95.3 | 79.2 / 90.2 / 93.1 / 95.3 |

| nearest+mean | current | flagged photos out | non-iNat records out | non-iNat out + flagged out |
|---|---|---|---|---|
| species strict | 52.7 / 70.6 / 77.0 / 83.2 | 52.8 / 70.7 / 76.9 / 83.2 | 52.9 / 70.6 / 77.0 / 83.1 | 52.8 / 70.7 / 76.9 / 83.2 |
| species complex (beta) | 55.5 / 72.6 / 78.6 / 84.4 | 55.6 / 72.6 / 78.5 / 84.3 | 55.6 / 72.5 / 78.6 / 84.3 | 55.6 / 72.6 / 78.5 / 84.3 |
| genus strict | 81.2 / 91.5 / 94.1 / 96.1 | 81.2 / 91.7 / 94.2 / 96.1 | 81.2 / 91.6 / 94.2 / 96.1 | 81.2 / 91.7 / 94.2 / 96.1 |
| genus *s.l.* | 82.0 / 91.7 / 94.1 / 96.1 | 81.9 / 91.8 / 94.2 / 96.2 | 81.9 / 91.8 / 94.2 / 96.1 | 81.9 / 91.8 / 94.2 / 96.2 |

Species top-1 / top-5 by the true species' reference records (bands from the current index):

| band (n) | nearest, current | nearest, every variant | nearest+mean, current | nearest+mean, non-iNat out | nearest+mean, + flagged out |
|---|---|---|---|---|---|
| 0 (99) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| 1-4 (307) | 12.4 / 34.5 | 12.4 / 34.5 | 16.0 / 40.4 | 16.0 / 40.4 | 16.0 / 40.4 |
| 5-19 (684) | 34.1 / 65.6 | 34.1 / 65.5-65.6 | 40.8 / 73.5 | 41.1 / 73.4 | 40.8 / 73.1 |
| 20-99 (1,344) | 59.4 / 86.1 | 59.5 / 86.1 | 65.6 / 87.5 | 65.8 / 87.6 | 65.9 / 87.6 |
| 100+ (484) | 70.2 / 91.3 | 70.2 / 91.3 | 67.8 / 91.3 | 67.8 / 91.3 | 67.6 / 91.3 |

Paired records fixed / broken (same method; top 1, then top 5):

| change | method | species top 1 | species top 5 | genus top 1 | genus top 5 |
|---|---|---|---|---|---|
| flagged photos out vs current | nearest | +1 / -0 | +0 / -1 | +4 / -5 | +4 / -2 |
| | nearest+mean | +6 / -4 | +2 / -3 | +6 / -7 | +6 / -4 |
| non-iNat records out vs current | nearest | +1 / -0 | +0 / -0 | +3 / -5 | +4 / -0 |
| | nearest+mean | +5 / -1 (p = 0.22) | +2 / -1 | +3 / -5 | +4 / -2 |
| flagged photos on genuine records out, vs non-iNat out | nearest | +1 / -1 | +0 / -1 | +1 / -0 | +2 / -0 |
| | nearest+mean | +2 / -3 | +0 / -2 | +2 / -2 | +2 / -2 |

No change is significant (McNemar p >= 0.2 everywhere). The label audit's own re-score on the
same 9,266-record list found the same (species top 1: nearest +1 / -1, nearest+mean +4 / -1).
iNat CV: not run (it does not depend on the reference index).

## Verdict
Most non-fungus reference photos are not stray uploads: they are whole records linked to the wrong
observation, because the export treated every numeric id as an iNat id. That is a record-source
bug, fixed upstream, and it matters for training, for the 1,773 species groups that exist only
through wrong photos, for reference depth counts and for the specimens shown to users, even
though it moves dev accuracy by nothing measurable. On genuine iNat records the photo-level
question has a small answer: perhaps 800-1,300 truly non-fungus photos (about 0.2% of the
reference), which an automatic filter cannot pick out (precision 14-23%), and leaving all
5,822 flagged ones out changes dev answers by 1-3 records either way.

## Decision
Pending (Steve). The record-source fix is already decided and under way in its own lane
(2026-10-09: keep Mushroom Observer records with their real Mushroom Observer photos, leave out
MyCoPortal and .com Sequences records).

## Next
Recommendation for Steve (who decides): **leave the flagged photos on genuine iNat records in, both
in the references and in training; adopt no zero-shot photo filter.** The mislinked records leave
both the references and training through the record-source fix (already decided). If a cleanup
of genuine records is wanted later, have a person review the strongest candidates (the
plant-only flags, about 3,070 photos, at about 37% precision, and the 248 records whose every
photo is flagged) and put only person-checked photos into the Dataset release v1 exclusion list.

The review list (photo, record, label, class, score, only photo, record source; private, under
data/audits) and the agreed list of non-iNat reference records are inputs to that release's
inclusion list. Re-run with experiments/non-fungus-photo-scan/run_all.py on the frozen release.
