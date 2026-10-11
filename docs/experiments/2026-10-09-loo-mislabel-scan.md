---
title: Leave-one-out scan of the reference set for mislabelled records
slug: loo-mislabel-scan
date: 2026-10-09
status: done
question: Which reference records do their own photos dispute, once each record is identified
  against all the others, and does leaving the strongest cases out change accuracy?
branch: exp/loo-mislabel-scan
commits: [5e78863, ebc267f, 7fda694, 06ecff8]
benchmark: reference snapshot a19ddac8464a (scan); heldout-2026-10-08 dev (effect)
split: dev
model: bioclip-2-ft-20261007-165400
methods: [nearest+mean]
headline: 'Flagged 1,686 probable wrong labels, 2,214 records in 1,280 two-name pairs, 3,291 wrong-photo
  records, 132 look-alikes of 154,135; DNA backs 46 of 50 hand-checked "wrong labels"; leaving
  1-2% out moves held-out dev species top-1 by at most 0.1 point (52.8%, n = 2,915).'
verdict: The scan finds photos that do not show the sequenced species, cryptic look-alikes and
  two-name pairs, rarely wrong names; removing flagged records does not change accuracy, so
  exclusions are data hygiene, decided record by record by a person.
decision: 'pending: Steve to choose whether, and who, checks the 187 exclusion candidates before
  the Dataset release v1 freeze.'
reproducibility: exploratory-pre-freeze
related: [depth-bias, heldout-benchmark, name-equivalence, full-run-finetune]
---

## Question
The reference set is the identifier: a nearest+mean answer is only as good as the labels
of the records it matches. Which reference records does the model, scoring each one as if
it were a new upload, confidently call something else, and are those wrong labels, one
species filed under two names, wrong photos, or genuine look-alikes?

## Why it matters
A wrong label in the reference costs twice: the record is never matched to its true
species, and it pulls uploads of the species it shows towards the wrong name.

## Setup
Reference snapshot a19ddac8464a (a private copy of the manifest taken 2026-10-09; 154,135
records, 590,971 photos, 18,070 species), nearest+mean (the site default), species
confidence temperature from comparison 20261008-012435-4ef7b0. CPU only. Reproducibility: exploratory, run before the Dataset release v1 freeze; one command (`scripts/loo_mislabel.py --manifest|--release`) re-runs it on the frozen release.

Each record is scored against the reference with hidden: its own photos, every record of
the same observer on the same day (35,640 such groups; the median group is one record, the
99th percentile 41), and any other record holding one of its photos. The hidden records
are masked rather than refitted; a test checks the masked scores equal a refit without them.
Kept per record: top 10 species with scores and confidence, the label's place, top 3
genera, the 10 nearest records, and per photo its best species and best-matching record.

Categories (rules fixed before the scan's results were read; `loo.Rules`):
(a) probably wrong label; (b) one species under two names; (c) wrong photos;
(d) hard look-alikes, not mislabels. Records already on another lane's list are tagged,
not re-derived: the non-iNat ids carrying wrong iNat photos, the label audit's stale names
and name pairs, the non-fungus photo scan.

**Effect on held-out dev, pre-registered before any dev scoring.** Rank the records in (a)
and (c) by evidence strength (`loo.strength`). Leave out the top K, K = 1% (1,541) and
2% (3,083) of the reference records (all of (a)+(c) if fewer), and re-score the held-out
dev split with nearest+mean against the same snapshot, unchanged vs reduced. Secondary
(descriptive): the same K with records already on the non-iNat list skipped. Dev only; the
test split is not scored.

## What we tried
Settings were fixed in `loo.Rules` before results were read (commit 5e78863), with one change,
recorded here. The first wrong-photo rule, "the record's photos point to different genera",
flagged 4,209 of the first 21,011 records scanned (20%): single photos of one record often
disagree. It was replaced after that partial scan and before any removal-variant dev scoring
(commit ebc267f) by "matches no other record (mean best match < 0.55)" or "a photo at cosine
>= 0.95 to a record of another genus" (58% of such near-identical matches cross genera, against
about 10% at 0.85-0.95). The evidence score was made 0 when the model agrees with the label.

Dev removal variants (all scored on the same snapshot as the unchanged reference): the two
pre-registered ones (top 1% and 2% of reference records, categories (a)+(c) by evidence
strength); the same with records on the shared non-iNat list skipped; and, added after the
scan but before any removal-variant dev scoring, all of (a). Five variants.

## Results
**Scan.** 154,135 records; leave-one-out species top-1 agrees with the label on 82,254 of
152,648 species-labelled records (53.9%, close to the held-out dev 52.8%).

| Category | Records | Already on another lane's list |
|---|---|---|
| (a) probably wrong label | 1,686 | 429 |
| (b) same species under two names | 2,214 (1,280 pairs) | 698 |
| (c) wrong photos | 3,291 (3,098 match nothing, 193 cross-genus duplicate photo) | 2,354 (1,987 non-iNat ids) |
| (d) hard look-alikes | 132 (61 pairs) | 53 |

(b) pairs by name relation: provisional beside a formal name in one genus 829, same complex
408, same epithet in another genus 29, writing variant 10, same species *s.l.* 3, no relation
but mutual >= 50% 1. Systematic pairs at the top (records of label A predicted as B): a
Hortiboletus temp code -> H. flavorubellus (17); two Sarcoscypha occidentalis codes (14);
Pleurotus ostreatus -> a P. pulmonarius code (12); two Coprinus comatus codes (11); Pholiota
aurivella -> P. sp. 'aurivella-limonella' (10 of 12 records); Armillaria sp. 'mellea-IN02' ->
A. mellea (7 of 7). Photos cannot tell a synonym from a DNA-split cryptic species, so (b) is a
list of candidate pairs for the label audit, not a verdict.

**Hand-check.** 50 records drawn at random (seed 20261009) from the top 500 of (a), checked by
Claude, not by a person: photos, the record's live legacy title and its DNA sequence names.
DNA and the live title agree with the current label on 46 of 50. Verdicts: wrong name (a stale
label, live title and DNA = the suggestion) 1; non-iNat record with wrong photos (already the
source lane's) 3; photos clearly of another species than the DNA-backed label 6, leaning 6;
photo dominated by another organism (host, parasite, neighbour) 3; look-alike or cryptic split,
label plausible 16; cannot tell 15. Precision for "wrong name" is 1/50; for "something is wrong
with the record as a photo reference" 13/50 clear, 19/50 including leaning.

**Effect on held-out dev** (heldout-2026-10-08 dev, development, not sealed; nearest+mean;
reference a19ddac8464a; 2,986 records scored, 2,915 with a species answer, 2,979 with a genus
answer). Unchanged reference:

| | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|
| species strict | 52.8 | 70.7 | 77.0 | 83.2 |
| species *s.l.* | 52.8 | 70.7 | 77.0 | 83.2 |
| species complex (beta) | 55.6 | 72.7 | 78.6 | 84.4 |
| genus strict | 81.3 | 91.6 | 94.2 | 96.1 |
| genus *s.l.* | 82.0 | 91.8 | 94.2 | 96.1 |

Species top-1 / top-5 by the true species' reference records (unchanged reference): 0 (99):
1.0 / 1.0; 1-4 (307): 16.0 / 40.4; 5-19 (682): 40.9 / 73.8; 20-99 (1,344): 65.6 / 87.5; 100+
(483): 67.9 / 91.3. This reproduces the stored held-out re-score of nearest+mean exactly.

| Variant | removed | species top 1 / 5 / 10 | genus top 1 | species fixed / broken | genus fixed / broken |
|---|---|---|---|---|---|
| unchanged | 0 | 52.8 / 77.0 / 83.2 | 81.3 | | |
| top 1% | 1,541 | 52.8 / 77.0 / 83.0 | 81.3 | 2 / 3 | 2 / 1 |
| top 2% | 3,083 | 52.8 / 77.0 / 83.0 | 81.3 | 3 / 4 | 2 / 1 |
| top 1%, non-iNat skipped | 1,541 | 52.8 / 77.0 / 83.0 | 81.3 | 3 / 3 | 2 / 1 |
| top 2%, non-iNat skipped | 2,945 | 52.7 / 76.9 / 83.0 | 81.5 | 12 / 15 | 9 / 4 |
| all of (a) | 1,686 | 52.7 / 77.0 / 83.1 | 81.5 | 8 / 12 | 7 / 2 |

No depth band moves more than 0.4 point in any variant. The pre-registered ranking mixes two
scales: (c)'s photo evidence (0.45-1.0) outranks (a)'s (median about 0.25), so the top 1% is 95%
(c) "matches nothing", 72% of it records the source lane already fixes. iNat computer vision:
not run (not relevant to a reference-set change).

## Verdict
The scan is a good finder of reference records whose photos do not show the sequenced
species and of two-name pairs, but almost never of wrong names: the legacy title and the DNA
back the label in 46 of 50 checked cases. Leaving the flagged records out does not change
accuracy on development (within 0.2 point, fixed and broken balanced), so there is no accuracy
case for bulk exclusion; an exclusion is a data-hygiene decision per record, by a person.

## Decision
Pending (Steve). Candidates for person review: 187 records (172 in cross-genus duplicate-photo
pairs, 15 from the hand-check), with the source lane's records left out. Only person-checked
rows go to the Dataset release v1 exclusion list (`scripts/loo_exclusions.py`). Relabels go to
the legacy database through the coordinator, with Steve's OK; the (b) pairs go to the label audit.

## Next
Re-run on the frozen release (`scripts/loo_mislabel.py --release`), where the source lane's
wrong-photo records are gone, so (c) shrinks to the genuine cases. A sample-swap check for the
"photos of another species, DNA backs the label" records belongs with the lab, not Vision.
