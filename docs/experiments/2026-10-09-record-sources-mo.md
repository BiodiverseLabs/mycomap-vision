---
title: 'Record sources: wrong-source photos out, Mushroom Observer photos in'
slug: record-sources-mo
date: '2026-10-09'
status: adopted
reproducibility: exploratory-pre-freeze
question: How much do the non-iNat records that carried an unrelated iNat observation's photos cost, and
  what do DNA-validated Mushroom Observer records with their own MO photos add?
branch: feat/record-sources-mo (merged 0b9d1f2); fix/nightly-feed-source (merged 2bd0682)
commits:
- 90428e1
- 33440bd
- 0f71684
- c91fcfa
- 173c270
benchmark: 'heldout-2026-10-08 (3,000 development records; 10,145 test records for one confirmation;
  not sealed)'
split: dev; test (confirmation only)
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
- nearest+prior@org
headline: 'Nearest + species average, species top-1: development 52.8 / 52.9 / 53.5 (current refs / wrong
  photos out / MO in; 2,915 records, a to c +38 / -17, p = 0.006); test confirmation 54.2 / 54.1 / 54.4
  (9,867 records, +105 / -88, p = 0.25).'
verdict: Removing the wrong-source photos is neutral on iNat queries; adding the MO records with their own
  photos helps species with few references, by less on test than on development.
decision: 'Steve 2026-10-09: include DNA-validated MO records with their real MO photos, leave out
  MyCoPortal (rarely field images) and .com Sequences; 2026-10-11: migrate the shared manifest for
  sources only (names unchanged until the dataset release), MO image files from a zip made from
  MO''s image store.'
related:
- heldout-benchmark
- depth-bias
- full-run-finetune
- not-fungus-gate
---

## Question
Until 2026-10-09 the export took every numeric record id for an iNaturalist id. Mushroom Observer
(MO) observation numbers, MyCoPortal occurrence ids and legacy .com sequence ids are numeric too,
so Vision fetched whatever iNat observation shared the number: a mallard as a tooth fungus. How
much did those photos cost, and what do the DNA-validated MO records add once they carry their
own MO photos?

## Why it matters
About 6% of the reference records (9,266 of 154,067 loaded) carried photos of unrelated
organisms, and 1,708 species had only wrong references; the fine-tune trained on them. MO
records are DNA-validated field finds that Vision could not use at all.

## Setup
One snapshot of the manifest (taken 2026-10-09), three reference sets, the same model, the
held-out benchmark's development split (3,000 iNat records Vision never saw; answer key = the
observation name), place for the prior from .org.

- (a) current references: reference hash 5dbfdb1d24a5, 154,067 records, 18,070 species.
- (b) wrong-source photos removed, MyCoPortal / .com Sequences / GenBank records left out (their
  photos unlinked with an audit row each): 47653c3bcaa3, 144,801 records, 16,362 species.
- (c) (b) plus the North American DNA-validated MO records with their own MO photos (32,440
  photos of 7,154 records, all Creative Commons or public domain; image files from a zip made
  from MO's image store, metadata from MO's images API at 5 s a request): 854991f79c5e, 151,870
  records, 17,262 species.

Sources are explicit everywhere (records.source from .org's observations.source; photo fetching,
embedding and use dispatch on it; non-iNat records keyed by source). Code: sources.py, mo.py.

## What we tried
Three configurations, three methods each, all reported (nothing chosen among them on these
numbers). Test confirmation: nearest + species average only, the served default.

## Results
Development split, species top-1 by method (n = 2,915 records with a species in the key):

| Method | (a) | (b) | (c) | (a) to (c): fixed / broken | McNemar p |
|---|---|---|---|---|---|
| nearest | 48.3 | 48.3 | 49.1 | +37 / -14 | 0.002 |
| nearest + species average | 52.8 | 52.9 | 53.5 | +38 / -17 | 0.006 |
| nearest + prior (.org place) | 51.3 | 51.2 | 52.2 | +45 / -16 | 0.0003 |

(a) to (b) alone: nearest +1 / -1, nearest + species average +5 / -1, with the prior +13 / -15
(none significant); genus within 0.2 points.

Standard tables, nearest + species average, top 1 / 3 / 5 / 10 (species n = 2,915, genus n =
2,963):

| | (a) | (b) | (c) |
|---|---|---|---|
| species strict | 52.8 / 70.7 / 77.0 / 83.2 | 52.9 / 70.7 / 77.0 / 83.2 | 53.5 / 71.2 / 77.3 / 83.2 |
| species s.l. | 52.8 / 70.7 / 77.0 / 83.2 | 52.9 / 70.7 / 77.0 / 83.2 | 53.5 / 71.2 / 77.3 / 83.2 |
| species complex (beta) | 55.6 / 72.6 / 78.6 / 84.4 | 55.7 / 72.6 / 78.7 / 84.4 | 56.4 / 73.2 / 79.1 / 84.4 |
| genus strict | 81.7 / 92.1 / 94.7 / 96.6 | 81.7 / 92.2 / 94.7 / 96.6 | 82.0 / 92.2 / 94.6 / 96.6 |
| genus s.l. | 82.4 / 92.3 / 94.7 / 96.6 | 82.4 / 92.3 / 94.7 / 96.6 | 82.8 / 92.4 / 94.6 / 96.6 |

Nearest: species strict 48.3 / 67.2 / 73.9 / 80.9 (a), 48.3 / 67.2 / 73.9 / 80.9 (b), 49.1 /
67.7 / 74.3 / 81.7 (c); genus strict 79.0 (a), 78.9 (b), 79.5 (c) at top 1. With the prior:
species strict 51.2 / 70.0 / 75.8 / 82.0 (a), 51.2 / 70.2 / 75.7 / 82.0 (b), 52.2 / 70.3 / 76.2 /
82.3 (c); genus strict 79.5 (a), 79.3 (b), 79.7 (c) at top 1.

Species by the true species' reference records, nearest + species average, top 1 / top 5 (each
configuration counts its own references, so a record can change band):

| References | (a) | (b) | (c) |
|---|---|---|---|
| 0 | 1.0 / 1.0 (99) | 0.9 / 0.9 (106) | 1.0 / 1.0 (100) |
| 1-4 | 16.0 / 40.4 (307) | 16.9 / 42.0 (326) | 17.5 / 39.9 (308) |
| 5-19 | 40.9 / 73.6 (682) | 41.9 / 75.0 (692) | 43.4 / 75.0 (687) |
| 20-99 | 65.6 / 87.5 (1,344) | 66.3 / 87.8 (1,338) | 65.7 / 87.7 (1,337) |
| 100+ | 67.9 / 91.3 (483) | 68.4 / 91.4 (453) | 68.1 / 91.5 (483) |

Fixed / broken from (a) to (c), nearest + species average, by (a)'s band: 1-4 +7 / -0, 5-19 +17 /
-3, 20-99 +12 / -10, 100+ +2 / -4.

Test split, confirmation only (scored once per configuration, nothing chosen on it), nearest +
species average, top 1 / 3 / 5 / 10 (species n = 9,867, genus n = 10,002):

| | (a) | (b) | (c) |
|---|---|---|---|
| species strict | 54.2 / 72.0 / 77.4 / 83.4 | 54.1 / 71.9 / 77.5 / 83.4 | 54.4 / 72.1 / 77.6 / 83.5 |
| species s.l. | 54.2 / 72.0 / 77.4 / 83.4 | 54.1 / 71.9 / 77.5 / 83.4 | 54.4 / 72.1 / 77.6 / 83.5 |
| species complex (beta) | 57.2 / 73.9 / 79.0 / 84.4 | 57.1 / 73.9 / 79.0 / 84.3 | 57.3 / 74.0 / 79.1 / 84.5 |
| genus strict | 83.5 / 92.8 / 94.7 / 96.3 | 83.6 / 92.9 / 94.7 / 96.3 | 83.6 / 92.9 / 94.8 / 96.3 |
| genus s.l. | 84.1 / 92.9 / 94.7 / 96.3 | 84.2 / 93.0 / 94.8 / 96.4 | 84.2 / 93.1 / 94.8 / 96.4 |

Species by the true species' reference records, test, top 1 / top 5 (n): (a) 0: 0.3 / 0.3 (344),
1-4: 19.2 / 39.6 (1,075), 5-19: 40.9 / 73.7 (2,241), 20-99: 66.2 / 88.8 (4,512), 100+: 73.2 / 91.6
(1,695); (c) 0: 0.3 / 0.3 (356), 1-4: 19.2 / 41.1 (1,073), 5-19: 42.2 / 74.2 (2,233), 20-99: 66.1 /
88.8 (4,513), 100+: 73.0 / 91.2 (1,692).

Fixed / broken on test, species top-1: (a) to (b) +9 / -19 (p = 0.09); (b) to (c) +104 / -77 (p =
0.053); (a) to (c) +105 / -88 (p = 0.25), by (a)'s band 1-4 +12 / -5, 5-19 +35 / -20, 20-99 +49 /
-52, 100+ +9 / -11.

The (a) test answers came from the 2026-10-09 re-score's code and (b), (c) from later code; 292
development records answered again with the later code gave the same ranked names at every rank,
so the code change does not move these numbers.

iNaturalist's computer vision: not run (it does not depend on Vision's reference set).

## Verdict
The wrong-source photos cost little on this benchmark: its queries are iNat photos of fungi, and a
duck or a plant rarely wins a match against them. Their removal is still required: they were
wrong references (and wrong training examples), and 1,708 species had no others. Adding the MO records
with their own photos helps, concentrated where it matters, in species with 1-19 references: +0.7
species top-1 on development (significant), +0.2 on the test confirmation (same direction, not
significant on its own; the 1-19 bands still gain, +47 / -25). Development and test of an
unsealed benchmark, not the paper's numbers.

## Decision
Steve 2026-10-09: MO records in with their real MO photos; MyCoPortal and .com Sequences out.
2026-10-11: the shared manifest migrated for sources only (16,795 records re-keyed, 13,166 wrong
photo links unlinked with an audit trail; names left as they were, so the development benchmark
stays usable), MO metadata and the zip's photos added. The full re-export (new records and
renames) waits for the dataset release.

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
The served reference index on the Vision box drops the wrong-source records when its nightly
update first runs with sources from mycomap.org (the .org feed change). The retrain uses the MO
photos (in the trainer's store). MO photos of records added later are downloaded from MO at MO's
pace.
