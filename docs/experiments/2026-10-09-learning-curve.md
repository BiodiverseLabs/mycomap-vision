---
title: Learning curve — accuracy against reference records per species
slug: learning-curve
date: '2026-10-09'
status: done
reproducibility: exploratory-pre-freeze
question: Is species accuracy still rising steeply at today's reference depth (so more DNA sequencing
  and validation is the lever) or flattening (so the model or the labels are), per depth band, and
  which species gain most from more sequenced records?
branch: exp/learning-curve
commits:
- 0a4e7e6
- 92e6070
benchmark: heldout-2026-10-08 (development split, 2,986 records scored; not sealed)
split: dev
model: bioclip-2-ft-20261007-165400
methods:
- nearest+mean
- nearest
headline: 'Cutting one species to N records while every other keeps all of its own (nearest + species
  average, deep species): species top-1 3 / 15 / 26 / 38 / 55 / 63% at 1 / 5 / 10 / 20 / 50 / 100
  records (68% at full depth); each extra record is worth +5.4 points at 1 record, +1.8 at 10, +0.7
  at 20 and +0.2 at 50; the knee is at about 20 records. Development split, 2,986 records.'
verdict: Below about 20 records per species, where 90% of reference species sit, accuracy is still
  rising steeply and sequencing is the lever; above about 50 it has flattened, and the model and labels
  are the lever.
decision: pending
related:
- depth-bias
- heldout-benchmark
- name-equivalence
---

## Question
Species accuracy tracks the true species' reference depth (held-out test: 19% at 1-4 records, 73%
at 100 or more, nearest + species average). Does it keep rising with depth, or has it flattened?
Which depths, and which species, gain most from one more DNA-validated record?

## Why it matters
It says where effort goes. If the curve is still steep, sequencing and validating more records is
what raises accuracy. If it has flattened, more of the same records won't help, and the model or
the labels need the work. A per-species answer turns it into a list of what to sequence next.

## Setup
**Reproducibility: exploratory, pre-freeze** (Steve, 2026-10-09). Every experiment before the
"Dataset release v1" freeze is exploratory. It is re-run on v1, and decisions stay provisional
until then.

- **Records:** the held-out development split, 3,000 records. 2,986 have photos, are not
  reference records and share no photo with the reference.
- **Model and methods:** fine-tuned BioCLIP 2. Nearest + species average (the site's default)
  and nearest, scored exactly as `methods.py` does. With the full reference, the re-scoring
  reproduces the stored answers of the morning re-score record for record: top-1 is equal on
  120 of 120 records checked, for both methods. Nothing was embedded again.
- **Reference:** snapshot hash `5dbfdb1d24a5`, the same as the re-score: 154,067 records,
  590,677 photos, 18,476 groups. It was read from a copy of the manifest taken 2026-10-09
  14:37 UTC.
- **Excluded references:** 9,259 records in this reference whose id .org holds only as a
  Mushroom Observer, MyCoPortal, legacy-site sequence or GenBank record, so their iNat photos
  belong to some other observation. They are left out of every condition, and once included,
  for comparison. The 72 ids green both as iNat and as a non-iNat record are kept.
- **Subsets** are whole records, never single photos. A species' records are kept in a fixed
  random order per seed (seeds 1000 to 1004), so smaller subsets nest inside larger ones.
- **Answer key** is the observation name, scored strict, *s.l.* and complex (`name_equiv`).

Re-run with one command (writes `report.json` / `report.md` / a chart; the provenance block
records code commit, reference hash, seeds, exclusions file and the exploratory label):
`mv learning-curve run --manifest <manifest> --sources <sources TSV> --out <folder>`, or
`--release <pulled release> --benchmark-db <manifest with the held-out tables>`. A 40-record run
through the command gave the same answers as the direct run (all 480 answer lists and 2,148 cut
scores identical).

## What we tried
1. **Fractions:** every species keeps 25 / 50 / 75 / 100% of its records (at least one where it
   has any). 5 seeds.
2. **Everyone capped:** every species keeps at most N = 1 / 3 / 10 / 30 records. 3 seeds.
3. **Add N more:** only the record's true species is cut to N records (1 to 200). Every other
   species keeps all of its records. 5 seeds. This is the planning number: what one more record
   of this species is worth while everything else stays as today. We also measured paired gains:
   the same records, their species cut to d and to d + k (nested), on records whose species has
   at least d + k records.
4. **Saturating fits** of acc(N) = ceiling × N / (N + half) per band. The knee is the depth where
   one more record adds under one point.
5. **Expected value of sequencing:** the measured gain from five more records, taken at the
   species' depth once the held-out records join the reference (they are released into
   training), times its share of the held-out arrivals (12,829 species-level records, both
   splits). A second ranking uses North American iNat observation counts instead.

## Results
Development split, records scored 2,986: species n 2,915, genus n 2,963. iNat's computer vision:
not run in this experiment (see heldout-benchmark).

**Standard table at 100%, wrong-photo references excluded** (% right in the top 1 / 3 / 5 / 10):

| Method | Row | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|---|
| nearest + species average | species strict | 52.9 | 70.7 | 77.0 | 83.2 |
| | species *s.l.* | 52.9 | 70.7 | 77.0 | 83.2 |
| | species complex | 55.7 | 72.6 | 78.7 | 84.4 |
| | genus strict / *s.l.* | 81.7 / 82.4 | 92.1 / 92.3 | 94.7 / 94.7 | 96.6 / 96.6 |
| nearest | species strict | 48.4 | 67.1 | 73.9 | 80.9 |
| | species *s.l.* | 48.4 | 67.1 | 73.9 | 80.9 |
| | species complex | 51.1 | 69.1 | 75.7 | 82.1 |
| | genus strict / *s.l.* | 79.0 / 79.6 | 90.5 / 90.7 | 93.4 / 93.5 | 95.7 / 95.7 |

Species top-1 (top-5) by the true species' reference records:

| References (finds) | nearest + species average | nearest |
|---|---|---|
| 0 (105) | 0.0 (0.0) | 0.0 (0.0) |
| 1-4 (326) | 16.9 (42.0) | 12.0 (35.3) |
| 5-19 (691) | 41.8 (75.0) | 34.9 (67.6) |
| 20-99 (1,340) | 66.3 (87.8) | 60.2 (86.4) |
| 100+ (453) | 68.4 (91.4) | 71.5 (91.6) |

With the 9,259 wrong-photo references left in, the overall numbers barely move: species top-1
52.8 and 48.4, genus 81.7 and 79.0, the same as the re-score. The removal mostly empties
species of their own. Ten references of one common Amanita are all such records, for example,
so its depth drops to 0 until the Mushroom Observer photo fix brings real photos back.

**Add N more** (nearest + species average). Species top-1 when only the true species keeps
min(N, its records); bands by full depth; mean over 5 seeds (sd ≤ 1.2):

| Full depth (n) | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | full |
|---|---|---|---|---|---|---|---|---|---|
| 1-4 (325) | 8.1 | 12.2 | 15.9 | 16.9 | | | | | 16.9 |
| 5-19 (688) | 7.0 | 12.9 | 19.1 | 28.1 | 38.1 | 41.9 | | | 41.9 |
| 20-99 (1,318) | 4.8 | 10.5 | 16.2 | 25.1 | 39.1 | 54.5 | 64.7 | 66.7 | 66.7 |
| 100+ (450) | 3.1 | 6.6 | 9.8 | 15.1 | 25.7 | 38.3 | 54.6 | 62.8 | 68.2 |

Under the same cut, genus top-1 for the 100+ band goes from 72.9 at 1 record to 87.4 at full;
for the 20-99 band from 68.9 to 87.0. Nearest shows the same shape, lower at every depth: its
100+ band reads 1.9 / 8.4 / 15.3 / 25.9 / 45.1 / 59.8 / 71.3 at 1 / 5 / 10 / 20 / 50 / 100 /
full.

Paired marginal gains (nearest + species average). Top-1 points; 95% bootstrap interval over
records; "not run" where the larger cut was not among those scored:

| At d records | +1 | +5 | +10 |
|---|---|---|---|
| 0 (not in the index) | +5.4 (4.8-6.0) | +24.1 (22.6-25.4) | +36.4 (34.6-38.1) |
| 1 | +5.4 (4.8-5.9) | +22.3 (20.9-23.6) | +33.2 (31.6-34.9) |
| 3 | +4.5 (4.1-4.9) | +16.7 (15.6-17.7) | not run |
| 5 | +3.3 (2.9-3.7) | +13.2 (12.3-14.0) | +22.0 (20.8-23.2) |
| 10 | +1.8 (1.4-2.1) | +8.9 (8.2-9.7) | +14.7 (13.7-15.7) |
| 20 | +0.7 (0.4-1.0) | +4.1 (3.6-4.6) | +6.9 (6.1-7.7) |
| 50 | +0.2 (0.0-0.3) | +1.1 (0.7-1.5) | +2.0 (1.6-2.6) |

The n per cell is 911 to 2,781. Nearest gains a little less early (+3.9 at 1, +1.6 at 10) but
more late (+1.1 at 20, +0.4 at 50).

Saturating fits (nearest + species average):

| Curve | Ceiling | Half at | Knee (<1 pt/record) |
|---|---|---|---|
| 5-19 band | 46.6% | 3.8 | 10 |
| 20-99 band | 76.3% | 9.9 | 18 |
| deep species (100+) cut to N | 75.5% | 19.6 | 19 |

Fitting nearest to the deep species gives ceiling 87.9%, half at 47.3 and knee 17.

**Depth, not difficulty.** At matched depth, species that naturally have few records do better
than deep species cut down to that depth. Nearest + species average, natural against cut:

| Depth | natural | deep cut to that depth |
|---|---|---|
| 1 | 10.3 (87 finds) | 3.1 |
| 3-4 | 20.3 (158) | 9.8 |
| 8-12 | 43.3 (240) | 25.7 |
| 40-60 | 69.0 (348) | 54.6 |

The deficit of sparse species is explained by their depth, and more than fully. They are not
intrinsically harder. A deep species cut down still competes with look-alikes at full depth, so
the cut numbers are a conservative guide.

**Fractions of everyone** (nearest + species average, mean ± sd over 5 seeds):

| Kept | 1-4 | 5-19 | 20-99 | 100+ | all species | genus |
|---|---|---|---|---|---|---|
| 25% | 12.1 ±1.2 | 29.0 ±1.2 | 59.2 ±0.6 | 68.6 ±0.8 | 46.1 ±0.5 | 79.0 ±0.5 |
| 50% | 13.8 ±1.2 | 36.6 ±0.8 | 63.7 ±1.1 | 68.8 ±0.7 | 50.2 ±0.6 | 80.6 ±0.3 |
| 75% | 15.7 ±0.5 | 40.0 ±1.4 | 65.8 ±0.7 | 68.9 ±0.9 | 52.2 ±0.7 | 81.5 ±0.4 |
| 100% | 16.9 | 41.8 | 66.3 | 68.4 | 52.9 | 81.7 |

Growing every species in proportion has nearly stopped paying overall. From 75% to 100% of
the reference, species top-1 rose only +0.7 points; the 100+ band is flat throughout. When
every species is capped at N records, the sparse species do slightly better than at full depth
(1-4 band 18.9% with everyone at 1 record, against 16.9%): deep neighbours take answers from
them.

**What to sequence next.** This is guidance for now and will be refreshed on Dataset release v1.
Of the expected gain from five more records per species, weighted by how often each species
arrives:

- 37% sits with species that will have 1-4 records once the held-out records join;
- 44% with species at 5-19;
- 17% at 20-99;
- 1% at 100+.

Top genera by summed expected gain: Cortinarius, Russula, Entoloma, Inocybe, Amanita, Mycena,
Lactarius, Agaricus, Ramaria, Hygrocybe, Pluteus, Psathyrella, Tricholoma, Phlegmacium,
Lactifluus. Each has hundreds of species, most of them under 20 records.

Top species by expected gain × arrivals (records now → after the held-out join):

- Collybiopsis confluens subsp. campanulata (0 → 11)
- Psathyrella sp. 'PEI01' (1 → 11)
- Calonarius elegantio-occidentalis (1 → 10)
- Lactarius montanus (2 → 15)
- Amanita sp. 'pallidorubescens' (1 → 8)
- Amanita muscaria var. flavivolvata (4 → 15)
- Clavulina chengdeensis (3 → 11)
- Ramaria highlandensis (3 → 11)
- Porphyrellus sp. 'porphyrosporus-CA01' (5 → 16)
- Mycena constans (2 → 8)
- Morchella importuna (7 → 26)
- Gymnopus sp. 'HI01' (5 → 15)

Ranked by North American iNat observations instead, the list is led by very common species with
almost no DNA-validated references: shaggy mane, common Amanitas, several lichens. People
sequence what is hard to name, so the commonest species stay thin in the reference. One
"Tubariua" name ranks too: a known legacy-site misspelling, to fix rather than sequence.

## Verdict
Both, by depth.

- **Below about 20 records,** accuracy is still steep. One more record is worth 2 to 5 points
  of top-1 for that species, five more are worth 9 to 24, and the deficit is caused by depth
  itself. 90% of reference species and about a third of finds are here, so targeted sequencing
  and validation of these species is the lever.
- **Above about 50 records,** the curve is flat: +0.2 points per record, and the 100+ band
  doesn't move when the reference is cut to a quarter. There the model, the scoring and the
  labels are the lever. At 100+ the default blend is below nearest (68.4 against 71.5), and
  that gap is the place to look.
- More records of the same mix help little overall (+0.7 points from 75% to 100%). Where the
  next records go matters more than how many there are.

## Decision
pending. A "what to sequence next" list is offered to Steve as guidance. All of this is
exploratory before the Dataset release v1 freeze.

## Next
- Re-run on Dataset release v1 with the one command; refresh the list.
- Look at the 100+ band: why the blend loses to nearest there, and look-alike pairs.
- Add the missing cuts (7, 12, 13 records) if the d = 2 to 3 cells matter.
- Count arrivals from the nightly feed rather than the held-out pool once it runs again.
- The genus-level list as a sampling plan for forays.
