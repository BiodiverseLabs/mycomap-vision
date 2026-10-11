---
title: Why species misses happen when the species is well referenced
slug: species-miss-deep-dive
date: '2026-10-09'
status: done
question: For held-out records whose true species has 6 or more reference records, why does Vision
  miss the species, apart from sparse reference data?
branch: exp/species-miss-deep-dive (one command, experiments/species-miss-deep-dive/run.py; id-level
  tables stay in the private data folder)
commits:
- 124fc42
reproducibility: exploratory-pre-freeze
benchmark: 'heldout-2026-10-08, truths with 6+ reference records (dev 2,439; test 8,224); stored
  answers of the 2026-10-09 06:13 UTC run, reference 5dbfdb1d24a5; code at main 935ce2f'
split: dev (diagnosis); test (descriptive confirmation only)
model: bioclip-2-ft-20261007-165400
methods:
- nearest+mean
- counterfactual reference edits
- blind hand labels
headline: 'Exploratory (pre-freeze). Dev, 6+ refs: species top-1 60.4% (n 2,439); in 64% of the 965
  misses the winner is in the same genus and the truth is at rank 2-5 in 64%. Label artefacts explain
  about 10%, contaminated references under 1%, the truth more than 1,500 km from any of its references
  2%; same-genus look-alikes ranked close about 32%.'
verdict: Most well-referenced misses are close look-alikes ranked a little ahead of the truth, many in
  genera separated by microscopy or DNA; the clearly fixable share is labels (about 10%); distance to
  the nearest same-species reference lowers accuracy steadily well inside 1,500 km; contamination,
  depth and photo choice explain almost nothing.
decision: pending
related:
- heldout-benchmark
- depth-bias
- name-equivalence
- uninformative-photos
- photo-views
- inat-cv-baseline
- guest-genera
- occurrence-prior
---

## Question
Reproducibility: **exploratory-pre-freeze**. Run before the "Dataset release v1" freeze; every number
here is to be re-run on that release, and any method recommendation is provisional until then.

On the held-out benchmark, what keeps Vision from the right species when its reference holds at
least 6 records of that species? Records with 5 or fewer references are left out: their misses
are mostly sparse data, which is understood.

## Why it matters
Species with 6+ references are 87% of the development records with a species-level truth in the
reference, and they hold most of the misses that more photos alone will not fix. Knowing which
misses are labels, references, scoring conventions or real look-alikes says where effort pays.

## Setup
- Records: heldout-2026-10-08 development split, truths that name a species with 6+ reference
  records (2,439 of 2,819 with the species in the reference). Test split (8,224) only to confirm a
  pattern descriptively, from the stored answers; nothing was chosen on it.
- Model and method: bioclip-2-ft-20261007-165400, nearest+mean, against the same reference the
  benchmark was scored on (154,067 records; the rebuilt copy matched every named species' record
  count, and a CPU re-score reproduced the stored top-1 on 2,437 of 2,437 comparable records).
- The re-score kept the truth's full rank, each photo's own ranking, and the reference records
  behind the winner's and the truth's best matches. Reference facts: distance from the query to
  the nearest same-species reference (distances only), season, observers, how tight each species
  is (mean cosine of its photos to its average) and its nearest species by average vector.
- Inputs from other lanes: the record-source list (non-iNat records fetched as iNat ids), the
  non-fungus photo scan, the label audit's name pairs and live titles, and the DNA sequence names
  of each record (read-only .com query; a hint only, never a key).
- Counterfactual re-scores change only the reference: without non-iNat records, without flagged
  non-fungus photos, both, and without the query observer's own reference records.
- A blind hand check of 60 photo sets (30 misses, 30 hits, shuffled) for what the photos show.
- Provenance: stored answers from the 2026-10-09 06:13 UTC re-score (reference hash 5dbfdb1d24a5);
  the reference was rebuilt from a 14:27 UTC manifest copy (hash a19ddac8464a: 68 more one-word or
  higher-rank records, the same record count for every named species); code at main 935ce2f.
- One command re-runs everything on a manifest or a release, from the stored per-record report:
  `experiments/species-miss-deep-dive/run.py --manifest <m> | --release <dir> --predictions
  <report-dev.csv> --reference-hash <hash> --out <private folder>` (README beside it). CPU only;
  it never imports torch, which makes numpy segfault on large CPU matrix products in the shared
  environment.

## What we tried
Every analysis below, including the ones that showed nothing:
- Rank of the truth in each miss (full ranking, not only the stored 10).
- Winner vs truth: same genus, family, reciprocal confusions, name-pair lists, shared epithet.
- Provisional codes vs formal names, by depth.
- Photo set: number of photos, first photo vs any single photo, photo size.
- Reference side: geography, season, observer count and dominance, species spread and nearest
  neighbour, the observer's own references, whether the winner is deeper.
- Label side: DNA sequence names, answer-key source (title vs index name), stale reference labels.
- Contamination and non-fungus photos, as counterfactual re-scores.
- A zero-shot photo-view tagger (cap, gills, pores, stem base, cut) from BioCLIP 2's text tower
  on the fine-tuned vectors: it failed a 30-photo eyeball check (about 8 right), so views are not
  described. A view breakdown would need hand labels.
- A curated list of macro-cryptic species groups, sized on dev and checked on test (below).

## Results
**Standard table, nearest+mean, truths with 6+ references** (% right within top k):

| Development (n 2,439) | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|
| species strict | 60.4 | 79.7 | 85.6 | 91.1 |
| species s.l. | 60.4 | 79.7 | 85.6 | 91.1 |
| species complex | 62.6 | 80.6 | 86.2 | 91.4 |
| genus strict | 85.8 | 92.6 | 95.0 | 96.6 |
| genus s.l. | 86.2 | 92.9 | 95.1 | 96.7 |

| Species strict by the true species' references | top 1 | top 3 | top 5 | top 10 | n |
|---|---|---|---|---|---|
| dev 6-19 | 43.2 | 66.0 | 76.9 | 85.4 | 611 |
| dev 20-99 | 65.6 | 83.6 | 87.5 | 92.0 | 1,344 |
| dev 100+ | 67.8 | 86.4 | 91.3 | 95.7 | 484 |
| test 6-19 | 42.5 | 66.5 | 75.4 | 84.4 | 2,017 |
| test 20-99 | 66.2 | 84.4 | 88.8 | 93.4 | 4,512 |
| test 100+ | 73.2 | 88.2 | 91.6 | 94.7 | 1,695 |

Test (n 8,224, descriptive): species strict 61.8 / 80.8 / 86.1 / 91.5, s.l. the same, complex
64.5 / 82.1 / 87.0 / 91.9, genus strict 87.6 / 93.9 / 95.4 / 97.0, genus s.l. 88.0 / 94.1 / 95.5
/ 97.1. iNaturalist's computer vision on the shared subsample: in the iNat section below.

**1. Where the truth is.** In 965 development misses the truth is at rank 2-5 in 63.6%, 6-10 in
13.9%, 11-20 in 7.6%, 21-100 in 9.9% and below 100 in 5.0% (median rank 4). By band, the
truth is in the top 5 in 59% (6-19), 64% (20-99) and 73% (100+) of misses. Test, from the stored
10: 63.5% / 14.2% / 22.3% beyond 10. Most misses are ranking failures, not representation
failures. In 17.7% of misses the winner leads by less than 0.01 (the median gap between first and
second is 0.013 in misses against 0.045 in hits).

**2. Look-alikes.** The winner is in the same genus in 64.0% of misses (test 67.4%). Misses are
spread thin: 928 distinct truth-winner pairs for 965 misses; pairs seen twice or more cover 7.3%
(test 17.1%), and pairs confused both ways 3.3% (test 8.1%). Twenty genera hold half the misses.
The seven genera usually named as microscopy- or DNA-dependent (*Cortinarius*, *Inocybe*,
*Russula*, *Lactarius*, *Entoloma*, *Galerina*, *Mycena*) are 22.7% of the records at 52.8% top-1
against 62.7% elsewhere, and 27.0% of misses (test: 24.2% of records, 53.6% vs 64.5%, 29.4% of
misses). By family: Russulaceae 11.4% of misses (53.8% top-1), Tricholomataceae 9.1%,
Cortinariaceae 6.8% (43.1%), Strophariaceae 6.5%, Inocybaceae 5.5%. Same-genus misses sit close
(median cosine of the two species' average vectors 0.84, truth in the top 5 for 72%); other-genus
misses are far (0.54; truth in the top 5 for 49%). In 20.9% of misses the winner is the truth's
nearest species by average vector. Typical same-genus pairs: *Pluteus cervinus* with *P. hongoi* /
*P. petasatus*, *Rhodocollybia asema* with *R. butyracea*, *Morchella importuna* with a
*sextelata* code, *Lactarius* sp. 'fallax-PNW03' with *L. fallax*, *Laccaria laccata* with *L.
nobilis* / *L. trichodermophora*, *Amanita bisporigera* with other destroying angels, *Galerina
marginata* with *G. castaneipes*, *Hydnum* and *Clavulina* pairs. Hosts and parasites: ten misses
(1.0%) are records of fungi that grow on other fungi (*Hypomyces*, *Naematelia*, *Tolypocladium*,
*Collybia tuberosa*); in about half the winner is the host or a look-alike of it.

**3. Provisional codes vs formal names.** Top-1, formal / provisional: 6-19 refs 43.3 / 43.2
(n 319 / 292); 20-99 68.0 / 60.4 (927 / 417); 100+ 66.8 / 71.4 (386 / 98). Test: 46.6 / 38.8,
69.3 / 59.9, 73.1 / 73.7. Codes do worse at 6-99 references. In same-genus misses the kinds mix
evenly (formal to formal 228, formal to code 143, code to code 145, code to formal 102). 54 misses
(5.6%) share an epithet stem with the winner (the complex row credits them): code to code 29 (for
example regional *semitale*, *micaceus* or *coralloides* codes), formal truth to its own code 16,
code to its formal parent 8. These are DNA-split taxa of one morphospecies, not mislabels. The DNA
names agree: in only 4 misses does a sequence name the winner.

**4. Photo sets.** Top-1 by photos: 1 54.5% (n 255), 2 58.0%, 3 58.2%, 4-5 63.8%, 6+ 64.2%. In
33.5% of multi-photo misses some single photo alone is right; the first photo is right in 14.4%
(in hits the first photo alone is right in 73.1%). The best single photo has the truth in its top 5
in 71.9% of misses. Photo size shows nothing (nearly all 500-799 px on the short side). The blind
hand check: photo sets with only a dried, tiny or distant specimen, or another subject, were 7 of
30 misses and 4 of 30 hits. Most misses have good fresh photos of the fertile surface. Far misses
(truth below 10) look different: 19% single-photo, and only 7% have any photo right.

**5. Reference side.**
- *Geography* matters beyond depth, and steadily. Top-1 by the distance from the query to the
  nearest reference of the true species: under 100 km 65.5% (n 1,702; 587 misses), 100-300 km
  53.4% (468; 218), 300-1,500 km 43.5% (246; 139), over 1,500 km 8.7% (23; 21); every record has
  one. With / without a same-species reference within 300 km: 6-19 refs 47.8 / 28.2 (n 469 / 142);
  20-99 66.9 / 50.5 (1,237 / 107); 100+ 67.5 / 75.0 (464 / 20). Test: 46.7 / 28.9, 67.3 / 54.2,
  73.6 / 59.6. In half of the misses with no truth reference within 300 km the winner has one.
- *Season*: with no truth reference within 30 days of the year, 36.2% (n 47); otherwise 55-62%.
- *Observers*: 96% of these truths have 6+ distinct observers, so a species seen by one or two
  observers is not a factor here (3-5 observers: 32.6%, n 95). The observer's own references help.
  Top-1 is 65.9% when the observer has references of the true species against 56.3% without (test
  68.3 vs 56.7). Leave-observer-out (below) measures it.
- *Spread*: species whose photos are loose (low cosine to their own average) or whose nearest
  species is close miss more. By quartile of separation (spread minus nearest-species cosine)
  53.0, 58.8, 60.9, 69.3%.
- *Depth* is not the story at 6+: the winner has more references than the truth in 48% of misses
  (median 34 vs 30).

**6. Labels.** A DNA sequence of the record names another species in 2.1% of misses against 1.0%
of hits; truth-side label noise is small. The truth-winner pair is in the label audit's pair list
in 6.0% of misses (test 5.9%), mostly stale reference labels. The winner's two best reference
records carry a stale label in 5.5% of misses (17 hits). Their live title is the truth in 2.0%.
Writing variants (for example *luteo-olivaceum* / *luteoolivaceum*) are 3 misses. Records whose
.com title differs from the index name (a recent rename, n 197) are right 52% vs 61%.

**Counterfactual re-scores** (only the reference changes; development, n 2,439, top-1 60.4):

| Reference change | top-1 | fixed / broken | sign test p |
|---|---|---|---|
| without non-iNat records (13,031 photos) | 60.6 | +4 / -1 | 0.38 |
| without flagged non-fungus photos (5,610) | 60.6 | +4 / -1 | 0.38 |
| both | 60.6 | +5 / -2 | 0.45 |
| without the query observer's own references | 59.5 | +32 / -56 | 0.014 |

Wrong-photo references never supplied the winner's best two matches in a 6+ miss. They matter for
sparse species, not here. Own references help more than they hurt (about 1 point of the
benchmark's top-1; 97% of development records have an observer with references).

**Causes, ranked** (development misses, n 965; one primary cause each by a fixed order, the
fixable ones first; the flags overlap and their own shares are in brackets). A regional gap is the
primary cause only beyond the occurrence prior's wide berth (over 1,500 km, or no reference with a
place); 300-1,500 km is a flag. With a 300 km cut instead, the regional gap took 134 misses (13.9%).

| Cause | Primary share | Flag share | What would fix it (provisional, confirm on v1) |
|---|---|---|---|
| Same-genus look-alike, truth in top 5, outside the 7 hard genera | 23.0% | 34.0% | partly irreducible; near ties (17.7% of misses) are where set matching or a place prior can move records |
| Other genus wins, truth at rank 2-10 | 19.6% | | method: visual neighbours across genera |
| Far miss, truth below rank 10 | 19.9% | 22.5% | mostly irreducible from these photos (few have any photo right; more single-photo) |
| Label artefacts (pair list, stale winner references, DNA names the winner, writing) | 9.6% | 9.6% | label fixes and .com refreshes, before the freeze |
| Same-genus look-alike, truth in top 5, in the 7 hard genera | 9.2% | 12.0% | irreducible from photos in most cases (microscopy, DNA) |
| Same genus, truth at rank 6-10 | 7.7% | | look-alikes ranked lower; as above |
| DNA-split taxa sharing an epithet (codes vs names) | 4.7% | 5.6% | scoring convention: the complex row already credits them |
| The observer's own references pull the wrong way | 2.9% | 3.4% | nothing (a shortcut that helps more than it hurts) |
| Regional gap: truth over 1,500 km from all its references | 1.8% | 2.2% | references from the region |
| (flag) no truth reference within 300 km | | 16.6% | more regional references; a graded place prior |
| Host or parasite in the photo | 1.0% | 1.0% | guest-genera rule, if these should be left out |
| Contaminated references | 0.6% | 0.6% | the record-source fix (in hand) |

Fixable from data: labels (about 10%), contamination (under 1%), and beyond-range gaps (2%).
Distance below 1,500 km is a steady drag rather than a cause one can cut off (65.5% to 43.5% from
under 100 to 300-1,500 km). Scoring convention: shared-epithet complexes (about 5%). Irreducible or
nearly so from photos: hard-genus look-alikes and far misses (about 29%). The rest are close
same-genus or cross-genus neighbours: the hard negatives for the retrain (the truth-winner pairs
are listed in the private miss table).

**Proposal (not adopted): a curated species-complex scoring row.** Groups separated mainly by
microscopy or DNA that met in development misses: black morels (*Morchella* Elata clade), white
destroying angels (*Amanita* sect. *Phalloideae*), the *Pluteus cervinus*, *Laccaria laccata*,
*Hydnum repandum*, *Clavulina coralloides*, *Gymnopilus sapineus*, *Paxillus involutus*,
*Cyanosporus caesius* and *Helvella lacunosa* groups, *Coprinellus micaceus* group, *Rhodocollybia
butyracea* with *R. asema*, *Galerina marginata* with *G. autumnalis*, *Ganoderma tsugae* with *G.
oregonense*, and *G. curtisii* with *G. sessile*. As a scoring-only extension of the complex row,
they credit +1.6 points on development (where they were picked) but only +0.7 on test. That is
the size of the selection effect. Worth adding only if a mycologist vets each group on the
literature, never from these counts.

**iNat on the shared subsample.** iNaturalist's computer vision on the development records of
the 2,000-record subsample whose truth has 6+ references and that iNat answered: 421 (288 formal
names; the run ended 2026-10-09 with 6 records deferred and 8 gone from iNat). iNat keeps 5
answers, so its top 10 is not stored (n.s.).

| n 421 | top 1 | top 3 | top 5 | top 10 |
|---|---|---|---|---|
| species strict: iNat photos + place / photos only / Vision | 32.5 / 31.6 / 58.7 | 43.2 / 43.0 / 79.8 | 48.5 / 48.7 / 85.3 | n.s. / n.s. / 90.5 |
| species s.l. | 32.5 / 31.6 / 58.7 | 43.2 / 43.0 / 79.8 | 48.5 / 48.7 / 85.3 | n.s. / n.s. / 90.5 |
| species complex | 42.3 / 40.4 / 60.8 | 53.7 / 53.0 / 81.0 | 58.7 / 59.1 / 86.7 | n.s. / n.s. / 90.7 |
| genus strict | 74.1 / 73.9 / 85.5 | 86.0 / 84.8 / 92.4 | 90.7 / 89.5 / 94.5 | n.s. / n.s. / 95.7 |
| genus s.l. | 74.3 / 74.1 / 85.7 | 86.5 / 85.0 / 92.4 | 91.0 / 89.5 / 94.5 | n.s. / n.s. / 95.7 |

Formal names only (n 288; iNat cannot name provisional codes): species strict top 1 / 3 / 5, iNat
with place 46.9 / 59.0 / 64.2, photos only 43.8 / 57.6 / 64.2, Vision 61.1 / 80.9 / 86.1; complex
49.3 vs 62.2; genus 76.7 vs 85.1. By band, species top-1 Vision / iNat with place: 6-19 46.4 /
17.0 (n 112), 20-99 62.4 / 33.8 (213), 100+ 64.6 / 47.9 (96). Of the 288 formal-name records:
both right 113, Vision only 63, iNat only 22, both wrong 90.

Where iNat is right and Vision wrong (22), Vision's misses are near misses: the truth is in its top
5 for 86% (typical: *Gymnopilus hybridus* / *G. sapineus*, *Inocybe pallidicremea* / its own PNW
code, *Paxillus involutus* / a *Paxillus* code; one host case, *Naematelia aurantia* answered
*Stereum hirsutum*). The 90 shared misses have Vision's general miss profile (same-genus winner
61%, truth in Vision's top 5 59%, 7 hard genera 26%, no truth reference within 300 km 21%); iNat
gets their genus in half and answers with a name in Vision's vocabulary in 85. These are hard for
both models, not specific to Vision; several shared misses are DNA-split taxa under one name
(*Pseudohydnum gelatinosum* subsp. *pusillum*, *Craterellus subperforatus* answered *C.
tubaeformis* by iNat and a *tubaeformis* code by Vision). Development subsample only; small
numbers are direction, not results. The test subsample was not used here.

## Verdict
At 6+ references Vision misses mostly by ranking a close relative a little ahead (truth in the
top 5 in two thirds of misses, same genus in two thirds), in groups that are often separated by
microscopy or DNA. Contaminated references, depth bias and photo choice explain almost nothing
here. Label artefacts (about 10%) are the clearly fixable share; distance to the nearest same-species
reference lowers accuracy steadily, but only 2% of misses lie beyond the 1,500 km wide berth.
Exploratory: to be re-run on Dataset release v1.

## Decision
pending.

## Next
- Feeds the freeze: label fixes from the label-audit lists (the 9.6%), and whether a vetted complex
  list becomes a scoring row.
- Feeds the retrain: the truth-winner pairs as hard negatives (provisional, confirm on v1).
- Distance: a graded place signal fits the data better than a cut at 300 km (the prior lane).
- Near ties and cross-genus rank 2-10 misses are the targets for observation-set matching and the
  place prior (their lanes).
- Re-run with run.py on Dataset release v1 (iNat on the test subsample, descriptive, then too).
