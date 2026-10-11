---
title: Label audit for systematic effects (training labels, references, answer key)
slug: label-audit
date: '2026-10-09'
status: done
question: Which labelling problems in the reference set, the training labels and the held-out answer key
  change Vision's measured accuracy systematically, by how much, and where does each fix belong?
branch: exp/label-audit
commits:
- db5231a
benchmark: heldout-2026-10-08 (development split for re-scores; label-only counts on both splits)
split: dev
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
headline: 'Development split (2,918 species answers): fixing the labels we can fix today (non-iNat refs out, refs renamed to the record title, writing variants merged) moves species top-1 48.4 -> 48.9 (nearest) and 53.0 -> 53.6 (nearest + species average); adding identical-ITS label merges gives 49.5 and 54.2.'
verdict: 'No label problem moves the development score by more than about 1 point today; the largest training-side problem (6% of references are another organism''s photos) is invisible to nearest-neighbour scoring, and a title/number mismatch on the legacy database makes 2-3% of answers much harder (20% right against 49%).'
decision: pending
reproducibility: exploratory-pre-freeze
related:
- heldout-benchmark
- name-equivalence
- depth-bias
---

## Question
Which labelling problems move Vision's numbers systematically, rather than adding random noise?
Three places can carry them: the reference/training labels, the reference photos behind a label,
and the held-out answer key. For each problem: how many records it touches, which way it pushes
accuracy, and what a fix would gain on the development split.

## Why it matters
Vision learns and answers by label. A species split over two labels, a stale label or a label on
the wrong photos changes the training signal and the reference index. A wrong answer key changes
the score itself. The paper's numbers rest on these labels.

## Setup
- **Labels audited**: the 154,067 reference records and 18,070 species of the held-out run's
  reference index (manifest exported 2026-10-06, rebuilt with `evaluate.load_records`; it matches
  the run's counts exactly) and the 13,145 held-out records (development 3,000, test 10,145).
- **Live sources, read only**: the legacy database's record titles, index names and index
  MycoBank numbers, project rows and hard synonyms; mycomap.org's observations and sequences (name,
  approval, a hash of the bases); MycoMap Barcode's Russulaceae ITS clusters (fine and mid levels).
- **Re-scorer**: for each development photo, its 3,000 nearest reference photos (fine-tuned
  BioCLIP 2 embeddings, the same vectors as the held-out run). A reference label can then be
  changed, dropped or merged and both methods re-scored. On the unchanged labels it reproduces the
  official development run within 0.3 points: species top-1 48.4 vs 48.3 (nearest) and 53.0 vs
  52.8 (nearest + species average); genus 79.2 vs 79.0 and 81.8 vs 81.7.
- **Merging two labels in scoring** is exact for nearest (the best match of the union is the best
  of the two bests) and close for nearest + species average. Merging can only add hits, so every
  merge was checked against a **null**: random merges between same-genus labels with similar
  record counts, the same number of them.
- **Reproducibility (exploratory, before the dataset freeze)**: a backup copy of the manifest
  taken 2026-10-09 about 13:56 UTC (sha256 prefix d31a555d694df7cc); its reference index has the
  run's 154,067 records and 18,070 species (reference hash 5dbfdb1d24a5 at predict time; the live
  index has since drifted, so every variant is paired against a baseline scored by this re-scorer
  on the same copy). One command re-runs any variant: `tools/label_audit_rescore.py --manifest
  <copy> --cache <scratch> [--exclude] [--relabel] [--merge]` (tests in
  `tests/test_label_audit_rescore.py`).
- **Test split**: only label checks (no predictions) were run on it; no test prediction was used to
  choose anything.

## What we tried
Seventeen checks, all listed with their numbers below (none was dropped for a poor result):
1. Reference records that are not iNaturalist records.
2. Reference labels against the live record title.
3. The title against the record's own MycoBank number.
4. Writing variants that stay separate labels (padding, rank words, quoted epithets, gender endings).
5. Registered hard synonyms in labels and answer key.
6. Same epithet in a sister genus.
7. Cortinarius / Inocybe *s.l.* genus splits.
8. Russulaceae labels against DNA clusters.
9. Identical ITS under different labels, every family.
10. Null control for 9.
11. Records whose sequences never name the label's genus.
12. The same photo file on records with different labels.
13. Development records near-identical to a reference photo.
14. The answer key against the live title, one-word truths, the Tubariua typo and the bulk-bug names.
15. Development truths with no references.
16. Project and observer naming conventions.
17. All label fixes together.

## Results
Species and genus top-1 on the development split, 2,918 records with a species answer (2,966 with
a genus), unless a row says otherwise. "+a/−b" = records fixed / broken at species top-1.

| # | Issue | Reference records | Dev / test answers | nearest top-1 | nearest+mean top-1 |
|---|---|---|---|---|---|
| — | Baseline (re-scorer) | 154,067 | 2,918 | 48.4 | 53.0 |
| 1 | Not iNat records: MO, MyCoPortal and sequence ids fetched as iNat ids (wrong organism's photos) | 9,266 (6.0%; 1,608 species have only these) | key unaffected | 48.4 (+1/−1) | 53.1 (+4/−1) |
| 2 | Reference label differs in substance from today's record title | 5,076 (3.5%) | — | 48.9 (+30/−15) | 53.6 (+42/−24) |
| 3 | Title and the record's own MycoBank number name different taxa (the key uses the title, the references the number's name) | 3,453 (2.4%) | 87 / 290 | those 79 dev: 20.3 → 25.3 with 2 | 21.5 → 29.1 |
| 4 | Writing variants left as separate labels | ~500 in minority spellings | — | 48.5 (+1/0) | 53.0 |
| 5 | Hard synonyms never resolved | 1,014 (56 labels; 30 with the accepted name also in use) | 36 / 106 | 48.6 (+4/0) | 53.1 (+3/0) |
| 6 | Same epithet in a sister genus (mixed: real synonyms and different species) | 2,739 in 87 pairs | — | with 4 and 7: 48.5 | 53.0 |
| 8 | Russulaceae: several labels in one DNA cluster | 484 (fine) / 1,209 (mid) of 12,369 | Russulaceae dev 264 | Russ. 42.4 → 45.8 (mid) | Russ. 49.2 → 53.4 (mid) |
| 9 | Identical ITS under different same-genus labels (≥1 shared record) | 2,933 records outside the main label | — | 50.7 (+66/0) | 55.2 (+65/0) |
| 10 | Null for 9: random same-genus merges | 3 seeds, ~735 labels | — | 48.8 to 49.1 | — |
| 11 | No sequence names the label's genus (mostly genus moves) | 1,157 (0.8%) | — | 48.3 (+2/−5) | 52.9 (+3/−6) |
| 12 | One photo file on records with different labels (always one observer) | 1,287 in 862 groups | — | 48.4 (+4/−5) | 53.1 (+8/−5) |
| 13 | Dev record with a reference photo at cosine ≥ 0.99 | — | 8 dev (3 of 8 right) | — | — |
| 17 | All label fixes (1, 2, 4) | — | — | 48.9 (+30/−15) | 53.6 (+41/−24) |
| 17+ | All label fixes + identical-ITS merges (≥2 shared) | — | — | 49.5 (+44/−12) | 54.2 (+58/−21) |

Details:
- **1, wrong photos.** mycomap.org's export sends no source, and the records step calls any
  numeric id an iNat id. Mushroom Observer numbers, MyCoPortal occurrence ids and legacy sequence
  ids are numeric too, so Vision stored whatever iNat observation shares the number. For the first
  8,868 the iNat taxon's genus matched the label 0 times (mammals, birds, plants, insects); 399
  more land on an iNat fungus of another species. Share of reference records by the species' depth:
  1-4 references 12.4%, 5-19 5.2%, 20-99 4.5%, 100+ 3.5%. Nearest never picks them for a fungus
  photo, so the development score does not move. The fine-tune trained on them; 1,608 species
  are known only from them, and photos that are not fungi land on them.
- **2 and 3, stale or split names.** A fresh pull from mycomap.org fixes about 1,750 of the 5,076.
  For about 2,900, mycomap.org's name today still differs from the title, because the legacy index
  kept the old MycoBank number when the title changed (often a temporary code replaced by a formal
  name). mycomap.org builds the name from the number, so the old name comes back. Records with
  this mismatch are hard cases: 20% right against 49% for the rest. Moving the references to the
  title gains most in the sparse bands (nearest, 1-4 references: 12.7 → 14.0).
- **8 and 9, one species under two labels.** Typical pairs: a temporary code and the formal name it
  became (Russula sp. 'IN67' / R. variata; R. salishensis / R. sp. 'Woo40'; Lactarius sp. 'OH01' /
  L. glutigriseus), regional codes for one clade, and rank-word variants. Identical-ITS merges add
  2.3 points against 0.4 to 0.7 for random merges, so about 1.7 points come from labels Vision
  cannot (and perhaps need not) tell apart. This is an upper bound on what relabelling could give:
  a shared ITS does not prove one species (Russula emetica group, Lactarius deliciosus group).
  Host and parasite pairs share ITS too (Pseudoboletus parasiticus / Scleroderma citrinum,
  Syzygospora / Gymnopus, Hypomyces on Helvella): one partner's records carry the other's sequence.
- **5, hard synonyms.** Most are a quote-style quirk (the synonym points from house style to an
  older quoted spelling, which Vision already merges). The substantive ones are temporary codes
  with a registered formal name, both in use as labels: Gymnopilus sp. 'IN01' / G. subdryophilus,
  Pseudoclitocybe sp. 'cyathiformis-IN01' / P. sabulophila, Phaeolus sp. 'schweinitzii-CA01' /
  P. hispidoides, Pholiota sp. 'mixta-PNW01' / P. pseudopulchella.
- **14, answer key.** Against today's titles, 35 of 3,000 development answers and 94 of 10,145
  test answers changed since the freeze (27 and 75 were one-word names refreshed to a species).
  One-word truths: 72 / 242. The Tubariua typo: 7 / 32. No answer is a bulk-bug name
  (Hymenoscyphus sp. 'BC02', Melanoleuca sp. 'BC01').
- **15, no references.** Of the 99 development truths with no reference, about 11 are known under
  another name: Tubariua (7), Lactifluus / Lactarius volemus, Hermanssonia / Phlebia centrifuga,
  and Amanita muscaria flavivolvata, which the references hold under three rank spellings. The rest
  are new to Vision.
- **16, conventions.** No project leans clearly to temporary codes or to formal names within the
  identical-ITS pairs. One-word labels are 0.7% of references overall; one sequencing run's project
  has 36%.

Standard tables for the combined runs (development, n as stated):

**Baseline (this re-scorer, unchanged labels)**, nearest / nearest+mean:

```
                    top 1  top 3  top 5 top 10      n
  nearest
  species strict     48.4   67.3   73.7   80.8   2918
  species sl         48.4   67.3   73.7   80.8   2918
  species complex    51.2   69.3   75.6   82.1   2918
  genus strict       79.2   90.6   93.3   95.7   2966
  genus sl           79.9   90.7   93.4   95.7   2966
  by depth (top 1 / top 5): 0 0.0/0.0 (99); 1-4 12.7/35.2 (307); 5-19 34.4/65.5 (684); 20-99 59.7/85.9 (1344); 100+ 69.6/91.1 (484)
  nearest+mean
  species strict     53.0   70.7   77.1   83.0   2918
  species sl         53.0   70.7   77.1   83.0   2918
  species complex    55.8   72.7   78.7   84.2   2918
  genus strict       81.8   92.0   94.6   96.7   2966
  genus sl           82.5   92.2   94.6   96.7   2966
  by depth (top 1 / top 5): 0 0.0/0.0 (99); 1-4 16.3/41.0 (307); 5-19 41.5/73.7 (684); 20-99 65.7/87.6 (1344); 100+ 68.0/91.1 (484)
```

**CLEAN: non-iNat out + refs = live title + writing merges**, nearest (species top-1: fixed 30 broken 15)

```
                    top 1  top 3  top 5 top 10      n
  species strict     48.9   68.4   74.2   81.2   2918
  species sl         48.9   68.4   74.2   81.2   2918
  species complex    51.7   70.1   75.8   82.2   2918
  genus strict       79.3   90.6   93.3   95.5   2966
  genus sl           80.0   90.8   93.4   95.6   2966
  species strict by true-species reference records (top 1 / top 5, n):
    0        1.0 /   3.0  (n=99)
    1-4     14.0 /  37.5  (n=307)
    5-19    35.2 /  65.9  (n=684)
    20-99   60.0 /  85.9  (n=1344)
    100+    69.6 /  91.3  (n=484)
```

**CLEAN: non-iNat out + refs = live title + writing merges**, nearest+mean (species top-1: fixed 41 broken 24)

```
                    top 1  top 3  top 5 top 10      n
  species strict     53.6   71.5   77.4   83.4   2918
  species sl         53.6   71.5   77.4   83.4   2918
  species complex    56.3   73.1   78.9   84.4   2918
  genus strict       82.0   92.2   94.6   96.6   2966
  genus sl           82.7   92.3   94.6   96.6   2966
  species strict by true-species reference records (top 1 / top 5, n):
    0        2.0 /   3.0  (n=99)
    1-4     17.3 /  42.7  (n=307)
    5-19    43.0 /  73.4  (n=684)
    20-99   66.0 /  87.9  (n=1344)
    100+    67.6 /  91.3  (n=484)
```

**CLEAN + identical-ITS same-genus merges (>=2 records; scoring)**, nearest (species top-1: fixed 44 broken 12)

```
                    top 1  top 3  top 5 top 10      n
  species strict     49.5   68.9   74.8   81.5   2918
  species sl         49.5   68.9   74.8   81.5   2918
  species complex    52.1   70.6   76.3   82.5   2918
  genus strict       79.3   90.6   93.3   95.5   2966
  genus sl           80.0   90.8   93.4   95.6   2966
  species strict by true-species reference records (top 1 / top 5, n):
    0        1.0 /   3.0  (n=99)
    1-4     14.3 /  38.1  (n=307)
    5-19    35.2 /  66.5  (n=684)
    20-99   60.7 /  86.5  (n=1344)
    100+    70.9 /  91.9  (n=484)
```

**CLEAN + identical-ITS same-genus merges (>=2 records; scoring)**, nearest+mean (species top-1: fixed 58 broken 21)

```
                    top 1  top 3  top 5 top 10      n
  species strict     54.2   72.0   77.8   83.7   2918
  species sl         54.2   72.0   77.8   83.7   2918
  species complex    56.9   73.6   79.2   84.6   2918
  genus strict       82.0   92.2   94.6   96.6   2966
  genus sl           82.7   92.3   94.6   96.6   2966
  species strict by true-species reference records (top 1 / top 5, n):
    0        2.0 /   3.0  (n=99)
    1-4     17.6 /  43.0  (n=307)
    5-19    43.0 /  73.7  (n=684)
    20-99   66.7 /  88.2  (n=1344)
    100+    69.6 /  92.1  (n=484)
```

iNaturalist's computer vision: not run for these variants (it does not depend on our labels; its
development numbers are in heldout-benchmark).

## Verdict
The labels are mostly sound for scoring: every check together moves development species top-1 by
0.5 to 1.2 points, most of it in the sparse bands. Two problems are larger than their score effect
suggests. One in 17 reference records shows another organism (a source bug in Vision's own records
step); nearest ignores those photos, but the fine-tune trained on them, and 1,608 species exist
only through them. A rename on the legacy database leaves the record's MycoBank number behind, so
the answer key (the title) and the references (the number's name) disagree on 2-3% of records, and
those records are scored far worse. Same-species label pairs (identical ITS, DNA clusters, hard
synonyms) are the main remaining lever, worth up to about 1.7 points above chance on
development, but deciding them is a taxonomic call, not a cleanup. These are exploratory results
from before the dataset freeze; they are re-run on the frozen release.

## Decision
Pending (Steve). Proposed fixes, by gain and cost:
1. **Vision records step (free, no outside write):** select `source` in the export and keep only
   iNaturalist records; drop the 9,266 from the reference and the next fine-tune. A separate
   session is adding Mushroom Observer photos for those records.
2. **Legacy database (code + refresh, then a full mycomap.org re-pull):** keep the index MycoBank
   number with the title on a rename, then refresh the about 3,450 green records where they
   disagree. This removes the key/reference name mismatch (gain: item 2, about +0.5).
3. **Labels for one species (a person's decision, legacy synonyms):** review the identical-ITS and
   Russulaceae DNA-cluster pairs and the hard synonyms that are not yet merged, and register
   synonyms where they are one species. Upper bound about +1.7 points beyond chance. Until then,
   report species *complex* beside strict (it already credits epithet-stem matches).
4. **Scoring only:** none of these should change the answer key beyond refreshing records on the
   legacy database; the Tubariua answers are fixed there.
5. **Not worth it:** writing variants, duplicate photo files, near-duplicate leakage (each ≤ 0.1).

## Next
Re-pull the manifest after fixes 1 and 2 land; re-score development, then retrain. Extend the DNA
cluster check (item 8) beyond Russulaceae once MycoMap Barcode clusters other families. Lists are
in the audit folder under data/audits (never committed).
