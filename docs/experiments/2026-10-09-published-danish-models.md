---
title: The published Danish Fungi and FungiTastic models on our records
slug: published-danish-models
date: '2026-10-09'
status: done
question: How do the Danish team's published fungal models do on North American DNA-verified records,
  compared fairly with Vision?
branch: feat/external-bvra-baselines (e90ef4f, not merged)
commits:
- e90ef4f
benchmark: heldout-2026-10-08 development split (3,000 records)
split: dev
model: 'external: FungiTastic BEiT-B 384, FungiTastic ViT-B 384, DF20 ViT-L 384; Vision bioclip-2-ft-20261007-165400'
methods:
- their models, logits averaged over photos
- Vision nearest
- nearest+prior@org
- nearest+mean
headline: They can name 13-19% of formally named North American records; on species both can name, Vision
  leads by 5-12 points at species top-1 (8-15 with nearest + species average).
verdict: Most of the gap is vocabulary, but Vision also leads on equal terms; this compares their published
  models, not their method trained on our data.
decision: Recorded for the paper (development benchmark).
related:
- picek-replication
- name-equivalence
- heldout-benchmark
- inat-cv-baseline
---

## Question
How far do the leading published fungal models carry to North America?

## Why it matters
Reviewers will expect the Danish Fungi 2020 and FungiTastic models as baselines; the comparison
must be fair to a model that cannot know our provisional names.

## Setup
The three published models, unchanged. Names matched exactly, and through a GBIF / Catalogue of
Life crosswalk as a check. Genus and family on all records; species on formal names only; a
same-vocabulary comparison on records whose true species is on the model's own list, with
Vision unlimited and limited to that list; each model's coverage reported. Full write-up and
code: replications/fungitastic/README.md on feat/replications-fungitastic.

## What we tried
Exact names and crosswalk; observation level (logits averaged over photos) and per image.

## Results
Coverage (development, 3,000): 36.8% carry provisional codes. Formal names on FungiTastic's list
17.0% (19.2% crosswalk), DF20's 13.8% (16.1%); across 159,388 North American records 16.2% (18.3%)
and 13.1% (15.4%).

Top 1 / 3 / 5 / 10, exact names:

| Model | Genus (2,963) | Family (2,950) | Species, formal (1,817) |
|---|---|---|---|
| FungiTastic BEiT-B | 54.6 / 68.6 / 73.8 / 78.7 | 66.8 / 81.0 / 86.1 / 91.3 | 14.0 / 20.0 / 21.7 / 23.3 |
| FungiTastic ViT-B | 50.8 / 66.0 / 71.6 / 77.7 | 61.8 / 79.1 / 84.3 / 89.9 | 13.5 / 18.0 / 19.7 / 22.5 |
| DF20 ViT-L | 52.4 / 65.7 / 69.2 / 73.7 | 64.7 / 80.4 / 84.6 / 89.6 | 12.6 / 17.0 / 18.1 / 19.5 |
| Vision nearest | 79.0 / 90.5 / 93.3 / 95.7 | 86.4 / 94.7 / 96.4 / 98.1 | 54.4 / 72.9 / 79.5 / 85.6 |
| Vision + DNA-record prior | 79.5 / 91.1 / 93.7 / 95.8 | 86.7 / 94.8 / 96.8 / 98.2 | 57.1 / 74.2 / 80.0 / 85.6 |
| Vision nearest + species average | 81.7 / 92.1 / 94.7 / 96.6 | 88.7 / 95.5 / 97.1 / 98.5 | 58.0 / 75.8 / 81.8 / 87.4 |

Same vocabulary, species top-1 and macro-F1:

| | FungiTastic list (508 finds, 252 species) | DF20 list (413 finds, 197 species) |
|---|---|---|
| their model | BEiT-B 50.0, F1 49.7; ViT-B 48.2, F1 45.4 | 55.5, F1 54.9 |
| Vision nearest | 60.4, F1 57.9 | 60.5, F1 58.7 |
| + DNA-record prior | 63.2, F1 61.2 | 63.2, F1 62.0 |
| nearest + species average | 63.2, F1 62.5 | 63.7, F1 63.8 |
| blend limited to their list | 80.9, F1 78.4 | 82.8, F1 79.8 |

Limited rows are a lower bound: for about 45% of records Vision's ten stored answers held fewer
than three of their names. Per image, as their papers report, their species top-1 is 10.5 / 9.8 /
9.2%. Through the crosswalk their formal-name species top-1 rises to 14.7-15.8%.

## Verdict
Vision leads at every rank; most of the species gap is vocabulary.

## Decision
Recorded for the paper (draft TABLES 7-8).

## Next
Spot-check the crosswalk (a suspect join: *Agaricus solidipes* to *Panaeolus antillarum*); the
replication of their method on our data (picek-replication).
