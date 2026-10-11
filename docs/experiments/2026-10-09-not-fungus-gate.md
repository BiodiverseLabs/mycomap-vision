---
title: 'Not a fungus: a gate in front of the identifier'
slug: not-fungus-gate
date: '2026-10-09'
status: done
question: Can Vision tell that a photo is not a fungus at all, cheaply at serve time, and hold back the
  species answer and its calibrated chance when it isn't, without costing true fungi?
branch: feat/not-fungus-gate (pushed, not merged)
commits:
- 903b65b
- fc47cdc
- 92b3cff
- d9afe24
- de00410
benchmark: heldout-2026-10-08 dev (2,989 records, 12,206 photos) + a 2,066-photo not-a-fungus set;
  125,405 held-out genuine reference records as a cost check
split: dev
model: bioclip-2-ft-20261007-165400 (served), bioclip-2 frozen (gate)
methods:
- answer confidence / margin / entropy
- nearest-similarity floor
- BioCLIP 2 zero-shot (text prompts)
- linear probe on frozen vectors
- zero-shot + probe (log-odds added)
headline: 'Zero-shot + probe on frozen BioCLIP 2 vectors, AUROC 0.997 on 1,582 everyday non-fungi vs
  12,206 DEV photos; at the chosen thresholds no DEV record loses its answer, 11 of 2,989 (0.37%) lose
  the stated chance, and 98.8% of everyday non-fungi get no species chance (ducks, birds, pets 100%).'
verdict: The served answer's own confidence cannot flag a non-fungus (it is higher for non-fungi);
  frozen BioCLIP 2 vectors can, at almost no cost to true fungi.
decision: 'Steve 2026-10-09: approach approved (frozen BioCLIP 2 "is this a fungus?" check). Thresholds
  and deployment: pending his review of the branch. Reproducibility: exploratory-pre-freeze; confirm on
  dataset release v1.'
related:
- uninformative-photos
- photo-views
- heldout-benchmark
- picek-replication
- depth-bias
---

## Question
A duck photo got Amanitaceae 100%, Amanita 100% and an Amanita temporary-code species at 98% "calibrated
chance" on a local preview. Calibration holds only for fungus photos: the chance is a softmax over the
fungi Vision knows. Can a cheap gate say "this doesn't look like a fungus" (or "not sure") and hold back
the species answer and the chance, per photo, so that a duck among mushroom photos doesn't sink the find?

## Why it matters
The public Identify page and the coming public API state a calibrated chance. Any photo that is not a
fungus (a pet, a person, a plant, a screenshot) gets a confident fungus name today. The cost of a gate
must be tiny on true fungi, including slip, microscope and habitat photos and rare species.

## Setup
- **Fungus side (tuning):** the held-out DEV split of heldout-2026-10-08, 2,989 records and 12,206
  photos, never in the reference set. test.csv was never read.
- **Not-a-fungus set (tuning):** 2,066 photos kept in data/ with a manifest (source ids, licence,
  owner, url, sha256). 1,586 iNaturalist research-grade North American photos, CC0 or CC BY only,
  from the open-data bucket (one photo per observation; about 35 API requests at 1/s), and 480
  Open Images V5 validation-split images (CC BY 2.0, none labelled Mushroom). 1,582 are everyday
  (animals 782, plants 420, objects 140, people 80, scenes 80, food 80); 484 are hard negatives:
  slime moulds 120, lichens 100, mushroom-like plants 145 (Monotropa, Conopholis, Epifagus,
  Pterospora, Sarcodes, Orobanche, Corallorhiza), oak galls 60, Nostoc 25, corals and sponges 34.
- **Cost check (not tuned on):** 125,405 genuine iNat-linked reference records the probe did not
  train on. The frozen model never trained on MycoMap labels, so these photos are as unseen to it
  as DEV.
- **Vectors:** the served model is bioclip-2-ft-20261007-165400, whose last 4 of 24 blocks are
  fine-tuned. Its vectors no longer line up with BioCLIP 2's text tower (fungus photos get a median
  zero-shot p(fungus) of 0.12), so the gate reads the FROZEN BioCLIP 2 vector. Both vectors come from
  one pass: the 20 shared blocks run once, then the last 4 run twice, with fine-tuned and with base
  weights (identical to the stored vectors, cos 1.0000).
- **Probe training (disjoint from every evaluation set):** positives are the photos of 20,000 random
  genuine reference records (79,763 photos). Negatives are 12,308 photos of 8,884 reference records
  that are not iNat records at all (ids of other sources fetched as iNat ids, so their photos are
  random iNat observations), where iNat files them under Animalia or Plantae, plus 2,000 Open Images
  test-split images. Class-balanced logistic regression, L2 1e-3, numpy Newton steps.
- **Reproduce:** `mv fungus-gate embed` (vectors through the serving fork) and `mv fungus-gate tune`
  (probe, scores, rule, gate, report) at de00410 on a manifest snapshot of 2026-10-11 (592,926
  reference photos, before the mislinked records leave). The report records the code version, the
  probe's training-set hash and the not-a-fungus manifest's sha256.
- **Rule, declared in code before the choice:** on the grid 0.05, 0.10, ... 0.95 of p(fungus), "not a
  fungus" is the largest value at which no DEV record loses its answer, and "not sure" the largest
  at which at most 0.5% of DEV records lose the stated chance. Among the modes that meet it, keep the
  one that lets the fewest everyday non-fungi through.

## What we tried
Every signal, photo level, DEV photos vs the not-a-fungus set:

| signal | AUROC everyday | AUROC with hard negatives | everyday passed at 95% DEV pass |
|---|---|---|---|
| species confidence of the served answer | 0.44 | 0.42 | 79% |
| species margin (top 1 - top 2) | 0.45 | 0.43 | 88% |
| answer entropy (lower = surer) | 0.49 | - | - |
| genus confidence | 0.62 | 0.60 | 81% |
| best match to any reference photo (fine-tuned) | 0.71 | 0.72 | 55% |
| same, mislinked references left out | 0.945 | - | 14% |
| zero-shot on fine-tuned vectors | broken (fungi median p 0.12) | - | - |
| zero-shot on frozen vectors | 0.993 | 0.967 | 2.0% |
| linear probe on frozen vectors | **0.998** | **0.993** | 0.5% |
| zero-shot + probe (log-odds added) | 0.997 | 0.985 | 0.5% |

The served answer is MORE confident on non-fungi: 36% of them get a species chance of 0.5 or more,
10.5% get 0.9 or more. Picek's warning "every suggestion below 0.5" would miss them. The
reference-similarity floor was weak because the reference set holds ~13,000 photos of mislinked
records (ducks, birds, plants); without them it reaches 0.945, still well short of the frozen gate.

Each mode under the rule (DEV 2,989 records; 1,582 everyday non-fungi):

| mode | not a fungus below | not sure below | DEV records with no answer | without a chance | with a photo left out | everyday: not a fungus / not sure / passed |
|---|---|---|---|---|---|---|
| zero-shot | no grid value keeps every DEV record | - | - | - | - | - |
| probe | 0.35 | 0.45 | 0 | 5 (0.17%) | 27 | 87.0% / 8.0% / 5.1% |
| **zero-shot + probe** | **0.05** | **0.70** | **0** | **11 (0.37%)** | **36 (1.2%)** | **82.1% / 16.7% / 1.2%** |

Also tried along the way: lichens counted as fungus, as other, or left out of the prompts (kept as
fungus: iNat files them under Fungi, and MycoMap holds DNA-verified lichens); a probe without the Open
Images negatives (everyday AUROC 0.996, weaker on people and objects); grids of hand-picked pairs (0.1
to 0.5) before the rule was fixed in code. All are in the report. The rule's choice was not changed
after seeing them.

## Results
The chosen gate: zero-shot + probe on frozen BioCLIP 2 vectors, "not a fungus" below p = 0.05, "not
sure" below 0.70 (provisional).

- **Not-a-fungus set, by group** (not a fungus / not sure / passed): ducks 1.00 / 0 / 0, birds 0.99 /
  0.01 / 0, pets 1.00 / 0 / 0, people 0.82 / 0.15 / 0.03, objects 0.78 / 0.22 / 0, vehicles 0.93 / 0.07 / 0,
  scenery 0.81 / 0.19 / 0, plants 0.80 / 0.19 / 0.01, insects 0.78 / 0.20 / 0.02, food 0.34 / 0.55 /
  0.11. Hard negatives: slime moulds 0.63 /
  0.30 / 0.07, mushroom-like plants 0.14 / 0.72 / 0.14, oak galls 0.25 / 0.58 / 0.17, Nostoc 0 / 0.68
  / 0.32, lichens (fungi) 0 / 0.05 / 0.95.
- **Cost on true fungi, DEV** (2,989 records), by the true species' reference records:

  | refs | records | no answer | without a chance | a photo left out |
  |---|---|---|---|---|
  | 0 | 173 | 0 | 1 | 4 |
  | 1-4 | 307 | 0 | 3 | 6 |
  | 5-19 | 682 | 0 | 2 | 9 |
  | 20-99 | 1,344 | 0 | 4 | 11 |
  | 100+ | 483 | 0 | 1 | 6 |

  Photos left out: 49 of 12,206 (0.4%). Species top-1 on the 36 records that lose a photo: +6 / -1.
- **Slip, microscope and habitat photos** (the 100-record audit's hand labels: 28 slip, 19 habitat, 4
  microscope or spore print, 2 basket): none is left out, and no audit record loses its answer.
- **Cost check, 125,405 held-out genuine reference records:** 169 (0.13%) would get no answer, 713
  (0.57%) names without a chance. 104 of the 169 carry no fungal phylum (bees and wasps, flowering
  plants, mosses, and records named only "Unknown" or "Fungi"), and 18 are slime moulds. The 47 real
  fungi are cases where the photo shows the host or substrate more than the fungus: insect pathogens
  (Beauveria, Metarhizium, Ophiocordyceps, Entomophthora, Massospora), plant pathogens (Claviceps,
  Taphrina, Balansia, smuts), sooty moulds, yeasts and moulds, and tiny Mycena, Galerina and crusts.
  They ought to be "not sure", not "not a fungus". Per depth band of the true species' other
  references: 0 refs 1.42%, 1-4 0.25%, 5-19 0.03%, 20-99 0.02%, 100+ 0.05% get no answer (and
  2.43%, 1.33%, 0.51%, 0.37%, 0.23% names without a chance).
- **Serving cost**, measured on the Vision box (2 vCPU, read-only test process, 2026-10-11): 0 MiB
  of private memory added. The base tail's 195 MiB of weights are read memory-mapped from the base
  checkpoint and stay as clean, reclaimable file pages (~199 MiB after use). Time per photo +29% for
  one photo (1.9 s to 2.4 s) and +15% per photo for three, at 2 threads; laptop CPU +7%.

## Verdict
Answer confidence and similarity floors can't keep non-fungi from getting a confident species
chance. A gate on the frozen BioCLIP 2 vector can: almost every everyday non-fungus loses the species
chance, and no DEV record loses its answer. The probe adds most against plants, people and objects;
zero-shot adds lichens back and turns mushroom-like plants into "not sure". Small sets (hard-negative
groups of 15-120 photos) are direction, not results.

## Decision
Approach approved by Steve on 2026-10-09. The thresholds (0.05 / 0.70, combined mode) and deployment
are pending his review of the branch. Reproducibility: exploratory-pre-freeze, to be confirmed on
dataset release v1.

## Next
- Confirm on dataset release v1, after the mislinked references leave. Refit the probe there; its
  negatives come from the excluded records' list, which travels with the release.
- Insect and plant pathogens: show "not sure" rather than "not a fungus" (a fungus-on-host prompt
  class, or a host-aware not-sure band).
- Slime moulds: Steve to say whether they are in scope. 63% get "not a fungus" today.
- The Identify page redesign keeps the gate's notice first in each result card.
