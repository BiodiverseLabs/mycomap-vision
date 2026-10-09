---
title: Which frozen image model? Screening eight backbones
slug: backbone-screen
date: '2026-09-28'
status: adopted
question: Which off-the-shelf image model gives the best nearest-specimen identification of fungi before
  any fine-tuning?
branch: main (models.py registry, mv compare)
commits: []
benchmark: 8,000-record development sample (comparisons 20260928-235515-60274d, 20260929-010339-60274d,
  20260929-031844-85e107)
split: sample
model: frozen backbones
methods:
- nearest
- species-mean
headline: BioCLIP 2 25.0 / 61.8 / 71.4 species / genus / family top-1 on 68 records; best general model
  DINOv3-L at 512 px 16.2 / 42.6; iNat CV 29.4 / 67.7 / 70.6 on the same records.
verdict: Nothing general-purpose comes near BioCLIP 2, which was trained on the tree of life; photo size
  (1024 vs 500 px) makes no measurable difference.
decision: 'Steve 2026-09-28: BioCLIP 2 is the backbone to fine-tune; screen many frozen models cheaply,
  fine-tune few.'
related:
- full-run-finetune
- inat-cv-baseline
- trained-heads
---

## Question
Which frozen backbone should the identifier start from?

## Why it matters
Fine-tuning costs GPU hours; screening frozen models costs one embedding pass each. The
choice sets the ceiling for everything after it.

## Setup
8,000 random green records (30,113 medium photos) on the laptop. Test = records validated in the
newest 28 days of the sample, references = everything earlier. Same test records and photos for
every model (only photos every backbone embedded). iNaturalist's computer vision on the same
records as an outside reference.

## What we tried
BioCLIP 2; DINOv2-B and -L (518 px, class token); DINOv3-B and -L at timm defaults (256 px,
average pooling) and again at 512 px on the class token; SigLIP 2 L/384; EVA-02 L/448;
ConvNeXt V2 L. Methods nearest specimen and species average; best per model reported. Medium
(500 px) vs large (1024 px) photos for the four leading models. Centring vectors on the
reference mean.

## Results
Same 68 test records (comparison 20260928-235515-60274d), top-1 %, best method each. A
direction only: 68 records.

| Model | Species | Genus | Family |
|---|---|---|---|
| iNat CV, with location | 29.4 | 67.7 | 70.6 |
| iNat CV, photo only | 27.9 | 64.7 | 67.7 |
| BioCLIP 2 | 25.0 | 61.8 | 71.4 |
| EVA-02 L | 13.2 | 33.8 | 44.6 |
| DINOv2-L | 10.3 | 36.8 | 44.6 |
| DINOv2-B | 10.3 | 30.9 | 41.1 |
| ConvNeXt V2 L | 8.8 | 29.4 | 32.1 |
| SigLIP 2 L | 5.9 | 19.1 | 28.6 |
| DINOv3-B (256 px, avg pool) | 5.9 | 11.8 | 21.4 |
| DINOv3-L (256 px, avg pool) | 4.4 | 14.7 | 21.4 |

DINOv3 at 512 px on the class token (comparison 20260929-010339-60274d): L 16.2 / 42.6, B
14.7 / 38.2 species / genus: the best general model once run on equal terms, still far behind
BioCLIP 2. Medium vs large photos on the same 77 records (20260929-031844-85e107): every change
was one or two records, in both directions. Centring changed nothing.

## Verdict
BioCLIP 2 by a wide margin. Large photos stay the default (no cost on the GPU, detail for
later work).

## Decision
BioCLIP 2 adopted as the backbone (Steve, 2026-09-28).

## Next
DINOv3-L 512 on the full data only if an ensemble test on the sample earns it.
