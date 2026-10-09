---
title: 'Their method, our data: replicating the Danish Fungi training recipe'
slug: picek-replication
date: '2026-10-09'
status: planned
question: Trained on exactly Vision's records, does the Danish Fungi / FungiTastic classifier recipe do
  better or worse than nearest-specimen retrieval?
branch: feat/picek-replication (586b16d, not merged); feat/replications-fungitastic
commits:
- 1f933af
- 586b16d
benchmark: heldout-2026-10-08 development split; the newest-weeks comparison
split: dev
model: BEiT-B/16 384 (timm), full fine-tune
methods:
- classifier
- classifier+month
- classifier+month-raw
- classifier+month+place
headline: Built, tested and smoke-run; the GPU run waits for the relabelled training data and Steve's
  compute go-ahead.
verdict: Not run.
decision: pending (launch after the relabel; compute estimate to Steve first).
related:
- published-danish-models
- trained-heads
- full-run-finetune
---

## Question
Separates method from data: if their recipe on our records matches Vision, the gap to the
published models is data; if it beats Vision, the method matters too.

## Why it matters
Our small classifier heads lost (trained-heads), but that never tested a classifier trained end to
end, which is how the Danish Fungi papers work.

## Setup
`picek.py`, preset fungitastic-beit-b384, from the published configuration and training code:
timm BEiT-B/16 at 384 x 384, the whole network fine-tuned with a new linear head over every
species; SGD (momentum 0.9) at 0.01 with a plateau schedule; effective batch 256; 50 epochs, keeping
the best validation macro-F1; Seesaw loss for the long tail; RandAugment(2, 20); photo logits
temperature-scaled and averaged per record; DF20's month prior (paper formula, smoothed toward
genus). Our place prior is a separate, labelled extension. Training on exactly the comparison's
reference records; the last 28 days before the cutoff as validation. Other presets: their ViT-B,
DF20's ViT-L, the 224 px BEiT. Full recipe and deviations: docs/PLAN.md (Picek replication) and
replications/fungitastic/README.md on feat/replications-fungitastic.

## What we tried
Unit and smoke tests; a dry-run launch.

## Results
None yet.

## Verdict
Not run.

## Decision
pending

## Next
Relabel from refreshed observation names; hours and cost on the g6.xlarge (4 vCPU loading limit)
to Steve; then the run and the comparison on the development split in the standard format.
