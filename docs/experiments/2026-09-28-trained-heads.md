---
title: Trained classifier heads versus nearest specimen
slug: trained-heads
date: '2026-09-28'
status: dropped
reproducibility: exploratory-pre-freeze
question: Does a classifier trained on the image features beat looking up the nearest DNA-verified specimen?
branch: 'main (evaluate.METHODS: linear, hybrid)'
commits: []
benchmark: 8,000-record sample; full-run comparison 20261008-012435-4ef7b0 (1,152 test / 152,915 reference)
split: sample; temporal
model: bioclip-2 frozen and fine-tuned
methods:
- nearest
- linear
- hybrid
- balanced-softmax linear
headline: 'Full run: fine-tuned nearest 34.5% species vs hybrid head 26.0% and linear head 17.7% (1,152
  records); sample: nearest 24%, linear 13%, balanced softmax 3%.'
verdict: Small classifiers trained on finished features lose at every rank; most species have too few
  records for them.
decision: 'Nearest specimen stays the method (2026-09-28); heads are not offered on the site. A full end-to-end
  classifier is a different test: see picek-replication.'
related:
- full-run-finetune
- picek-replication
- depth-bias
---

## Question
Can a classifier head (linear, or linear blended with nearest specimen) on the image features
do better than nearest-specimen lookup?

## Why it matters
Classifiers are the standard approach in fungal image recognition (Danish Fungi 2020,
FungiTastic). If they won, the design would change.

## Setup
Heads trained on the reference records' embeddings only. Sample first (median 4 photos per
species), then the full run on the AWS GPU trainer.

## What we tried
Plain linear (softmax) head; balanced-softmax linear head; hybrid (head blended with nearest);
on frozen BioCLIP 2, the sample fine-tune and the full-run fine-tune.

## Results
Sample, species top-1: nearest 24%, plain linear 13%, balanced softmax 3%, hybrid 14-16%. On the
sample fine-tune (91 records): linear 9.9%, hybrid 17.6%.

Full run, comparison 4ef7b0 (1,152 test records), top-1 % species / genus / family:

| Model | Method | Species | Genus | Family |
|---|---|---|---|---|
| fine-tuned | nearest | 34.5 | 71.6 | 79.7 |
| fine-tuned | hybrid | 26.0 | 66.5 | 74.6 |
| fine-tuned | linear | 17.7 | 55.8 | 66.4 |
| frozen | hybrid | 15.9 | 51.2 | 62.8 |
| frozen | linear | 7.3 | 27.1 | 35.2 |

Balanced softmax was the wrong default: test records arrive at natural frequencies.

## Verdict
Small heads on fixed features lose. This does not test a classifier trained end to end with the
image model (the whole network retrained, as in the Danish Fungi papers): that is
picek-replication. The fine-tune's own classifier heads were not saved, so they have not been
scored either.

## Decision
Dropped as a method on the site; nearest specimen is the default.

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
Save the classifier heads at the next fine-tune and score them; the Picek replication.
