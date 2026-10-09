---
title: Leave-one-out scan of the reference set for mislabelled records
slug: loo-mislabel-scan
date: 2026-10-09
status: running
question: Which reference records do their own photos dispute, once each record is identified
  against all the others, and does leaving the strongest cases out change accuracy?
branch: exp/loo-mislabel-scan
commits: []
benchmark: reference snapshot a19ddac8464a (scan); heldout-2026-10-08 dev (effect)
split: dev
model: bioclip-2-ft-20261007-165400
methods: [nearest+mean]
headline: Pre-registered; results to follow.
verdict: Pending.
decision: pending
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
To follow.

## Results
To follow.

## Verdict
Pending.

## Decision
Pending. Fixes go to the legacy database through the coordinator, with Steve's OK.

## Next
To follow.
