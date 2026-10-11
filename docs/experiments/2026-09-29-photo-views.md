---
title: 'Photo views: crops, slip and habitat tags'
slug: photo-views
date: '2026-09-29'
status: dropped
reproducibility: exploratory-pre-freeze
question: Do voucher slips, microscope and habitat photos hurt identification, and do crops or view tags
  help?
branch: feat/photo-views (deleted 2026-09-29; only record is the memory note and this entry)
commits: []
benchmark: 8,000-record sample, 180-day split (comparison 20260929-174054-views-71ae2c, 1,398 test)
split: sample
model: bioclip-2 frozen
methods:
- nearest
- view-filtered nearest
- crop / mask / crop+mask embeddings
headline: Every effect about 1 point; best combination (crop + greyed slips + no microscope/habitat) family
  +25 records net (P = 0.02).
verdict: Microscope and habitat photos carry about nothing; slip photos still carry the specimen; greying
  slips hurts; crops are neutral for frozen BioCLIP 2.
decision: 'Steve 2026-09-29: skip the whole concept; effects too small. Branch, tables and embeddings
  deleted.'
related:
- uninformative-photos
---

## Question
Should photos be tagged by view (slip, microscope, habitat), cropped to the fungus, or filtered?

## Why it matters
Community records mix specimen photos with slips, microscopy and habitat shots; if those
confuse the identifier, filtering or an annotation step would be worth building.

## Setup
SigLIP 2 zero-shot view tags from stored embeddings (93% agreement with 208 hand labels); OWLv2
boxes for fungus and slips on all 25,520 sample photos; BioCLIP 2 embeddings of crops, masks
and both. Paired sign tests; an observer-disjoint rerun and a same-size random control.

## What we tried
Dropping microscope and habitat photos; dropping slip photos; greying slips with rectangles;
crop, mask and crop+mask embeddings; combinations.

## Results
Microscope and habitat photos are worth about zero alone; dropping them is a small safe gain.
Slip photos still carry signal (the specimen is usually on them). Greying slips with rectangles
hurts (the grey blocks become their own signature). Crops are neutral for frozen BioCLIP 2.
Removing test observers costs the same as removing as many random records (the observer shortcut
is small). Best combination: family +25 records net (P = 0.02). All effects about one point.

## Verdict
Not worth building.

## Decision
Dropped (Steve, 2026-09-29). Re-tested on the full-run model as uninformative-photos (also
dropped).

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
None.
