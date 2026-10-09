---
title: Leave out records named for a guest organism
slug: guest-genera
date: '2026-10-07'
status: adopted
reproducibility: exploratory-pre-freeze
question: Should records whose DNA name is a yeast or parasite living in or on the photographed fungus
  be left out?
branch: feat/guest-genera-exclusion (merged 9dd37ec)
commits:
- f53d027
- 9dd37ec
benchmark: comparison 20261007-072214-7752c1 (306 test records)
split: temporal
model: bioclip-2; bioclip-2-ft-sample
methods:
- nearest
headline: '20 hidden-guest and 89 on-host records left out, 480 visible-guest records kept; 306-record
  comparison: fine-tuned 22.2 / 62.8 / 71.6 species / genus / family.'
verdict: The photos of such records show the host, so the label teaches the wrong thing; a rule by genus
  group is safer than one by class.
decision: 'Steve 2026-10-07: adopted (on-host default = left out).'
related:
- genus-gap
---

## Question
How should records be handled when the sequenced organism is not the fungus in the photo?

## Why it matters
A yeast sequenced from inside a puffball teaches the identifier that puffballs look like a
yeast.

## Setup
`guests.py` lists genera in three groups, applied wherever records are loaded and in the
advance-prediction report; `mv guests` reports coverage.

## What we tried
- Hidden guests, always out: *Teunomyces*, *Candida*, *Meyerozyma*, *Rhodotorula*,
  *Vishniacozyma*, *Saitozyma*, *Geotrichum*, *Dipodascus*.
- On-host parasites, out unless MV_KEEP_ON_HOST=1: *Spinellus*, *Syzygites*, *Mycogone*,
  *Sepedonium*, *Cladobotryum*.
- Visible, kept: *Trichoderma*, *Penicillium*, *Aspergillus*, *Botrytis*, *Pilobolus*,
  *Phycomyces*, *Modicella*, *Purpureocillium*, *Lecanicillium*, *Microbotryum*, *Taphrina*.

A rule by class was rejected (*Trichoderma* has 303 visible records; parasites are often the
photographed subject).

## Results
Coverage: 20 hidden + 89 on-host records out, 480 visible kept. Comparison 7752c1 (306
records): fine-tuned 22.2 / 62.8 / 71.6, frozen 21.9 / 60.9 / 71.0 species / genus / family. A
different record set from the 307-record comparison, so not a paired before/after: direction
only.

## Verdict
Adopt.

## Decision
Adopted (Steve, 2026-10-07).

Provisional: an exploratory result from before the dataset freeze; to be re-run on dataset release v1 (docs/PLAN.md, "a reproducible dataset release").

## Next
Scan for further guest genera as the reference set grows.
