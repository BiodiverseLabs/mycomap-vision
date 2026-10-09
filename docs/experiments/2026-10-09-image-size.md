---
title: Fine-tuning BioCLIP 2 at larger input sizes
slug: image-size
date: '2026-10-09'
status: planned
reproducibility: exploratory-pre-freeze
question: Does fine-tuning BioCLIP 2 at 336 or 448 px input, instead of the 224 px it was trained at, identify fungi better?
branch: none yet
commits: []
benchmark: dataset release v1 (planned arm of the v1 retrain)
split: v1 development; confirmed on the new experiment dataset
model: bioclip-2 fine-tuned at 224, 336 and 448 px
methods: [nearest+mean, nearest]
headline: Planned; never tried. Earlier 'large vs medium' tests changed the source photo size only; both were resized to 224 px before the model.
verdict: Not run.
decision: pending (planned by Steve 2026-10-09 as an arm of the v1 retrain; about 2-4x the fine-tuning compute).
related: [full-run-finetune, backbone-screen]
---

## Question
BioCLIP 2 has only ever seen our photos at 224 x 224 px. Would a larger input, where gill spacing,
pores, textures and small features survive, give better identifications?

## Why it matters
Fine detail is diagnostic in fungi. The source-photo test (backbone-screen) compared 1024 px and
500 px originals but resized both to 224 px, so it said nothing about the model's input size. For
DINOv3, moving from 256 to 512 px raised species top-1 from 4.4 to 16.2% on the sample.

## Setup
Planned as an arm of the full fine-tune on dataset release v1: the same recipe (last 4 of 24
blocks, cosine heads, two epochs), with the positional embeddings interpolated to the new grid, at
224 (control), 336 and 448 px. Every variant embeds the same v1 photos and is scored with the
served method and nearest alone, in the standard format.

## What we tried
Nothing yet.

## Results
None yet.

## Verdict
Not run.

## Decision
pending

## Next
Cost check on the GPU trainer: patch count grows with the square of the side (336 px = 2.25x,
448 px = 4x the tokens), so about 2-4x the fine-tuning and embedding time, and the serving box
would need the same input size at identification time (CPU cost to measure).
