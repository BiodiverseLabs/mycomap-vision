# Retrain on dataset release v1: runbook and pre-declared evaluation

Status: **pre-declared 2026-10-12, before any v1 result.** Owner: the Vision retrain lane.
Only the Vision Coordinator merges to main and deploys vision.mycomap.org. Steve approves
the freeze, the spending and the adoption.

Steve, 2026-10-11: everything before v1 is preliminary. v1 is the first real training dataset,
so this model's numbers on v1 are the **first official numbers**. They are **not compared with
the model served today** (`bioclip-2-ft-20261007-165400`, trained on a drifting manifest). Every
earlier experiment is re-run on v1 afterwards (section 6).

## 1. What is decided

- **Splits** (docs/dataset-release.md §5): `train` = first green validation on or before the
  cutoff (the newest 8 weeks are `val`, Steve's v1 default); `val` = after it. The 13,145
  held-out development records are ordinary records (PLAN G7). `v1-test` = Steve's new records,
  validated after the cut, frozen as their own release, never trained on.
- **Training**: the fine-tune learns from `train` only. `val` is never trained on: it is where
  the model is scored, calibrated (per-rank temperatures, likely sets) and sanity-checked, and
  where the gate and place prior are re-tuned. `val` records are still specimens in the served
  reference index (`reference_records()` = train + val).
- **Model**: BioCLIP 2, last 4 of 24 blocks, 2 epochs, the settings of
  `finetune.FinetuneConfig` at the launch commit; cosine heads at species, genus and family,
  **saved** (`<model>.heads.npz`). AWS g6.xlarge On-Demand, us-east-2.
- **Name**: the public name stays "MycoMap Vision"; `version` = the training-data snapshot
  date (`release.snapshot_date`: the release's `label_snapshot_at`), written into the model's
  metadata. `mv release` adds `.2`, `.3` for later builds of the same data (API lane).

## 2. The official numbers (pre-declared)

**Records.** v1's `val` split, every included record with at least one embedded photo; then
`v1-test` **once**, when Steve's records are cut. Nothing is tuned on `v1-test`.

**Comparison.** One command, on the laptop, after the run is pulled:

```
mv compare --release v1 --name-scores \
  --backbones bioclip-2,bioclip-2-ft-<run> \
  --methods nearest,species-mean,nearest+mean,nearest+mean+place,ft-head
mv standard-summary --comparison <id>
```

- `bioclip-2` (frozen) on the same release is the backbone-screen row of the re-run (section
  6, item 2), not a comparison with the served model.
- `ft-head` scores the fine-tune's own classifier (research only; never served unless adopted).
- `nearest+mean+place` needs `data/occurrence/inat-fungi-na.npz` (md5 `a599fb51…`). Its val
  numbers count each record's own iNat occurrence in the store's far-penalty counts: a small
  inflation, stated in the entry (the prior lane re-tunes it on v1 with its leave-one-out
  index, section 6, item 6).

**Format** (Steve, 2026-10-09; `mv standard-summary`), every table with its n:
1. Per model and method, **photos only** and **photos + place and date** in separate groups:
   top 1 / 3 / 5 / 10 × species strict, species *s.l.*, species complex (beta), genus strict,
   genus *s.l.*;
2. species top-1 and top-5 by the true species' reference records: 0, 1-4, 5-19, 20-99, 100+;
3. iNat CV on the same records, or "not run" (iNat runs on `v1-test`, section 6, item 8).

**The served default** stays what it is today (`nearest+mean+place` when a place or date is
given, else `nearest+mean`) unless a re-run in section 6 changes it.

## 3. The sanity gate (pre-declared; not a comparison)

`mv model-sanity --release v1 --backbone bioclip-2-ft-<run> --comparison <id>` (sanity.py)
must print PASS before the model is proposed for serving. Thresholds are fixed in sanity.py:

| Check | Pass when |
|---|---|
| files | weights, `.heads.npz` and metadata present |
| release | the model names v1's `release_hash`, was trained on `train`, val cutoff = v1's |
| version | `version` = v1's snapshot date |
| heads | finite, no all-zero class vector, a head for every species of `train` |
| embeddings | every included photo of v1 has a finite unit vector (≤ 0.2% missing: undecodable) |
| species top-1 | `nearest+mean`, photos only, on `val`: between 35% and 90% |
| genus / family top-1 | at least 65% / 70% |
| top-k rises | top 1 ≤ 3 ≤ 5 ≤ 10 |
| depth order | the 100+ band above the 1-4 band |
| classes | no species is the top answer for > 2% of `val`; ≥ 30% of `val`'s true species appear as a top answer |
| calibration | no rank's temperature at the edge of the searched grid (ranks with ≥ 50 records) |
| cpu | with no GPU (`CUDA_VISIBLE_DEVICES=-1`, 2 threads, `--cpu-photos 20`): ≤ 3.0 s per photo to embed, ≤ 5.0 s per search of the whole index, every photo answered |

Then, by hand, before the coordinator deploys:
- **Box memory** fits (section 5); the coordinator's call with Steve.
- **Not-a-fungus gate**: re-tuned on v1 (`mv fungus-gate tune`, gate lane), since it forks the
  served fine-tune after its last frozen block. After deploy, `/api/health` shows the new gate's
  `fungus_gate` fingerprint (not today's `73ce44b9afa55ff6`) and `verify` is all ok.
- **Occurrence store** present in the serving release (md5 `a599fb51…`).

A FAIL stops the line: report it to the coordinator with the check's detail; nothing is served.

## 4. Commands, in order

Run from a worktree of origin/main with `MV_DATA_DIR=<main checkout>/data`, `PYTHONPATH=src`
and the AWS settings from the main `.env`. Spending lines are marked $.

**B. Cut v1** (only after the coordinator relays Steve's freeze approval; the ~$1.50 below was
approved in advance on 2026-10-11):
1. Full re-export into the shared manifest (the ~15.5k new records incl. the 13,145; renames).
   Back up `data/manifest.sqlite` first.
2. $ Fetch the new records' photos to S3: `mv aws-launch-downloader` (t3.small, self-terminating;
   ~50k photos ≈ 21 GB, 3-4 GB from the all-rights-reserved host, within its 20 GB/day cap),
   then `mv aws-pull-manifest`.
3. $ The derived-hash pass and build in-region: `mv aws-launch-dataset-build --name v1
   --freeze-approved "<who, when>"` (c7i.2xlarge, self-terminating; ~650k originals ≈ 10
   core-hours, ~1.5-2 h, ~$1; S3 reads in-region cost nothing, against ~$25 of download fees
   to do it on the laptop). The instance verifies the release and uploads its kept copy to
   `research-releases/v1/`; `dataset-builds/<build>/result.json` (written last) has its
   release_hash. Then `mv dataset pull v1` brings it home, private part included, verified.
4. `mv dataset show v1` (the kept copy is already in S3; `mv dataset upload v1` only for a
   release built on the laptop).
5. Report to the coordinator: release_hash, counts by source and split, val cutoff.

**C. Retrain** ($ ~$8.50; ask Steve through the coordinator before launching):
1. `mv aws-launch-trainer --release v1 --backbones bioclip-2 --finetune bioclip-2
   --methods nearest,species-mean,nearest+mean,ft-head --instance-type g6.xlarge
   --max-hours 13`. The launcher checks the release and its S3 photos first and estimates
   (~11 h for ~610k photos: embed 1.8 h, fine-tune 6.6 h, embed 1.8 h, setup + compare 1 h).
2. Watch `runs/<run>/progress.json` and `train.log`. A stopped or Spot-interrupted run resumes
   with `--resume <run>`.
3. `mv aws-pull-trainer --run <run>`: the frozen and fine-tuned embeddings of every v1 photo
   (the reference re-embedded with the new model), the weights, the heads, the run's own
   comparison.
4. Section 2's comparison and summary; section 3's sanity gate.
5. Write the registry entry `docs/experiments/<date>-v1-finetune.md`:
   `reproducibility: reproduced-on-v1`, `dataset_release`, `reference_hash` and `code_commit`
   from the comparison's `dataset_release` block, `reproduce_command` = section 2's commands.
6. Report to the coordinator: the standard summary, the sanity result, box memory. Steve decides
   adoption; the coordinator builds the serving release (`mv release`) and deploys.

## 5. Box memory (measured 2026-10-12, projected for v1)

Measured read-only on the box (Lightsail 4 GB: 3.83 GB RAM + 4 GB swap; the service's cgroup
is capped at `MemoryMax` 3.5 GB), serving `bioclip-2-ft-20261007-165400` with
`nearest,nearest+mean,nearest+mean+place` and the not-a-fungus gate:

| | now | v1 (projected) |
|---|---|---|
| reference photos | 592,926 | ~660k (draft 609,527 + ~50k from the re-export) |
| vectors (float16 x 768 = 1.5 KB per photo, memory-mapped) | 874 MiB | ~970 MiB (+~100 MB) |
| model: base image tower (mmap) + trained tail | ~1.2 GB + 205 MB | same (same architecture; heads are not shipped) |
| anon (Python, torch, index arrays, record tables) | 1.11 GB | ~1.17 GB (records +8%) |
| service total now / peak | 2.99 / 3.10 GB, swap 0.74 GB (peak 1.29 GB) | ~3.15 / ~3.3 GB, more swap |
| working set (anon + vectors + weights: what a search touches) | ~3.4 GB | ~3.6 GB, above the 3.5 GB cap |

Reading: today's working set already sits at the cap, which is why the box swaps and
preloads take 67-114 s. v1 adds ~0.15-0.2 GB. It will most likely still run (vectors and
weights are file pages the kernel can drop and re-read), but searches slow down when vector
pages are evicted, and the nightly index swap peaks higher. Nightly growth is small (~6 MB of
vectors a week).

Options for Steve, cheapest first:
1. **8 GB box** (Lightsail plan change, roughly +$20 a month): the working set fits with room
   for the gate and API sessions. Recommended once v1 is adopted.
2. **int8 reference vectors** (code change + a check on v1's `val` that accuracy is unchanged):
   vectors ~970 → ~490 MB, working set ~3.1 GB, under the cap.
3. **Serve fewer methods** (drop `nearest` from `MV_METHODS`): saves little (<0.1 GB); a stopgap.
Decide after the sanity gate, with the model's measured CPU timings.

## 6. Re-running every earlier experiment on v1

Each re-run ships one command taking `--release v1` and a registry entry
`reproduced-on-v1`. A decision stands if its re-run agrees within its interval; otherwise it is
revisited and the entry says so. Order (PLAN §3: what defines the benchmarks first). Owners
are suggestions for the coordinator; times are the laptop's unless marked AWS.

| # | Re-run | Registry entries | Suggested owner | Estimate |
|---|---|---|---|---|
| 1 | Scoring definitions: name-equivalence rule check, the time splits, the likely-list protocol; cut `v1-test` when Steve's records are validated | name-equivalence, likely-sets | retrain lane | 1 h; v1-test cut ~1 h + photo fetch |
| 2 | Backbone screen (frozen backbones) on v1 | backbone-screen | benchmarking lane | bioclip-2 comes free with the retrain; others on a 20k-record stratified subset of `train`/`val` ≈ 3-4 h (all ~610k photos per backbone is 3-13 h each on the laptop) |
| 3 | Fine-tune (this run), its heads scored, the small trained heads, the Picek replication on v1's training records, the image-size arm (224/336/448) | full-run-finetune, trained-heads, picek-replication, image-size | retrain lane; Picek and image-size: benchmarking lane | retrain ~11 h AWS ~$8.50; trained heads 1 h; Picek ~44 h AWS ~$36; image-size ~2-4x the retrain, ~$20-35 (each needs Steve's OK) |
| 4 | Scoring method: nearest vs species average vs nearest + species average | depth-bias, nearest-mix | retrain lane (in section 2's comparison) | 0 extra |
| 5 | Calibration and likely lists, fitted on `val`, checked on `v1-test` | likely-sets | retrain lane | in section 2's comparison; v1-test check 0.5 h |
| 6 | Priors: DNA-record prior, occurrence prior, the served place boost re-tuned on `val`; the not-a-fungus gate re-tuned | dna-range-prior, occurrence-prior, prior-tuning, not-fungus-gate | prior lane; gate lane | prior grid with CV 3-4 h CPU; gate 1-2 h GPU |
| 7 | Photo handling: per-photo votes, photo views, uninformative photos, 8-bit model, test-time flips | nearest-mix, photo-views, uninformative-photos, int8-onnx | benchmarking lane | ~1 h each; flips ~3 h GPU |
| 8 | Outside comparisons on `v1-test`: iNat CV (identifications snapshotted first, model version recorded) and the published Danish models | inat-cv-baseline, published-danish-models | benchmarking lane | iNat ~2,000 records ≈ 7-17 h of paced calls (needs a fresh JWT from Steve); Danish models 1-2 h GPU |
| 9 | Descriptive: genus gap, depth over time, learning curve, leakage and stress slices, guest-genera counts, observation sets | genus-gap, guest-genera, learning-curve, leakage-tta-stress, observation-sets | plan-review / paper lane | 1-3 h each |

Not re-run, by design: the label audit, the LOO scan and the non-fungus photo scan are data
reviews whose fixes are inputs to v1 (person-checked lists), not experiments on it.
