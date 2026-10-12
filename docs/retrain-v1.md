# Retrain on dataset release v1: runbook and pre-declared evaluation

Status: **pre-declared 2026-10-12, before any v1 result.** Owner: the retrain owner.
Only the project coordinator merges to main and deploys vision.mycomap.org. Steve approves
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
  metadata. `mv release` adds `.2`, `.3` for later builds of the same data (the API's model card).

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
  inflation, stated in the entry (the prior owner re-tunes it on v1 with its leave-one-out
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
- **Not-a-fungus gate**: re-tuned on v1 (`mv fungus-gate tune`, gate owner), since it forks the
  served fine-tune after its last frozen block. After deploy, `/api/health` shows the new gate's
  `fungus_gate` fingerprint (not today's `73ce44b9afa55ff6`) and `verify` is all ok.
- **Occurrence store** present in the serving release (md5 `a599fb51…`).

A FAIL stops the line: report it to the coordinator with the check's detail; nothing is served.

## 4. Commands, in order

Run from a worktree of origin/main with `MV_DATA_DIR=<main checkout>/data`, `PYTHONPATH=src`
and the AWS settings from the main `.env`. Spending lines are marked $.

**B. Cut v1** (only after the coordinator relays Steve's freeze approval; the ~$1.50 below was
approved in advance on 2026-10-11). Owner: the retrain owner.

**LOO gate** (Steve, 2026-10-11: "review first, then build"): v1 is not built until (a) Steve has
filled `decision` and `checked_by` on every row of
`data/audits/loo-mislabel-2026-10-09/candidates-for-person-review.csv` (187 records, 229 rows);
(b) `scripts/loo_exclusions.py` has turned it into the release exclusion TSV against the frozen
manifest copy of step 3 below, with exit 0 (any key that copy lacks stops it, and is reported
to Steve, never dropped); (c) the TSV's path, rows and sha256 are recorded and passed to the
build as `--loo-exclusions`. "No LOO exclusions" counts only on Steve's explicit word,
recorded with `--no-loo-exclusions "<Steve, date>"`. The build refuses a frozen release with
neither, and the release's MANIFEST.json records which (file, rows, sha256, or the word).

1. Full re-export into the shared manifest (`mv export-records`: at the 2026-10-12 audit,
   15,398 new records incl. all 13,145 held-out, 100 removed, 6,943 renamed). Back up
   `data/manifest.sqlite` first.
2. Read what the sources say, all read-only and at each source's pace:
   - `mv fetch-place-fallback`: records .org gives no place (892 at the audit) get their region
     from their own source (iNat `place_ids`, MO's location name); North American ones then
     count like any other (Steve, 2026-10-12). It also runs `fetch-mo-points` for every MO
     record: its PUBLIC point (shown GPS, else the centre of its location box). Steve,
     2026-10-12 (D1, docs/atlas-environment.md): v1's `private.sqlite` freezes every record's
     public point in `record_public_places` (iNat: the point the anonymous API showed, obscured
     where iNat obscures it; never .org's true point), so the later `v1-env` sidecar samples
     Atlas layers only there. A frozen build refuses an included record whose public point
     was never read; `public_kind = 'none'` is an explicit answer.
   - `mv fetch-inat` (new and newly North American iNat records) and `mv fetch-mo` (753 MO
     records at the audit), then `mv fetch-mo-points` again for MO records the export added.
   Then the photos into S3. Nothing already in S3 is fetched again, and nothing we hold is
   fetched twice:
   - the held-out records' photos (54,352 at the audit) already sit on the laptop in the
     benchmark's folder: `mv adopt-benchmark-photos --benchmark heldout-2026-10-08`, then
     `mv copy-photos --source <the folder it prints> --dest s3://<bucket>/ --size large`
     (every file re-hashed before upload; ~24 GB from the laptop's uplink);
   - $ the photos held nowhere (4,490 iNat at the audit, 799 of them all-rights-reserved):
     `mv aws-launch-downloader` (t3.small, self-terminating; minutes), then
     `mv aws-pull-manifest`;
   - the new MO records' images (4,655 from 1,088 observations at the audit): from MO's zip
     (list `data/audits/v1-cut-2026-10-12/mo-images-for-zip-v1.csv`, files named
     `<mo_image_id>.jpg`) with `mv import-mo-zip`, else downloaded at MO's pace (~6 h).
   A record counts only with a kept original in S3: skipping this would drop the held-out
   records from v1.
3. Freeze the manifest: `mv manifest-freeze --out data/manifest-v1-freeze.sqlite` (a consistent
   copy; prints its sha256). With its path and sha256, the retrain owner runs
   `scripts/loo_exclusions.py --manifest data/manifest-v1-freeze.sqlite` once Steve's review is
   complete, and returns the TSV's path, rows and sha256.
4. $ The derived-hash pass and build in-region: `mv aws-launch-dataset-build --name v1
   --freeze-approved "<who, when>" --manifest data/manifest-v1-freeze.sqlite
   --loo-exclusions <the TSV>` (c7i.2xlarge, self-terminating; ~650k originals ≈ 10
   core-hours, ~1.5-2 h, ~$1; S3 reads in-region cost nothing, against ~$25 of download fees
   to do it on the laptop). The launch ships that exact manifest file and LOO list (the
   instance refuses either if its sha256 differs). The instance verifies the release and
   uploads its kept copy to `research-releases/v1/`; `dataset-builds/<build>/result.json`
   (written last) has its release_hash. Then `mv dataset pull v1` brings it home, private
   part included, verified.
5. `mv dataset show v1` (the kept copy is already in S3; `mv dataset upload v1` only for a
   release built on the laptop).
6. Report to the coordinator: release_hash, counts by source and split, val cutoff, the LOO
   list's rows and sha256. After the cut, the record-sources experiment copies
   `data/sources-exp/{a,b,c}` may be deleted (their entry is merged).

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
5. Write the registry entry `experiments/<date>-v1-finetune.md` in the private research
   checkout (`config.RESEARCH_DIR`, never this repo):
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

**Decided (Steve, 2026-10-12): int8 reference vectors; the box stays at 4 GB** ("keep it
low for now"). The serving release is made with `mv release --backbones <model> --int8-vectors`:
each vector ships as int8 codes plus a float32 scale (772 bytes instead of 1,536), the server
maps and scores them (serving.MappedVectors), the nightly update's own shards stay float16,
and `/api/health` shows `vectors` per served index (`int8_rows`, `float_rows`). Measured on
the served model's 625,366 vectors: 922 MB → 467 MB, worst round-trip cosine 0.99983; an index
built on CPU from them gives the same species top-1 as the float16 one on 40 of 40 sampled
photos; on the draft release's val split (5,156 records) `nearest+mean` species top-1 moved
44.89 → 44.83% (3 vs 0 discordant, p 0.25): "no loss" by the pre-declared rule. Before v1 is
served, the check runs again on v1's val with the new model, and must pass:
`mv int8-check --release v1 --backbone bioclip-2-ft-<run> --methods nearest,nearest+mean`
("no loss" = `nearest+mean` species top-1 moves by less than 0.3 points and the paired sign
test is not significant, p ≥ 0.05). Projected v1 working set ~3.1 GB, under the 3.5 GB cap.
Fallbacks if it does not fit: serve fewer methods (<0.1 GB), then the 8 GB box (~+$20/month).

**ONNX image tower: not for the 4 GB box** (measured on the box 2026-10-12, 2 threads: torch
tower 539 MB anon + 1,007 MB file-mapped weights, 1.76 s/photo; float32 ONNX 1,989 MB anon,
1.98 s/photo: +1.45 GB anon and no speed gain). `MV_EMBED_ONNX` stays off and v1 skips
`mv export-onnx`. If an export is made later, `mv release` ships `models/<b>.onnx` +
`.onnx.json` and refuses one exported from other weights than the `.pt` it ships.

**Fast int8 search (`MV_SEARCH_INT8`) on the box only after a measurement.** It searches a
packed int8 copy (~0.48 GB anon) and re-reads only each query's top rows from the mapped
shards, so the hot set moves from reclaimable file pages to anon memory that can only be
swapped. Before it is switched on for v1: a dark run on the box with it on across one
nightly update, reading the service cgroup's `memory.current`, `memory.stat` (anon vs file)
and `memory.events` (high, oom) before, during the index swap and after, first on the current
release, then on v1's int8 release before it serves. If v1's peak nears the 3.5 GB cap, it
stays off for v1 (one setting), or `MV_SEARCH_REFINE` is lowered. **Measured 2026-10-12 on the box:** on since
04:31 UTC with today's release, through a full nightly index swap: anon peak 1.81 GB, memory
events high/max/oom 0, search 1.61 → 0.061 s (1 photo), identical answers. The packed copy
costs ~800 bytes a reference row; v1's ~660k rows (+~12%) stay under the 25% re-measure line,
so it stays on for v1, and the same watch reruns on v1's first nightly as a check.

## 6. Re-running every earlier experiment on v1

Each re-run ships one command taking `--release v1` (code in this repo) and a registry entry
`reproduced-on-v1` in the private research checkout (`config.RESEARCH_DIR`, never this repo). A decision stands if its re-run agrees within its interval; otherwise it is
revisited and the entry says so. Order (PLAN §3: what defines the benchmarks first). Owners
are suggestions for the coordinator; times are the laptop's unless marked AWS.

| # | Re-run | Registry entries | Suggested owner | Estimate |
|---|---|---|---|---|
| 1 | Scoring definitions: name-equivalence rule check, the time splits, the likely-list protocol; cut `v1-test` when Steve's records are validated | name-equivalence, likely-sets | retrain owner | 1 h; v1-test cut ~1 h + photo fetch |
| 2 | Backbone screen (frozen backbones) on v1 | backbone-screen | benchmarking owner | bioclip-2 comes free with the retrain; others on a 20k-record stratified subset of `train`/`val` ≈ 3-4 h (all ~610k photos per backbone is 3-13 h each on the laptop) |
| 3 | Fine-tune (this run), its heads scored, the small trained heads, the Picek replication on v1's training records, the image-size arm (224/336/448) | full-run-finetune, trained-heads, picek-replication, image-size | retrain owner; Picek and image-size: benchmarking owner | retrain ~11 h AWS ~$8.50; trained heads 1 h; Picek ~44 h AWS ~$36; image-size ~2-4x the retrain, ~$20-35 (each needs Steve's OK) |
| 4 | Scoring method: nearest vs species average vs nearest + species average | depth-bias, nearest-mix | retrain owner (in section 2's comparison) | 0 extra |
| 5 | Calibration and likely lists, fitted on `val`, checked on `v1-test` | likely-sets | retrain owner | in section 2's comparison; v1-test check 0.5 h |
| 6 | Priors: DNA-record prior, occurrence prior, the served place boost re-tuned on `val`; the not-a-fungus gate re-tuned | dna-range-prior, occurrence-prior, prior-tuning, not-fungus-gate | prior owner; gate owner | prior grid with CV 3-4 h CPU; gate 1-2 h GPU |
| 7 | Photo handling: per-photo votes, photo views, uninformative photos, 8-bit model, test-time flips | nearest-mix, photo-views, uninformative-photos, int8-onnx | benchmarking owner | ~1 h each; flips ~3 h GPU |
| 8 | Outside comparisons on `v1-test`: iNat CV (identifications snapshotted first, model version recorded) and the published Danish models | inat-cv-baseline, published-danish-models | benchmarking owner | iNat ~2,000 records ≈ 7-17 h of paced calls (needs a fresh JWT from Steve); Danish models 1-2 h GPU |
| 9 | Descriptive: genus gap, depth over time, learning curve, leakage and stress slices, guest-genera counts, observation sets | genus-gap, guest-genera, learning-curve, leakage-tta-stress, observation-sets | paper owner | 1-3 h each |

Not re-run, by design: the label audit, the LOO scan and the non-fungus photo scan are data
reviews whose fixes are inputs to v1 (person-checked lists), not experiments on it.
