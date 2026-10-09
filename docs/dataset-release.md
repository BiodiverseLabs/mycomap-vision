# Dataset releases: a frozen, verifiable snapshot for research

Status: **specification, for Steve's review (2026-10-09)**. Code on `feat/dataset-release`.
Release v1 is **not** cut until the coordinator relays Steve's freeze trigger
(docs/PLAN.md, "a reproducible dataset release": freeze gate G1–G8).

Steve, 2026-10-09: every experiment so far is exploratory because the data drifted. Once
labelling and photo inclusion are final, we cut an immutable **dataset release** and re-run
every experiment on it, reproducible by anyone. The Picek/FungiTastic replication and Vision
train on the same release.

A *dataset release* (this document) is not a *serving release* (`release.py`, `mv release`:
what the site serves). A serving release says which dataset release its model came from.

## 1. Names and commands

- Frozen releases are `v1`, `v2`, … A correction never edits a release; it makes the next one.
- The public variant of `vN` is `vN-cc` (§9). It is built **from** `vN`, never from the
  manifest, so it is a strict subset by construction.
- Anything else (`dry-20261009-1830`, …) is a **draft**: same layout, marked
  `"draft": true` in its manifest, refused by every research command unless `--allow-draft`.

`mv release` is taken (serving releases), so the commands are `mv dataset …`:

| Command | Does |
|---|---|
| `mv dataset build --name v1 [--exclusions F.tsv …] [--cutoffs val=…,test=…] [--recipe long512-q90 …] [--dry-run]` | writes the snapshot from the manifest (read-only toward it) |
| `mv dataset verify v1 [--photos N\|all]` | recomputes every file hash, every content hash and `release_hash`; with `--photos`, re-reads originals from the store and checks their sha256 |
| `mv dataset derive v1 --photo <key> --recipe long512-q90 [--out F]` | regenerates one derived image from our original and checks it against the stored hash |
| `mv dataset diff v1 v2` | records/photos added, removed, relabelled, re-included, moved between splits; per-component hash changes |
| `mv dataset public v1` | writes `v1-cc` (§9) |
| `mv dataset show v1` | counts by source, split, inclusion reason, licence class |

`--dry-run` builds into a scratch folder with id `dry-<stamp>`, prints counts and the would-be
hashes, and labels every line "dry run, not v1".

## 2. On disk

Local: `data/research-releases/<id>/` (git-ignored, like all of `data/`). Kept copy:
S3 `research-releases/<id>/` beside the photo store; never under `releases/`, which the
serving box may read.

```
<id>/
  release.sqlite      the dataset: records, labels, inclusion, photos, splits, recipes,
                      derived-image hashes, models (all tables below)
  private.sqlite      what never leaves us: coordinates, owners' numeric ids,
                      permission answers, review-list notes
  MANIFEST.json       id, draft flag, created, code commit, inputs, counts, every content
                      hash, release_hash, every file's bytes + sha256. Written LAST:
                      a folder without it is unfinished and nothing reads it
  SHA256SUMS          `sha256sum -c` format, for anyone without our code
  DATASET_CARD.md     datasheet: what is in it, how it was built, gaps, biases, licences
  inputs/             the exclusion lists the build read, copied verbatim
```

**Immutable.** Every file is set read-only on disk after `MANIFEST.json` is written; the
reader opens SQLite with `mode=ro&immutable=1`, so any write raises; `build` refuses an
existing id; `verify` recomputes everything and fails on any difference.

## 3. Contents

### `records` — every candidate record, in or out

| Column | Meaning |
|---|---|
| `record_key` | manifest key: the iNat id for iNat, `mo:<n>` for MO (sources.record_key) |
| `source`, `source_id`, `org_source` | from `records` after migration `record-sources-v1` (feat/record-sources-mo); a build refuses a manifest without that migration |
| `org_name` | `.org` `observations.scientific_name` at the snapshot |
| `mycobank_number` | the number `.org` holds for that name (null until the export carries it) |
| `label` | the training label: `org_name` after the spelling merges (names.manifest_labels) |
| `label_rank`, `species`, `genus`, `family` | the unit scored (taxonomy.labels_for; one-word names count at genus/family only) |
| `name_kind` | `formal` / `provisional` (temp code) / `one-word` |
| `label_rule` | `"2026-10-09: iNat Species Name Override > Provisional; MycoBank number looked up from the name"` |
| `label_exported_at` | when the export read this record's name (records.exported_at) |
| `validated_on`, `green_projects` | earliest green date; projects that marked it green |
| `observed_on`, `country`, `state`, `north_america` | public-level place and date (no coordinates) |
| `included` | 1 / 0 |
| `reason` | `ok`, or the **first** exclusion reason in §4's order |
| `split` | §5; null when excluded |
| `benchmark`, `benchmark_split` | held-out membership (e.g. `heldout-2026-10-08` / `dev`) |

### `photos` — every photo of every candidate record

| Column | Meaning |
|---|---|
| `photo_key` | `inat:<photo id>` or `mo:<image id>` (the manifest's internal MO offset never leaks) |
| `record_key`, `position` | owner record, order on the record |
| `source`, `source_photo_id`, `source_url` | where it came from; the public URL of the photo |
| `original_sha256`, `original_bytes`, `original_ext` | **our kept original**: the S3 `large` copy (iNat 1024 px, MO 960 px) |
| `license_code`, `license_class` | at the freeze (`open` / `nc` / `arr`); `license_checked_at` |
| `owner_login`, `owner_name`, `attribution` | needed for CC attribution |
| `included`, `reason` | `ok` or the first photo reason in §4 |

`private.sqlite` holds the rest: record coordinates (needed by the location priors),
`owner_user_id`, the photo-permission answer per ARR owner at the freeze, and review-list
notes. It is hashed into `release_hash` (so v1 is one thing) but never published.

### Other tables

- `splits`: the cutoffs and how they were chosen; counts per split.
- `recipes`: name, the full JSON recipe, its sha256 (§7).
- `derived`: `photo_key`, `recipe`, `derived_sha256`, `pixels_sha256`, `width`, `height`
  for every included photo × every recipe built.
- `models`: filled after the fact by `mv dataset add-model` into a separate
  `models.json` beside the release (the release itself never changes): model name, file
  sha256, training command, code commit. The registry and serving releases cite these.
- `info`: key/value copy of `MANIFEST.json`'s inputs.

## 4. Inclusion: one reason per record and per photo

Every candidate gets exactly one reason (the first that applies), so the counts add up.

**Records**, in this order:

| Reason | Rule |
|---|---|
| `source_mycoportal` | MyCoPortal (Steve: rarely field images) |
| `source_com_sequence` | .com Sequences: no photos of their own |
| `source_genbank`, `source_unknown` | no photos of their own |
| `wrong_source_photos` | photos were fetched for the wrong record (`source_removals`) and none of its own remain |
| `not_north_america` | outside North America (Vision is North-American today; **Steve to confirm**) |
| `label_conflict` | .org holds more than one name |
| `no_label` | no usable rank ("Unknown", "Agaricales") |
| `guest` | DNA name is a guest of the fungus in the photo (guests.py) |
| `review:<list>` | listed by a person-checked review list (label audit, LOO scan, …): `inputs/<list>.tsv` |
| `no_photos` | no included photo left after the photo rules |
| `ok` | in |

**Photos**, in this order: `record_excluded`, `no_original` (no kept original with a
sha256), `permission_withdrawn` (ARR, owner withdrew on mycomap.org), `review:<list>`
(non-fungus scan, wrong-photo lists, after a person's check), `ok`.

ARR photos with no answer yet stay **in** for training (Steve, 2026-09-28), counted apart.

Review lists are TSV files: `kind` (`record`|`photo`), `key`, `reason`, `note`. The build
copies each into `inputs/` and records its sha256; a key it cannot find fails the build
(a stale list must not silently exclude nothing).

## 5. Splits

One split per included record, by the earliest green validation date (time-based, never
random; CLAUDE.md):

- `train`: validated on or before `val_cutoff`; records with no validation date go here.
- `val`: after `val_cutoff`, on or before `test_cutoff` (model selection, tuning).
- `test`: after `test_cutoff` (reported once per experiment).
- `sealed`: the paper's fresh ~1,000-record test set (G7), from a **sealed** heldout set.
  Never in `train`/`val`/`test`, never in a reference index; only the paper's final run
  reads it.

Default cutoffs: the build fixes calendar dates and writes them into the release; the
proposal is `test` = the newest 8 weeks of green validations and `val` = the 8 weeks
before (Steve to confirm; today's comparisons use 28 days).

The development benchmark `heldout-2026-10-08` (13,145, released 2026-10-08) keeps its
membership in `benchmark`/`benchmark_split`. Its role in v1 is G7 (open): either its records
take their split by date like everyone else (current state) or they form their own split.
The builder supports both (`--benchmark-role date|own-split`).

**Guards (tests):** `training_records()` returns only `train` (and `val` when asked);
anything `sealed` or excluded raises `SealedLeak`; `reference_records()` never returns
`test` or `sealed`.

## 6. Hashes (exact definitions)

All hashes are SHA-256, lower-case hex.

**Canonical rows.** A table's content is the sequence of its rows sorted by primary key,
each written as `json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
+ `"\n"`, UTF-8. Values are text, integers or null (no floats in hashed columns;
coordinates are stored as integer micro-degrees in `private.sqlite`). The stream starts with
the line `"<table>:<schema version>\n"`.

| Hash | Covers |
|---|---|
| `records_hash` | `records` (every candidate, labels, inclusion, splits) |
| `labels_hash` | `(record_key, label, label_rank, mycobank_number)` of included records — what the Picek guard checks |
| `photos_hash` | `photos` (every candidate photo: original sha256, licence, inclusion) |
| `splits_hash` | `splits` |
| `recipes_hash` | `recipes` |
| `derived_hash` | `derived` |
| `private_hash` | the tables of `private.sqlite` |
| `reference_hash` | `(record_key, label, sorted original sha256s of its included photos)` over `train`+`val`: the reference index's records + labels + photos (the registry's `reference_hash`) |
| **`release_hash`** | SHA-256 of the lines `"<name>=<hash>\n"` for the seven component hashes above, in that order |

`release_hash` depends only on content: the build time, machine and file layout do not
enter it, so building v1 twice from the same inputs gives the same hash, and any change to a
record, label, photo, licence, inclusion, split or recipe changes it. The registry cites
`dataset_release: v1`, `reference_hash` (first 12 shown), `code_commit`.

File hashes (`SHA256SUMS`, `MANIFEST.json.files`) cover the bytes on disk; SQLite files are
written with fixed page size and `VACUUM`ed, but the content hashes above are the contract.

## 7. Images: originals kept, derived by recipe

No zip archives (Steve). Our core store is the S3 `large` copy of each photo (~624k photos,
~274 GB), **kept permanently**: iNat photos get deleted and relicensed. A release records each
photo's `original_sha256`. A derived image is never stored in the release; it is made on
demand by a **recipe**, and its hash is stored so the result can be checked.

A recipe is immutable JSON; once a release uses it, its name is never reused for anything
else:

```json
{"name": "long512-q90", "version": 1,
 "steps": ["exif_transpose", "convert:RGB",
           "resize_long_side:512:lanczos:no_upscale", "jpeg:quality=90:subsampling=4:2:0:baseline:no_metadata"],
 "library": {"pillow": "12.3.0", "libjpeg": "<PIL.features.version('jpg')>"}}
```

Shipped recipes: `original` (identity: the derived hash is the original's), `long512-q90`
(public CC derivative, small), `long384-q95` (the size Picek's BEiT-B 384 reads, if Steve
wants a pre-made cache). More can be added to a later release.

Two hashes per derived image: `derived_sha256` (the JPEG bytes) and `pixels_sha256` (the
raw RGB pixels after resizing, with width and height). Same library versions → both match,
byte for byte. A different JPEG library may change the bytes but not the pixels; `derive`
says which matched, and names the library versions the recipe was made with.

Model-time augmentation (random crops, flips) is code, not a recipe: it is fixed by
`code_commit` and the training seed recorded with each model.

## 8. Reader API

```python
from mycomap_vision.dataset_release import load_release
rel = load_release("v1")               # verifies MANIFEST.json + file hashes on open (fast)
rel.id, rel.release_hash, rel.reference_hash, rel.code_commit
rel.records(split="train")              # included records of a split
rel.training_records(include_val=False) # raises on sealed / excluded
rel.reference_records()                 # train + val, never test or sealed
rel.photos(record_key)                  # included photos with original sha256 + licence
rel.photo_ids()                         # manifest photo ids, for embeddings lookups
rel.derive(photo_key, "long512-q90")    # bytes, hash-checked
rel.cite()                              # the dict experiments write into their outputs
```

Experiments take `--release v1` (`mv heldout`, `mv compare`, `mv finetune`, the Picek
launcher): records, labels, photos and splits come from the release; embeddings and photo
bytes from the store, checked against `original_sha256`. Outputs carry `rel.cite()`.

## 9. Public vs private

| | `v1` (ours) | `v1-cc` (public) |
|---|---|---|
| record ids, sources, labels, label provenance, splits, inclusion reasons | yes | yes (only records with ≥1 CC photo) |
| photo ids, licences, owners' logins and attribution, public URLs | all | CC only (`open` + `nc`) |
| original sha256 + derived hashes | all | CC only |
| ARR photos (files, derivatives, or their hashes) | used for training | **never** |
| coordinates | `private.sqlite` only | **never** (not even rounded) |
| code, recipes, dataset card | yes | yes |

Anyone can rebuild `v1-cc` from public URLs, or ask us for the on-demand CC derivatives,
verify each against `derived_sha256`, and re-run our commands. The paper reports both.

## 10. Open decisions for Steve

1. North America only, as Vision is today?
2. Split cutoffs: 8 + 8 weeks (proposed), or 28 days as in today's comparisons.
3. Held-out dev benchmark in v1: dated like everyone else, or its own split (G7).
4. `v1-cc` includes NC and ND photos (all CC, per "CC-licensed only"); for ND, only the
   unmodified photo and a plain resize (no crops), which CC treats as a format change.
5. Owners' logins in the public variant (CC-BY requires attribution) — yes, proposed.
6. Pre-made recipes: `long512-q90` (public) and `long384-q95` (Picek), or others.

## Not in this branch

Cutting v1; uploading to S3; `--release` wiring in each experiment command (the reader is
ready; each lane adds the flag to its command); `add-model` after the first v1 training run.
