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
- The test set of `vN` is `vN-test` (§5): the fresh records Steve validates for testing,
  cut later as its own release, disjoint from `vN`.
- Anything else (`dry-20261009-1830`, …) is a **draft**: same layout, marked
  `"draft": true` in its manifest, refused by every research command unless `--allow-draft`.

`mv release` is taken (serving releases), so the commands are `mv dataset …`:

| Command | Does |
|---|---|
| `mv dataset build --name v1 --freeze-approved "<who, when>" [--exclusions F.tsv …] [--val-weeks 8 \| --val-cutoff D] [--recipes original,long500-q90]` | writes the snapshot from a private copy of the manifest (never writes to it); without `--name` (or with `--dry-run`): a draft |
| `mv dataset build --name v1-test --test-for v1 --ids new.csv --freeze-approved …` | the test release: only those records, all `test`; refused if any record, or any photo (by bytes), is in `v1` |
| `mv dataset verify v1 [--photos N\|all]` | recomputes every file hash, every content hash and `release_hash`; with `--photos`, re-reads originals from the store and checks their sha256 |
| `mv dataset derive v1 --photo <key> --recipe long500-q90 [--out F]` | regenerates one derived image from our original and checks it against the stored hash |
| `mv dataset diff v1 v2` | records/photos added, removed, relabelled, re-included, moved between splits; per-component hash changes |
| `mv dataset public v1` | writes `v1-cc` (§9) |
| `mv dataset show v1` | counts by source, split, inclusion reason, licence class |

A frozen id (`v1`, `v1-test`, …) is refused without `--freeze-approved` (who gave the trigger,
and when; recorded in the release) and without every derived hash (`--derive all`). A draft
(`--dry-run`) may skip or sample derived hashes (`--derive none|N`); its output starts
"DRY RUN, not v1". Experiments take `--release <id>` (`mv compare`, `mv heldout predict`
today) and refuse a draft unless `--allow-draft`.

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
| `observer_login`, `inat_uuid` | per-observer reporting; the uuid lets occurrence priors leave the record out |
| `included` | 1 / 0 |
| `reason` | `ok`, or the **first** exclusion reason in §4's order |
| `split` | §5; null when excluded |
| `benchmark`, `benchmark_split` | held-out membership (e.g. `heldout-2026-10-08` / `dev`) |

### `photos` — every photo of every candidate record

| Column | Meaning |
|---|---|
| `photo_key` | `inat:<photo id>` or `mo:<image id>` |
| `manifest_photo_id` | the manifest's id, to find the photo's vectors (zeroed in the public variant) |
| `record_key`, `position` | owner record, order on the record |
| `source`, `source_photo_id`, `source_url` | where it came from; the public URL of the photo |
| `original_path`, `original_sha256`, `original_bytes` | **our kept original**: the `large` copy (iNat 1024 px, MO 960 px), S3 first; the store itself is named only in `private.sqlite` |
| `license_code`, `license_class` | at the freeze (`open` / `nc` / `arr`); `license_checked_at` |
| `owner_login`, `owner_name`, `attribution` | needed for CC attribution |
| `public_trainable` | 1 when the photo is CC (NC included) or its photographer granted permission: what the final public model may learn from (Steve, 2026-10-09); internal benchmarking models may use every included photo |
| `included`, `reason` | `ok` or the first photo reason in §4 |

`private.sqlite` holds the rest: record coordinates (needed by the location priors),
owners' numeric ids, the photo-permission answer per ARR photo at the freeze
(`granted` / `no_answer` / `withdrawn`), and the photo store each original is in. It is
hashed into `release_hash` (so v1 is one thing) but never published.

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
| `held_out` | listed in `benchmark_holdouts` (a sealed benchmark) |
| `not_north_america` | outside North America (Steve, 2026-10-09: North America only, as today; fetch-inat and fetch-mo read North America only, so these were never fetched) |
| `not_fetched` | no iNat / MO details read for it yet |
| `missing_at_source` | the iNat or MO observation is gone, private or not returned (status `missing`) |
| `label_conflict` | .org holds more than one name |
| `no_label` | no usable rank ("Unknown", "Agaricales") |
| `guest` | DNA name is a guest of the fungus in the photo (guests.py) |
| `review:<list>` | listed by a person-checked review list (label audit, LOO scan, …): `inputs/<list>.tsv` |
| `no_photos` | no included photo left after the photo rules |
| `ok` | in |

**Photos**, in this order: `record_excluded`, `no_original` (no kept original with a
sha256), `permission_withdrawn` (ARR, owner withdrew on mycomap.org; the rule of
`evaluate.load_records`, read from permissions.py), `review:<list>` (non-fungus scan,
wrong-photo lists, after a person's check), `ok`.

Wrong-photo records need no reason of their own: a build refuses a manifest without the
`record-sources-v1` migration, which deleted each wrongly keyed record (an MO number taken for
an iNat id) and brought it back under its own key (`mo:123`, `mycoportal:…`) without the
unrelated iNat photos; the trail stays in the manifest's `source_removals`. The public variant adds
`no_cc_photos` for a record whose only photos are all-rights-reserved.

ARR photos with no answer yet stay **in** for training (Steve, 2026-09-28), counted apart.

Review lists are TSV files: `kind` (`record`|`photo`), `key`, `reason`, `note`. A record key may be written `inat:<id>` (the manifest key is the bare id) or `mo:<n>`; photo keys are `inat:<photo id>` / `mo:<image id>`. The build
copies each into `inputs/` and records its sha256; a key it cannot find fails the build
(a stale list must not silently exclude nothing).

## 5. Splits

Steve, 2026-10-09: "I will provide a new dataset for testing of new records." So:

- `vN` holds `train` and `val`, by the earliest green validation date (time-based, never
  random; CLAUDE.md). `val` = validated after `val_cutoff` (model selection and tuning);
  `train` = on or before it, plus records with no validation date. The cutoff is a calendar
  date written into the release: by default the newest 8 weeks of validations are `val`
  (`--val-weeks`), or `--val-cutoff` is chosen at the freeze.
- `vN-test` holds `test`: Steve's fresh records, validated after `vN` was cut, frozen the same
  way (labels, photos, hashes) as their own release. The build refuses it when any of its
  records is in `vN`, or any of its photos is byte-identical to one in `vN`. It is never
  trainable (`training_records()` raises `SealedLeak`).
- The development benchmark `heldout-2026-10-08` (13,145) is dated like every other record
  (Steve, 2026-10-09); its membership stays in `benchmark` / `benchmark_split`, so it can
  still be reported.
- A record held out for a sealed benchmark (`benchmark_holdouts`) is excluded (`held_out`).

**Guards (tests):** `training_records()` returns `train` (and `val` when asked) only;
`reference_records()` is `train` + `val`; excluded records have no split; a test release is
disjoint from its base and never trainable.

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
record, label, photo, licence, inclusion, split or recipe changes it. MANIFEST.json carries
`dataset_release`, `reference_hash`, `release_hash` and `code_commit` (the commit that cut
the release) at its top level. A registry entry copies `dataset_release` and
`reference_hash` from `rel.cite()`, with its own `code_commit` (the experiment's code) and
`reproduce_command`.

**How a comparison's `record_set` relates.** `evaluate.record_set_hash` (sha1, 12 hex) names
the records one comparison scored: its reference and test record ids, units and the photos
every compared backbone embedded. It is not the release's `reference_hash`: on a release, a
comparison's reference is `train` and its test is `val`, and photos without a vector drop out,
so `record_set` is a function of (`reference_hash`'s records and photos, the backbones'
vectors). Both are written: `dataset_release` + `reference_hash` say which data, `record_set`
says which subset of it a comparison could score. `record_set_hash` is left unchanged, so
`mv compare --into` keeps working on saved comparisons.

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
{"name": "long500-q90", "version": 1, "long_side": 500, "quality": 90,
 "steps": ["exif_transpose", "convert:RGB", "resize_long_side:500:lanczos:no_upscale",
           "jpeg:quality=90:subsampling=4:2:0:baseline:no_metadata"],
 "library": {"pillow": "12.3.0", "libjpeg": "<PIL.features.version('jpg')>"}}
```

Shipped recipes: `original` (identity: the derived hash is the original's) and
`long500-q90` (the usual public size: iNat "medium", FungiTastic's 500p). Picek's BEiT-B 384
reads originals and resizes in code (a pre-made cache was measured useless at 384), so it
needs no recipe. Later releases may add recipes under new names.

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
rel.training_records(include_val=False) # train only; a test release raises SealedLeak
rel.reference_records()                 # train + val, never test
rel.photos(record_key)                  # included photos with original sha256 + licence
rel.photo_ids(public_trainable_only=False)   # manifest photo ids, for embeddings lookups
rel.as_records(("train", "val"), photo_row, public_trainable_only=False)  # evaluate.Records
load_release("v1", without_private=True)     # a trainer copy shipped without private.sqlite
rel.check_photos_match(conn)            # refuses vectors of a photo whose original changed
rel.derive(photo_key, "long500-q90")    # bytes, hash-checked
rel.cite()                              # the dict experiments write into their outputs
```

Wired today: `evaluate.load_records(..., release=)` (so `Identifier`, the reference index and
everything built on them), `evaluate.shared_records` / `mv compare --release` (reference =
`train`, test = `val`), and `mv heldout predict --release`. Records, labels, photos and
splits come from the release; vectors are found by photo id in the manifest, and refused
when that photo's kept original no longer has the release's sha256. Outputs carry
`rel.cite()`. Still to add, by their owners: `mv finetune` and the Picek launcher (whose
guard checks `labels_hash`).

**The final public model** (Steve, 2026-10-09): trains with `public_trainable_only=True`
(CC photos, NC included, plus photos whose photographer granted permission). Internal
benchmarking models may use every included photo.

**A trainer copy** (an AWS instance) may ship `release.sqlite`, `MANIFEST.json`,
`SHA256SUMS` and the card **without `private.sqlite`**: `load_release(id,
without_private=True)` checks every other file; coordinates come back as None (a trainer
needs none); anything that needs the private part (location priors, `derive` without a
named store) says so. `verify(id, without_private=True)` recomputes every content hash but
the private one, which it takes from MANIFEST.json.

**On-demand images on the site** (`dataset_api.py`; Steve, 2026-10-09):

| Request | Returns |
|---|---|
| `GET /api/dataset/v1-cc` | JSON: `dataset_release`, `release_hash`, `reference_hash`, `labels_hash`, recipes, counts, the photo URL pattern |
| `GET /api/dataset/v1-cc/recipes/long500-q90` | JSON: the recipe document and its sha256 |
| `GET /api/dataset/v1-cc/photos/inat:12345678?recipe=long500-q90` | the JPEG, made from our kept original; headers `X-Derived-SHA256`, `X-Original-SHA256`, `X-Recipe`, `X-Dataset-Release`, `X-License`, `X-Attribution` (percent-encoded), `Link: <source URL>; rel="via"`, `ETag`, `Cache-Control: public, max-age=31536000, immutable` |

Only a public variant (`-cc`, and marked as one) is served, and only its CC photos (the
licence class is checked again per request); a full release is never even opened. Rate limit
`MV_DATASET_RATE` per address per minute (default 60); the last 256 images are kept in
memory. Settings on the box: `MV_DATASET_ROOT` (where releases are), `MV_DATASET_PHOTO_STORE`
(where originals are; the box's AWS user, today limited to `releases/*`, then needs read
access to the photos' `large/` prefix: a policy change for Steve), and `MV_DATASET_PUBLIC=1`
to serve these images without sign-in (otherwise they follow the site's sign-in rule).
Verify: `sha256sum` of the bytes equals `X-Derived-SHA256` and the release's `derived` row.

## 9. Public vs private

Steve, 2026-10-09: "whatever is the standard." The standard for image datasets built from
iNaturalist and MO (iNat open data, FungiTastic, DF20) is every CC licence, each photo with
its licence code, attribution and source URL; resized copies are shared under the photo's own
licence (a resize is a format change under CC, so ND photos are included as plain resizes,
never crops). All-rights-reserved photos are never shared.

| | `v1` (ours) | `v1-cc` (public) |
|---|---|---|
| record ids, sources, labels, label provenance, splits, inclusion reasons | yes | yes (only records with ≥1 CC photo) |
| photo ids, licences, owners' logins and attribution, public URLs | all | CC only (`open` + `nc`) |
| original sha256 + derived hashes | all | CC only |
| ARR photos (files, derivatives, or their hashes) | used for training | **never** |
| coordinates | `private.sqlite` only | **never** (not even rounded) |
| code, recipes, dataset card | yes | yes |

Anyone can rebuild `v1-cc` from public URLs, or fetch the on-demand CC derivatives (§8),
verify each against `derived_sha256`, and re-run our commands. The paper reports both.

**Zenodo** (Steve, 2026-10-09): each public variant is a Zenodo record with a DOI; a new
release is a new version of the same concept record. `mv dataset public v1` writes a
Zenodo-ready bundle in `v1-cc/zenodo/`:
- `metadata.json`: Zenodo deposit metadata (title, `upload_type: dataset`, description,
  creators: Russell, Stephen + MycoMap contributors (names TBC), `version` = release id,
  keywords, `license` for the tables (cc-by-4.0, to confirm), related identifiers: the code
  repository; the paper and the previous version are added at upload);
- `records.csv` and `photos.csv` (a `license_code` per photo, attribution, source URL,
  original sha256 and a `<recipe>_sha256` column per recipe);
- `recipes.json`;
- `files.json`: every file of the upload with sha256, size and licence ("per photo" for
  `photos.csv`; the record licence for the rest). Zenodo lists md5; sha256 stays the
  release's primary hash. CC photos only; no photo files, no ARR row, no coordinates.

## 10. Decisions (Steve, 2026-10-09) and what is still open

- North America only: **yes**.
- Testing: **Steve provides a new dataset of new records**, cut as `v1-test` (§5).
- Held-out development benchmark: **dated like every record**, membership kept.
- Public variant: **the standard** (§9), approved as drafted (16:05 UTC).
- The held-out 13,145 are ordinary training and reference records in v1; no held-out split.
- Weights: internal benchmarking models may use every photo; the final public model trains on
  `public_trainable` photos only.
- Public files on **Zenodo** with a DOI (16:35 UTC).
- On-demand images on the site (§8).
- Open: the `val` window (8 weeks by default, or a calendar cutoff chosen at the freeze);
  whether the images are public without sign-in (`MV_DATASET_PUBLIC`); the box's S3 read
  access to the originals; the table licence and the Zenodo creators.

## Not in this branch

Cutting v1 (waits for Steve's freeze trigger, relayed by the coordinator); uploading a release
to S3 or Zenodo; deploying the image endpoint (box settings + AWS policy); `--release` in
`mv finetune` and the Picek launcher; `add-model` after the first training run on v1.
