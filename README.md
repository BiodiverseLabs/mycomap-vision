# MycoMap Vision

Fungal identification from photos, trained only on DNA-validated MycoMap records
("green in a project"), using every photo of a record rather than one.

It is a separate platform from mycomap.org. .org is read, never written: records
come from the read-only production SQL route, and .org will call this platform's
API when it needs a prediction.

## Status: phase 0 (data)

Phase 0 runs on a local machine with a GPU. Its goal is one number: does a
multi-photo model on DNA-validated labels beat iNat's model on the newest weeks
of green records?

1. `mv export-records`: pull green records from mycomap.org into the manifest.
2. `mv fetch-inat`: photo lists, licenses and owners from iNat (1 request/s).
3. `mv aws-launch-downloader`: large (1024 px) photos straight into the private S3
   bucket from a self-terminating EC2 instance, within iNat's limits. See
   [deploy/aws/README.md](deploy/aws/README.md). (`mv download-photos` does the
   same into a local folder or any `--dest`.)
4. Embeddings, baseline and comparison: next.

`mv status` prints counts. `mv contributors [--arr-only]` writes the list of
photographers, with their all-rights-reserved photo counts, for permission requests.

## Setup (Windows, Git Bash or PowerShell)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

`mv export-records` needs the `SQL-ROUTE-HOST` SSH host (read-only .org route).

## Data

Everything lives under `data/` (git-ignored; override with `MV_DATA_DIR`):

- `manifest.sqlite`: records, iNat observations, photos, observation-photo
  links and license history.
- `raw/`: the records export and every iNat API answer (gzipped), as fetched.
- `photos/<size>/<id % 1000>/<photo_id>.<ext>`: the images, under the same path in
  `s3://YOUR-BUCKET/` when they are downloaded to S3.
- `reports/`: contributor lists.

### Labels

The label is the record's `scientific_name` on .org. A record is included when
any of its three flattened validation slots on .org is `yes`. Records green only
in a fourth or later project are missed (small). Records with more than one name
on .org are flagged `label_conflict` and are not used for training.

### Photos and licenses

All photos of green records are pulled, including all-rights-reserved ones
(Steve's decision, 2026-09-28: he will ask those photographers for permission).
Every photo keeps its owner, license and a license history, so any contributor's
photos can be listed or removed. CC-licensed photos come from the iNaturalist
Open Data bucket on AWS; all-rights-reserved ones come from iNat's static host,
capped at 4 GB/hour and 20 GB/day (iNat's rule is under 5 and 24).
