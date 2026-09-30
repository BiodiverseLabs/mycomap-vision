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
4. `mv embed --backbone <alias or timm:/open_clip: spec>`: one vector per photo.
5. `mv compare --backbones a,b --methods nearest,species-mean`: score models on the
   same photos (newest 28 days vs older records); `mv scoreboard` lists results;
   `mv models` lists backbones and methods. Plan: [docs/PLAN.md](docs/PLAN.md).

6. `mv release --backbones bioclip-2 --make-current`: publish what the public site
   serves to S3; the server box pulls it with `mv pull-release`. The site
   (vision.mycomap.org, its own Lightsail box, sign-in with a mycomap.org account)
   is set up and deployed as in [docs/deploy.md](docs/deploy.md).

`mv status` prints counts. `mv contributors [--arr-only]` writes the list of
photographers, with their all-rights-reserved photo counts, for permission requests.
`mv name-spellings [--json]` lists the names written more than one way: how many
Vision merges, and the ones a person has to decide (read-only).
`mv fetch-taxonomy [--refresh | --older-than DAYS]` asks iNat for the family, order,
class and phylum of every genus the records use (1 request/s, resumable, kept in
`data/taxonomy/inat_genera.sqlite`); `mv taxonomy` reports what that changes and writes
the genera in doubt to `data/reports/taxonomy-doubts.csv` (no network). The weekly
`mv refresh` asks only about genera no lookup has answered yet, for at most 20 minutes
(`--taxonomy-minutes`; the next refresh carries on), and goes on without it when iNat
is down; `--no-taxonomy` skips it.

### Photo permission

Photographers answer on mycomap.org (its `docs/vision-photo-permissions.md`).
`mv serve` reads the answers every 5 minutes from mycomap.org's
`/api/vision/photo-permissions` (settings `MV_ORG_BASE_URL`, `MV_ORG_VISION_KEY`;
`mv permissions --sync` does it by hand, `mv permissions` shows where they stand):

- an all-rights-reserved photo is shown as an example match only with its
  owner's grant, read within the last hour; otherwise the card says it isn't shown;
- a photographer who declines or withdraws has their all-rights-reserved photos
  left out of the reference set at once, and out of comparisons and training
  (`evaluate.load_records`); no answer yet = still used while permission is sought;
- licences are read when the photo is shown, and `mv refresh-licenses` (or
  `MV_LICENSE_REFRESH_HOURS=24` in `mv serve`) re-reads them from iNat, so a
  licence change there is picked up.

## Setup (Windows, Git Bash or PowerShell)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r requirements/dev.txt
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest -q
```

For embeddings, install PyTorch for your GPU first (e.g. the CUDA build from
download.pytorch.org), then `pip install -e ".[embed]"`. The web app is in `web/`
(`pnpm install`, `pnpm build`); `mv serve` serves the API and the built site.
See SECURITY.md for what stays private.

Copy `.env.example` to `.env` and fill it in. `mv export-records` needs a read-only
SQL route to the mycomap.org database (`MV_ORG_SQL_SSH_HOST`); the AWS commands need
the AWS settings.

## Data

Everything lives under `data/` (git-ignored; override with `MV_DATA_DIR`):

- `manifest.sqlite`: records, iNat observations, photos, observation-photo
  links and license history.
- `raw/`: the records export and every iNat API answer (gzipped), as fetched.
- `photos/<size>/<id % 1000>/<photo_id>.<ext>`: the images, under the same path in
  the S3 bucket (`MV_S3_BUCKET`) when they are downloaded to S3.
- `reports/`: contributor lists, the genera in doubt (`taxonomy-doubts.csv`).
- `taxonomy/inat_genera.sqlite`: iNat's answer for each genus and every taxon fetched
  (never inside the manifest; a release ships it).

### Labels

The label is the record's `scientific_name` on .org. A record is included when
any of its three flattened validation slots on .org is `yes`. Records green only
in a fourth or later project are missed (small). Records with more than one name
on .org are flagged `label_conflict` and are not used for training.

Spellings of one name are one label: `Inocybe PNW18` and `Inocybe "sp-PNW18"` are
both `Inocybe sp. 'PNW18'`. Only differences in writing are merged; a code that is
also in use as a plain name, IN7 / IN07 and described codes stay separate until
someone decides on mycomap.org. The manifest keeps the name as .org spells it
(`names.py`, the same rule as mycomap.org's `services/nameVariants.ts`).

Family (and order, class, phylum) comes from iNaturalist's taxonomy, one answer per
genus, so every record of a genus has the same family (`taxonomy.py`, Steve
2026-09-30). The genus is looked up within Fungi only; a genus in any doubt (not on
iNat, two fungal genera of that name, inactive, provisional, not a Latin name) keeps
.org's family and is listed by `mv taxonomy`, as is any genus whose iNat family
differs from what most of its records say on .org (iNat's is applied).

A one-word name ("Russula", "Agaricales") is never a species: the record has no
species label, adds no species class, is never a species candidate and isn't scored
at species. It counts at genus when the word is a genus (iNat says so, or, when iNat
can't, the genus column agrees) and at family. A record left with no label at family,
genus or species ("Fungi", "Agaricales", "Unknown") is left out.

### Photos and licenses

All photos of green records are pulled, including all-rights-reserved ones
(Steve's decision, 2026-09-28: he will ask those photographers for permission).
Every photo keeps its owner, license and a license history, so any contributor's
photos can be listed or removed. CC-licensed photos come from the iNaturalist
Open Data bucket on AWS; all-rights-reserved ones come from iNat's static host,
capped at 4 GB/hour and 20 GB/day (iNat's rule is under 5 and 24).
