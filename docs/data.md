# Data: records, labels, photos and permissions

How MycoMap Vision decides which records it learns from, what their labels are, and
which photos it may use and show. The data itself is private (see
[SECURITY.md](../SECURITY.md)); this page describes the rules, which are all in code.

## Where it lives

Everything lives under `data/` (git-ignored; override with `MV_DATA_DIR`):

- `manifest.sqlite`: records, iNat observations, photos, observation-photo links and
  licence history.
- `raw/`: the records export and every iNat and Mushroom Observer API answer
  (gzipped), as fetched.
- `photos/<size>/<id % 1000>/<photo_id>.<ext>`: the images, under the same path in
  the private S3 bucket (`MV_S3_BUCKET`) when they are downloaded to S3.
- `reports/`: contributor lists, the genera in doubt (`taxonomy-doubts.csv`).
- `taxonomy/inat_genera.sqlite`: iNat's answer for each genus and every taxon fetched
  (never inside the manifest; a serving release ships it).

## Which records

A record comes from [mycomap.org](https://mycomap.org) and is used when its DNA
sequence confirmed its name: a project on mycomap.org marked it validated ("green").
mycomap.org is only read, never written, through a read-only SQL route
(`MV_ORG_SQL_SSH_HOST`, used by `mv export-records`).

### Sources

A record's source is mycomap.org's `observations.source`, stored as `records.source`:
`inat`, `mo` (Mushroom Observer), `mycoportal`, `com_sequence` (sequences from
MycoMap's legacy site), `genbank` or `unknown` (`sources.py`). Ids are never read for
it: MO, MyCoPortal and iNat numbers look alike, and until 2026-10-09 the export took
every number for an iNat id, so MO and MyCoPortal records carried photos of whatever
iNat observation shared their number. Non-iNat records are keyed `<source>:<id>`.

Vision takes photos from iNat records (from iNat) and MO records (from MO, decided
2026-10-09). MyCoPortal records are left out (they rarely have field photos), and so
are legacy-site sequences, GenBank and unknown-source records (no photos of their
own). The first export with this code migrates an older manifest: the iNat photos and
details fetched for non-iNat records come off them, each written to `source_removals`,
and `manifest_migrations` gets `record-sources-v1`; nothing is fetched, downloaded or
embedded before that.

## Labels

The label is the record's `scientific_name` on mycomap.org. A record is included when
any of its first three validation slots on mycomap.org is `yes` (mycomap.org flattens
a record's project validations into numbered slots). Records validated only in a
fourth or later project are missed (few). Records with more than one name on
mycomap.org are flagged `label_conflict` and are not used for training. No
iNaturalist or Mushroom Observer community identification is ever used as a label.

Spellings of one name are one label: `Inocybe PNW18` and `Inocybe "sp-PNW18"` are
both `Inocybe sp. 'PNW18'`. Only differences in writing are merged; a code that is
also in use as a plain name, IN7 / IN07, and described codes stay separate until
someone decides on mycomap.org. The manifest keeps the name as mycomap.org spells it
(`names.py`, the same rule as mycomap.org's `services/nameVariants.ts`).

Family (and order, class, phylum) comes from iNaturalist's taxonomy, one answer per
genus, so every record of a genus has the same family (`taxonomy.py`, decided
2026-09-30). The genus is looked up within Fungi only; a genus in any doubt (not on
iNat, two fungal genera of that name, inactive, provisional, not a Latin name) keeps
mycomap.org's family and is listed by `mv taxonomy`, as is any genus whose iNat
family differs from what most of its records say on mycomap.org (iNat's is applied).

A one-word name ("Russula", "Agaricales") is never a species: the record has no
species label, adds no species class, is never a species candidate and isn't scored
at species. It counts at genus when the word is a genus (iNat says so, or, when iNat
can't, the genus column agrees) and at family. A record left with no label at family,
genus or species ("Fungi", "Agaricales", "Unknown") is left out.

Records whose DNA name belongs to a yeast or parasite living on the fungus in the
photo ("guests") are handled by `guests.py` (`mv guests` lists them).

## Photos and licences

All photos of validated records are pulled, including all-rights-reserved ones
(decided 2026-09-28: MycoMap asks those photographers for permission). Every photo
keeps its owner, licence and a licence history, so any contributor's photos can be
listed or removed. CC-licensed photos come from the iNaturalist Open Data bucket on
AWS; all-rights-reserved ones come from iNat's static host, capped at 4 GB/hour and
20 GB/day by default (iNat asks for under 5 and 24).

Mushroom Observer photos (`mo.py`) are read from MO's images API and image host, one
request at a time and at least 5 seconds apart (MO asks anonymous clients for 20 a
minute at most), with MO's licence name, owner and `ok_for_export` flag kept per photo.
An MO photo is shown only when it is Creative Commons; it is used only when MO marks it
ok for export and, for now, only when it is not all rights reserved. No MO location is
read.

## Photo permission

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

`mv contributors [--arr-only]` writes the list of photographers, with their
all-rights-reserved photo counts, for permission requests (private, under `data/`).

## Locations

Records can carry the true coordinates of observations that are obscured on iNat,
often for rare or sensitive species. No API response, page, export or model may
reveal them.
