"""A frozen dataset release's erratum: records found wrong after the cut (2026-10-13).

A release is never edited (dataset_release.py). When a fault is found in one, the fix is
an erratum FILE kept beside it, never in it: a TSV in the review-list format (kind, key,
reason, note) listing the release's records to leave out. `--errata FILE` on a command
that reads the release (compare, model-sanity, int8-check, finetune, heldout predict,
external run) drops those records from every split it reads, so they are neither scored,
used as references, nor trained on; the file's sha256 goes into the release's cite(), and
so into every report and registry entry made with it. A key that is not a record of the
release refuses the erratum: an erratum is for one release.

What the fine-tune already learnt is not undone: its heads still hold a class for a label
it saw only in erratum records, but a method answers only with the reference's labels, so
such a class can no longer be an answer (`masked_head_labels` counts them).

Reasons:
- `mo_twin_as_inat` (v1): a Mushroom Observer find that mycomap.org also holds as an
  iNaturalist record under the same number, in the same validation project. The iNat
  record's name and photos are the unrelated iNat observation with that number (a duck, a
  panda...). Rule: sources.mo_twins. The nightly drops such records from the live
  manifest (records.save_records), so no later release or served index holds them.
- `nonfungal_label` (v1, Steve 2026-10-13): a record whose label's genus is no fungus
  (taxonomy.nonfungal: no fungal genus of that name on iNat, every taxon of that name in
  Animalia, Plantae or Bacteria): a bee from a bee project, the host plant of a pooled
  sample. Loading records leaves them out from now on (evaluate.load_records), and the
  builder lists them as `nonfungal_label`. A record that is also an MO twin keeps that
  reason. The rule reads the taxonomy cache, whose sha256 the erratum's writer reports.
- `crosspost_twin` (v1, Steve 2026-10-18, erratum rev 4): the val side of one find posted to
  both Mushroom Observer and iNaturalist whose other side is in train. A pair
  (crosspost_twin_pairs) is an included MO record and an included iNat record with the SAME
  release label and a photo within TWIN_MAX_BITS bits of each other in the release's dHash
  index. Template hashes (one hash on the photos of TEMPLATE_RECORDS or more records:
  microscope fields, slip forms) and the photos of a given exclusion list (micrographs) never
  make a pair. Only a pair with one side in train and one in val is listed, by its val side.
  Found by the v1 miss review (2026-10-18): 99% of such val finds were answered right,
  because their own photo was in the reference. Twins within one split are not a leak
  (train/train duplicates a reference; val/val holds no reference) and stay.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .dataset_release import REVIEW_COLUMNS, ReleaseError, sha256_file

REASONS = ("mo_twin_as_inat", "nonfungal_label", "crosspost_twin")
TWIN_MAX_BITS = 3          # dHash bits at or under which two photos are one picture
TEMPLATE_RECORDS = 4       # a hash on the photos of this many records is a template


@dataclass(frozen=True)
class Errata:
    file: str
    sha256: str
    reasons: dict[str, str] = field(default_factory=dict)     # record_key -> reason

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.reasons)

    def cite(self) -> dict:
        return {"file": self.file, "sha256": self.sha256, "records": len(self.reasons),
                "by_reason": dict(sorted(Counter(self.reasons.values()).items()))}


def read(path: Path) -> Errata:
    """Read an erratum TSV; refuse another layout, a photo row or an unknown reason."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
        header = rows[0].keys() if rows else []
    if rows and list(header) != REVIEW_COLUMNS:
        raise ReleaseError(f"{path.name}: columns must be {REVIEW_COLUMNS}")
    reasons: dict[str, str] = {}
    for r in rows:
        if r["kind"] != "record":
            raise ReleaseError(f"{path.name}: an erratum lists records, not {r['kind']!r}")
        if r["reason"] not in REASONS:
            raise ReleaseError(f"{path.name}: unknown reason {r['reason']!r} (known: {REASONS})")
        reasons[r["key"].strip()] = r["reason"]
    if not reasons:
        raise ReleaseError(f"{path.name} lists no records")
    return Errata(path.name, sha256_file(path), reasons)


def check(errata: Errata, release) -> None:
    unknown = errata.keys - set(release.record_keys())
    if unknown:
        raise ReleaseError(f"{errata.file} names {len(unknown)} records that are not in "
                           f"{release.id} (e.g. {sorted(unknown)[:5]}): an erratum is for one "
                           "release")


def mo_twin_rows(release) -> list[dict]:
    """The erratum rows for `mo_twin_as_inat`: every record of the release (included or
    not) that sources.mo_twins finds, with its twin and shared projects in the note."""
    from . import sources
    recs = [dict(r) for r in release.db.execute(
        "select record_key, source, source_id, green_projects from records")]
    by_id = {r["source_id"]: r for r in recs if r["source"] == sources.MO}
    out = []
    for key in sorted(sources.mo_twins(recs, key="record_key")):
        r = next(x for x in recs if x["record_key"] == key)
        twin = by_id[r["source_id"]]
        shared = sorted(set(json.loads(r["green_projects"] or "[]"))
                        & set(json.loads(twin["green_projects"] or "[]")))
        out.append({"kind": "record", "key": key, "reason": "mo_twin_as_inat",
                    "note": f"twin {twin['record_key']}; projects {', '.join(shared)}"})
    return out


def nonfungal_rows(release, tax, skip: set[str] = frozenset()) -> list[dict]:
    """The erratum rows for `nonfungal_label`: every record of the release (included or
    not) whose genus taxonomy.nonfungal names no fungus, but those in `skip` (already
    listed for another reason)."""
    from .taxonomy import nonfungal
    out = []
    for key, genus, label in release.db.execute(
            "select record_key, genus, label from records order by record_key"):
        kingdom = nonfungal(tax, genus or "")
        if kingdom and key not in skip:
            out.append({"kind": "record", "key": key, "reason": "nonfungal_label",
                        "note": f"{label}: genus {genus} is {kingdom} on iNat, no fungus"})
    return out


@dataclass(frozen=True)
class TwinPair:
    mo_record: str
    inat_record: str
    label: str
    mo_split: str
    inat_split: str
    mo_photo: str
    inat_photo: str
    bits: int


def template_hashes(hash_records) -> set[int]:
    """Hashes on the photos of TEMPLATE_RECORDS or more different records (microscope
    fields, slip forms): they say nothing about the specimen, so they never make a twin.
    `hash_records`: (dhash, record key) for every photo considered."""
    from collections import defaultdict
    seen = defaultdict(set)
    for h, r in hash_records:
        seen[int(h)].add(r)
    return {h for h, rs in seen.items() if len(rs) >= TEMPLATE_RECORDS}


def match_twins(photos, templates=frozenset()) -> dict[tuple[str, str], tuple[str, str, int]]:
    """The cross-post rule on photos given as (photo_key, record_key, label, source, dhash):
    an 'mo' and an 'inat' record with the same label and a photo within TWIN_MAX_BITS bits,
    template hashes left out. Returns {(mo record, inat record): (mo photo, inat photo,
    bits)}, the closest photos of each record pair. Shared by the erratum (crosspost_twin_pairs)
    and the test-release builder (dataset_release.drop_crosspost_twins)."""
    from collections import defaultdict
    usable = [(k, r, lab, src, int(h)) for k, r, lab, src, h in photos
              if int(h) not in templates]
    # Two hashes within 3 bits agree exactly on at least one of their four 16-bit blocks.
    blocks = [defaultdict(list) for _ in range(4)]
    for x in usable:
        if x[3] == "inat":
            for b in range(4):
                blocks[b][(x[4] >> (16 * b)) & 0xFFFF].append(x)
    best: dict[tuple[str, str], tuple[str, str, int]] = {}
    for mk, mr, ml, msrc, mh in usable:
        if msrc != "mo":
            continue
        cand = {}
        for b in range(4):
            for y in blocks[b].get((mh >> (16 * b)) & 0xFFFF, ()):
                cand[y[0]] = y
        for ik, ir, il, _, ih in cand.values():
            bits = bin(mh ^ ih).count("1")
            if bits > TWIN_MAX_BITS or ml != il:
                continue
            old = best.get((mr, ir))
            if old is None or (bits, mk, ik) < (old[2], old[0], old[1]):
                best[(mr, ir)] = (mk, ik, bits)
    return best


def crosspost_twin_pairs(release, index: dict, exclude_photos=frozenset(),
                         skip=frozenset()) -> list[TwinPair]:
    """Every MO <-> iNat cross-post pair of the release's included records, in any splits
    (rule in the module docstring; match_twins), one per record pair (its closest photos),
    sorted. `index` is the release's dHash index (dhash.load_index; checked against the
    release); `exclude_photos` holds photo keys that never pair (micrographs); records in
    `skip` (already in an erratum, so in no split) never pair."""
    from . import dhash
    dhash.check_index(index, release)
    recs = {k: (label, split, source) for k, label, split, source in release.db.execute(
        "select record_key, label, split, source from records where included = 1")}
    rec_of = dict(release.db.execute("select photo_key, record_key from photos "
                                     "where included = 1"))
    keys, hashes = index["photo_key"], [int(h) for h in index["dhash"]]
    templates = template_hashes((h, rec_of.get(k)) for k, h in zip(keys, hashes))
    photos = [(k, rec_of[k], recs[rec_of[k]][0], recs[rec_of[k]][2], h)
              for k, h in zip(keys, hashes)
              if rec_of.get(k) in recs and rec_of[k] not in skip and k not in exclude_photos]
    pairs = match_twins(photos, templates)
    return [TwinPair(mo, inat, recs[mo][0], recs[mo][1], recs[inat][1], mp, ip, bits)
            for (mo, inat), (mp, ip, bits) in sorted(pairs.items())]


def crosspost_twin_rows(release, index: dict, exclude_photos=frozenset(),
                        skip=frozenset()) -> list[dict]:
    """The erratum rows for `crosspost_twin`: the val record of every pair that straddles
    train and val (crosspost_twin_pairs), listed once, with its first twin in the note."""
    out = {}
    for p in crosspost_twin_pairs(release, index, exclude_photos, skip):
        if {p.mo_split, p.inat_split} != {"train", "val"}:
            continue
        val_mo = p.mo_split == "val"
        val_key, train_key = (p.mo_record, p.inat_record) if val_mo else (p.inat_record,
                                                                          p.mo_record)
        val_photo, train_photo = (p.mo_photo, p.inat_photo) if val_mo else (p.inat_photo,
                                                                            p.mo_photo)
        out.setdefault(val_key, {
            "kind": "record", "key": val_key, "reason": "crosspost_twin",
            "note": f"twin {train_key} (train), {p.label}; photo {val_photo} ~ {train_photo}, "
                    f"{p.bits} bits"})
    return list(out.values())


def read_rows(path: Path) -> list[dict]:
    """An earlier erratum's rows exactly as written, to carry into a new revision."""
    read(path)                                     # refuses a malformed one
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def read_photo_list(path: Path) -> frozenset[str]:
    """Photo keys from a TSV with a `photo_key` column (the micrograph list)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if rows and "photo_key" not in rows[0]:
        raise ReleaseError(f"{Path(path).name}: needs a photo_key column")
    return frozenset(r["photo_key"].strip() for r in rows)


def write(rows: list[dict], path: Path) -> Path:
    """Write an erratum TSV (LF line ends, sorted by key: the same rows give the same
    bytes, so the same sha256). Refuses to overwrite: an erratum is fixed once cited."""
    path = Path(path)
    if path.exists():
        raise ReleaseError(f"{path} exists: an erratum is never rewritten (write a new file)")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in sorted(rows, key=lambda r: r["key"]):
        w.writerow(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue().encode("utf-8"))
    return path


def masked_head_labels(release, errata: Errata, head_names) -> list[str]:
    """Species classes of a fine-tune's head (trained on `release`'s train split) whose
    only training records are erratum records: classes it can no longer answer with."""
    keep, gone = set(), set()
    for r in release.db.execute("select record_key, species from records where included = 1 "
                                "and split = 'train' and species is not null and species != ''"):
        (gone if r[0] in errata.keys else keep).add(r[1])
    return sorted((gone - keep) & set(head_names))
