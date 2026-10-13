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
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .dataset_release import REVIEW_COLUMNS, ReleaseError, sha256_file

REASONS = ("mo_twin_as_inat",)


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
