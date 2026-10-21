"""The photo exclusions people made in Human Feedback, as a file the paper can freeze by its
sha256 (the paper coordinator's E34, 2026-10-21; cut-off 2026-11-14 23:59 UTC).

One row per photo item of the "Photos that show no specimen" queue that a person decided on or
before `as_of`: its LATEST such decision. A decision saved after `as_of` changes nothing, so a
run after the cut-off gives the same bytes. Items nobody decided (a flag alone) are not in it.

It reads a COPY of the store (made with sqlite3's .backup), opened read-only; it never opens
the live store and writes nothing to the copy. Reviewers appear only as an HMAC pseudonym (a key
the caller holds), never by name or id; the summary gives the key's fingerprint,
HMAC(key, "mycomap-e34"), and a run given --expect-key-fingerprint refuses any other key. A photo of a v1-test record is refused: the paper's
seal says no queue may hold one (feedback.SealGate), so finding one is an error to look into.

Columns (fixed, pinned by the paper coordinator in curation_v1_1):
  photo_key, manifest_photo_id, record_key, v1_split, decision, reason, decision_id,
  decided_at, reviewer_pseudonym, confirmation, flag_source
`confirmation` comes from the decision's note: "shown" for the Summary's default Exclude saved
once the photo had been on screen ("summary: shown"), "flipped" for any other Summary save (the
reviewer changed the tile or picked a reason), and "record" for a save in the record view.
`flag_source` is what brought the photo to review: each part of the item's rule without its
numbers ("R6 micrograph probe; R7 uv probe"), or the reviewer's own pick for a photo no rule
flagged."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import feedback as fb
from . import photo_review as pr

COLUMNS = ("photo_key", "manifest_photo_id", "record_key", "v1_split", "decision", "reason",
           "decision_id", "decided_at", "reviewer_pseudonym", "confirmation", "flag_source")
NOTE_SHOWN = "summary: shown"
FINGERPRINT_LABEL = b"mycomap-e34"      # the paper pins HMAC(key, this), never the key
_NUMBERS = re.compile(r"\s*(p(\(fungus\))?=[0-9.]+|\([^)]*\))")


class ExportError(Exception):
    pass


def parse_as_of(text: str) -> float:
    """An ISO time with a zone ('2026-11-14T23:59:59Z') as epoch seconds."""
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ExportError("give --as-of with its zone, e.g. 2026-11-14T23:59:59Z")
    return t.timestamp()


def confirmation(note: str | None) -> str:
    note = (note or "").strip()
    if note == NOTE_SHOWN:
        return "shown"
    if note.startswith("summary:"):
        return "flipped"
    return "record"


def flag_source(rule: str) -> str:
    parts = [_NUMBERS.sub("", p).strip() for p in (rule or "").split(";")]
    return "; ".join(p for p in parts if p)


def key_fingerprint(key: bytes) -> str:
    """What the paper's config records for the key: HMAC(key, "mycomap-e34")."""
    return hmac.new(key, FINGERPRINT_LABEL, hashlib.sha256).hexdigest()


def pseudonym(key: bytes, user: str) -> str:
    return hmac.new(key, (user or "").encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _open_copy(path: Path, live: Path | None) -> sqlite3.Connection:
    path = Path(path)
    if not path.exists():
        raise ExportError(f"no store copy at {path}")
    if live is not None and path.resolve() == Path(live).resolve():
        raise ExportError("that is the live store: export from a copy made with "
                          "sqlite3 <store> '.backup <copy>'")
    for side in ("-wal", "-shm"):
        if Path(str(path) + side).exists():
            raise ExportError(f"{path.name} has a {side} file beside it, so it is a live "
                              "database, not a .backup copy")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _splits(release: Path) -> dict[str, str]:
    """photo_key -> its record's v1 split, from release v1 (every photo it lists)."""
    rel = sqlite3.connect(f"file:{Path(release).as_posix()}?mode=ro", uri=True)
    split = dict(rel.execute("select record_key, split from records"))
    return {k: split.get(rk) or "" for k, rk in rel.execute(
        "select photo_key, record_key from photos")}


def export(store_copy: Path, release: Path, as_of: str, key: bytes,
           live: Path | None = None, expect_fingerprint: str | None = None
           ) -> tuple[bytes, dict]:
    """The export's bytes (UTF-8, LF, a header row, rows sorted by photo_key) and a summary
    with its sha256 and counts by decision, reason, confirmation and v1 split."""
    if not key:
        raise ExportError("no HMAC key: reviewers are exported only as pseudonyms")
    if expect_fingerprint is not None and key_fingerprint(key) != expect_fingerprint.strip():
        raise ExportError("the HMAC key's fingerprint is not the one expected: a different key "
                          "would give different pseudonyms")
    cut = parse_as_of(as_of)
    conn = _open_copy(store_copy, live)
    split_of = _splits(release)
    items = {r["id"]: dict(r) for r in conn.execute(
        "select id, subject, rule, record_keys, links from items where item_type = ?",
        (pr.ITEM_TYPE,))}
    latest: dict[int, sqlite3.Row] = {}
    for d in conn.execute("select * from decisions where at <= ? order by at, id", (cut,)):
        if d["item_id"] in items:
            latest[d["item_id"]] = d                       # the last one on or before the cut
    rows, test = [], []
    for item_id, d in latest.items():
        it = items[item_id]
        key_ = it["subject"]
        sp = split_of.get(key_, "")
        if sp == "test":
            test.append(key_)
            continue
        rk = pr.record_of_item({"record_keys": json.loads(it["record_keys"] or "[]"),
                                "links": json.loads(it["links"] or "[]")}) or ""
        pid = fb.photo_id_of(key_)
        rows.append({
            "photo_key": key_, "manifest_photo_id": "" if pid is None else str(pid),
            "record_key": rk, "v1_split": sp or "none", "decision": d["decision"],
            "reason": (d["value"] or "") if d["decision"] == "exclude_photo" else "",
            "decision_id": str(d["id"]),
            "decided_at": datetime.fromtimestamp(d["at"], timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"),
            "reviewer_pseudonym": pseudonym(key, d["by_user"]),
            "confirmation": confirmation(d["note"]), "flag_source": flag_source(it["rule"])})
    conn.close()
    if test:
        raise ExportError(f"{len(test)} decided photo(s) belong to v1-test records, which no "
                          f"queue may hold (the paper's seal): {sorted(test)[:20]}")
    rows.sort(key=lambda r: r["photo_key"])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    summary = {"as_of": as_of, "rows": len(rows), "sha256": hashlib.sha256(data).hexdigest(),
               "key_fingerprint": key_fingerprint(key),
               "by_decision": dict(sorted(Counter(r["decision"] for r in rows).items())),
               "exclusions_by_confirmation": dict(sorted(Counter(
                   r["confirmation"] for r in rows if r["decision"] == "exclude_photo").items())),
               "exclusions_by_reason": dict(sorted(Counter(
                   r["reason"] for r in rows if r["decision"] == "exclude_photo").items())),
               "by_split": dict(sorted(Counter(r["v1_split"] for r in rows).items()))}
    return data, summary
