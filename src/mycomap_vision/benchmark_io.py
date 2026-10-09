"""Carry a held-out benchmark's results from the laptop to the server box, so the site's
Models page can chart them (Steve, 2026-10-09: every approach compared on the website).

    laptop:  mv benchmark-export --report data/benchmarks/<name>/reports/report-dev-<t>.json > dev.json
    box:     mv benchmark-import - < dev.json

`mv heldout report` runs on the laptop, where the benchmark's photos and answers live. Only
its aggregates travel: rates, counts and intervals per model and rank, species by the true
species' reference records, paired tests, calibration bins, likely-set coverage and the
standard summary. Never a record id, name, observer, coordinate, label audit or file path:
the export is built from a fixed list of fields, so anything new in a report stays home until
someone adds it here on purpose. Importing a benchmark split again replaces it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

KIND = "mycomap-vision benchmark summary"
VERSION = 1
SPLITS = ("dev", "test", "all")
HEAD = ("benchmark", "sealed", "records", "scored_records", "code_version", "released_at")
DEPTH = "species reference records"

SCHEMA = """
create table if not exists published_benchmarks (
  benchmark text not null,
  split text not null,
  report text,
  imported_at text not null,
  payload_json text not null,
  primary key (benchmark, split)
);
"""


class ImportRefused(ValueError):
    """The summary was not imported; nothing was changed."""


def export_summary(report: dict, report_name: str) -> dict:
    """The publishable part of one `mv heldout report` JSON."""
    if not isinstance(report, dict) or not report.get("benchmark") \
            or not isinstance(report.get("models"), dict) or not report["models"]:
        raise ValueError("not an mv heldout report (no benchmark or models)")
    depth = {m: d[DEPTH] for m, d in (report.get("breakdowns") or {}).items()
             if isinstance(d, dict) and d.get(DEPTH)}
    return {"kind": KIND, "version": VERSION, **{k: report.get(k) for k in HEAD},
            "split": report.get("split") or "all", "report": report_name,
            "models": report["models"], "species_by_reference_records": depth,
            "paired": report.get("paired") or [], "calibration": report.get("calibration") or {},
            "likely_sets": report.get("likely_sets") or {}, "summary": report.get("summary")}


def import_summary(conn: sqlite3.Connection, payload: dict) -> dict:
    conn.executescript(SCHEMA)
    if not isinstance(payload, dict) or payload.get("kind") != KIND \
            or payload.get("version") != VERSION:
        raise ImportRefused("not a benchmark export (mv benchmark-export)")
    name, split = payload.get("benchmark"), payload.get("split")
    if not isinstance(name, str) or not name:
        raise ImportRefused("the export names no benchmark")
    if split not in SPLITS:
        raise ImportRefused(f"unknown split {split!r}")
    if not isinstance(payload.get("models"), dict) or not payload["models"]:
        raise ImportRefused("the export holds no models")
    with conn:
        replaced = conn.execute("delete from published_benchmarks where benchmark = ? and split = ?",
                                (name, split)).rowcount
        conn.execute("insert into published_benchmarks values (?,?,?,?,?)",
                     (name, split, payload.get("report"),
                      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      json.dumps(payload)))
    return {"benchmark": name, "split": split, "models": len(payload["models"]),
            "replaced": bool(replaced)}


def published(conn: sqlite3.Connection) -> list[dict]:
    """Every imported benchmark split, newest import first."""
    conn.executescript(SCHEMA)
    rows = conn.execute("select payload_json, imported_at from published_benchmarks "
                        "order by imported_at desc, benchmark, split").fetchall()
    return [{**json.loads(p), "imported_at": t} for p, t in rows]
