"""Carry a held-out benchmark's results from the laptop to the server box, so the site's
Models page can chart them (Steve, 2026-10-09: every approach compared on the website).

    laptop:  mv benchmark-export --report data/benchmarks/<name>/reports/report-dev-<t>.json > dev.json
    box:     mv benchmark-import - < dev.json

A comparison on a dataset release (`mv compare --release v1`, a trainer run) travels the same
way, so the Results page shows the release's numbers beside (and above) the pre-v1 benchmarks:

    laptop:  mv benchmark-export --comparison <id> --split val > v1-val.json

`mv heldout report` runs on the laptop, where the benchmark's photos and answers live. Only
its aggregates travel: rates, counts and intervals per model and rank, species by the true
species' reference records, paired tests, calibration bins, likely-set coverage and the
standard summary. Never a record id, name, observer, coordinate, label audit or file path:
the export is built from a fixed list of fields, so anything new in a report stays home until
someone adds it here on purpose. Importing a benchmark split again replaces it.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

KIND = "mycomap-vision benchmark summary"
VERSION = 1
SPLITS = ("dev", "test", "all", "val")
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


def export_summary(report: dict, report_name: str,
                   reproducibility: str = "exploratory-pre-freeze") -> dict:
    """The publishable part of one `mv heldout report` JSON. `reproducibility` says whether
    it was measured before the dataset freeze (docs/PLAN.md) or on a named release."""
    if not isinstance(report, dict) or not report.get("benchmark") \
            or not isinstance(report.get("models"), dict) or not report["models"]:
        raise ValueError("not an mv heldout report (no benchmark or models)")
    depth = {m: d[DEPTH] for m, d in (report.get("breakdowns") or {}).items()
             if isinstance(d, dict) and d.get(DEPTH)}
    return {"kind": KIND, "version": VERSION, **{k: report.get(k) for k in HEAD},
            "split": report.get("split") or "all", "report": report_name,
            "models": report["models"], "species_by_reference_records": depth,
            "paired": report.get("paired") or [], "calibration": report.get("calibration") or {},
            "likely_sets": report.get("likely_sets") or {}, "summary": report.get("summary"),
            "reproducibility": reproducibility}


def _rate(n: int, rate: float | None) -> dict:
    """A share as the site's Rate: n, right, rate and a Wilson 95% interval."""
    if rate is None or not n:
        return {"n": n, "rate": None}
    z, p = 1.959964, rate
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"n": n, "right": round(p * n), "rate": rate,
            "ci95": [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]}


def export_comparison(conn: sqlite3.Connection, comparison_id: str, split: str = "val",
                      sealed: bool = False) -> dict:
    """The publishable part of a comparison measured on a frozen dataset release, in the
    same form as export_summary: per model the standard table (top 1/3/5/10, s.l. and complex
    where scored), species, genus and family top-1/top-5, and species by the true species'
    reference records. A comparison off any release, or on a draft, is refused: only frozen
    releases are published as reproducible."""
    from . import evaluate, standard_report
    if split not in ("val", "test"):
        raise ValueError("a release comparison is scored on val, or on a test release")
    s = standard_report.summary(conn, comparison_id)
    rel = s.get("dataset_release")
    if not rel:
        raise ValueError(f"comparison {comparison_id} is not on a dataset release: publish "
                         "held-out reports with --report")
    if rel.get("draft"):
        raise ValueError(f"comparison {comparison_id} is on a draft release: not publishable")
    rows = conn.execute("select backbone, method, n_test, report_json from eval_runs "
                        "where comparison_id = ? order by backbone, method",
                        (comparison_id,)).fetchall()
    models, depth, tables = {}, {}, {}
    for backbone, method, n_test, report_json in rows:
        report = json.loads(report_json)
        std = report["all_photos"].get("standard") or {}
        key = f"{backbone}/{method}"
        models[key] = {"records": n_test}
        for rank in ("species", "genus", "family"):
            c = (std.get(rank) or {}).get("all")
            if c:
                models[key][rank] = {k: _rate(c["n"], c.get(k)) for k in ("top1", "top5")}
        depth[key] = {band: {"species": _rate(c["n"], c.get("top1"))}
                      for band in std.get("bands", [])
                      if (c := (std.get("species") or {}).get(band))}
        table = standard_report.model_table(report)
        tables[key] = {"rows": {label: {f"top{k}": _rate(row["n"], row.get(f"top{k}"))
                                        for k in evaluate.STANDARD_K}
                                for label, row in table["rows"].items() if "note" not in row}}
    return {"kind": KIND, "version": VERSION, "benchmark": rel["dataset_release"],
            "sealed": sealed, "records": s["test_records"], "scored_records": s["test_records"],
            "code_version": rel.get("release_code_commit"), "released_at": None,
            "split": split, "report": f"comparison {comparison_id}",
            "dataset_release": rel, "reference_records": s["reference_records"],
            "models": models, "species_by_reference_records": depth, "paired": [],
            "calibration": {}, "likely_sets": {}, "summary": {"models": tables},
            "reproducibility": f"reproduced-on-{rel['dataset_release']}"}


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
