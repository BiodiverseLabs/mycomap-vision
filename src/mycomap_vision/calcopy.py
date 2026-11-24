"""A model's calibration copied onto another model (`mv calibration copy`).

Steve, 2026-11-07: MycoMap Vision v2 (the opt-in beta) states its confidences with the
calibration fitted for the step-1 BioCLIP 2.5 fine-tune, until it has its own. Serving reads a
model's calibration from `eval_runs` (evaluate.latest_calibration): the newest row of the
model and the method whose calibration the served method shares, or, with
MV_PLACE_CALIBRATION=own, the place method's own newest row scored under the place setting
in force. This module writes exactly such rows for the target model:

- from another model's rows (`--from <backbone>`, read from `--from-manifest`, default the
  same manifest): for each method, the rows latest_calibration would read for the source;
- or from a calibration file (`--from-file`, the format of the step-1 file
  `<backbone>.calibration-v1-val.json`: a `photos_only` and a `place` block, each with its
  method, place setting and per-rank temperature, floor and target).

What is written is the calibration only: no score columns (species_top1 and the rest are
null) and no report beyond the calibration, so a copied row never reads as a measurement of
the target. Each row records where it came from, when, who and why (`calibration_copy` in
its report_json). The scoreboard leaves these rows out (evaluate.scoreboard), so no results
page or "How good is it" shows them as the target's results.

Safety: rows for the target are never overwritten or deleted. If the target already has a
row for a method the copy would write, the copy is refused unless `replace` is given; then
the copies are added as newer rows, which serving reads first, and the older ones stay.
Everything is written in one transaction. A dry run opens the manifest read-only and writes
nothing.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import config, evaluate, placeprior

MARK = "calibration_copy"            # the report_json key that marks a copied row
RANKS = ("species", "genus", "family")
FILE_BLOCKS = ("photos_only", "place")
NOT_NULL = ("cutoff", "test_days", "n_reference", "n_test", "record_set")


class Refused(ValueError):
    """The copy would overwrite, or its source is not usable: nothing was written."""


@dataclass
class Planned:
    """One eval_runs row the copy writes for the target."""
    method: str
    place_setting: str | None
    calibration: dict                       # {rank: {temperature, n, sets?}}
    source: dict                            # where it came from (into calibration_copy)
    columns: dict = field(default_factory=dict)   # cutoff, test_days, n_reference, ...

    def key(self) -> tuple[str, str | None]:
        return self.method, self.place_setting


def _report(conn: sqlite3.Connection, row_id: int) -> dict:
    (text,) = conn.execute("select report_json from eval_runs where id = ?", (row_id,)).fetchone()
    return json.loads(text)


def _has_eval_runs(conn: sqlite3.Connection) -> bool:
    return conn.execute("select 1 from sqlite_master where type = 'table' and name = 'eval_runs'"
                        ).fetchone() is not None


def rows_read_for(conn: sqlite3.Connection, backbone: str) -> list[int]:
    """The ids of `backbone`'s rows serving can read a calibration from: per method its
    newest row (latest_calibration), and for a method that shares another's calibration
    (evaluate.calibration_method) also its newest row per place setting
    (own_calibration_row, MV_PLACE_CALIBRATION=own). Newest first per key."""
    if not _has_eval_runs(conn):
        return []
    seen: set = set()
    out = []
    for row_id, method, setting in conn.execute(
            "select id, method, json_extract(report_json, '$.place_setting') from eval_runs "
            "where backbone = ? order by id desc", (backbone,)):
        shares = evaluate.calibration_method(method) != method
        key = (method, setting if shares else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(row_id)
    return sorted(out)


def plan_from_rows(src: sqlite3.Connection, source: str,
                   methods: set[str] | None = None) -> tuple[list[Planned], list[str]]:
    """What copying `source`'s calibration rows would write, and what was skipped (a newest
    row with no calibration: serving reads none for that method, so none is copied)."""
    planned, skipped = [], []
    for row_id in rows_read_for(src, source):
        cols = dict(zip(("id", "comparison_id", "method", "cutoff", "test_days", "n_reference",
                         "n_test", "record_set", "code_version", "created_at"),
                        src.execute("select id, comparison_id, method, cutoff, test_days, "
                                    "n_reference, n_test, record_set, code_version, created_at "
                                    "from eval_runs where id = ?", (row_id,)).fetchone()))
        if methods and cols["method"] not in methods:
            continue
        report = _report(src, row_id)
        cal = (report.get("all_photos") or {}).get("calibration")
        if not cal:
            skipped.append(f"row {row_id} ({cols['method']}): no calibration")
            continue
        setting = report.get("place_setting")
        planned.append(Planned(
            method=cols["method"], place_setting=setting,
            calibration={rank: {k: c[k] for k in ("temperature", "n", "sets") if k in c}
                         for rank, c in cal.items()},
            source={"from_backbone": source, "from_run_id": row_id,
                    "from_comparison_id": cols["comparison_id"],
                    "from_created_at": cols["created_at"],
                    "from_code_version": cols["code_version"]},
            columns={k: cols[k] for k in NOT_NULL}))
    return planned, skipped


def plan_from_file(path: Path, source: str) -> list[Planned]:
    """What importing a calibration file (the step-1 `*.calibration-v1-val.json` format)
    would write: one row per block (photos_only, place), keyed by its method and place
    setting as serving reads them. Refused when the file names another model than `source`,
    or a block lacks a method, a known place setting, or a temperature per rank."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("model") != source:
        raise Refused(f"{Path(path).name} is the calibration of {doc.get('model')!r}, "
                      f"not {source!r}")
    out = []
    for name in FILE_BLOCKS:
        block = doc.get(name)
        if not isinstance(block, dict):
            raise Refused(f"{Path(path).name} has no {name!r} block")
        method = block.get("method")
        if method not in evaluate.METHODS:
            raise Refused(f"{name}: unknown method {method!r}")
        setting = block.get("place_setting")
        if setting is not None and setting not in placeprior.PLACE_SETTINGS:
            raise Refused(f"{name}: unknown place setting {setting!r}")
        values, achieved = block.get("values") or {}, block.get("achieved") or {}
        cal = {}
        for rank in RANKS:
            v, a = values.get(rank) or {}, achieved.get(rank) or {}
            t = v.get("temperature")
            if not isinstance(t, (int, float)) or not t > 0:
                raise Refused(f"{name}: no temperature for {rank}")
            entry = {"temperature": float(t), "n": int(a.get("n_nll") or a.get("n") or 0)}
            if v.get("floor") is not None and a.get("coverage") is not None:
                # A likely set as likely.fit_and_check stores one (identify._likely reads
                # its floor; apiv1.likely_coverage its achieved coverage).
                entry["sets"] = {"method": "lac", "target": v.get("target"),
                                 "coverage": a["coverage"], "floor": float(v["floor"]),
                                 "n": a.get("n"), "mean_size": a.get("mean_size")}
            cal[rank] = entry
        out.append(Planned(
            method=method, place_setting=setting, calibration=cal,
            source={"from_backbone": source, "from_file": Path(path).name,
                    "from_file_sha256": hashlib.sha256(raw).hexdigest(),
                    "fit_on": doc.get("fit_on"), "answer_key": doc.get("answer_key"),
                    "block": name},
            columns={"cutoff": "", "test_days": 0,
                     "n_reference": int(block.get("reference_records") or 0),
                     "n_test": int(block.get("test_records") or 0),
                     "record_set": str(block.get("record_set") or "")}))
    return out


def clashes(conn: sqlite3.Connection, target: str, planned: list[Planned]) -> list[str]:
    """The target's own rows for any method the copy would write (a copy would shadow
    them): each as "row <id> <method> [<place setting>]"."""
    if not _has_eval_runs(conn):
        return []
    methods = sorted({p.method for p in planned})
    if not methods:
        return []
    marks = ",".join("?" for _ in methods)
    return [f"row {i} {m}" + (f" [{s}]" if s else "")
            for i, m, s in conn.execute(
                "select id, method, json_extract(report_json, '$.place_setting') from eval_runs "
                f"where backbone = ? and method in ({marks}) order by id", (target, *methods))]


def rows_to_write(target: str, planned: list[Planned], note: str, by: str | None = None,
                  now: datetime | None = None) -> list[dict]:
    """The eval_runs rows (column: value) the copy inserts, ids left to SQLite."""
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    at = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    who = by or getpass.getuser()
    rows = []
    for p in planned:
        report = {"backbone": target, "method": p.method,
                  "all_photos": {"calibration": p.calibration},
                  MARK: {**p.source, "to_backbone": target, "copied_at": at, "by": who,
                         "note": note, "code_version": config.code_version()}}
        if p.place_setting is not None:
            report["place_setting"] = p.place_setting
        rows.append({"comparison_id": f"{stamp}-calcopy", "backbone": target,
                     "method": p.method, **p.columns,
                     "species_top1": None, "genus_top1": None, "family_top1": None,
                     "species_top1_first_photo": None,
                     "report_json": json.dumps(report, sort_keys=True),
                     "code_version": config.code_version(), "created_at": at})
    return rows


def methods_reading(planned: list[Planned]) -> list[str]:
    """The methods whose served calibration the planned rows set: their own methods, and
    every method that shares one of them (evaluate.calibration_method)."""
    own = {p.method for p in planned}
    return sorted(own | {m for m in evaluate.METHODS if evaluate.calibration_method(m) in own})


def insert(conn: sqlite3.Connection, rows: list[dict],
           check: "Callable[[sqlite3.Connection], list[str]] | None" = None) -> list[int]:
    """Insert `rows` in one transaction (all or nothing); their new ids. `check` runs inside
    the transaction, before anything is written: anything it returns refuses the copy."""
    if not _has_eval_runs(conn):
        conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    ids = []
    conn.execute("begin immediate")
    try:
        found = check(conn) if check else []
        if found:
            raise Refused("the target has rows for these methods now: " + "; ".join(found))
        for row in rows:
            cols = list(row)
            cur = conn.execute(f"insert into eval_runs ({', '.join(cols)}) values "
                               f"({', '.join('?' for _ in cols)})", [row[c] for c in cols])
            ids.append(cur.lastrowid)
        conn.execute("commit")
    except BaseException:
        conn.execute("rollback")
        raise
    return ids


def served_reads(conn: sqlite3.Connection, target: str, rows: list[dict],
                 methods: list[str]) -> dict:
    """What evaluate.latest_calibration would return for `target` and each of `methods`
    once `rows` are added (under this environment's MV_PLACE_CALIBRATION and
    MV_PLACE_SETTING), worked out on an in-memory copy of the target's rows: nothing is
    written to `conn`."""
    mem = sqlite3.connect(":memory:")
    mem.executescript(evaluate.SCOREBOARD_SCHEMA)
    cols = ["id", "comparison_id", "backbone", "method", "cutoff", "test_days", "n_reference",
            "n_test", "record_set", "report_json", "code_version", "created_at"]
    top = 0
    if _has_eval_runs(conn):
        for r in conn.execute(f"select {', '.join(cols)} from eval_runs where backbone = ?",
                              (target,)):
            mem.execute(f"insert into eval_runs ({', '.join(cols)}) values "
                        f"({', '.join('?' for _ in cols)})", r)
        top = conn.execute("select coalesce(max(id), 0) from eval_runs").fetchone()[0]
    for n, row in enumerate(rows, 1):
        mem.execute(f"insert into eval_runs (id, {', '.join(row)}) values "
                    f"(?, {', '.join('?' for _ in row)})", [top + n, *row.values()])
    out = {}
    for m in methods:
        cal = evaluate.latest_calibration(mem, target, m)
        out[m] = None if cal is None else {
            "run_id": cal["run_id"], "comparison_id": cal["comparison_id"],
            "temperatures": cal["temperatures"],
            "floors": {r: s.get("floor") for r, s in cal["sets"].items()}}
    mem.close()
    return out


def describe(planned: list[Planned], rows: list[dict]) -> list[str]:
    """The rows a copy writes, one line each, for the command's output."""
    lines = []
    for p, row in zip(planned, rows):
        temps = ", ".join(f"{r} T {c['temperature']:.6f}"
                          + (f" floor {c['sets']['floor']:.6f}" if c.get("sets") else "")
                          for r, c in p.calibration.items())
        lines.append(f"  {row['backbone']} / {p.method}"
                     + (f" [place setting {p.place_setting}]" if p.place_setting else "")
                     + f": {temps}; from {json.dumps(p.source, sort_keys=True)}")
    return lines
