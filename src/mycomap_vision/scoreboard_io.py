"""Carry scoreboard rows from one machine to another.

The iNat baseline runs on the laptop (it needs a person's 24-hour iNat token and the
laptop's photo access), but the site shows the scoreboard of the server box. So:

    laptop:  mv scoreboard-export --comparison <id> --backbone external:inat-cv > inat.json
    box:     mv scoreboard-import - < inat.json

An import only joins a comparison the box already has, tested on the same records (its
record set, cutoff and number of test records must match the box's rows of that
comparison): numbers measured on other records next to ours would mislead. Importing a
model's rows again replaces them, so a corrected run can follow a broken one.
"""

from __future__ import annotations

import json
import sqlite3

from .evaluate import SCOREBOARD_SCHEMA

KIND = "mycomap-vision scoreboard rows"
VERSION = 1
COLUMNS = ["comparison_id", "backbone", "method", "cutoff", "test_days", "n_reference", "n_test",
           "record_set", "species_top1", "genus_top1", "family_top1", "species_top1_first_photo",
           "report_json", "code_version", "created_at"]
SAME_TEST = ("record_set", "cutoff", "n_test")


class ImportRefused(ValueError):
    """The rows were not imported; nothing was changed."""


def export_runs(conn: sqlite3.Connection, comparison_id: str, backbone: str) -> dict:
    conn.executescript(SCOREBOARD_SCHEMA)
    rows = conn.execute(f"select {', '.join(COLUMNS)} from eval_runs where comparison_id = ? "
                        "and backbone = ? order by id", (comparison_id, backbone)).fetchall()
    if not rows:
        raise ValueError(f"no {backbone} rows in comparison {comparison_id}")
    return {"kind": KIND, "version": VERSION, "comparison_id": comparison_id,
            "backbone": backbone, "rows": [dict(zip(COLUMNS, r)) for r in rows]}


def import_runs(conn: sqlite3.Connection, payload: dict) -> dict:
    """Add the rows to this machine's scoreboard (see the module notes for what is refused)."""
    conn.executescript(SCOREBOARD_SCHEMA)
    if not isinstance(payload, dict) or payload.get("kind") != KIND \
            or payload.get("version") != VERSION:
        raise ImportRefused("not a scoreboard export (mv scoreboard-export)")
    cid, backbone, rows = payload.get("comparison_id"), payload.get("backbone"), payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ImportRefused("the export holds no rows")
    for r in rows:
        if not isinstance(r, dict) or set(r) != set(COLUMNS):
            raise ImportRefused("a row does not have the scoreboard's columns")
        if r["comparison_id"] != cid or r["backbone"] != backbone:
            raise ImportRefused("the rows are not all of one comparison and model")
        json.loads(r["report_json"])
    here = conn.execute(f"select {', '.join(SAME_TEST)} from eval_runs where comparison_id = ? "
                        "and backbone <> ? limit 1", (cid, backbone)).fetchone()
    if here is None:
        raise ImportRefused(f"comparison {cid} is not on this machine; import rows only into "
                            "a comparison it already has")
    theirs = tuple(rows[0][k] for k in SAME_TEST)
    if any(tuple(r[k] for k in SAME_TEST) != theirs for r in rows) or tuple(here) != theirs:
        raise ImportRefused(f"the rows were measured on other records than comparison {cid} "
                            f"here ({dict(zip(SAME_TEST, theirs))} vs "
                            f"{dict(zip(SAME_TEST, here))})")
    with conn:
        replaced = conn.execute("delete from eval_runs where comparison_id = ? and backbone = ?",
                                (cid, backbone)).rowcount
        conn.executemany(f"insert into eval_runs ({', '.join(COLUMNS)}) values "
                         f"({', '.join('?' for _ in COLUMNS)})",
                         [[r[c] for c in COLUMNS] for r in rows])
    return {"comparison_id": cid, "backbone": backbone, "imported": len(rows),
            "replaced": replaced}
