"""Export the DNA-validated ("green in a project") records from mycomap.org.

The source is the live .org database through a read-only SQL route reached over
SSH (host named by MV_ORG_SQL_SSH_HOST): it takes one SQL statement and answers a
header line followed by rows.

A record is a training candidate when any of its three flattened validation slots
says 'yes'. Records validated in a fourth or later project and in none of the first
three are missed; that gap is small and noted in the README.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from . import config
from .dates import real_date

EXPORT_SQL = """
select row_to_json(t)::text from (
  select observation_id, scientific_name, phylum, class, "order", family, genus, species,
         infraspecies, latitude, longitude, observed_on, state, country, continent,
         sequence_id,
         validation_project_1, validation_status_1, validation_date_1,
         validation_project_2, validation_status_2, validation_date_2,
         validation_project_3, validation_status_3, validation_date_3
  from observations
  where 'yes' in (coalesce(validation_status_1, ''), coalesce(validation_status_2, ''),
                  coalesce(validation_status_3, ''))
  order by observation_id
) t
"""

# North America as the project scopes it: the continent plus Central America and
# the Caribbean. Used only when .org's own continent column is blank.
NA_COUNTRIES = {
    "US", "CA", "MX", "GL", "BM", "PM",
    "GT", "BZ", "SV", "HN", "NI", "CR", "PA",
    "PR", "VI", "VG", "CU", "DO", "HT", "JM", "BS", "TC", "KY", "AG", "BB", "DM",
    "GD", "KN", "LC", "VC", "TT", "AW", "CW", "SX", "MF", "BL", "GP", "MQ", "AI", "MS",
}


def is_north_america(continent: str | None, country: str | None,
                     lat: float | None, lng: float | None) -> bool:
    if continent:
        return continent == "North America"
    if country:
        return country.upper() in NA_COUNTRIES
    if lat is not None and lng is not None:
        return 7.0 <= lat <= 84.0 and -170.0 <= lng <= -50.0
    return False


_MDY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})")


def parse_validation_date(s: str | None) -> str | None:
    """.org stores validation dates as text, usually 'M/D/YYYY ...'. Returns ISO or None
    (also for the 1970-01-01 placeholder, see dates.py)."""
    if not s:
        return None
    s = s.strip()
    m = _MDY.match(s)
    try:
        if m:
            iso = date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat()
        else:
            iso = date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None
    return real_date(iso)


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def green_slots(row: dict) -> list[tuple[str | None, str | None]]:
    """(project, ISO date) for each validation slot marked 'yes'."""
    out = []
    for i in (1, 2, 3):
        if (row.get(f"validation_status_{i}") or "").strip().lower() == "yes":
            out.append((row.get(f"validation_project_{i}"),
                        parse_validation_date(row.get(f"validation_date_{i}"))))
    return out


def build_records(rows: list[dict], exported_at: str) -> list[dict]:
    """Collapse export rows to one record per observation_id.

    .org can hold several rows for one observation (e.g. two sequences). When they
    disagree on the name the record is flagged as a label conflict and kept out of
    training until someone resolves it.
    """
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        oid = str(r.get("observation_id") or "").strip()
        if oid:
            by_id[oid].append(r)

    records = []
    for oid, group in by_id.items():
        first = group[0]
        names = sorted({(g.get("scientific_name") or "").strip() for g in group} - {""})
        projects: list[str] = []
        dates: list[str] = []
        for g in group:
            for project, d in green_slots(g):
                if project and project not in projects:
                    projects.append(project)
                if d:
                    dates.append(d)
        lat, lng = _num(first.get("latitude")), _num(first.get("longitude"))
        seq = first.get("sequence_id")
        records.append({
            "observation_id": oid,
            "source": "inat" if oid.isdigit() else "other",
            "scientific_name": names[0] if len(names) == 1 else (first.get("scientific_name") or None),
            "phylum": first.get("phylum"), "class": first.get("class"),
            "order": first.get("order"), "family": first.get("family"),
            "genus": first.get("genus"), "species": first.get("species"),
            "infraspecies": first.get("infraspecies"),
            "latitude": lat, "longitude": lng,
            # 1970-01-01 is .org's placeholder for "no date": stored as none, so
            # evaluation falls back to iNat's date.
            "observed_on": real_date(first.get("observed_on")),
            "state": first.get("state"), "country": first.get("country"),
            "continent": first.get("continent"),
            "north_america": int(is_north_america(first.get("continent"), first.get("country"), lat, lng)),
            "sequence_id": int(seq) if seq not in (None, "") else None,
            "green_projects": json.dumps(projects),
            "validated_on": min(dates) if dates else None,
            "label_conflict": int(len(names) > 1),
            "names_json": json.dumps(names),
            "exported_at": exported_at,
        })
    return records


def parse_export(text: str) -> list[dict]:
    """The SQL route answers a header line, then one JSON object per line."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def fetch_export(sql: str = EXPORT_SQL) -> str:
    """Run one read-only statement on the .org route; the answer is a header line and
    one JSON object per line (see parse_export)."""
    result = subprocess.run(
        ["ssh", config.required("MV_ORG_SQL_SSH_HOST", "the read-only SQL route for records"),
         " ".join(sql.split())],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"export failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


_COLUMNS = [
    "observation_id", "source", "scientific_name", "phylum", "class", "order", "family",
    "genus", "species", "infraspecies", "latitude", "longitude", "observed_on", "state",
    "country", "continent", "north_america", "sequence_id", "green_projects", "validated_on",
    "label_conflict", "names_json", "exported_at",
]


def save_records(conn: sqlite3.Connection, records: list[dict]) -> dict:
    """Upsert the export. Records no longer green on .org leave the candidate list;
    their iNat metadata and photos stay in the manifest in case they turn green again.

    Returns what changed: records new to the list (stamped first_seen_at with this
    export), records removed, and records whose name changed on .org (a new label).
    """
    before = dict(conn.execute("select observation_id, scientific_name from records"))
    cols = ", ".join(f'"{c}"' for c in _COLUMNS) + ', "first_seen_at"'
    marks = ", ".join("?" for _ in _COLUMNS) + ", ?"
    # first_seen_at is set on insert only, never on update.
    updates = ", ".join(f'"{c}" = excluded."{c}"' for c in _COLUMNS if c != "observation_id")
    with conn:
        conn.executemany(
            f"insert into records ({cols}) values ({marks}) "
            f"on conflict(observation_id) do update set {updates}",
            [[r[c] for c in _COLUMNS] + [r["exported_at"]] for r in records],
        )
        keep = {r["observation_id"] for r in records}
        stale = [oid for oid in before if oid not in keep]
        conn.executemany("delete from records where observation_id = ?", [(s,) for s in stale])
    return {
        "new": sum(r["observation_id"] not in before for r in records),
        "removed": len(stale),
        "renamed": sum(r["observation_id"] in before
                       and before[r["observation_id"]] != r["scientific_name"] for r in records),
    }


def export_records(conn: sqlite3.Connection) -> dict:
    config.ensure_dirs()
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = fetch_export()
    raw_path = config.RAW_DIR / f"records-{exported_at[:10]}.jsonl"
    Path(raw_path).write_text(text, encoding="utf-8")
    rows = parse_export(text)
    records = build_records(rows, exported_at)
    changes = save_records(conn, records)
    return {
        "exported_at": exported_at,
        **changes,
        "rows": len(rows),
        "records": len(records),
        "north_america": sum(r["north_america"] for r in records),
        "inat": sum(r["source"] == "inat" for r in records),
        "label_conflicts": sum(r["label_conflict"] for r in records),
        "raw_file": str(raw_path),
    }
