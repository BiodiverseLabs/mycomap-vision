"""Predictions made before the DNA answer exists, checked once it does.

Candidates are records on mycomap.org that carry a sequence but haven't been
assessed in any project yet. Each is identified from its iNat photos by a chosen
model, and the prediction is saved with its time and code version. When the
record later turns green, the saved prediction is compared with the DNA-
validated name. Only predictions made before the record first appeared green
count, so nothing can have seen the answer.

Photos are fetched from iNat and processed in memory, never stored. The place
used is iNat's public (possibly obscured) location.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import requests

from . import config, guests, names, taxonomy
from .embed import decode
from .inat import parse_observation
from .licenses import sized_url, taken_down
from .prior import Context
from .ratelimit import MinInterval
from .records import is_north_america, parse_export

CANDIDATE_SQL = """
select row_to_json(t)::text from (
  select observation_id, scientific_name, continent, country, latitude, longitude
  from observations
  where coalesce(validation_status_1, '') not in ('yes', 'no')
    and coalesce(validation_status_2, '') not in ('yes', 'no')
    and coalesce(validation_status_3, '') not in ('yes', 'no')
    and observation_id ~ '^[0-9]+$'
  order by observation_id
) t
"""

SCHEMA = """
create table if not exists candidates (
  observation_id  text primary key,
  org_name        text,              -- the name on .org before validation (not used to predict)
  north_america   integer not null,
  first_seen_at   text not null,
  last_seen_at    text not null
);
create table if not exists predictions (
  observation_id  text not null,
  backbone        text not null,
  method          text not null,
  predicted_at    text not null,
  code_version    text,
  reference_records integer,
  photos          integer,
  result_json     text not null,     -- per rank: top candidates with confidence
  primary key (observation_id, backbone, method)
);
"""

RANKS = ("family", "genus", "species")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_candidates(conn: sqlite3.Connection, rows: list[dict], seen_at: str) -> int:
    conn.executescript(SCHEMA)
    with conn:
        for r in rows:
            lat = r.get("latitude")
            lon = r.get("longitude")
            na = int(is_north_america(r.get("continent"), r.get("country"),
                                      float(lat) if lat not in (None, "") else None,
                                      float(lon) if lon not in (None, "") else None))
            conn.execute(
                "insert into candidates values (?, ?, ?, ?, ?) on conflict(observation_id) do "
                "update set org_name = excluded.org_name, last_seen_at = excluded.last_seen_at",
                (str(r["observation_id"]), r.get("scientific_name"), na, seen_at, seen_at))
    return len(rows)


def export_candidates(conn: sqlite3.Connection, fetch: Callable[[str], str] | None = None) -> int:
    from .records import fetch_export
    text = (fetch or fetch_export)(CANDIDATE_SQL)
    return save_candidates(conn, parse_export(text), now_iso())


def unpredicted(conn: sqlite3.Connection, backbone: str, method: str,
                limit: int | None = None, seen_since: str | None = None) -> list[str]:
    """North American candidates, still not green, with no prediction from this model.
    `seen_since`: only candidates an export at or after that time still listed (a
    record rejected since then stays in the table but is no longer pending)."""
    conn.executescript(SCHEMA)
    sql = """
      select c.observation_id from candidates c
      left join predictions p on p.observation_id = c.observation_id
        and p.backbone = ? and p.method = ?
      left join records r on r.observation_id = c.observation_id
      where c.north_america = 1 and p.observation_id is null and r.observation_id is null
        and (? is null or c.last_seen_at >= ?)
      order by cast(c.observation_id as integer) desc
    """
    if limit:
        sql += f" limit {int(limit)}"
    return [r[0] for r in conn.execute(sql, (backbone, method, seen_since, seen_since))]


class PhotoFetcher:
    """iNat observations and their medium photos, fetched politely, kept in memory."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.api = MinInterval(1.0)
        self.media = MinInterval(0.25)

    def observations(self, ids: list[str]) -> dict[str, dict]:
        from .inat import fetch_batch
        out = {}
        for i in range(0, len(ids), 200):
            for obs in fetch_batch(self.session, ids[i:i + 200], self.api):
                out[str(obs["id"])] = obs
        return out

    def photo(self, url: str) -> bytes | None:
        if taken_down(url):
            return None
        self.media.wait()
        try:
            r = self.session.get(sized_url(url, "medium"), timeout=60)
        except (requests.RequestException, ValueError):
            return None
        return r.content if r.status_code == 200 else None


def summarise(result: dict, top: int = 5) -> dict:
    """Keep what the report needs from an identify result: the top candidates per rank."""
    return {rank: [{"name": c["name"], "confidence": c["confidence"]}
                   for c in result["ranks"][rank][:top]] for rank in RANKS}


def predict_pending(conn: sqlite3.Connection, identifier, backbone_model, ids: list[str],
                    fetcher, log=print) -> dict:
    """Identify each candidate from its iNat photos and save the prediction."""
    conn.executescript(SCHEMA)
    stats = Counter()
    observations = fetcher.observations(ids)
    for i, oid in enumerate(ids, 1):
        obs = observations.get(oid)
        if obs is None:
            stats["gone from iNat"] += 1
            continue
        row, photos = parse_observation(obs)
        images = []
        for p in photos:
            body = fetcher.photo(p["source_url"])
            if body:
                try:
                    images.append(decode(body))
                except Exception:
                    pass
        if not images:
            stats["no usable photos"] += 1
            continue
        ctx = Context(row["inat_latitude"], row["inat_longitude"], row["observed_on"])
        result = identifier.identify(backbone_model, images, context=ctx)
        with conn:
            conn.execute(
                "insert or replace into predictions values (?, ?, ?, ?, ?, ?, ?, ?)",
                (oid, identifier.backbone, identifier.method, now_iso(), config.code_version(),
                 identifier.records, len(images), json.dumps(summarise(result))))
        stats["predicted"] += 1
        if i % 25 == 0:
            log(f"  {i}/{len(ids)} candidates, {stats['predicted']} predicted")
    return dict(stats)


def report(conn: sqlite3.Connection) -> list[dict]:
    """Per model: how its advance predictions fared once the DNA answer came in.

    A prediction counts only if made before the record first appeared green here
    (records.first_seen_at), so the answer can't have been known. Names are compared
    by label (names.py), so an answer spelled another way than the DNA name is right.
    """
    conn.executescript(SCHEMA)
    rows = conn.execute("""
      select p.backbone, p.method, p.result_json, r.scientific_name, r.genus, r.family,
             p.predicted_at, r.first_seen_at
      from predictions p join records r on r.observation_id = p.observation_id
      where r.first_seen_at is not null and p.predicted_at < r.first_seen_at
        and r.label_conflict = 0
    """).fetchall()
    totals = Counter()
    for backbone, method in conn.execute("select backbone, method from predictions"):
        totals[(backbone, method)] += 1
    by_model: dict[tuple, Counter] = {}
    conf: dict[tuple, list] = {}
    results = [json.loads(r[2]) for r in rows]
    # A model that predicted before a name was respelled answers with the old spelling.
    labels = names.manifest_labels(conn, extra=[top[0]["name"] for result in results
                                                if (top := result.get("species"))])
    label = lambda name: labels.get(name, name)  # noqa: E731
    tax = taxonomy.for_manifest(conn)
    for (backbone, method, _json, name, genus, family, _p, _f), result in zip(rows, results):
        key = (backbone, method)
        c = by_model.setdefault(key, Counter())
        # The same labels as training and comparisons: a one-word name has no species
        # answer to check, and family is iNat's for the genus when the cache has it.
        lab = taxonomy.labels_for(label(name) or "", genus, family,
                                  (label(name) or "").strip() != (name or "").strip(), tax)
        if guests.excluded(lab.genus):       # the DNA name is a guest, not the photo's fungus
            continue
        species = lab.species
        truth = {"species": species, "genus": lab.genus, "family": lab.family}
        c["resolved"] += 1
        for rank in RANKS:
            top = result.get(rank) or []
            if truth[rank] and top:
                c[f"{rank}_n"] += 1
                c[f"{rank}_top1"] += label(top[0]["name"]) == truth[rank]
        top_sp = (result.get("species") or [None])[0]
        if top_sp and species:
            conf.setdefault(key, []).append((top_sp["confidence"],
                                             label(top_sp["name"]) == species))
    out = []
    for key in sorted(set(totals) | set(by_model)):
        c = by_model.get(key, Counter())
        pairs = conf.get(key, [])
        out.append({
            "backbone": key[0], "method": key[1], "predicted": totals[key],
            "resolved": c["resolved"],
            **{f"{rank}_top1": round(c[f"{rank}_top1"] / c[f"{rank}_n"], 4) if c[f"{rank}_n"]
               else None for rank in RANKS},
            # Calibration check: stated confidence vs how often the top species was right.
            "mean_species_confidence": round(float(np.mean([p[0] for p in pairs])), 4)
            if pairs else None,
        })
    return out
