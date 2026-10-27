"""Per-find attributes for `mv paper-score` that are data, not model output (`mv paper-attrs`):
written once per scored set as a TSV that paper-score reads with `--attrs`. Everything is
measured on PUBLIC points only (record_public_places: obscured where iNat obscures, MO's
box centre), for the find and its references alike; the release's true points are never
read here.

- distance_band: great-circle km from the find's public point to the nearest reference
  record of its true species (the reference used to score it: val against train, a test
  release against train + val; an erratum given to the base release leaves its records
  out): <100, 100-300, 300-1500, >1500, or "no reference" / "no point".
- in_season (the leakage/stress experiment's definition, so v1 re-checks it like for
  like): over the true species' dated references, the share observed within SEASON_DAYS of
  the find's day of year (circular, any year): "core" when at least CORE_SHARE of them,
  "edge" when some but fewer, "off" when none; "no reference" / "no date" when it can't
  be said.
- ecoregion: the CEC Level I terrestrial ecoregion holding the find's public point
  (ecoregions.py; "(none)" outside every polygon, e.g. the Caribbean islands; "no point").

For E14's secondary rows (Steve, 27 Oct), when asked:
- inat_observation: "yes" when the find has an iNat observation (an iNat record, or one
  cross-posted to iNat), "none" otherwise (an MO-only find was never in iNat's training).
- created_at: when the find's observation was created (UTC ISO): iNat's, from the E31
  identification snapshot's created_at.tsv; MO's, from MO's public API (mopoints.py's
  anonymous, paced, cached read; nothing is ever written to MO); the earlier of the two for
  a cross-post; empty when unknown.
- rg_at_inat_export: "yes" / "no" / "unknown", Research Grade by identifications alone as it
  stood on a date (iNat CV's training export), from the snapshot (rg_by_identifications);
  a find with no iNat observation is "no".
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

EARTH_KM = 6371.0088
BANDS_KM = ((100, "<100"), (300, "100-300"), (1500, "300-1500"), (math.inf, ">1500"))
SEASON_DAYS = 30
CORE_SHARE = 0.5
FIELDS = ("record_key", "distance_band", "distance_km", "in_season", "ecoregion")
# Optional columns, written when computed (e.g. beit_vocab for E15).


def km_to_nearest(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> float:
    """Great-circle distance (haversine) to the nearest of the points."""
    p1, p2 = math.radians(lat), np.radians(lats)
    dphi = p2 - p1
    dl = np.radians(lons) - math.radians(lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))).min())


def band_of_km(km: float) -> str:
    return next(label for limit, label in BANDS_KM if km < limit)


def day_of_year(day: str | None) -> int | None:
    from .dates import real_date
    d = real_date(day)
    if not d:
        return None
    try:
        return date.fromisoformat(d[:10]).timetuple().tm_yday
    except ValueError:
        return None


def circular_days(a: int, b: np.ndarray) -> np.ndarray:
    d = np.abs(np.asarray(b) - a) % 366
    return np.minimum(d, 366 - d)


def attributes(scored: list[dict], refs: list[dict], public: dict[str, tuple],
               season_days: int = SEASON_DAYS) -> dict[str, dict]:
    """{record key: {distance_band, distance_km, in_season}} for the scored rows (dicts
    with record_key, species, observed_on) against the reference rows, from `public`
    {record key: (lat, lon) or (None, None)}."""
    # Pooled by answer key (PLAN rule 7): a species with its varieties and subspecies, a
    # hybrid code on its own; the nearest reference and the season use every pooled record.
    from .name_equiv import answer_key
    pts, doys, known = defaultdict(list), defaultdict(list), set()
    for r in refs:
        sp = answer_key(r.get("species") or "")
        if not sp:
            continue
        known.add(sp)
        lat, lon = public.get(str(r["record_key"]), (None, None))
        if lat is not None:
            pts[sp].append((lat, lon))
        d = day_of_year(r.get("observed_on"))
        if d is not None:
            doys[sp].append(d)
    arr = {sp: np.array(v, float) for sp, v in pts.items()}
    out = {}
    for r in scored:
        key, sp = str(r["record_key"]), answer_key(r.get("species") or "")
        lat, lon = public.get(key, (None, None))
        if not sp or sp not in known:
            band, km = "no reference", None
        elif lat is None:
            band, km = "no point", None
        elif sp not in arr:
            band, km = "no reference point", None
        else:
            km = km_to_nearest(lat, lon, arr[sp][:, 0], arr[sp][:, 1])
            band = band_of_km(km)
        d = day_of_year(r.get("observed_on"))
        if not sp or not doys.get(sp):
            season = "no reference"
        elif d is None:
            season = "no date"
        else:
            share = float((circular_days(d, np.array(doys[sp])) <= season_days).mean())
            season = "core" if share >= CORE_SHARE else "edge" if share > 0 else "off"
        out[key] = {"distance_band": band, "distance_km": None if km is None else round(km, 1),
                    "in_season": season}
    return out


def in_vocabulary(species: str, classes: set[str], credited: set[tuple[str, str]]) -> bool:
    """A true species (taken at species rank) is in a published model's vocabulary when it
    is one of the model's classes by name (heldout.name_key), or a credited pair of the
    reviewed crosswalk joins one of its classes to it (the same rule E15 scores by)."""
    from .heldout import name_key
    from .paper_outcomes import species_rank
    sp = species_rank(species)
    if not sp:
        return False
    if name_key(sp) in classes:
        return True
    return any(our in (species, sp) and name_key(theirs) in classes
               for theirs, our in credited)


def model_classes(conn, short: str) -> set[str]:
    """A published model's species classes as Vision names them (name_key)."""
    from .heldout import Labeller, name_key
    from .replications.fungitastic import published as external
    m = external.model_for(short)
    cn = external.vision_names(external.read_labels(m), Labeller(conn))
    return {name_key(n) for n in cn.species if n and len(n.split()) > 1}


def credited_pairs(review: Path) -> set[tuple[str, str]]:
    with open(review, encoding="utf-8") as fh:
        return {(r["their_answer"], r["our_label"]) for r in csv.DictReader(fh, delimiter="\t")
                if (r.get("review") or "").strip().lower() == "credit"}


def public_points(rel) -> dict[str, tuple]:
    if not rel.has_private:
        raise ValueError(f"{rel.id} has no private.sqlite here: no public points")
    return {str(k): ((lat / 1e6, lon / 1e6) if lat is not None and kind != "none"
                     else (None, None))
            for k, lat, lon, kind in rel.private.execute(
                "select record_key, public_lat_micro, public_lon_micro, public_kind "
                "from record_public_places")}


def for_release(base, test, regions=None) -> dict[str, dict]:
    """The attributes of `test` ("val": base's val against its train; or a test release
    of base: its test split against base's train + val). `regions`: ecoregions.Regions."""
    from .paper_score import scored_and_reference
    scored_rel, split, ref_splits, _sealed = scored_and_reference(base, test)
    public = public_points(base)
    if scored_rel is not base:
        public.update(public_points(scored_rel))
    scored = [dict(r) for r in scored_rel.records(split)]
    out = attributes(scored, [dict(r) for r in base.records(ref_splits)], public)
    keys = [str(r["record_key"]) for r in scored]
    with_point = [k for k in keys if public.get(k, (None, None))[0] is not None]
    names = (regions.lookup([public[k][0] for k in with_point], [public[k][1] for k in with_point])
             if regions is not None and with_point else [])
    region_of = dict(zip(with_point, names))
    for k in keys:
        out[k]["ecoregion"] = (region_of.get(k, "no point") if regions is not None
                               else "not computed")
    return out


RG_AGREEMENT = 2 / 3
SPECIES_LEVEL = 10


def rg_by_identifications(identifications: list[dict], as_of: str) -> bool:
    """iNat's Research Grade read from identifications alone, as it stood at `as_of` (an ISO
    date or time): each identifier's latest identification made before then; the community
    taxon is a taxon more than 2/3 of them place the find in, with at least two agreeing (an
    identification of one of its ancestors does not count against it); Research Grade when
    such a taxon is at species rank or finer. The snapshot holds no data-quality votes,
    captive flags or date/place checks, so this is an approximation, named as such."""
    latest: dict = {}
    for i in sorted(identifications, key=lambda r: (r.get("created_at") or "", r.get("id") or 0)):
        if i.get("created_at") and i["created_at"] < as_of and i.get("taxon_path"):
            latest[i["user_id"]] = i
    ids = list(latest.values())
    paths = [list(i["taxon_path"]) for i in ids]
    level = {i["taxon_path"][-1]: i.get("rank_level") for i in ids}
    for t in {t for p in paths for t in p}:
        if level.get(t) is None or level[t] > SPECIES_LEVEL:
            continue
        agree = sum(t in p for p in paths)
        line = next(p[:p.index(t) + 1] for p in paths if t in p)
        disagree = sum(1 for p in paths if t not in p and p[-1] not in line)
        if agree >= 2 and agree / (agree + disagree) > RG_AGREEMENT:
            return True
    return False


def read_created_tsv(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """({record key: created_at}, {iNat observation id: record key}) from the E31
    snapshot's created_at.tsv."""
    created, key_of = {}, {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            created[str(r["record_key"])] = r.get("created_at") or ""
            key_of[str(r["observation_id"])] = str(r["record_key"])
    return created, key_of


def mo_created_at(mo_ids: list[int], raw_dir: Path, fetch=None, offline: bool = False,
                  log=print) -> tuple[dict[int, str], dict]:
    """MO observation number -> created_at, from MO's public API answers kept in `raw_dir`
    (mopoints.py's cache, data/raw/mo/observations-*.json.gz); numbers not cached are read
    anonymously at MO's pace (mopoints.fetch_batch, unless `offline`) and cached the same
    way. One MO doesn't have is "" (missing). Read only: nothing is written to MO."""
    import gzip
    import json
    from datetime import datetime, timezone
    raw_dir = Path(raw_dir)
    got: dict[int, str] = {}
    for f in sorted(raw_dir.glob("observations-*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for o in json.load(fh) or []:
                if o.get("id") is not None and o.get("created_at"):
                    got[int(o["id"])] = o["created_at"]
    want = sorted(set(int(i) for i in mo_ids))
    todo = [i for i in want if i not in got]
    stats = {"asked": len(want), "cached": len(want) - len(todo), "fetched": 0, "missing": 0,
             "offline": offline}
    if todo and not offline:
        import requests

        from . import config
        from .inat import chunks
        from .mo import MissingObservation, Pacer
        from .mopoints import BATCH, fetch_batch
        fetch = fetch or fetch_batch
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
        pacer = Pacer()
        raw_dir.mkdir(parents=True, exist_ok=True)
        for batch in chunks(todo, BATCH):
            ask = list(batch)
            while True:
                try:
                    answers = fetch(session, ask, pacer) if ask else []
                    break
                except MissingObservation as e:
                    if e.mo_id not in ask:
                        raise
                    ask.remove(e.mo_id)
            day = datetime.now(timezone.utc).isoformat(timespec="seconds")[:10]
            with gzip.open(raw_dir / f"observations-{batch[0]}-{day}.json.gz", "wt",
                           encoding="utf-8") as fh:
                json.dump(answers, fh)
            for o in answers:
                if o.get("created_at"):
                    got[int(o["id"])] = o["created_at"]
                    stats["fetched"] += 1
            log(f"  MO created_at: {stats['fetched']:,} read of {len(todo):,}")
    out = {i: got.get(i, "") for i in want}
    stats["missing"] = sum(1 for v in out.values() if not v)
    return out, stats


def secondary_attrs(scored: list[dict], inat_created: dict[str, str],
                    mo_created: dict[int, str], snapshot: dict[str, dict] | None = None,
                    rg_as_of: str | None = None) -> dict[str, dict]:
    """Per scored record: inat_observation, created_at (the earlier of its iNat and MO
    creation; "" unknown) and, with a snapshot ({record key: snapshot row}) and a date,
    rg_at_inat_export ("no" when the find has no iNat observation; "unknown" when its iNat
    observation is not in the snapshot or is gone)."""
    out = {}
    for r in scored:
        k = str(r["record_key"])
        has_inat = r.get("source") == "inat" or bool(r.get("inat_uuid"))
        times = [inat_created.get(k) or ""]
        if r.get("source") == "mo" and str(r.get("source_id") or "").isdigit():
            times.append(mo_created.get(int(r["source_id"])) or "")
        times = [t for t in times if t]
        a = {"inat_observation": "yes" if has_inat else "none",
             "created_at": min(times) if times else ""}
        if rg_as_of is not None:
            row = (snapshot or {}).get(k)
            a["rg_at_inat_export"] = ("no" if not has_inat else
                                      "unknown" if row is None or row.get("missing") else
                                      "yes" if rg_by_identifications(
                                          row.get("identifications") or [], rg_as_of) else "no")
        out[k] = a
    return out


def sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, attrs: dict[str, dict]) -> Path:
    extra = sorted({c for a in attrs.values() for c in a} - set(FIELDS))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow([*FIELDS, *extra])
        for k in sorted(attrs):
            a = attrs[k]
            w.writerow([k, a["distance_band"], "" if a["distance_km"] is None else a["distance_km"],
                        a["in_season"], a.get("ecoregion", "not computed"),
                        *[a.get(c, "") for c in extra]])
    return path
