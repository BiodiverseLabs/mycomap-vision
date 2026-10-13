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
    pts, doys, known = defaultdict(list), defaultdict(list), set()
    for r in refs:
        sp = r.get("species")
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
        key, sp = str(r["record_key"]), r.get("species") or ""
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
