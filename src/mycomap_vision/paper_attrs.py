"""Per-find attributes for `mv paper-score` that are data, not model output (`mv paper-attrs`):
written once per scored set as a TSV that paper-score reads with `--attrs`. Everything is
measured on PUBLIC points only (record_public_places: obscured where iNat obscures, MO's
box centre), for the find and its references alike; the release's true points are never
read here.

- distance_band: great-circle km from the find's public point to the nearest reference
  record of its true species (the reference used to score it: val against train, a test
  release against train + val; an erratum given to the base release leaves its records
  out): <100, 100-300, 300-1500, >1500, or "no reference" / "no point".
- in_season: "in" when a reference record of the true species was observed within
  SEASON_DAYS of the find's day of year (circular, any year), else "off"; "no reference" /
  "no date" when it can't be said.
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
FIELDS = ("record_key", "distance_band", "distance_km", "in_season")


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
            season = "in" if (circular_days(d, np.array(doys[sp])) <= season_days).any() else "off"
        out[key] = {"distance_band": band, "distance_km": None if km is None else round(km, 1),
                    "in_season": season}
    return out


def public_points(rel) -> dict[str, tuple]:
    if not rel.has_private:
        raise ValueError(f"{rel.id} has no private.sqlite here: no public points")
    return {str(k): ((lat / 1e6, lon / 1e6) if lat is not None and kind != "none"
                     else (None, None))
            for k, lat, lon, kind in rel.private.execute(
                "select record_key, public_lat_micro, public_lon_micro, public_kind "
                "from record_public_places")}


def for_release(base, test) -> dict[str, dict]:
    """The attributes of `test` ("val": base's val against its train; or a test release
    of base: its test split against base's train + val)."""
    from .paper_score import scored_and_reference
    scored_rel, split, ref_splits, _sealed = scored_and_reference(base, test)
    public = public_points(base)
    if scored_rel is not base:
        public.update(public_points(scored_rel))
    return attributes([dict(r) for r in scored_rel.records(split)],
                      [dict(r) for r in base.records(ref_splits)], public)


def write(path: Path, attrs: dict[str, dict]) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(FIELDS)
        for k in sorted(attrs):
            a = attrs[k]
            w.writerow([k, a["distance_band"], "" if a["distance_km"] is None else a["distance_km"],
                        a["in_season"]])
    return path
