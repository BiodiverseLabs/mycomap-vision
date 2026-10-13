"""Stronger place penalties for nearest+mean+place, tuned on dataset release v1's val.

Steve, 2026-10-13: place should push far species down. That reverses his 2026-10-10
"never bury far finds", so every setting here is reported with what it costs records
whose species is known only far away (the distance bands), and the far-find cue
(placeprior.PlaceCue) stays beside every candidate whatever the setting.

The grid is declared here before any result was seen (GRID_DECLARED). Every setting keeps
the served boost (placeprior.BOOST: DNA place boost 75 km capped at log 5, season 0.5 at
10 days, -2 beyond 1,500 km by the iNat out-of-range rule) and adds (placeprior's
named-setting keys, so a chosen setting is served exactly as tuned):

- shrink 3 (served) / 10 / 30: a sparse species' place boost pulled to its genus's
  (the 3-record species boosted +1.68 by 2 nearby records, the Coprinus find);
- a graded distance penalty: slope 0 / 1 / 2 / 4 / 8 nats per 1,000 km beyond
  start 300 / 500 km from the candidate's nearest known record, capped at 2 / 4 / 8 nats;
  known = DNA records, or DNA records and iNat open-data finds; damped by n / (n + k),
  k 0 / 5, n = the candidate's DNA records with a place;
- a season penalty 0 / 1 / 2 nats when out of a known season (the cue's in_season false).

Selection (as exp/prior-tuning): species top-1, then genus top-1, then the gentler
setting; 5-fold cross-validation grouped by observer, each fold scored with the setting
chosen on the other four, so every reported number is out of fold. Distance bands use the
true species' nearest train DNA record. Nothing here reads or writes a coordinate to an
output: components hold scores and distances to candidates only, reported as bands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from . import placeprior as pp
from .prior import Context, RangeSeasonPrior

PHOTO_TEMPERATURE = pp.PHOTO_TEMPERATURE
TOP_K = 1000
FOLDS = 5
SEED = 20261013
GRID_DECLARED = "2026-10-13, before any place-penalty result (feat/place-scoring)"
MARGIN = 0.02            # Steve's 300-1,500 km non-inferiority margin (2026-10-12)

SHRINKS = (3.0, 10.0, 30.0)
SEASON_PENALTIES = (0.0, 1.0, 2.0)
PENALTY_GRID = {
    "pen_start_km": (300.0, 500.0),
    "pen_slope": (1.0, 2.0, 4.0, 8.0),
    "pen_cap": (2.0, 4.0, 8.0),
    "pen_known": ("dna", "dna+inat"),
    "damp_k": (0.0, 5.0),
}
SLOPES = (0.0,) + PENALTY_GRID["pen_slope"]
NONE = {"family": "none"}                         # P0: photos only
SERVED = {"family": "place", **pp.NO_EXTRAS}      # P1: what is served today

BANDS = [(0, 100, "< 100 km"), (100, 300, "100-300 km"), (300, 500, "300-500 km"),
         (500, 1500, "500-1,500 km"), (1500, math.inf, "> 1,500 km")]
RULE_BAND = (300.0, 1500.0)                       # the band Steve's rule judges


def settings() -> list[dict]:
    """Every declared setting, in a fixed order (index 0 = the served one)."""
    out = []
    for shrink, season in product(SHRINKS, SEASON_PENALTIES):
        base = {**SERVED, "shrink": shrink, "season_penalty": season}
        out.append(base)
        for start, slope, cap, known, k in product(*PENALTY_GRID.values()):
            out.append({**base, "pen_start_km": start, "pen_slope": slope, "pen_cap": cap,
                        "pen_known": known, "damp_k": k})
    return out


def gentleness(s: dict) -> float:
    """Smaller is closer to what is served: the tie-break after species and genus."""
    if s["family"] == "none":
        return -1.0
    return (s["pen_slope"] * s["pen_cap"] / 8.0 + s["season_penalty"]
            + abs(math.log(s["shrink"] / 3.0)) + (0.1 if s["pen_known"] != "dna" else 0.0)
            - 0.05 * s["damp_k"] / 5.0 + (s["pen_start_km"] == 300.0) * 0.01 * s["pen_slope"])


def describe(s: dict) -> str:
    if s["family"] == "none":
        return "photos only"
    parts = [f"shrink {s['shrink']:g}"]
    if s["pen_slope"] and s["pen_cap"]:
        parts.append(f"-{s['pen_slope']:g}/1,000 km past {s['pen_start_km']:g} km, cap "
                     f"{s['pen_cap']:g}, known {s['pen_known']}"
                     + (f", damped k {s['damp_k']:g}" if s["damp_k"] else ""))
    if s["season_penalty"]:
        parts.append(f"off season -{s['season_penalty']:g}")
    return "; ".join(parts)


# --- components --------------------------------------------------------------------------

@dataclass
class Components:
    """Per record (n) and candidate (K): photo score and every term the grid needs."""
    S: np.ndarray                     # nearest + species average score
    cand: np.ndarray                  # group index (-1 = padding)
    SP: np.ndarray                    # candidate is a species group
    GG: np.ndarray                    # candidate's genus id (-1 none)
    truth_at: np.ndarray              # (n,) candidate column of the true species, -1 none
    has_sp: np.ndarray                # (n,) the answer key names a species
    truth_g: np.ndarray               # (n,) true genus id (-3 none)
    boost: dict = field(default_factory=dict)     # shrink -> (n, K) DNA place log-ratio
    season: dict = field(default_factory=dict)    # shrink -> (n, K) DNA season log-ratio
    berth: np.ndarray | None = None               # (n, K) bool: served out-of-range rule
    km_dna: np.ndarray | None = None              # (n, K) km to nearest DNA cell (inf none)
    km_inat: np.ndarray | None = None             # (n, K) km to nearest iNat find (inf none)
    n_place: np.ndarray | None = None             # (n, K) DNA records with a place
    off_season: np.ndarray | None = None          # (n, K) bool


def candidates(scores: np.ndarray, truth: int, k: int = TOP_K) -> np.ndarray:
    """The k best groups by photo score, best first, plus the truth when it isn't one."""
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    if truth >= 0 and truth not in set(top.tolist()):
        top = np.append(top, truth)
    return top


class TermBuilder:
    """The terms of one record's candidates, from the reference (train) records, made by
    the serving code itself (placeprior) so the tuned setting is the served one."""

    def __init__(self, records, species: list[str], store=None):
        from .occprior import OccParams, OccurrencePrior
        self.dna = {}
        for sh in SHRINKS:
            p = RangeSeasonPrior(bandwidth_km=pp.BOOST["place_km"],
                                 bandwidth_days=pp.BOOST["season_days"], cap=pp.BOOST["cap"],
                                 shrink=sh)
            p.fit(records, species)
            self.dna[sh] = p
        # The penalties' tables (snapped DNA cells, record counts, dated days).
        self.extra = pp.PlaceBoost(store=None, setting=pp.DEFAULT_SETTING)
        self.extra.n_groups = len(species)
        self.extra._fit_extras(records, species)
        self.occ = None
        if store is not None:
            # As served (placeprior.PlaceBoost.fit), but keeping the pair counts so a
            # record's own iNat observation can be taken back out of its nearest find.
            self.occ = OccurrencePrior(store, OccParams(
                radius_km=pp.BOOST["radius_km"], genus_rule=False,
                min_dna_effort=pp.BOOST["min_dna_effort"], density_weight=0.0,
                season_weight=0.0), out_of_range_only=False)
            self.occ.fit(records, species)

    def terms(self, ctx: Context) -> dict[str, np.ndarray]:
        """Per group (full length): every term, zeros / inf / False where unknown."""
        G = self.extra.n_groups
        out = {}
        for sh, p in self.dna.items():
            place, season = p.terms(ctx)
            out[f"boost_{sh:g}"], out[f"season_{sh:g}"] = place, season
        place_ok, date_ok = pp.has_place(ctx), pp.has_date(ctx)
        out["berth"] = np.zeros(G, bool)
        out["km_dna"] = np.full(G, np.inf)
        out["km_inat"] = np.full(G, np.inf)
        if place_ok:
            out["km_dna"] = self.extra.nearest_dna_km(ctx)
            if self.occ is not None:
                r = pp.BOOST["radius_km"]
                parts = self.occ.parts(Context(ctx.latitude, ctx.longitude, None, ctx.uuid),
                                       (r,))
                far = parts.out(r, genus_rule=False, min_dna_effort=pp.BOOST["min_dna_effort"])
                out["berth"] = np.asarray(far, bool)
                out["km_inat"] = pp.nearest_inat_km(self.occ, ctx.latitude, ctx.longitude,
                                                    ctx.uuid)
        out["n_place"] = self.extra.n_place
        out["off_season"] = np.zeros(G, bool)
        if date_ok:
            from .prior import day_distance
            e = self.extra
            within = day_distance(ctx.day_of_year, e.doy) <= pp.SEASON_WINDOW_DAYS
            near = np.bincount(e.doy_group, weights=within.astype(float), minlength=G)
            out["off_season"] = pp.off_season(near, e.dated)
        return out


def combine(c: Components, s: dict, T: float = PHOTO_TEMPERATURE) -> np.ndarray:
    """(n, K) combined scores of a setting, exactly as placeprior.PlaceBoost serves it."""
    z = c.S / T
    if s["family"] == "none":
        return z
    cap = pp.BOOST["cap"]
    sh = s["shrink"]
    z = (z + pp.BOOST["place_weight"] * np.clip(c.boost[sh], 0.0, cap)
         + pp.BOOST["season_weight"] * np.clip(c.season[sh], -cap, cap)
         - pp.BOOST["far_penalty"] * c.berth)
    if s["pen_slope"] and s["pen_cap"]:
        km = c.km_dna if s["pen_known"] == "dna" else np.minimum(c.km_dna, c.km_inat)
        z = z - (pp.distance_penalty(km, s["pen_start_km"], s["pen_slope"], s["pen_cap"])
                 * pp.count_damping(c.n_place, s["damp_k"]))
    if s["season_penalty"]:
        z = z - s["season_penalty"] * c.off_season
    return z


def top1(c: Components, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(species right, genus right) per record at top 1 (species among species groups)."""
    rows = np.arange(len(z))
    sp_best = np.argmax(np.where(c.SP, z, -np.inf), axis=1)
    sp = c.has_sp & (c.truth_at >= 0) & (sp_best == c.truth_at)
    ge = c.GG[rows, np.argmax(z, axis=1)] == c.truth_g
    return sp, ge


def score_all(c: Components, grid: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    sp = np.zeros((len(grid), len(c.S)), dtype=bool)
    ge = np.zeros_like(sp)
    for j, s in enumerate(grid):
        sp[j], ge[j] = top1(c, combine(c, s))
    return sp, ge


# --- selection -----------------------------------------------------------------------------

def band_of(km: np.ndarray, lo: float, hi: float) -> np.ndarray:
    km = np.asarray(km, dtype=np.float64)
    return np.isfinite(km) & (km >= lo) & (km < hi)


def holds_band(sp: np.ndarray, base: np.ndarray, rows: np.ndarray, in_band: np.ndarray,
               margin: float = MARGIN) -> np.ndarray:
    """(settings,) bool: on `rows`, the band's right count is no more than `margin` of the
    band's records below the base's (a point rule inside the folds)."""
    b = rows[in_band[rows]]
    if not len(b):
        return np.ones(len(sp), dtype=bool)
    delta = sp[:, b].sum(axis=1) - base[b].sum()
    return delta >= -margin * len(b)


def choose(sp: np.ndarray, ge: np.ndarray, rows: np.ndarray, grid: list[dict],
           allowed: np.ndarray | None = None) -> int:
    """Most species right on `rows`, then genus, then the gentlest, then the first."""
    allowed = np.ones(len(grid), dtype=bool) if allowed is None else allowed
    idx = np.flatnonzero(allowed)
    if not len(idx):
        raise ValueError("no setting allowed")
    s_ok = sp[idx][:, rows].sum(axis=1)
    g_ok = ge[idx][:, rows].sum(axis=1)
    gentle = np.array([gentleness(grid[j]) for j in idx])
    order = np.lexsort((idx, gentle, -g_ok, -s_ok))
    return int(idx[order[0]])


def observer_folds(observers: list, k: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """A fold per record, all of one observer's records in one fold (largest first into
    the smallest fold, after a seeded shuffle); no observer: its own group."""
    groups: dict = {}
    for i, o in enumerate(observers):
        key = o if o not in (None, "", "None") else f"record:{i}"
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))
    fold = np.full(len(observers), -1, dtype=np.int64)
    size = np.zeros(k, dtype=np.int64)
    for g in keys:
        f = int(np.argmin(size))
        fold[groups[g]] = f
        size[f] += len(groups[g])
    return fold


def nested_cv(sp: np.ndarray, ge: np.ndarray, folds: np.ndarray, grid: list[dict],
              allowed: np.ndarray | None = None, band: np.ndarray | None = None,
              base: np.ndarray | None = None, margin: float = MARGIN) -> dict:
    """Each fold scored with the setting chosen on the other four. With `band` and `base`
    (P0's species right), only settings holding the band within `margin` on the training
    folds may be chosen; when none does, the fold gets the base (photos only)."""
    n = sp.shape[1]
    cv_sp, cv_ge = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    chosen = []
    allowed = np.ones(len(grid), dtype=bool) if allowed is None else allowed
    for f in range(int(folds.max()) + 1):
        test = folds == f
        train = np.flatnonzero(~test)
        ok = allowed.copy()
        if band is not None:
            ok &= holds_band(sp, base, train, band, margin)
        if not ok.any():
            chosen.append(None)
            cv_sp[test] = base[test]
            continue
        j = choose(sp, ge, train, grid, ok)
        chosen.append(j)
        cv_sp[test], cv_ge[test] = sp[j, test], ge[j, test]
    return {"chosen": chosen, "species_right": cv_sp, "genus_right": cv_ge}


def fixed_broken(a: np.ndarray, b: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """b against a on `rows`: records b gets right that a didn't (fixed), and the reverse."""
    if rows is not None:
        a, b = a[rows], b[rows]
    return {"n": int(len(a)), "a": int(a.sum()), "b": int(b.sum()),
            "fixed": int((b & ~a).sum()), "broken": int((a & ~b).sum())}


def band_table(a: np.ndarray, b: np.ndarray, km: np.ndarray, keep: np.ndarray) -> list[dict]:
    """fixed / broken of b against a in every distance band (records in `keep`)."""
    out = []
    for lo, hi, label in BANDS:
        rows = np.flatnonzero(keep & band_of(km, lo, hi))
        out.append({"band": label, **fixed_broken(a, b, rows)})
    out.append({"band": "no train DNA record / no place",
                **fixed_broken(a, b, np.flatnonzero(keep & ~np.isfinite(km)))})
    return out


def band_lower_bound(a: np.ndarray, b: np.ndarray, in_band: np.ndarray, folds: np.ndarray,
                     n_boot: int = 10_000, seed: int = SEED) -> dict:
    """One-sided 95% lower bound of (right b - right a) / n_band, paired bootstrap of the
    band's records stratified by observer fold (atlas-environment.md fix 2)."""
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(in_band)
    d = b[rows].astype(np.int8) - a[rows].astype(np.int8)
    if not len(rows):
        return {"n": 0, "delta": None, "lower95": None}
    by_fold = [np.flatnonzero(folds[rows] == f) for f in np.unique(folds[rows])]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        tot = 0
        for idx in by_fold:
            tot += d[idx[rng.integers(0, len(idx), len(idx))]].sum()
        boots[i] = tot / len(rows)
    return {"n": int(len(rows)), "delta": float(d.mean()),
            "lower95": float(np.quantile(boots, 0.05)),
            "passes_margin": bool(np.quantile(boots, 0.05) > -MARGIN)}
