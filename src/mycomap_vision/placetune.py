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


def band_lower_bound_clustered(a: np.ndarray, b: np.ndarray, in_band: np.ndarray,
                               observers: list, n_boot: int = 10_000,
                               seed: int = SEED) -> dict:
    """The paper's version (E8): the same bound by a paired CLUSTER bootstrap, observers
    resampled with replacement (each with all its band records); the ratio of sums."""
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(in_band)
    if not len(rows):
        return {"n": 0, "delta": None, "lower95": None}
    d = b[rows].astype(np.int64) - a[rows].astype(np.int64)
    keys = [o if o not in (None, "", "None") else f"record:{i}"
            for i, o in zip(rows.tolist(), np.asarray(observers, dtype=object)[rows].tolist())]
    _, cluster = np.unique(np.array(keys, dtype=str), return_inverse=True)
    sums = np.bincount(cluster, weights=d)
    sizes = np.bincount(cluster)
    k = len(sums)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        boots[i] = sums[pick].sum() / sizes[pick].sum()
    lb = float(np.quantile(boots, 0.05))
    return {"n": int(len(rows)), "observers": int(k), "delta": float(d.mean()),
            "lower95": lb, "passes_margin": bool(lb > -MARGIN)}


def label_table(a: np.ndarray, b: np.ndarray, labels: np.ndarray, keep: np.ndarray,
                order: tuple[str, ...]) -> list[dict]:
    """fixed / broken of b against a per band label (the paper's distance_band)."""
    labels = np.asarray(labels, dtype=str)
    return [{"band": lab, **fixed_broken(a, b, np.flatnonzero(keep & (labels == lab)))}
            for lab in order]


PAPER_BANDS = ("<100", "100-300", "300-1500", ">1500", "no reference", "no point",
               "no reference point")
PAPER_RULE_BAND = "300-1500"


# --- confidence and answers for the standard tables ----------------------------------------

T_GRID = np.geomspace(0.05, 20.0, 80)       # on the combined (log-probability) scale


def _species_logits(c: Components, z: np.ndarray) -> np.ndarray:
    return np.where(c.SP, z, -np.inf)


def nll_curve(c: Components, z: np.ndarray, rows: np.ndarray,
              grid: np.ndarray = T_GRID) -> tuple[np.ndarray, int]:
    """Summed species NLL of the truth at each temperature, over `rows` whose true
    species is a candidate."""
    zz = _species_logits(c, z)
    use = [i for i in np.asarray(rows).tolist() if c.has_sp[i] and c.truth_at[i] >= 0
           and c.SP[i, c.truth_at[i]]]
    total = np.zeros(len(grid))
    for i in use:
        row = zz[i][np.isfinite(zz[i])]
        t = zz[i, c.truth_at[i]]
        x = row[None, :] / grid[:, None]
        m = x.max(axis=1)
        total += m + np.log(np.exp(x - m[:, None]).sum(axis=1)) - t / grid
    return total, len(use)


def fit_temperature(c: Components, z: np.ndarray, rows: np.ndarray) -> float:
    total, used = nll_curve(c, z, rows)
    return float(T_GRID[int(np.argmin(total))]) if used else 1.0


def top_confidence(c: Components, z: np.ndarray, T) -> np.ndarray:
    """(n,) stated confidence of the top species at temperature T (one, or one per row)."""
    T = np.asarray(T, dtype=np.float64)
    zz = _species_logits(c, z) / (T[:, None] if T.ndim == 1 else T)
    m = np.max(zz, axis=1, keepdims=True)
    return 1.0 / np.exp(zz - m).sum(axis=1)


def calibration_report(conf: np.ndarray, right: np.ndarray, keep: np.ndarray) -> dict:
    from .heldout_report import calibration
    pairs = [(float(a), bool(b)) for a, b, k in zip(conf, right, keep) if k]
    cal = calibration(pairs)
    return {"n": cal["n"], "ece": cal["ece"],
            "mean_confidence": round(float(np.mean([p for p, _ in pairs])), 4) if pairs else None,
            "accuracy": round(float(np.mean([r for _, r in pairs])), 4) if pairs else None,
            "share_stated_99": round(float(np.mean([p >= 0.99 for p, _ in pairs])), 4)
            if pairs else None}


def ranks(z_row: np.ndarray, cand: np.ndarray, sp_mask: np.ndarray, group_names: list[str],
          group_genus: list[str], group_family: list[str], T: float, top: int = 10) -> dict:
    """An identify-like answer from one record's candidate scores: the top species
    (species groups only), genera and families (best group of each), with confidence."""
    ok = cand >= 0

    def rank_list(keys: list[str], mask: np.ndarray) -> list[dict]:
        best: dict[str, float] = {}
        for j in np.flatnonzero(mask & ok):
            name = keys[j]
            if name and (name not in best or z_row[j] > best[name]):
                best[name] = float(z_row[j])
        if not best:
            return []
        names_ = list(best)
        v = np.array([best[x] for x in names_]) / T
        p = np.exp(v - v.max())
        p /= p.sum()
        order = np.argsort(-v, kind="stable")[:top]
        return [{"name": names_[o], "confidence": round(float(p[o]), 4)} for o in order]
    c_names = [group_names[g] if g >= 0 else "" for g in cand.tolist()]
    c_genus = [group_genus[g] if g >= 0 else "" for g in cand.tolist()]
    c_family = [group_family[g] if g >= 0 else "" for g in cand.tolist()]
    return {"species": rank_list(c_names, sp_mask), "genus": rank_list(c_genus, np.ones_like(ok)),
            "family": rank_list(c_family, np.ones_like(ok)), "top_k": top}


# --- Atlas arms (docs/atlas-environment.md section 1; evaluated on R_train only) -------------
#
# Declared in the design doc (2026-10-12, Steve) and coded here before any Atlas result.
# Every arm keeps the served season term and, except P2d for mapped taxa, the served
# out-of-range rule; "mapped" = the candidate has a strong Atlas map. The rank factor f is
# learned on the training folds only (`learn_log_f`), as is each arm's setting.

ATLAS_DECLARED = "docs/atlas-environment.md section 1 (2026-10-12); coded 2026-10-13"
ATLAS_WEIGHTS = (0.25, 0.5, 1.0)
IN_REACH_ARMS = ("P2a", "P2b", "P2d")


def atlas_settings() -> list[dict]:
    out = []
    for w, app in product(ATLAS_WEIGHTS, ("cut", "graded")):
        out.append({"family": "atlas", "arm": "P2c", "weight": w, "applicability": app,
                    "trust": True})
    for w, fallback in product(ATLAS_WEIGHTS, ("served", "none")):
        out.append({"family": "atlas", "arm": "P2a" if fallback == "served" else "P2b",
                    "weight": w, "applicability": "cut", "trust": True})
    for pen, w in product((0.5, 1.0, 2.0), (0.5, 1.0)):
        out.append({"family": "atlas", "arm": "P2d", "weight": w, "applicability": "cut",
                    "trust": True, "outside_penalty": pen})
    for w in ATLAS_WEIGHTS:
        out.append({"family": "atlas", "arm": "P3", "weight": w, "applicability": "cut",
                    "trust": True})
    return out


def p2c_u(chosen: dict) -> dict:
    """P2c-u: P2c's chosen weight and applicability rule, trust = 1."""
    return {**chosen, "arm": "P2c-u", "trust": False}


@dataclass
class AtlasComponents:
    """Per record and candidate, from an AtlasTable at the record's place (uint8 values,
    atlasprior.NO_DATA where none). `mapped`: the candidate has a strong map at all."""
    mapped: np.ndarray
    rank: np.ndarray
    aoa: np.ndarray
    models: np.ndarray
    rank_sd: np.ndarray
    reach: np.ndarray
    strength: np.ndarray


def atlas_terms(table, groups_col: np.ndarray, ctx: Context, cand: np.ndarray) -> dict:
    """One record's Atlas values for its candidates (`groups_col`: the Atlas column of
    every group, -1 unmapped; atlasprior.AtlasTable.columns_for)."""
    from . import atlasprior as ap
    K = len(cand)
    col = np.where(cand >= 0, groups_col[np.maximum(cand, 0)], -1)
    out = {"mapped": col >= 0, "rank": np.full(K, ap.NO_DATA, np.uint8),
           "aoa": np.full(K, ap.NO_DATA, np.uint8), "models": np.full(K, ap.NO_DATA, np.uint8),
           "rank_sd": np.full(K, ap.NO_DATA, np.uint8), "reach": np.zeros(K, bool),
           "strength": np.where(col >= 0, table.map_strength[np.maximum(col, 0)], 0.0)}
    view = table.at(ctx.latitude, ctx.longitude) if pp.has_place(ctx) else None
    if view is not None:
        ok = col >= 0
        for name in ("rank", "aoa", "models", "rank_sd", "reach"):
            out[name][ok] = getattr(view, name)[col[ok]]
    return out


def learn_log_f(c: Components, a: AtlasComponents, rows: np.ndarray, in_reach: bool,
                top_k: int = 20) -> np.ndarray:
    """log f per rank bin from `rows` (training folds only): the true taxon's rank at its
    record's cell against its top-k photo candidates' ranks (mapped ones with a rank)."""
    from . import atlasprior as ap
    ok_rank = (a.rank != ap.NO_DATA) & a.mapped
    if in_reach:
        ok_rank = ok_rank & a.reach
    truths, looks = [], []
    for i in np.asarray(rows).tolist():
        t = c.truth_at[i]
        if c.has_sp[i] and t >= 0 and ok_rank[i, t]:
            truths.append(int(a.rank[i, t]))
        top = np.argsort(-c.S[i])[:top_k]
        top = top[ok_rank[i, top]]
        looks.extend(a.rank[i, top].astype(int).tolist())
    return ap.rank_factor(np.array(truths, int), np.array(looks, int))


def combine_atlas(c: Components, a: AtlasComponents, s: dict, log_f: np.ndarray,
                  T: float = PHOTO_TEMPERATURE) -> np.ndarray:
    """(n, K) scores of an Atlas arm. The DNA boost is the served one (shrink 3)."""
    from . import atlasprior as ap
    cap = pp.BOOST["cap"]
    dna = pp.BOOST["place_weight"] * np.clip(c.boost[3.0], 0.0, cap)
    tr = ap.trust(a.strength, a.models, a.rank_sd, a.aoa, s["applicability"], s["trust"])
    arm = s["arm"]
    rank = np.where(a.reach, a.rank, ap.NO_DATA) if arm in IN_REACH_ARMS else a.rank
    atl = ap.atlas_boost(rank, s["weight"] * tr, log_f)
    berth = c.berth.astype(float)
    if arm in ("P2c", "P2c-u", "P2a"):
        place = np.where(a.mapped, atl, dna)
    elif arm == "P2b":
        place = np.where(a.mapped, atl, 0.0)
    elif arm == "P2d":
        place = np.where(a.mapped, np.where(a.reach, atl, -s["outside_penalty"]), dna)
        berth = np.where(a.mapped, 0.0, berth)        # the area replaces the 1,500 km berth
    elif arm == "P3":
        place = np.where(a.mapped, np.maximum(dna, atl), dna)
    else:
        raise ValueError(f"arm {arm!r}")
    return (c.S / T + place + pp.BOOST["season_weight"] * np.clip(c.season[3.0], -cap, cap)
            - pp.BOOST["far_penalty"] * berth)


def choose_atlas(sp, ge, rows, grid, allowed) -> int:
    idx = np.flatnonzero(allowed)
    s_ok = sp[idx][:, rows].sum(axis=1)
    g_ok = ge[idx][:, rows].sum(axis=1)
    gentle = np.array([grid[j]["weight"] + grid[j].get("outside_penalty", 0.0) for j in idx])
    return int(idx[np.lexsort((idx, gentle, -g_ok, -s_ok))[0]])


def nested_cv_atlas(c: Components, a: AtlasComponents, folds: np.ndarray, grid: list[dict],
                    band: np.ndarray | None = None, base: np.ndarray | None = None,
                    margin: float = MARGIN) -> dict:
    """As nested_cv, with f learned on each fold's training records too."""
    n = len(c.S)
    cv_sp, cv_ge = np.zeros(n, bool), np.zeros(n, bool)
    chosen, factors = [], []
    for f in range(int(folds.max()) + 1):
        test = folds == f
        train = np.flatnonzero(~test)
        lf = {r: learn_log_f(c, a, train, r) for r in (False, True)}
        sp = np.zeros((len(grid), n), bool)
        ge = np.zeros_like(sp)
        for j, s in enumerate(grid):
            sp[j], ge[j] = top1(c, combine_atlas(c, a, s, lf[s["arm"] in IN_REACH_ARMS]))
        ok = np.ones(len(grid), bool)
        if band is not None:
            ok &= holds_band(sp, base, train, band, margin)
        if not ok.any():
            chosen.append(None)
            factors.append(None)
            cv_sp[test] = base[test]
            continue
        j = choose_atlas(sp, ge, train, grid, ok)
        chosen.append(j)
        factors.append(lf[grid[j]["arm"] in IN_REACH_ARMS].round(4).tolist())
        cv_sp[test], cv_ge[test] = sp[j, test], ge[j, test]
    return {"chosen": chosen, "log_f": factors, "species_right": cv_sp, "genus_right": cv_ge}


def primary_test(p0_sp: np.ndarray, arm_sp: np.ndarray, has_sp: np.ndarray,
                 band: np.ndarray, folds: np.ndarray, n_boot: int = 10_000) -> dict:
    """The pre-declared intersection-union test (fix 1-2): overall superiority by McNemar
    one-sided p < 0.05 AND the 300-1,500 km band's 95% lower bound above -2%."""
    from .heldout_report import mcnemar
    fb = fixed_broken(p0_sp, arm_sp, np.flatnonzero(has_sp))
    p = mcnemar(fb["broken"], fb["fixed"])
    one_sided = p / 2 if fb["fixed"] > fb["broken"] else 1 - p / 2
    nb = band_lower_bound(p0_sp, arm_sp, band & has_sp, folds, n_boot)
    return {"overall": {**fb, "mcnemar_one_sided_p": round(float(one_sided), 6),
                        "passes": bool(one_sided < 0.05)},
            "band_300_1500": nb,
            "passes": bool(one_sided < 0.05 and nb.get("passes_margin", False))}
