"""Tune a place-and-date prior on top of nearest+mean, on a development set only.

exp/prior-tuning (Steve, 2026-10-09: CPU only, dev only). Three prior families are
tried on top of the nearest+mean photo scores (cosine-like, put on the log-probability
scale at PHOTO_TEMPERATURE):

- "dna": the range-and-season score from our own DNA-verified records (prior.py,
  `+prior`), with its place and season terms weighted and capped separately;
- "occ": the iNat occurrence prior (occprior.py, `+occ`): a wide berth for out of
  range (Steve's ~1,500 km rule), gentle density and season;
- "dna+occ": the occurrence prior's wide berth and density with the DNA records'
  place term, and season from one source or the other.

The grid is fixed here, before any result was seen (GRID_DECLARED), and every setting
in it is scored and recorded, so the selection is never hidden. Settings are chosen by
species top-1 (then genus top-1, then the gentler setting) with 5-fold cross-validation
grouped by observer: each fold is scored with the setting chosen on the other four, so
the reported development number is not the number it was tuned on.

Nothing here reads or writes a coordinate to an output: components hold scores only,
and the range-edge check reports counts per distance band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

PHOTO_TEMPERATURE = 0.02        # AsLogProb's: score = cosine / T + prior
TOP_K = 1000                    # candidates kept per record (plus the truth)
FOLDS = 5
SEED = 20261009
LOG5, LOG20 = math.log(5.0), math.log(20.0)
MIN_DNA_EFFORT = 500            # occprior default, fixed here
GRID_DECLARED = "2026-10-09, before any prior-tuning result (commit 'pre-declared grid')"

DNA_GRID = {
    "place_km": [75.0, 150.0, 300.0],
    "season_days": [10.0, 20.0, 40.0],
    "place_weight": [0.0, 0.25, 0.5, 1.0],
    "season_weight": [0.0, 0.25, 0.5, 1.0],
    "cap": [LOG5, LOG20],
}
OCC_GRID = {
    "radius_km": [1000.0, 1500.0, 2000.0],
    "penalty": [0.0, 2.0, 6.0, 50.0],          # 50 = outright exclusion (occprior.EXCLUDE)
    "genus_rule": [True, False],
    "density_km": [75.0, 150.0, 300.0],
    "density_weight": [0.0, 0.25, 0.5, 1.0],
    "season_weight": [0.0, 0.25, 0.5, 1.0],
}
COMBINED_GRID = {
    "penalty": [0.0, 2.0, 6.0, 50.0],
    "density_weight": [0.0, 0.25, 0.5],
    "dna_place_weight": [0.0, 0.25, 0.5, 1.0],
    "season_source": ["dna", "occ"],
    "season_weight": [0.0, 0.25, 0.5, 1.0],
}
# Held fixed in the combined family: Steve's radius, the genus rule, the default kernels.
COMBINED_FIXED = {"radius_km": 1500.0, "genus_rule": True, "density_km": 150.0,
                  "place_km": 150.0, "season_days": 20.0, "cap": LOG20}
# The DNA prior as nearest+prior ships it: place 150 km + season 20 days, summed, capped
# at log 20, weight 1. Scored as the reference, never chosen.
AS_SHIPPED = {"family": "dna-as-shipped"}

# What the declared procedure chose on all of heldout-2026-10-08's development split
# (2,986 records; the whole grid, family included), as RangeSeasonPrior settings for
# methods' `nearest+mean+prior`. Cross-validated estimate of that procedure: species
# top-1 55.1% vs 52.8% for nearest+mean (docs/experiments/2026-10-09-prior-tuning.md).
CHOSEN_DNA_PRIOR = {"bandwidth_km": 150.0, "bandwidth_days": 10.0, "cap": LOG20,
                    "place_weight": 0.5, "season_weight": 0.5}
# Its confidence temperature on the log-probability scale, fitted on the same records.
CHOSEN_CONFIDENCE_TEMPERATURE = 1.52

RADII = tuple(OCC_GRID["radius_km"])
PLACE_KM = tuple(DNA_GRID["place_km"])
SEASON_DAYS = tuple(DNA_GRID["season_days"])
DENSITY_KM = tuple(OCC_GRID["density_km"])


def settings() -> list[dict]:
    """Every setting of the declared grid, in a fixed order, each with its family."""
    out = []
    for fam, grid in (("dna", DNA_GRID), ("occ", OCC_GRID), ("dna+occ", COMBINED_GRID)):
        keys = list(grid)
        for values in product(*(grid[k] for k in keys)):
            s = {"family": fam, **dict(zip(keys, values))}
            if fam == "dna+occ":
                s = {**s, **COMBINED_FIXED}
            out.append(s)
    return out


def gentleness(s: dict) -> float:
    """Smaller is gentler: how far a setting moves the photo scores (ties go to it)."""
    w = sum(float(s.get(k, 0.0)) for k in ("place_weight", "season_weight", "density_weight",
                                             "dna_place_weight"))
    return w + float(s.get("penalty", 0.0)) / 6.0


def describe(s: dict) -> str:
    fam = s["family"]
    if fam == "none":
        return "no prior"
    if fam == "dna-as-shipped":
        return "DNA prior as shipped (150 km + 20 d, cap log 20, weight 1)"
    if fam == "dna":
        return (f"DNA place {s['place_weight']:g} @ {s['place_km']:g} km, season "
                f"{s['season_weight']:g} @ {s['season_days']:g} d, cap log {math.exp(s['cap']):.0f}")
    if fam == "occ":
        return (f"iNat occ: berth {s['radius_km']:g} km pen {s['penalty']:g}"
                f"{' +genus' if s['genus_rule'] else ''}, density {s['density_weight']:g} @ "
                f"{s['density_km']:g} km, season {s['season_weight']:g}")
    return (f"combined: berth 1500 km pen {s['penalty']:g} +genus, iNat density "
            f"{s['density_weight']:g}, DNA place {s['dna_place_weight']:g}, season "
            f"{s['season_weight']:g} ({s['season_source']})")


# --- components ------------------------------------------------------------------------

@dataclass
class Components:
    """Per record, its candidates' photo score and every prior term the grid needs,
    uncapped and unweighted (n records x K candidates). No coordinates."""
    S: np.ndarray                     # base photo score (cosine-like)
    cand: np.ndarray                  # group index of each candidate (-1 = padding)
    SP: np.ndarray                    # candidate is a species group
    truth_at: np.ndarray              # (n,) candidate column of the true species, -1 none
    has_sp: np.ndarray                # (n,) the answer key names a species
    GG: np.ndarray                    # candidate's genus id (-1 none)
    truth_g: np.ndarray               # (n,) true genus id (-3 none)
    dna_place: dict = field(default_factory=dict)       # km -> (n, K)
    dna_season: dict = field(default_factory=dict)      # days -> (n, K)
    occ_out: dict = field(default_factory=dict)         # radius -> (n, K) bool, own evidence
    occ_genus_out: dict = field(default_factory=dict)   # radius -> (n, K) bool
    occ_dna_out: dict = field(default_factory=dict)     # radius -> (n, K) bool
    occ_dna_effort: dict = field(default_factory=dict)  # radius -> (n,)
    occ_density: dict = field(default_factory=dict)     # km -> (n, K)
    occ_season: np.ndarray | None = None                # (n, K)


def candidates(scores: np.ndarray, truth: int, k: int = TOP_K) -> np.ndarray:
    """The k best groups by photo score, best first, plus the truth when it isn't one."""
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    if truth >= 0 and truth not in set(top.tolist()):
        top = np.append(top, truth)
    return top


def combine(c: Components, s: dict, T: float = PHOTO_TEMPERATURE) -> np.ndarray:
    """(n, K) combined scores of a setting: photo / T + weighted, capped prior terms."""
    z = c.S / T
    fam = s["family"]
    if fam == "none":
        return z
    if fam == "dna-as-shipped":
        return z + np.clip(c.dna_place[150.0] + c.dna_season[20.0], -LOG20, LOG20)
    if fam == "dna":
        cap = s["cap"]
        if s["place_weight"]:
            z = z + s["place_weight"] * np.clip(c.dna_place[s["place_km"]], -cap, cap)
        if s["season_weight"]:
            z = z + s["season_weight"] * np.clip(c.dna_season[s["season_days"]], -cap, cap)
        return z
    r = s["radius_km"]
    if s.get("density_weight"):
        z = z + s["density_weight"] * np.clip(c.occ_density[s["density_km"]], -LOG5, LOG5)
    if fam == "occ":
        if s["season_weight"] and c.occ_season is not None:
            z = z + s["season_weight"] * np.clip(c.occ_season, -LOG5, LOG5)
    else:
        if s["dna_place_weight"]:
            z = z + s["dna_place_weight"] * np.clip(c.dna_place[s["place_km"]], -LOG20, LOG20)
        if s["season_weight"]:
            season = (np.clip(c.dna_season[s["season_days"]], -LOG20, LOG20)
                      if s["season_source"] == "dna"
                      else np.clip(c.occ_season, -LOG5, LOG5))
            z = z + s["season_weight"] * season
    if s["penalty"]:
        z = z - s["penalty"] * out_of_range(c, r, s["genus_rule"])
    return z


def out_of_range(c: Components, radius: float, genus_rule: bool) -> np.ndarray:
    """occprior.Parts.out, over all records: the species' own evidence, DNA-only evidence
    where DNA sampling is dense, and the genus rule."""
    out = c.occ_out[radius] | (c.occ_dna_out[radius]
                               & (c.occ_dna_effort[radius] >= MIN_DNA_EFFORT)[:, None])
    if genus_rule:
        out = out | c.occ_genus_out[radius]
    return out


def top1(c: Components, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(species right, genus right) per record at top 1. Species are ranked among species
    groups only; genus is the genus of the best group of any kind."""
    rows = np.arange(len(z))
    sp_best = np.argmax(np.where(c.SP, z, -np.inf), axis=1)
    sp = c.has_sp & (c.truth_at >= 0) & (sp_best == c.truth_at)
    ge = c.GG[rows, np.argmax(z, axis=1)] == c.truth_g
    return sp, ge


# --- search and cross-validation ---------------------------------------------------------

def score_all(c: Components, grid: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(settings, n) bool matrices: species and genus top-1 right, per setting."""
    sp = np.zeros((len(grid), len(c.S)), dtype=bool)
    ge = np.zeros_like(sp)
    for j, s in enumerate(grid):
        sp[j], ge[j] = top1(c, combine(c, s))
    return sp, ge


def choose(sp: np.ndarray, ge: np.ndarray, rows: np.ndarray, grid: list[dict],
           allowed: np.ndarray | None = None) -> int:
    """The setting with most species right on `rows`, then most genus right, then the
    gentlest, then the first declared."""
    allowed = np.ones(len(grid), dtype=bool) if allowed is None else allowed
    idx = np.flatnonzero(allowed)
    s_ok = sp[idx][:, rows].sum(axis=1)
    g_ok = ge[idx][:, rows].sum(axis=1)
    gentle = np.array([gentleness(grid[j]) for j in idx])
    order = np.lexsort((idx, gentle, -g_ok, -s_ok))
    return int(idx[order[0]])


def observer_folds(observers: list, k: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """A fold per record, every record of one observer in the same fold. Observers are
    shuffled, then placed largest first in the fold with fewest records so far.
    Records with no observer are each their own group."""
    groups: dict = {}
    for i, o in enumerate(observers):
        key = o if o not in (None, "", "None") else f"record:{i}"
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))           # stable: ties keep the shuffle
    fold = np.full(len(observers), -1, dtype=np.int64)
    size = np.zeros(k, dtype=np.int64)
    for g in keys:
        f = int(np.argmin(size))
        fold[groups[g]] = f
        size[f] += len(groups[g])
    return fold


def nested_cv(sp: np.ndarray, ge: np.ndarray, folds: np.ndarray, grid: list[dict],
              allowed: np.ndarray | None = None) -> dict:
    """Each fold scored with the setting chosen on the others. Returns the chosen setting
    per fold and the pooled out-of-fold right vectors."""
    n = sp.shape[1]
    cv_sp = np.zeros(n, dtype=bool)
    cv_ge = np.zeros(n, dtype=bool)
    chosen = []
    for f in range(int(folds.max()) + 1):
        test = folds == f
        j = choose(sp, ge, np.flatnonzero(~test), grid, allowed)
        chosen.append(j)
        cv_sp[test] = sp[j, test]
        cv_ge[test] = ge[j, test]
    return {"chosen": chosen, "species_right": cv_sp, "genus_right": cv_ge}


# --- confidence --------------------------------------------------------------------------

T_GRID = np.geomspace(0.05, 20.0, 80)       # on the combined (log-probability) scale


def _species_logits(c: Components, z: np.ndarray) -> np.ndarray:
    return np.where(c.SP, z, -np.inf)


def nll_curve(c: Components, z: np.ndarray, rows: np.ndarray,
              grid: np.ndarray = T_GRID) -> tuple[np.ndarray, int]:
    """Summed species NLL of the truth at each temperature, over `rows` whose true
    species is a candidate."""
    zz = _species_logits(c, z)
    use = [i for i in rows.tolist() if c.has_sp[i] and c.truth_at[i] >= 0
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
    e = np.exp(zz - m)
    return 1.0 / e.sum(axis=1)


def calibration_report(conf: np.ndarray, right: np.ndarray, keep: np.ndarray,
                       nll: float | None = None) -> dict:
    from .heldout_report import calibration
    pairs = [(float(a), bool(b)) for a, b, k in zip(conf, right, keep) if k]
    cal = calibration(pairs)
    return {"n": cal["n"], "ece": cal["ece"], "nll": None if nll is None else round(nll, 4),
            "mean_confidence": round(float(np.mean([p for p, _ in pairs])), 4) if pairs else None,
            "accuracy": round(float(np.mean([r for _, r in pairs])), 4) if pairs else None,
            "share_stated_99": round(float(np.mean([p >= 0.99 for p, _ in pairs])), 4)
            if pairs else None, "bins": cal["bins"]}


# --- ranks for the standard tables ---------------------------------------------------------

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


def distance_band(km: float | None) -> str:
    """The true species' nearest known find (iNat or DNA), as a band: never a place."""
    if km is None or not np.isfinite(km):
        return "no known find"
    for hi, label in ((100, "< 100 km"), (300, "100-300 km"), (1000, "300-1,000 km"),
                      (1500, "1,000-1,500 km")):
        if km < hi:
            return label
    return ">= 1,500 km"


DISTANCE_BANDS = ("< 100 km", "100-300 km", "300-1,000 km", "1,000-1,500 km", ">= 1,500 km",
                  "no known find")
