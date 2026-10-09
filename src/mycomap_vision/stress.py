"""Leakage, test-time augmentation and stress slices for a held-out benchmark
(experiment 2026-10-09-leakage-tta-stress; drivers in scripts/leakage_tta_stress/).

Scores are made the way identify.Identifier makes them (methods.NearestSpecimen,
methods.NearestAndMean, prior.WithPrior over AsLogProb), but from one similarity matrix
per query with some reference records masked out, so a condition that removes, for
each query, the records of its own observer (or of the same collection) costs a mask,
not a rebuilt index. A masked record is gone for every method: its photos can't be a
nearest match, it leaves its species' average vector, and it leaves the place-and-date
score's counts. A species whose every record is masked has no score (-inf), as if it
had never been in the reference.

numpy only: importing torch next to numpy's big CPU matrix products segfaults in this
project's environment, so the GPU embedding step (TTA) runs in its own process.

Nothing here returns a coordinate: distances are used to decide a mask or a bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prior import Context, RangeSeasonPrior, day_distance, haversine_km

NEG = np.float32(-np.inf)
SAME_COLLECTION_KM = 1.0       # same observer, within this distance ...
SAME_COLLECTION_DAYS = 1       # ... and this many days: one collection trip
NEAR_DUPLICATE_COSINE = 0.98   # a reference photo this close to a query photo


# --- per-species scores with a column mask -------------------------------------------------

def best_two(sims: np.ndarray, starts: np.ndarray, keep: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """(q, groups) best and second-best similarity of each query photo within each group
    (reference photos sorted by group, `starts` the first column of each group). Columns
    with keep == False don't count. A group with no kept photo has -inf for both; with
    one, -inf for the second. A tie for best counts twice, as a sort would."""
    s = sims if keep is None else np.where(keep[None, :], sims, NEG)
    m1 = np.maximum.reduceat(s, starts, axis=1)
    counts = np.diff(np.append(starts, s.shape[1]))
    at_best = s == np.repeat(m1, counts, axis=1)
    n_best = np.add.reduceat(at_best & np.isfinite(s), starts, axis=1)
    m2 = np.maximum.reduceat(np.where(at_best, NEG, s), starts, axis=1)
    m2 = np.where(n_best >= 2, m1, m2)
    return m1, m2


def best_two_masked(sims: np.ndarray, starts: np.ndarray, col_group: np.ndarray,
                    base: tuple[np.ndarray, np.ndarray], masked_cols: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """best_two with the `masked_cols` columns left out, recomputing only the groups those
    columns are in (`base`: best_two of the same sims with nothing masked)."""
    m1, m2 = base
    if not len(masked_cols):
        return m1, m2
    groups = np.unique(col_group[masked_cols])
    ends = np.append(starts[1:], sims.shape[1])
    lengths = ends[groups] - starts[groups]
    local = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    cols = np.repeat(starts[groups] - local, lengths) + np.arange(lengths.sum())
    dropped = np.zeros(sims.shape[1], dtype=bool)
    dropped[masked_cols] = True
    s1, s2 = best_two(sims[:, cols], local, ~dropped[cols])
    m1, m2 = m1.copy(), m2.copy()
    m1[:, groups], m2[:, groups] = s1, s2
    return m1, m2


def top2_mean(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """methods.top_k_species_scores at k = 2, per photo: the mean of the two best matches,
    or the best alone where the group has one photo."""
    return np.where(np.isfinite(m2), (m1 + m2) / 2, m1)


def species_sums(vectors: np.ndarray, starts: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """(groups, dim) float32 sum of each group's reference vectors (in group order)."""
    n = len(starts)
    out = np.empty((n, vectors.shape[1]), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lo = starts[s]
        hi = starts[e] if e < n else len(vectors)
        block = np.asarray(vectors[lo:hi], dtype=np.float32)
        out[s:e] = np.add.reduceat(block, starts[s:e] - lo, axis=0)
    return out


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def mean_sims(query: np.ndarray, sums: np.ndarray, sizes: np.ndarray, vectors: np.ndarray,
              col_group: np.ndarray, masked_cols: np.ndarray,
              base: np.ndarray | None = None) -> np.ndarray:
    """(q, groups) similarity of each query photo to each group's average vector, with
    the `masked_cols` reference photos taken out of their groups' averages (a group left
    with none: -inf). `sizes`: photos per group; `base`: the unmasked result, when
    already made."""
    q = np.asarray(query, dtype=np.float32)
    out = (q @ unit(sums).T) if base is None else base.copy()
    if len(masked_cols):
        masked_cols = np.sort(masked_cols)     # columns are in group order, so runs
        g = col_group[masked_cols]
        run = np.flatnonzero(np.diff(g, prepend=-1) != 0)
        groups = g[run]
        removed = np.add.reduceat(np.asarray(vectors[masked_cols], dtype=np.float32), run,
                                  axis=0)
        left = sums[groups] - removed
        norms = np.linalg.norm(left, axis=1)
        alive = (sizes[groups] - np.diff(np.append(run, len(g)))) > 0
        sims = q @ (left / norms.clip(1e-12)[:, None]).T
        out[:, groups] = np.where(alive[None, :], sims, NEG)
    return out


NEAREST_AND_MEAN_WEIGHT = 0.6     # methods.NearestAndMean.weight
LOGPROB_TEMPERATURE = 0.02        # methods.AsLogProb's default


def nearest_scores(m1: np.ndarray) -> np.ndarray:
    """methods.NearestSpecimen: mean over photos of each photo's best match."""
    return m1.mean(axis=0)


def nearest_mean_scores(m1: np.ndarray, m2: np.ndarray, msims: np.ndarray) -> np.ndarray:
    """methods.NearestAndMean: 0.6 x mean of the two best + 0.4 x the species average."""
    w = NEAREST_AND_MEAN_WEIGHT
    return w * top2_mean(m1, m2).mean(axis=0) + (1 - w) * msims.mean(axis=0)


def log_softmax(z: np.ndarray) -> np.ndarray:
    finite = np.isfinite(z)
    if not finite.any():
        return z
    m = z[finite].max()
    out = np.full_like(z, -np.inf, dtype=np.float64)
    zz = z[finite] - m
    out[finite] = zz - np.log(np.exp(zz).sum())
    return out


def prior_weights(prior: RangeSeasonPrior, ctx: Context | None) -> list:
    """The kernel weights RangeSeasonPrior.log_prior gives each record for this place and
    date, with which records have a place / date: [(weights, known), ...]."""
    parts = []
    if ctx is None:
        return parts
    if ctx.latitude is not None and ctx.longitude is not None:
        d = haversine_km(ctx.latitude, ctx.longitude, np.nan_to_num(prior.lat),
                         np.nan_to_num(prior.lon))
        parts.append((np.exp(-0.5 * (d / prior.bandwidth_km) ** 2), ~np.isnan(prior.lat)))
    doy = ctx.day_of_year
    if doy is not None:
        d = day_distance(doy, np.nan_to_num(prior.doy))
        parts.append((np.exp(-0.5 * (d / prior.bandwidth_days) ** 2), ~np.isnan(prior.doy)))
    return parts


def log_prior_from(prior: RangeSeasonPrior, parts: list,
                   exclude: np.ndarray | None = None) -> np.ndarray:
    """log_prior from prior_weights, with the records `exclude` (indices into the
    prior's records) left out of every count, as if they weren't in the reference."""
    out = np.zeros(prior.n_species)
    if not parts:
        return out
    drop = np.zeros(len(prior.sp), dtype=bool)
    if exclude is not None and len(exclude):
        drop[exclude] = True
    for weights, known in parts:
        out += prior._log_ratio(weights, known & ~drop)
    return np.clip(out, -prior.cap, prior.cap)


def log_prior_excluding(prior: RangeSeasonPrior, ctx: Context | None,
                        exclude: np.ndarray | None = None) -> np.ndarray:
    """prior.RangeSeasonPrior.log_prior with the records `exclude` left out."""
    return log_prior_from(prior, prior_weights(prior, ctx), exclude)


def nearest_prior_scores(m1: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """nearest+prior: log_softmax(nearest / 0.02) + the place-and-date score."""
    return log_softmax(nearest_scores(m1).astype(np.float64) / LOGPROB_TEMPERATURE) + log_prior


# --- ranks ---------------------------------------------------------------------------------

def rank_scores(scores: np.ndarray, label_of: np.ndarray, n_labels: int,
                species: bool) -> np.ndarray:
    """evaluate.rank_scores: species groups only at species; the best group per label
    above it."""
    keep = label_of >= 0
    if species:
        return scores[keep]
    out = np.full(n_labels, -np.inf, dtype=np.float64)
    np.maximum.at(out, label_of[keep], scores[keep])
    return out


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best finite scores, best first (ties by index, as argsort)."""
    finite = np.isfinite(scores)
    order = np.argsort(-np.where(finite, scores, -np.inf), kind="stable")[:k]
    return order[finite[order]]


# --- who and what is the same collection ----------------------------------------------------

def same_collection(user: int | None, lat: float | None, lon: float | None,
                    day: int | None, ref_user: np.ndarray, ref_lat: np.ndarray,
                    ref_lon: np.ndarray, ref_day: np.ndarray,
                    km: float = SAME_COLLECTION_KM, days: int = SAME_COLLECTION_DAYS
                    ) -> np.ndarray:
    """Reference records (bool mask) by the same observer within `km` and `days` of the
    query: one collecting trip, likely the same or a sister collection. `day` and
    `ref_day` are day numbers (date.toordinal); unknown places or dates never match."""
    out = np.zeros(len(ref_user), dtype=bool)
    if user is None or lat is None or lon is None or day is None:
        return out
    cand = np.flatnonzero((ref_user == user) & ~np.isnan(ref_lat) & (ref_day >= 0)
                          & (np.abs(ref_day - day) <= days))
    if len(cand):
        d = haversine_km(lat, lon, ref_lat[cand], ref_lon[cand])
        out[cand[d <= km]] = True
    return out


def matched_control(same: np.ndarray, group: np.ndarray, seed: int) -> np.ndarray:
    """Records (bool mask) to remove as a control for removing `same`: per group, as many
    records as `same` removes there, drawn at random from the group's other records (all
    of them when it has fewer). The control takes as much depth from each species as the
    condition, so what differs between the two is only whose records went."""
    out = np.zeros(len(same), dtype=bool)
    k = np.bincount(group[same], minlength=int(group.max()) + 1)
    cand = np.flatnonzero(~same & (k[group] > 0))
    if not len(cand):
        return out
    key = np.random.default_rng(seed).random(len(cand))
    c = cand[np.lexsort((key, group[cand]))]
    g = group[c]
    rank = np.arange(len(c)) - np.searchsorted(g, g)
    out[c[rank < k[g]]] = True
    return out


# --- slices --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Bands:
    edges: tuple[float, ...]       # upper bounds (inclusive), ascending
    labels: tuple[str, ...]        # one more label than edges: the last is "above"

    def of(self, x) -> str:
        for e, lab in zip(self.edges, self.labels):
            if x <= e:
                return lab
        return self.labels[-1]


OBSERVER_REFS = Bands((0, 9, 99), ("0", "1-9", "10-99", "100+"))
PHOTOS = Bands((1, 3, 6), ("1", "2-3", "4-6", "7+"))
NEAREST_SAME_SPECIES_KM = Bands((10, 50, 200, 1000), ("<=10 km", "10-50 km", "50-200 km",
                                                     "200-1000 km", ">1000 km"))
REFS_WITHIN_100KM = Bands((99, 999, 4999), ("<100", "100-999", "1,000-4,999", "5,000+"))


def season_share(doy: int | None, species_doys: np.ndarray, window: int = 30) -> float | None:
    """Share of the true species' dated reference records within `window` days of the
    query's day of the year (circular), or None when either has no date."""
    doys = species_doys[species_doys >= 0]
    if doy is None or not len(doys):
        return None
    return float((day_distance(doy, doys) <= window).mean())


def season_band(share: float | None) -> str:
    if share is None:
        return "n/a (no date or no dated refs)"
    if share == 0:
        return "off season (0% of refs within 30 d)"
    if share < 0.2:
        return "edge (<20%)"
    if share < 0.5:
        return "in season (20-50%)"
    return "peak (50%+)"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))
