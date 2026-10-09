"""Where and when: a range-and-season score per species from DNA-verified records.

For a place and date, each species gets the log of how much more often it has
been DNA-verified near that place (and near that time of year) than DNA
sampling in general would predict there. Dividing by all records' density is
the effort correction: a heavily sampled state doesn't make every species in it
look likely. Species with few records borrow strength from their genus, and the
genus from "no information", so one record doesn't draw a sharp range.

The score is capped (default: a factor of 20 either way), so location can shift
an answer but never overrule clear photos. It is computed from the records' true
coordinates but only ever returns scores, never a location.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .dates import is_placeholder_date

EARTH_KM = 6371.0

# Confidence for log-probability scores (a +prior or +occ method) before a comparison
# has calibrated them. In comparison 20261008-012435-4ef7b0 the fitted species
# temperature of nearest on the fine-tuned BioCLIP 2 was 0.037 on cosine scores: 1.9
# on the log-probability scale AsLogProb's 0.02 puts them on. At the cosine default
# (0.02) every answer read ~100%, right or wrong.
LOGPROB_CONFIDENCE_TEMPERATURE = 1.9


@dataclass
class Context:
    latitude: float | None = None
    longitude: float | None = None
    observed_on: str | None = None       # ISO date
    # The query's own iNat observation, when it is one: an occurrence prior takes it
    # back out of its counts, so a record never finds itself (occprior.py).
    uuid: str | None = None

    @property
    def day_of_year(self) -> int | None:
        """None when there is no date, including the 1970-01-01 placeholder (dates.py)."""
        if is_placeholder_date(self.observed_on):
            return None
        try:
            return date.fromisoformat((self.observed_on or "")[:10]).timetuple().tm_yday
        except ValueError:
            return None


def haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat), np.radians(lats)
    dphi, dlmb = p2 - p1, np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def day_distance(a: int, days: np.ndarray) -> np.ndarray:
    d = np.abs(days - a)
    return np.minimum(d, 365 - d)


class RangeSeasonPrior:
    bandwidth_km = 150.0
    bandwidth_days = 20.0
    shrink = 3.0            # records' worth of weight given to the level above
    cap = float(np.log(20.0))

    def fit(self, records, species: list[str]) -> None:
        """records: reference records (with latitude, longitude, observed_on, genus);
        species: the index's group order (evaluate.Index.species), which scores follow.
        A record named with one word is in its own group; a group with no genus
        borrows strength from itself only."""
        pos = {s: i for i, s in enumerate(species)}
        genus_of = lambda r: r.genus or r.unit  # noqa: E731
        genera = sorted({genus_of(r) for r in records})
        gpos = {g: i for i, g in enumerate(genera)}
        rows = [(pos[r.unit], gpos[genus_of(r)], r.latitude, r.longitude,
                 Context(observed_on=r.observed_on).day_of_year)
                for r in records if r.unit in pos]
        self.n_species, self.n_genera = len(species), len(genera)
        self.sp = np.array([r[0] for r in rows], dtype=np.int64)
        self.ge = np.array([r[1] for r in rows], dtype=np.int64)
        self.lat = np.array([np.nan if r[2] is None else r[2] for r in rows], dtype=np.float64)
        self.lon = np.array([np.nan if r[3] is None else r[3] for r in rows], dtype=np.float64)
        self.doy = np.array([np.nan if r[4] is None else r[4] for r in rows], dtype=np.float64)
        self.genus_of_species = np.zeros(len(species), dtype=np.int64)
        for s, g in zip(self.sp, self.ge):
            self.genus_of_species[s] = g

    def _log_ratio(self, weights: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Per species: log(species density / overall density), shrunk to genus, then to 0."""
        w = np.where(known, weights, 0.0)
        n_all = max(known.sum(), 1)
        overall = w.sum() / n_all
        if overall <= 0:
            return np.zeros(self.n_species)
        n_sp = np.bincount(self.sp[known], minlength=self.n_species).astype(float)
        k_sp = np.bincount(self.sp, weights=w, minlength=self.n_species)
        n_ge = np.bincount(self.ge[known], minlength=self.n_genera).astype(float)
        k_ge = np.bincount(self.ge, weights=w, minlength=self.n_genera)
        # Ratios, each pulled towards the level above by `shrink` records' worth.
        r_ge = (k_ge / overall + self.shrink * 1.0) / (n_ge + self.shrink)
        r_up = r_ge[self.genus_of_species]
        r_sp = (k_sp / overall + self.shrink * r_up) / (n_sp + self.shrink)
        return np.log(np.clip(r_sp, 1e-6, None))

    def log_prior(self, ctx: Context | None) -> np.ndarray:
        """(n_species,) capped log-ratio for this place and date; zeros when unknown."""
        out = np.zeros(self.n_species)
        if ctx is None:
            return out
        if ctx.latitude is not None and ctx.longitude is not None:
            known = ~np.isnan(self.lat)
            d = haversine_km(ctx.latitude, ctx.longitude, np.nan_to_num(self.lat),
                             np.nan_to_num(self.lon))
            out += self._log_ratio(np.exp(-0.5 * (d / self.bandwidth_km) ** 2), known)
        doy = ctx.day_of_year
        if doy is not None:
            known = ~np.isnan(self.doy)
            d = day_distance(doy, np.nan_to_num(self.doy))
            out += self._log_ratio(np.exp(-0.5 * (d / self.bandwidth_days) ** 2), known)
        return np.clip(out, -self.cap, self.cap)


class WithPrior:
    """A log-probability method plus the range-and-season score."""
    needs_context = True
    confidence_temperature = LOGPROB_CONFIDENCE_TEMPERATURE   # identify.temperature

    def __init__(self, base_cls, weight: float = 1.0):
        self.base = base_cls()
        self.name = f"{self.base.name}+prior"      # AsLogProb keeps its base's name
        self.weight = weight

    def fit(self, vectors: np.ndarray, index, records=None, state: dict | None = None) -> None:
        if hasattr(self.base, "state"):
            self.base.fit(vectors, index, state=state)
        else:
            self.base.fit(vectors, index)
        self.prior = RangeSeasonPrior()
        self.prior.fit(records or [], index.species)

    @property
    def trainable(self) -> bool:
        return hasattr(self.base, "state")

    def state(self) -> dict:
        return self.base.state()

    def species_scores(self, query: np.ndarray, context: Context | None = None) -> np.ndarray:
        return self.base.species_scores(query) + self.weight * self.prior.log_prior(context)

    def photo_sims(self, query: np.ndarray) -> np.ndarray:
        return self.base.photo_sims(query)
