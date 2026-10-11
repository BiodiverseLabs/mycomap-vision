"""Place and date for the served identifier (Steve, 2026-10-11): `nearest+mean+place`.

The boost prior from exp/prior-tuning (docs/experiments/2026-10-09-prior-tuning.md),
added to nearest + species average only when an identification has a place, a date, or
both. With neither, the answer is nearest + species average's, exactly: same scores, same
confidences (it shares that method's calibration).

The prior, on the log-probability scale of photo scores / 0.02 (the settings chosen on the
held-out development split and confirmed once on its test split; never tuned on test):

- place: our DNA records near the place rank a species UP (the positive part of its
  place score, 75 km kernel, capped at log 5); having none nearby costs nothing;
- date: the species' share of DNA records near this day of the year (10-day kernel,
  capped at log 5 either way), weight 0.5;
- beyond the wide berth: -2 for a species with enough finds (iNat open-data occurrences
  plus our DNA records) and none within 1,500 km (occprior.OccurrencePrior, no genus rule).

Measured on the held-out test split (10,105 records, confirmation only): species top-1
54.3 -> 56.1%, genus 83.0 -> 83.4%; it still costs some records whose species is known
only 100-1,500 km away (see the experiment entry), which is why every candidate also
carries a cue (PlaceCue) saying how far its nearest DNA record is and whether the date is
in its known season, so a far find stands out instead of being buried.

Nothing here returns a coordinate. Distances in the cue are measured to the CENTRE of the
0.2-degree cell a record lies in (the grid iNat obscures sensitive records to) and rounded
to 10 km, so no answer can place a record more finely than iNat's public obscured cell.
"""

from __future__ import annotations

import math

import numpy as np

from .prior import Context, RangeSeasonPrior, day_distance, haversine_km

PHOTO_TEMPERATURE = 0.02            # the scale the prior was tuned on: photo score / 0.02
LOG5 = math.log(5.0)
BOOST = {"place_km": 75.0, "place_weight": 1.0, "cap": LOG5,
         "season_days": 10.0, "season_weight": 0.5,
         "far_penalty": 2.0, "radius_km": 1500.0, "min_dna_effort": 500}

# The cue.
FAR_KM = 300.0        # beyond this a top answer was right under a third of the time on the
                      # held-out test (30.8% at 300-1,500 km vs 50.1% at 100-300 km and
                      # 64.3% under 100 km, nearest + species average), and it is where the
                      # prior starts to cost: worth a second look, possibly a range extension
SEASON_WINDOW_DAYS = 30   # in season: a DNA record within 30 days of this day of the year
                          # (any year), as the stress lane's off-season slice
MIN_DATED_FOR_SEASON = 5  # fewer dated records: no known season (in_season null)
CELL_DEG = 0.2            # records snap to the centre of iNat's obscuring cell
KM_STEP = 10              # distances rounded to this
FAR_NOTE = "far from known records (possible range extension)"
SEASON_NOTE = "outside its known season"


def has_place(ctx: Context | None) -> bool:
    return ctx is not None and ctx.latitude is not None and ctx.longitude is not None


def has_date(ctx: Context | None) -> bool:
    return ctx is not None and ctx.day_of_year is not None


class PlaceBoost:
    """The prior. `far` is None when this machine has no occurrence store; then
    `ready()` is False and the method is not offered (methods.method_ready)."""

    def __init__(self, store=None):
        self._store = store

    @staticmethod
    def ready() -> bool:
        from .occprior import default_store_path
        return default_store_path().is_file()

    def fit(self, records, species: list[str]) -> None:
        from .occprior import OccParams, OccurrencePrior
        from .occurrence import OccurrenceStore
        self.dna = RangeSeasonPrior(bandwidth_km=BOOST["place_km"],
                                    bandwidth_days=BOOST["season_days"], cap=BOOST["cap"])
        self.dna.fit(records, species)
        store, own_store = self._store, False
        if store is None and self.ready():
            store, own_store = OccurrenceStore.load(), True
        self.far = None
        if store is not None:
            # Out of range only: no density or season terms, and no season table built.
            self.far = OccurrencePrior(store, OccParams(
                radius_km=BOOST["radius_km"], genus_rule=False,
                min_dna_effort=BOOST["min_dna_effort"], density_weight=0.0,
                season_weight=0.0), out_of_range_only=True)
            self.far.fit(records, species)
            if own_store:
                # Fitted, out of range reads only the store's grid: let the tables it was
                # fitted from go (~25 MB the 4 GB box would otherwise hold).
                for name in ("pair_cell", "pair_count", "week_unit", "week_band",
                             "week_week", "week_count", "excluded_hashes"):
                    setattr(store, name, None)
                store.names = []

    def log_prior(self, ctx: Context | None) -> np.ndarray | None:
        """(n_groups,) on the photo-score / 0.02 scale, or None without place and date."""
        place_ok, date_ok = has_place(ctx), has_date(ctx)
        if not (place_ok or date_ok):
            return None
        place, season = self.dna.terms(ctx)
        cap = BOOST["cap"]
        out = (BOOST["place_weight"] * np.clip(place, 0.0, cap)
               + BOOST["season_weight"] * np.clip(season, -cap, cap))
        if self.far is not None and place_ok:
            r = BOOST["radius_km"]
            parts = self.far.parts(Context(ctx.latitude, ctx.longitude), (r,))
            far = parts.out(r, genus_rule=False, min_dna_effort=BOOST["min_dna_effort"])
            if far is not None:
                out = out - BOOST["far_penalty"] * far
        return out


class NearestMeanPlace:
    """nearest + species average, plus the boost prior when the context has a place or a
    date. Its confidence uses nearest + species average's calibration (`calibration_from`):
    on the held-out test that temperature also fits the boosted scores best (ECE 0.020
    against 0.047 at the temperature fitted on them)."""
    name = "nearest+mean+place"
    needs_context = True
    calibration_from = "nearest+mean"

    def __init__(self, store=None):
        from .methods import NearestAndMean
        self.base = NearestAndMean()
        self.prior = PlaceBoost(store)

    @staticmethod
    def ready() -> bool:
        return PlaceBoost.ready()

    def fit(self, vectors: np.ndarray, index, records=None) -> None:
        self.base.fit(vectors, index)
        self.prior.fit(records or [], index.species)

    def place_shift(self, context: Context | None) -> np.ndarray | None:
        """What the place and date add to every photo's score, or None (no place, no date)."""
        lp = self.prior.log_prior(context)
        return None if lp is None else PHOTO_TEMPERATURE * lp

    def per_photo_scores(self, photo_sims: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Photos only (nearest + species average's own), for identify's shared path."""
        return self.base.per_photo_scores(photo_sims, query)

    def species_scores(self, query: np.ndarray, context: Context | None = None) -> np.ndarray:
        scores = self.base.species_scores(query)
        shift = self.place_shift(context)
        return scores if shift is None else scores + shift

    def photo_sims(self, query: np.ndarray) -> np.ndarray:
        return self.base.photo_sims(query)


class PlaceCue:
    """Per taxon: how far its nearest DNA record is from a place, and whether a date is in
    its known season. Built from the reference records (any method); answers in bands
    coarse enough that no record can be placed (module notes)."""

    def fit(self, records, index) -> None:
        pos = {s: i for i, s in enumerate(index.species)}
        n = len(index.species)
        self.n_groups = n
        self.label_of = {rank: index.label_of[rank] for rank in ("species", "genus")}
        self.attach(index)
        cells: dict[tuple[int, float, float], None] = {}
        dated: list[tuple[int, int]] = []
        for r in records:
            g = pos.get(r.unit)
            if g is None:
                continue
            if r.latitude is not None and r.longitude is not None:
                cells[(g, *snap(r.latitude, r.longitude))] = None
            doy = Context(observed_on=r.observed_on).day_of_year
            if doy is not None:
                dated.append((g, doy))
        c = np.array(list(cells), dtype=np.float64).reshape(-1, 3)
        self.cell_group = c[:, 0].astype(np.int32)
        self.cell_lat = c[:, 1].astype(np.float32)
        self.cell_lon = c[:, 2].astype(np.float32)
        d = np.array(dated, dtype=np.int32).reshape(-1, 2)
        self.doy_group, self.doy = d[:, 0], d[:, 1].astype(np.int16)
        self.dated = np.bincount(self.doy_group, minlength=n)

    def attach(self, index) -> None:
        """Look names up in `index` (call again after Identifier remakes its name lists,
        so the cue holds none of the strings made while the records were read)."""
        self.label_pos = {rank: {name: i for i, name in enumerate(index.labels[rank])}
                          for rank in ("species", "genus")}

    def _by_rank(self, per_group: np.ndarray, rank: str, how: str) -> np.ndarray:
        """Per label of `rank`: min (distance) or sum (counts) over its groups."""
        lab = self.label_of[rank]
        keep = lab >= 0
        size = len(self.label_pos[rank])
        if how == "min":
            out = np.full(size, np.inf)
            np.minimum.at(out, lab[keep], per_group[keep])
        else:
            out = np.zeros(size)
            np.add.at(out, lab[keep], per_group[keep])
        return out

    def tables(self, ctx: Context | None) -> dict | None:
        """{rank: (km per label, in-season count per label, dated count per label)} for
        this context, or None without place and date."""
        place_ok, date_ok = has_place(ctx), has_date(ctx)
        if not (place_ok or date_ok):
            return None
        out = {}
        km_g = np.full(self.n_groups, np.inf)
        if place_ok:
            d = haversine_km(ctx.latitude, ctx.longitude, self.cell_lat.astype(np.float64),
                             self.cell_lon.astype(np.float64))
            np.minimum.at(km_g, self.cell_group, d)
        near_g = np.zeros(self.n_groups)
        if date_ok:
            within = day_distance(ctx.day_of_year, self.doy.astype(np.int64)) \
                <= SEASON_WINDOW_DAYS
            near_g = np.bincount(self.doy_group, weights=within.astype(float),
                                 minlength=self.n_groups)
        for rank in ("species", "genus"):
            out[rank] = (self._by_rank(km_g, rank, "min") if place_ok else None,
                         self._by_rank(near_g, rank, "sum") if date_ok else None,
                         self._by_rank(self.dated.astype(float), rank, "sum"))
        return out

    def cue(self, tables: dict | None, rank: str, name: str) -> dict | None:
        """The cue for one candidate, or None without place and date (or at family)."""
        if tables is None or rank not in tables:
            return None
        i = self.label_pos[rank].get(name)
        km_t, near_t, dated_t = tables[rank]
        km = None
        if km_t is not None and i is not None and np.isfinite(km_t[i]):
            km = int(round(km_t[i] / KM_STEP) * KM_STEP)
        in_season = None
        if near_t is not None and i is not None and dated_t[i] >= MIN_DATED_FOR_SEASON:
            in_season = bool(near_t[i] > 0)
        far = None if km is None else km > FAR_KM
        notes = [n for n, on in ((FAR_NOTE, far), (SEASON_NOTE, in_season is False)) if on]
        return {"nearest_record_km": km, "in_season": in_season, "far_from_known": far,
                "note": "; ".join(notes) or None}


def snap(lat: float, lng: float) -> tuple[float, float]:
    """The centre of the 0.2-degree cell a point lies in."""
    return ((math.floor(lat / CELL_DEG) + 0.5) * CELL_DEG,
            (math.floor(lng / CELL_DEG) + 0.5) * CELL_DEG)
