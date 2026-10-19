"""Place and date for the served identifier (Steve, 2026-10-11): `nearest+mean+place`.

The boost prior from exp/prior-tuning (entry 2026-10-09-prior-tuning, private research repo),
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

# Named settings for the place prior (Steve, 2026-10-13: "place should push far species
# down", reversing 2026-10-10). Each adds to BOOST, which every setting keeps:
# - shrink: records' worth of weight pulling a species' place boost to its genus's
#   (prior.RangeSeasonPrior.shrink; 3 as served). Higher damps the boost a species with
#   few records gets from one or two of them lying nearby;
# - distance penalty: slope nats per 1,000 km beyond start_km from the candidate's nearest
#   known record, capped at pen_cap; known = our DNA records ("dna") or those and iNat's
#   open-data finds ("dna+inat"); times n / (n + damp_k), n = its DNA records with a
#   place, so a species known from a few records is pushed down less;
# - season_penalty: nats off a species with a known season (the cue's in_season false).
# A candidate pushed down for distance still carries the far-find cue (PlaceCue).
# MV_PLACE_SETTING picks one; unset, the served setting of 2026-10-11 (no change).
SETTING_ENV = "MV_PLACE_SETTING"
DEFAULT_SETTING = "boost-2026-10-11"
NO_EXTRAS = {"shrink": 3.0, "pen_slope": 0.0, "pen_start_km": 300.0, "pen_cap": 0.0,
             "pen_known": "dna", "damp_k": 0.0, "season_penalty": 0.0}
PLACE_SETTINGS: dict[str, dict] = {
    DEFAULT_SETTING: dict(NO_EXTRAS),
    # The candidates from the v1 re-tune (experiment place-penalty, 2026-10-13; v1 val,
    # bioclip-2-ft-20261013-003428, nested 5-fold CV by observer, out of fold), for
    # Steve's pick. Species top-1: photos only 54.9%, served 57.1%.
    # Picked under Steve's band rule (the 300-1,500 km band within 2% of photos only on
    # the training folds): 57.7%, the band -1.8% against photos only (served -2.6%).
    "penalty-500-2026-10-13": {"shrink": 10.0, "pen_slope": 2.0, "pen_start_km": 500.0,
                               "pen_cap": 8.0, "pen_known": "dna", "season_penalty": 1.0},
    # Best overall, no band rule: 57.8%, the band -3.6% against photos only.
    "penalty-300-2026-10-13": {"shrink": 10.0, "pen_slope": 2.0, "pen_start_km": 300.0,
                               "pen_cap": 8.0, "pen_known": "dna", "season_penalty": 1.0},
    # Boost shrinkage alone (the sparse-species fix; no new penalty): 57.0%.
    "shrink-2026-10-13": {"shrink": 10.0},
    # The band-rule pick on record-grouped scoring (E32 arm c, train cross-post twins
    # grouped; v1 val, erratum r4, the answer key; experiment v1-c-place-recheck,
    # 2026-10-19): penalty-500 with shrink 30 (grid setting 748). It beat penalty-500 by
    # one find at species top-1 on all val and held the 300-1,500 km band better; Steve
    # followed the declared rule (PLAN O9). Out of fold 57.1% against photos only 54.1%,
    # the band -1.6% against photos only. penalty-500-2026-10-13 stays for rollback.
    "penalty-500-s30-2026-10-19": {"shrink": 30.0, "pen_slope": 2.0, "pen_start_km": 500.0,
                                   "pen_cap": 8.0, "pen_known": "dna", "season_penalty": 1.0},
}


# Whose calibration the place method states its confidence and likely sets with (experiment
# v1-place-calibration, 2026-10-18): "shared" (default, as served since 2026-10-11) =
# nearest+mean's; "own" = its own comparison row, when one was fitted under the place
# setting in force (else nearest+mean's still: a row from another setting is not its own).
CALIBRATION_ENV = "MV_PLACE_CALIBRATION"
CALIBRATIONS = ("shared", "own")


def calibration_choice() -> str:
    """MV_PLACE_CALIBRATION, "shared" when unset; an unknown value is refused."""
    from . import config
    choice = config.setting(CALIBRATION_ENV) or "shared"
    if choice not in CALIBRATIONS:
        raise ValueError(f"{CALIBRATION_ENV}={choice!r}: one of {', '.join(CALIBRATIONS)}")
    return choice


def place_setting(name: str | None = None) -> tuple[str, dict]:
    """(name, setting): `name`, else MV_PLACE_SETTING, else the served default. An unknown
    name is refused, so a typo never serves a prior nobody chose."""
    from . import config
    name = name or config.setting(SETTING_ENV) or DEFAULT_SETTING
    if name not in PLACE_SETTINGS:
        raise ValueError(f"{SETTING_ENV}={name!r} is not a place setting "
                         f"({', '.join(sorted(PLACE_SETTINGS))})")
    return name, {**NO_EXTRAS, **PLACE_SETTINGS[name]}


def distance_penalty(km: np.ndarray, start_km: float, slope: float, cap: float) -> np.ndarray:
    """Nats taken off for lying `km` from the nearest known record: `slope` per 1,000 km
    beyond `start_km`, at most `cap`. No known record (inf or NaN): nothing, since there is
    no evidence of where it lives."""
    km = np.asarray(km, dtype=np.float64)
    known = np.isfinite(km)
    over = np.where(known, np.maximum(np.nan_to_num(km, posinf=0.0) - start_km, 0.0), 0.0)
    return np.minimum(slope * over / 1000.0, cap)


def count_damping(n: np.ndarray, k: float) -> np.ndarray:
    """n / (n + k): 1 for k = 0; a taxon known from few records is pushed down less."""
    n = np.asarray(n, dtype=np.float64)
    if not k:
        return np.ones_like(n)
    return n / (n + k)


def off_season(near: np.ndarray, dated: np.ndarray) -> np.ndarray:
    """A known season (MIN_DATED_FOR_SEASON+ dated records) with no record within
    SEASON_WINDOW_DAYS of this day of the year: PlaceCue's in_season false."""
    return (np.asarray(dated) >= MIN_DATED_FOR_SEASON) & (np.asarray(near) == 0)


def nearest_inat_km(far, lat: float, lon: float, uuid: str | None = None) -> np.ndarray:
    """(n_groups,) km from a place to each group's nearest iNat open-data find (cell
    centre), inf with none. The query's own observation, when the store counted it and
    the pair counts are kept (tuning), is taken back out. `far`: a fitted OccurrencePrior."""
    from .prior import haversine_km
    d = haversine_km(lat, lon, far.cell_lat, far.cell_lon)
    dp = d[far.pair_pos]
    n_local = len(far.total)
    if getattr(far, "pair_count", None) is not None and uuid:
        own = far.store.own_contribution(uuid)
        if own:
            at = int(far.pos_of_cell[own.cell])
            mine = {far.local[u] for u in own.units if u in far.local}
            lone = (far.pair_pos == at) & np.isin(far.pair_local, list(mine)) \
                & (far.pair_count <= 1)
            dp = np.where(lone, np.inf, dp)
    per_local = np.full(n_local, np.inf)
    if len(dp):
        # Pairs are grouped by unit in order (OccurrencePrior.fit): one reduce per unit.
        pl = np.asarray(far.pair_local)
        starts = np.flatnonzero(np.r_[True, pl[1:] != pl[:-1]])
        per_local[pl[starts]] = np.minimum.reduceat(dp, starts)
    out = np.full(far.n_groups, np.inf)
    ok = far.g_sp >= 0
    out[ok] = per_local[far.g_sp[ok]]
    return out

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

    def __init__(self, store=None, setting: str | None = None):
        self._store = store
        self.setting_name, self.setting = place_setting(setting)

    @staticmethod
    def ready() -> bool:
        from .occprior import default_store_path
        return default_store_path().is_file()

    def fit(self, records, species: list[str]) -> None:
        from .occprior import OccParams, OccurrencePrior
        from .occurrence import OccurrenceStore
        s = self.setting
        self.dna = RangeSeasonPrior(bandwidth_km=BOOST["place_km"],
                                    bandwidth_days=BOOST["season_days"], cap=BOOST["cap"],
                                    shrink=s["shrink"])
        self.dna.fit(records, species)
        self.n_groups = len(species)
        if self.has_extras():
            self._fit_extras(records, species)     # the default setting holds none of it
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

    def _fit_extras(self, records, species: list[str]) -> None:
        """What the setting's penalties read: each group's DNA records as 0.2-degree cell
        centres (as the cue measures), its records with a place, and its dated days."""
        pos = {s_: i for i, s_ in enumerate(species)}
        self.n_groups = len(species)
        cells: dict[tuple[int, float, float], None] = {}
        n_place = np.zeros(self.n_groups)
        dated: list[tuple[int, int]] = []
        for r in records:
            g = pos.get(r.unit)
            if g is None:
                continue
            if r.latitude is not None and r.longitude is not None:
                cells[(g, *snap(r.latitude, r.longitude))] = None
                n_place[g] += 1
            doy = Context(observed_on=r.observed_on).day_of_year
            if doy is not None:
                dated.append((g, doy))
        c = np.array(list(cells), dtype=np.float64).reshape(-1, 3)
        self.cell_group = c[:, 0].astype(np.int64)
        self.cell_lat, self.cell_lon = c[:, 1], c[:, 2]
        self.n_place = n_place
        d = np.array(dated, dtype=np.int64).reshape(-1, 2)
        self.doy_group, self.doy = d[:, 0], d[:, 1]
        self.dated = np.bincount(self.doy_group, minlength=self.n_groups)

    def nearest_dna_km(self, ctx: Context) -> np.ndarray:
        """(n_groups,) km to each group's nearest DNA record's cell centre (inf: none)."""
        out = np.full(self.n_groups, np.inf)
        if len(self.cell_group):
            d = haversine_km(ctx.latitude, ctx.longitude, self.cell_lat, self.cell_lon)
            np.minimum.at(out, self.cell_group, d)
        return out

    def known_km(self, ctx: Context) -> np.ndarray:
        """Distance to the nearest known record, by the setting's `pen_known`."""
        km = self.nearest_dna_km(ctx)
        if self.setting["pen_known"] == "dna+inat" and self.far is not None:
            km = np.minimum(km, nearest_inat_km(self.far, ctx.latitude, ctx.longitude,
                                                ctx.uuid))
        return km

    def has_extras(self) -> bool:
        s = self.setting
        return bool((s["pen_slope"] and s["pen_cap"]) or s["season_penalty"])

    def extras(self, ctx: Context | None) -> np.ndarray | None:
        """The setting's added penalties (nats, <= 0), or None when it adds none."""
        s = self.setting
        if not self.has_extras():
            return None
        out = np.zeros(self.n_groups)
        used = False
        if s["pen_slope"] and s["pen_cap"] and has_place(ctx):
            pen = distance_penalty(self.known_km(ctx), s["pen_start_km"], s["pen_slope"],
                                   s["pen_cap"])
            out -= pen * count_damping(self.n_place, s["damp_k"])
            used = True
        if s["season_penalty"] and has_date(ctx):
            within = day_distance(ctx.day_of_year, self.doy) <= SEASON_WINDOW_DAYS
            near = np.bincount(self.doy_group, weights=within.astype(float),
                               minlength=self.n_groups)
            out -= s["season_penalty"] * off_season(near, self.dated)
            used = True
        return out if used else None

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
            # Place only (out of range reads no date), but the find's uuid goes along: the
            # store takes the find's own iNat observation back out of its counts.
            parts = self.far.parts(Context(ctx.latitude, ctx.longitude, uuid=ctx.uuid), (r,))
            far = parts.out(r, genus_rule=False, min_dna_effort=BOOST["min_dna_effort"])
            if far is not None:
                out = out - BOOST["far_penalty"] * far
        extra = self.extras(ctx)
        return out if extra is None else out + extra


class NearestMeanPlace:
    """nearest + species average, plus the boost prior when the context has a place or a
    date. Its confidence uses nearest + species average's calibration (`calibration_from`):
    on the held-out test that temperature also fits the boosted scores best (ECE 0.020
    against 0.047 at the temperature fitted on them)."""
    name = "nearest+mean+place"
    needs_context = True
    calibration_from = "nearest+mean"

    base_method = "nearest+mean"     # the photo scoring the place shift is added to

    def __init__(self, store=None):
        from .methods import METHODS
        self.base = METHODS[self.base_method]()
        self.prior = PlaceBoost(store)

    @staticmethod
    def ready() -> bool:
        return PlaceBoost.ready()

    def fit(self, vectors: np.ndarray, index, records=None) -> None:
        self.base.fit(vectors, index)
        self.prior.fit(records or [], index.species)

    def scoreboard_extra(self) -> dict:
        """Saved with a comparison row (evaluate.compare): the place setting the row was
        scored under, so an own calibration is only ever read from a row of that setting."""
        return {"place_setting": self.prior.setting_name}

    def place_shift(self, context: Context | None) -> np.ndarray | None:
        """What the place and date add to every photo's score, or None (no place, no date)."""
        lp = self.prior.log_prior(context)
        return None if lp is None else PHOTO_TEMPERATURE * lp

    def per_photo_scores(self, photo_sims: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Photos only (nearest + species average's own), for identify's shared path."""
        return self.base.per_photo_scores(photo_sims, query)

    def leave_out(self, per_photo, photo_sims, query, hidden):
        """nearest + species average's (photos only: a caller leaving a reference record
        out sends no place or date, since that record is part of the prior)."""
        return self.base.leave_out(per_photo, photo_sims, query, hidden)

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
