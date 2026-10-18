"""Identify uploaded photos against every DNA-verified record a model has embedded.

A model is a backbone plus a method (see evaluate.METHODS). The whole embedded
set is the reference: records, species and higher ranks all come from .org's
DNA-validated names.

Confidence is a softmax over each rank's candidate scores. Its temperature per
rank comes from the model's latest comparison (`mv compare`), fitted so the
stated confidence matches how often the answer was right on the newest weeks.
Without a comparison it falls back to a fixed temperature and says so.
"""

from __future__ import annotations

import gc
import hashlib
import sqlite3
from bisect import bisect_left
from typing import Callable, NamedTuple
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import config, fungusgate, likely, taxonomy
from .embed import normalise
from .methods import rows_at
from .serving import map_embeddings
from .evaluate import (METHODS, RANKS, NearestSpecimen, build_index, load_records, make_method,
                       rank_scores, with_rows)
from .licenses import photo_url
from .higher import HIGHER_RANKS, HigherRanks
from .placeprior import PlaceCue
from .sources import INAT, record_url

CONFIDENCE_TEMPERATURE = 0.02
ORG_SPECIES_URL = "https://mycomap.org/species/"


def softmax_confidence(scores: np.ndarray, temperature: float = CONFIDENCE_TEMPERATURE) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    z = (s - np.max(s)) / temperature
    e = np.exp(z)
    return e / e.sum()


def improvement_hints(n_photos: int, ranks: dict[str, list[dict]],
                      top_species_refs: int) -> list[str]:
    """Plain advice on how to firm up this identification."""
    hints: list[str] = []
    species = ranks.get("species") or []
    genus = ranks.get("genus") or []
    sp_conf = species[0]["confidence"] if species else 0.0
    ge_conf = genus[0]["confidence"] if genus else 0.0
    if n_photos == 1:
        hints.append("Add more photos: the cap from above, the underside (gills, pores or "
                     "teeth) and the stem. Most fungi need at least two views to identify.")
    if species and sp_conf < 0.5 and ge_conf >= 0.7 and len(species) > 1:
        hints.append(f"The genus looks settled, but {species[0]['name']} and "
                     f"{species[1]['name']} are close. Clear photos of the underside and stem "
                     "base, or DNA sequencing, would separate them.")
    if genus and ge_conf < 0.4:
        hints.append("Nothing in the DNA-verified set is a clear match, even at genus level. "
                     "It may be a species we have no sequenced record of yet; sequencing it "
                     "would help, and would add it to the reference set.")
    if species and top_species_refs == 1:
        hints.append(f"The best species match, {species[0]['name']}, rests on a single "
                     "DNA-verified specimen, so treat the species-level answer with caution.")
    return hints


def cache_key(backbone: str, method: str, photo_ids: np.ndarray, species: list[str]) -> str:
    """Identifies a trained model: the backbone, the method, and exactly which reference
    photos (in order) and species it was trained on."""
    h = hashlib.sha1(f"{backbone}|{method}|".encode())
    h.update(np.asarray(photo_ids, dtype=np.int64).tobytes())
    h.update("\x1f".join(species).encode())
    return h.hexdigest()[:16]


class NotTrainedHere(RuntimeError):
    """A trained method with no saved training for this reference set, on a machine
    that must not train (MV_FIT_ON_DEMAND=0)."""


def fit_cached(model, vectors: np.ndarray, index, records, cache_dir: Path, key: str,
               train: bool = True) -> bool:
    """Fit the model, reusing saved trained weights when this exact reference set has been
    trained before. Returns True when loaded from the cache. With train=False a
    trainable model without saved weights raises NotTrainedHere instead of training."""
    kw = {"records": records} if getattr(model, "needs_context", False) else {}
    # WithPrior always has state(), but only a trained base has anything to save.
    trainable = getattr(model, "trainable", hasattr(model, "state"))
    path = cache_dir / f"{key}.npz"
    if trainable and path.exists():
        with np.load(path) as saved:
            model.fit(vectors, index, state={k: saved[k] for k in saved.files}, **kw)
        return True
    if trainable and not train:
        raise NotTrainedHere(f"{getattr(model, 'name', 'this method')} has no saved training "
                             "for this reference set, and this server doesn't train")
    model.fit(vectors, index, **kw)
    if trainable:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **model.state())
        tmp.replace(path)
    return False


# Licence classes whose photos may be shown in results without asking: Creative
# Commons (with or without non-commercial / no-derivatives terms). An
# all-rights-reserved photo is shown only with its owner's grant from mycomap.org
# (permissions.PermissionView.may_show); otherwise it is for training only while
# permission is sought.
SHOWN_LICENSES = frozenset({"open", "nc"})


def pick_best(scores, allowed: list[bool]) -> int | None:
    """Index of the best-scoring photo that may be shown, or None if none may."""
    best = None
    for i, ok in enumerate(allowed):
        if ok and (best is None or scores[i] > scores[best]):
            best = i
    return best


def pick_shown(scores, licenses: list[str | None]) -> int | None:
    """pick_best by licence alone (no photographers' answers at hand)."""
    return pick_best(scores, [lic in SHOWN_LICENSES for lic in licenses])


@dataclass
class PhotoInfo:
    source_url: str
    license_class: str
    owner_login: str | None
    owner_user_id: int | None = None
    source: str = INAT                  # photos.source: picks the URL rule (licenses.photo_url)


def only_open_licences(info: PhotoInfo) -> bool:
    """The showing rule when no photographers' answers are at hand: CC photos only."""
    return info.license_class in SHOWN_LICENSES


def photo_info_rows(rows) -> dict[int, PhotoInfo]:
    return {int(pid): PhotoInfo(url, lic, owner, uid, src)
            for pid, url, lic, owner, uid, src in rows}


PHOTO_INFO_SQL = ("select photo_id, source_url, license_class, owner_login, owner_user_id, "
                  "source from photos")


class Specimen(NamedTuple):
    """What a result needs of a reference record."""
    observation_id: str
    species: str
    genus: str
    family: str
    source: str = INAT                   # records.source
    source_id: str = ""                  # the id at the source


class Specimens:
    """The reference record behind each column, kept as arrays. The Records a build
    reads (dates, places, projects, photo lists) are Python objects; keeping them, or
    any small object made alongside them, held ~0.5 GB at the full photo set, because
    the allocator can't hand back memory a survivor shares. These arrays: ~10 MB."""

    def __init__(self, records: list, col_rows: list[int]):
        at = {id(r): i for i, r in enumerate(records)}
        rec_of_row = {row: at[id(r)] for r in records for row in r.photo_rows}
        self.rec = np.fromiter((rec_of_row[c] for c in col_rows), dtype=np.int32,
                               count=len(col_rows))
        names = sorted({n for r in records for n in (r.species, r.genus, r.family)})
        pos = {n: i for i, n in enumerate(names)}
        self.names = np.array(names or [""])
        self.obs = np.array([r.observation_id for r in records] or [""])
        self.source = np.array([getattr(r, "source", INAT) for r in records] or [INAT])
        self.source_id = np.array([getattr(r, "source_id", "") or r.observation_id
                                   for r in records] or [""])
        self.taxa = np.array([(pos[r.species], pos[r.genus], pos[r.family]) for r in records],
                             dtype=np.int32).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.rec)

    def __getitem__(self, col: int) -> Specimen:
        r = int(self.rec[col])
        sp, ge, fa = (str(self.names[i]) for i in self.taxa[r])
        return Specimen(str(self.obs[r]), sp, ge, fa, str(self.source[r]),
                        str(self.source_id[r]))

    def starts(self) -> np.ndarray:
        """Where each record's run of columns starts (a record's photos are contiguous)."""
        return np.flatnonzero(np.diff(self.rec, prepend=-1) != 0).astype(np.int64)


class Identifier:
    """One backbone + method over all of its embedded, DNA-verified records.

    The reference vectors are memory-mapped from the embedding files (serving.py), not
    read into memory. `photo_info=False` skips the owner and licence of every reference
    photo, for callers that look them up per identification (the API does).
    `layer_root` holds the shards the nightly update added (nightly.py)."""

    def __init__(self, conn: sqlite3.Connection, backbone: str, method: str,
                 embeddings_root: Path | None = None, calibration: dict | None = None,
                 model_cache: Path | None = None, train: bool = True, photo_info: bool = True,
                 layer_root: Path | None = None, release=None,
                 gate: "fungusgate.FungusGate | None" = None):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.backbone, self.method = backbone, method
        # The not-a-fungus gate (fungusgate.py), or None: answers are then ungated.
        self.gate = gate
        ids, vecs = map_embeddings(conn, backbone, embeddings_root, layer_root)
        if not len(ids):
            raise ValueError(f"no embeddings for {backbone!r}")
        self.embedded = len(ids)
        row_of = {int(p): i for i, p in enumerate(ids.tolist())}
        by_id = load_records(conn, {int(p): int(p) for p in ids.tolist()}, release=release)
        records = with_rows(by_id, row_of)
        self.index = build_index(records)
        self.model = make_method(method, backbone)
        self.uses_context = getattr(self.model, "needs_context", False)
        self.loaded_from_cache = fit_cached(
            self.model, vecs, self.index, records,
            model_cache or config.DATA_DIR / "models",
            cache_key(backbone, method, ids[self.index.cols], self.index.species),
            train=train)
        self.nearest = self.model if isinstance(self.model, NearestSpecimen) else NearestSpecimen()
        if self.nearest is not self.model:
            self.nearest.fit(vecs, self.index)
        # Per reference column: photo id and record.
        self.col_photo = ids[self.index.cols]
        self.col_record = Specimens(records, self.index.cols.tolist())
        self.rec_starts = self.col_record.starts()
        self.calibration = calibration
        self.photos = (self._photo_info(conn, set(int(p) for p in self.col_photo.tolist()))
                       if photo_info else {})
        self.records = len(records)
        # How far each taxon's nearest DNA record is, and its known season (placeprior.py):
        # a cue beside every species and genus candidate when a place or date is given.
        self.cue = PlaceCue()
        self.cue.fit(records, self.index)
        self.rank_counts = {rank: Counter() for rank in RANKS}
        for r in records:
            for rank in RANKS:
                self.rank_counts[rank][{"species": r.species, "genus": r.genus,
                                        "family": r.family}[rank]] += 1
        # The names the index keeps were made while the records were read, scattered
        # among them, and would keep that memory (~0.5 GB at the full set) from going
        # back to the OS. Park them in arrays, let the records go, and make them again.
        del records, by_id, row_of
        self._remake_names()
        # Order, class and phylum (higher.py), from the taxonomy cache beside the manifest;
        # None without one. Only callers that ask for them get them (the public API).
        self.higher = HigherRanks.from_taxonomy(taxonomy.for_manifest(conn),
                                                self.index.labels["genus"],
                                                self.rank_counts["genus"])
        self.cue.attach(self.index)

    def _remake_names(self) -> None:
        index = self.index

        def park(counter):
            return np.array(list(counter), dtype=str), np.array(list(counter.values()), np.int64)
        species = np.array(index.species, dtype=str)
        refs = np.array([index.ref_count[s] for s in index.species], dtype=np.int64)
        labels = {rank: np.array(names, dtype=str) for rank, names in index.labels.items()}
        counts = {rank: park(c) for rank, c in self.rank_counts.items()}
        index.species = index.ref_count = index.labels = self.rank_counts = None
        gc.collect()
        index.species = species.tolist()
        index.ref_count = Counter(dict(zip(index.species, refs.tolist())))
        index.labels = {rank: names.tolist() for rank, names in labels.items()}
        self.rank_counts = {rank: Counter(dict(zip(names.tolist(), n.tolist())))
                            for rank, (names, n) in counts.items()}

    def can_leave_out(self) -> bool:
        """Whether this method can answer with reference columns left out (hidden=)."""
        return self.model is self.nearest or hasattr(self.model, "leave_out")

    def left_out(self, source: str, source_id: str, photo_ids=()) -> tuple[np.ndarray, np.ndarray]:
        """(columns to leave out, the record's own columns) for an observation identified
        by its number: the reference record(s) it is (source + id at the source), and any
        column holding one of its photos, those records' or `photo_ids` (the same photo
        filed under another observation). Both empty when it is in no way part of the
        reference."""
        recs = self.col_record
        mine = np.flatnonzero((recs.source == source) & (recs.source_id == str(source_id)))
        own = (np.flatnonzero(np.isin(recs.rec, mine)) if len(mine)
               else np.empty(0, dtype=np.int64))
        ids = np.concatenate([np.asarray([int(p) for p in photo_ids],
                                         dtype=self.col_photo.dtype), self.col_photo[own]])
        same = (np.flatnonzero(np.isin(self.col_photo, ids)) if len(ids)
                else np.empty(0, dtype=np.int64))
        return np.union1d(own, same).astype(np.int64), own.astype(np.int64)

    def reference_vectors(self, cols: np.ndarray) -> np.ndarray:
        """(len(cols), dim) unit float32 vectors of these reference columns, as served
        (an int8 row as its codes x scale)."""
        v = rows_at(self.nearest.scorer.ref, cols)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)

    def vector_storage(self) -> dict:
        """How the reference vectors are held: int8 and float rows (serving.MappedVectors),
        or all float when they are an ordinary array."""
        ref = self.nearest.scorer.ref
        storage = getattr(getattr(ref, "vectors", None), "storage", None)
        return storage() if storage else {"int8_rows": 0, "float_rows": int(len(ref))}

    def warm(self) -> None:
        """Score one blank photo: reads every reference vector once, which proves the
        files are whole and brings them into the page cache, so the first identification
        after a (re)build doesn't wait on the disk."""
        self.nearest.photo_sims(np.zeros((1, self.nearest.scorer.ref.shape[1]), np.float16))

    @staticmethod
    def _photo_info(conn: sqlite3.Connection, photo_ids: set[int]) -> dict[int, PhotoInfo]:
        infos = photo_info_rows(conn.execute(PHOTO_INFO_SQL))
        return {pid: info for pid, info in infos.items() if pid in photo_ids}

    def identify_vectors(self, query: np.ndarray, top_k: int = 5,
                         n_specimens: int = 6, context=None,
                         photo_lookup: Callable[[list[int]], dict[int, PhotoInfo]] | None = None,
                         may_show: Callable[[PhotoInfo], bool] = only_open_licences,
                         n_photo_specimens: int = 3, gate: list[dict] | None = None,
                         higher_ranks: bool = False, sims: np.ndarray | None = None,
                         hidden: np.ndarray | None = None) -> dict:
        """identify_vectors_ungated, behind the fungus gate (fungusgate.py) when `gate`
        holds a verdict per photo. A photo judged not a fungus is left out of the answer
        (a duck among mushroom photos doesn't sink the find); when every photo is, there
        is no answer at all. When no photo is clearly a fungus, the names are given but
        no stated chance: calibration holds only for fungus photos. `result["gate"]` is
        None when the gate didn't run. `higher_ranks` and `sims` as in
        identify_vectors_ungated (only the kept photos' rows of `sims` are used), and
        `hidden`."""
        kw = dict(top_k=top_k, n_specimens=n_specimens, context=context,
                  photo_lookup=photo_lookup, may_show=may_show,
                  n_photo_specimens=n_photo_specimens, higher_ranks=higher_ranks,
                  hidden=hidden)
        if gate is None:
            return {**self.identify_vectors_ungated(query, sims=sims, **kw), "gate": None,
                    "confidence_withheld": False}
        if len(gate) != len(query):
            raise ValueError(f"{len(gate)} gate verdicts for {len(query)} photos")
        verdicts = [g["verdict"] for g in gate]
        kept = [i for i, v in enumerate(verdicts) if v != fungusgate.NOT_FUNGUS]
        overall = fungusgate.observation_verdict(verdicts)
        if kept:
            out = self.identify_vectors_ungated(
                query[kept], sims=None if sims is None else sims[kept], **kw)
            renumber(out, kept, len(query))
        else:
            out = self.no_answer(len(query))
        if higher_ranks:                     # left-out photos and no answer list them empty
            for ranks in [out["ranks"], *(p["ranks"] for p in out["per_photo"])]:
                for rank in HIGHER_RANKS:
                    ranks.setdefault(rank, [])
        for p in out["per_photo"]:
            p["gate"] = gate[p["photo"]]
        out["confidence_withheld"] = False
        if overall != fungusgate.FUNGUS:
            withhold_confidence(out)
        like = fungusgate.find_looks_like(gate, overall)
        out["hints"] = fungusgate.gate_hints(verdicts, overall, like) + (
            out["hints"] if overall == fungusgate.FUNGUS else [])
        out["gate"] = {"verdict": overall,
                       "message": (fungusgate.LOOKS_LIKE[like]["message"] if like
                                   else fungusgate.MESSAGES.get(overall)),
                       "looks_like": like,
                       "left_out": [i for i in range(len(query)) if i not in kept],
                       "photos": gate}
        return out

    def no_answer(self, n_q: int) -> dict:
        """What a find that isn't a fungus gets: no names, no chance, no specimens."""
        return {
            "model": {"backbone": self.backbone, "method": self.method},
            "reference": {"records": self.records, "species": len(self.index.species),
                          "photos": self.embedded},
            "photos": n_q,
            "ranks": {rank: [] for rank in RANKS},
            "likely": {},
            "specimens": [],
            "per_photo": [{"photo": i, "ranks": {rank: [] for rank in RANKS},
                           "overall_top": {rank: None for rank in RANKS}, "specimens": []}
                          for i in range(n_q)],
            "hints": [],
            "species_url": None,
            "calibration": self.calibration,
            "confidence_note": self.confidence_note(),
            "prior": {"applied": False, "method": self.method},
            "photo_only_top": {},
        }

    def identify_vectors_ungated(self, query: np.ndarray, top_k: int = 5,
                                 n_specimens: int = 6, context=None,
                                 photo_lookup: Callable[[list[int]], dict[int, PhotoInfo]] | None = None,
                                 may_show: Callable[[PhotoInfo], bool] = only_open_licences,
                                 n_photo_specimens: int = 3, higher_ranks: bool = False,
                                 sims: np.ndarray | None = None,
                                 hidden: np.ndarray | None = None) -> dict:
        """Query photo vectors (already normalised) -> ranks, specimens, hints.
        `context` (place and date) is used by methods with a range-and-season score.
        `higher_ranks` adds order, class and phylum (higher.py) to the combined and
        per-photo ranks. `sims`: the photos' similarities to every reference photo, when
        the caller already has them (a batch scored together, workqueue.py). `hidden`:
        reference columns left out of this answer, as if never there (left_out: an
        observation that is itself a reference record must not match itself); only for
        methods that can (can_leave_out), and with no context.

        The main answer uses all the photos together. `per_photo` gives what the same
        model says from each photo on its own, so a person can see which photo pulled
        the answer which way (an underside may match well while the cap does not).

        A specimen's photo is shown only if `may_show` allows it (permissions.py:
        CC licence, or the owner's grant). `photo_lookup` reads licence and owner
        now rather than when the model was built, so a licence change on iNat or a
        withdrawal on mycomap.org applies to the next identification."""
        n_q = int(query.shape[0])
        if sims is None:
            sims = self.nearest.photo_sims(query)             # (q, reference photos)
        leaving_out = hidden is not None and len(hidden) > 0
        if leaving_out:
            if not self.can_leave_out():
                raise ValueError(f"{self.method} can't leave reference records out")
            if context is not None and (context.latitude is not None
                                        or context.observed_on):
                raise ValueError("a left-out record's own place and date are in the prior")
            unmasked, sims = sims, np.array(sims, dtype=np.float32, copy=True)
            sims[:, hidden] = -np.inf
        if self.model is self.nearest:
            # The nearest-specimen score is the mean of each photo's own scores, so
            # both come from the one similarity matrix.
            per_photo = np.maximum.reduceat(sims, self.index.starts, axis=1)
            scores = per_photo.mean(axis=0)
        elif hasattr(self.model, "place_shift"):
            # nearest + species average with place and date: the same shared path, plus
            # the prior's shift on every photo when a place or date is given. Without
            # them, exactly nearest + species average.
            per_photo = self.model.per_photo_scores(unmasked if leaving_out else sims, query)
            if leaving_out:
                per_photo = self.model.leave_out(per_photo, unmasked, query, hidden)
            photo_only = per_photo.mean(axis=0)
            shift = self.model.place_shift(context)
            if shift is not None:
                per_photo = per_photo + shift
            scores = per_photo.mean(axis=0)
        elif hasattr(self.model, "per_photo_scores") and not self.uses_context:
            # nearest + species average (the default): its specimen half comes from the
            # same similarities, so nothing is computed twice; the record's score is the
            # mean of each photo's own, exactly as species_scores gives it.
            per_photo = self.model.per_photo_scores(unmasked if leaving_out else sims, query)
            if leaving_out:
                per_photo = self.model.leave_out(per_photo, unmasked, query, hidden)
            scores = per_photo.mean(axis=0)
        else:
            scores = self._species_scores(query, context)
            per_photo = (np.stack([self._species_scores(query[i:i + 1], context)
                                   for i in range(n_q)]) if n_q > 1 else scores[None, :])
        ranks = self._ranks(scores, top_k, higher=higher_ranks)
        applied = shift is not None if hasattr(self.model, "place_shift") else False
        # Where the place and date changed the top answer, what the photos alone said.
        photo_only_top = {}
        if applied:
            alone = self._ranks(photo_only, 1)
            photo_only_top = {rank: alone[rank][0]["name"] for rank in RANKS
                              if alone[rank] and ranks[rank]
                              and alone[rank][0]["name"] != ranks[rank][0]["name"]}
        cue_tables = self._cue_tables(context)
        self._add_place_cue({rank: ranks[rank] for rank in RANKS}, cue_tables)
        likely_sets = self._likely(scores)
        self._add_place_cue({rank: s["names"] for rank, s in likely_sets.items()}, cue_tables)
        # Records are ranked by the same rule as species: for each of your photos,
        # its best match among the record's photos, averaged over your photos.
        per_record = np.maximum.reduceat(sims, self.rec_starts, axis=1)   # (q, records)
        picks = self._pick_specimens(sims, per_record.mean(axis=0), None, n_specimens)
        photo_picks = [self._pick_specimens(sims, per_record[i], i, n_photo_specimens)
                       for i in range(n_q)]
        wanted = [p for pk in [picks, *photo_picks] for *_, ids, _ in pk for p in ids]
        infos = photo_lookup(sorted(set(wanted))) if photo_lookup else self.photos
        top = {rank: (ranks[rank][0]["name"] if ranks[rank] else None) for rank in RANKS}
        photos = []
        for i in range(n_q):
            own = self._ranks(per_photo[i], top_k, with_position_of=top, higher=higher_ranks)
            self._add_place_cue({rank: own[rank]["candidates"] for rank in RANKS}, cue_tables)
            photos.append({
                "photo": i,
                "ranks": {rank: own[rank]["candidates"] for rank in own},
                "overall_top": {rank: own[rank]["position_of"] for rank in RANKS},
                "specimens": self._specimens(photo_picks[i], infos, may_show),
            })
        top_species = top["species"]
        return {
            "model": {"backbone": self.backbone, "method": self.method},
            "reference": {"records": self.records, "species": len(self.index.species),
                          "photos": self.embedded},
            "photos": n_q,
            "ranks": ranks,
            "likely": likely_sets,
            "specimens": self._specimens(picks, infos, may_show),
            "per_photo": photos,
            "hints": improvement_hints(n_q, ranks, self.index.ref_count.get(top_species, 0)),
            "species_url": (ORG_SPECIES_URL + quote(top_species)) if top_species else None,
            "calibration": self.calibration,
            "confidence_note": self.confidence_note(),
            # Place and date (placeprior.py): whether a prior moved the scores, and the
            # photos-only top where it changed it ({} otherwise).
            "prior": {"applied": applied, "method": self.method},
            "photo_only_top": photo_only_top,
        }

    def _cue_tables(self, context):
        cue = getattr(self, "cue", None)
        return cue.tables(context) if cue is not None else None

    def _add_place_cue(self, ranks: dict, tables) -> None:
        """Each species and genus name ({rank: [{"name": ...}]}) gets `place`
        (PlaceCue.cue): None without a place and date, and at family."""
        cue = getattr(self, "cue", None)
        for rank, names in ranks.items():
            for c in names:
                c["place"] = cue.cue(tables, rank, c["name"]) if cue is not None else None

    def _species_scores(self, query: np.ndarray, context) -> np.ndarray:
        if self.uses_context:
            return self.model.species_scores(query, context)
        return self.model.species_scores(query)

    def _ranks(self, scores: np.ndarray, top_k: int, with_position_of: dict | None = None,
               higher: bool = False) -> dict:
        """Top candidates at every rank. With `with_position_of` ({rank: name}), each rank
        is {"candidates": [...], "position_of": where that name places here, 1-based}.
        `higher` adds order, class and phylum after family (no position_of there)."""
        ranks: dict = {}
        for rank in RANKS:
            # Species candidates are species only: never a one-word name (evaluate.Index).
            names = self.index.labels[rank]
            rs = rank_scores(scores, self.index, rank)
            finite = np.isfinite(rs)
            conf = np.zeros_like(rs, dtype=np.float64)
            conf[finite] = softmax_confidence(rs[finite], self.temperature(rank))
            order = np.argsort(-np.where(finite, rs, -np.inf))[:top_k]
            candidates = [{"name": names[i], "score": round(float(rs[i]), 4),
                           "confidence": round(float(conf[i]), 4),
                           "reference_records": self.rank_counts[rank][names[i]]}
                          for i in order if finite[i]]
            if with_position_of is None:
                ranks[rank] = candidates
                continue
            ranks[rank] = {"candidates": candidates,
                           "position_of": self._position(with_position_of.get(rank), names,
                                                         rs, finite, conf)}
        if higher:
            above = (self.higher.candidates(rank_scores(scores, self.index, "genus"), top_k)
                     if getattr(self, "higher", None) else {r: [] for r in HIGHER_RANKS})
            for rank, candidates in above.items():
                ranks[rank] = (candidates if with_position_of is None
                               else {"candidates": candidates, "position_of": None})
        return ranks

    def _likely(self, scores: np.ndarray) -> dict:
        """Each rank's likely set (likely.py): the names at or above the rank's fitted
        probability floor (and the top one), which held the right name about `coverage` of
        the time on calibration records held back from the fit. Ranks without a fitted set (no comparison
        yet, or no useful target reached) are left out."""
        fitted = (self.calibration or {}).get("sets") or {}
        out = {}
        for rank in RANKS:
            fit = fitted.get(rank)
            if not fit:
                continue
            names = self.index.labels[rank]
            probs = likely.probabilities(rank_scores(scores, self.index, rank),
                                         self.temperature(rank))
            chosen, capped = likely.likely_set(probs, fit["floor"])
            out[rank] = {
                # How often such a set held the truth on records it was not fitted on
                # (likely.stated_coverage): what was achieved, never the target alone.
                "coverage": likely.stated_coverage(fit),
                "checked_coverage": (fit.get("crosscheck") or {}).get("coverage"),
                "names": [{"name": names[i], "confidence": round(float(probs[i]), 4),
                           "reference_records": self.rank_counts[rank][names[i]]}
                          for i in chosen],
                "capped": capped,
            }
        return out

    @staticmethod
    def _position(name: str | None, names: list[str], rs: np.ndarray, finite: np.ndarray,
                  conf: np.ndarray) -> dict | None:
        """Where `name` places among these scores (1 = first), and its confidence here."""
        if name is None:
            return None
        j = bisect_left(names, name)                      # names are sorted (build_index)
        if j >= len(names) or names[j] != name or not finite[j]:
            return None
        return {"name": name, "position": int((rs[finite] > rs[j]).sum()) + 1,
                "confidence": round(float(conf[j]), 4)}

    def _pick_specimens(self, sims: np.ndarray, record_score: np.ndarray,
                        photo: int | None, n: int) -> list:
        """The n best records by `record_score`. With `photo`, only that query photo's
        matches count when choosing which of the record's photos to show."""
        n = min(n, len(record_score))
        if n <= 0:
            return []
        top = np.argpartition(-record_score, n - 1)[:n]
        picked = []
        for ri in top[np.argsort(-record_score[top], kind="stable")].tolist():
            lo = int(self.rec_starts[ri])
            hi = int(self.rec_starts[ri + 1]) if ri + 1 < len(self.rec_starts) else sims.shape[1]
            block = sims[:, lo:hi] if photo is None else sims[photo:photo + 1, lo:hi]
            q_best, c_best = np.unravel_index(int(block.argmax()), block.shape)
            col = lo + int(c_best)
            photo_ids = [int(p) for p in self.col_photo[lo:hi].tolist()]
            picked.append((float(record_score[ri]), int(q_best) if photo is None else photo,
                           self.col_record[col], photo_ids, block.max(axis=0)))
        return picked

    @staticmethod
    def _specimens(picked: list, infos: dict, may_show: Callable[[PhotoInfo], bool]) -> list[dict]:
        specimens = []
        for score, q_best, rec, photo_ids, photo_scores in picked:
            # The photo shown: the record's best match among those that may be shown (a
            # CC licence, or the owner's grant), which need not be the photo that matched best.
            candidates = [infos.get(p) for p in photo_ids]
            shown = pick_best(photo_scores, [i is not None and may_show(i) for i in candidates])
            info = candidates[shown] if shown is not None else None
            specimens.append({
                "observation_id": rec.observation_id,
                # A record named only to genus or family has no species; `name` is what it
                # is named, at the finest rank it has.
                "name": rec.species or rec.genus or rec.family,
                "species": rec.species, "genus": rec.genus, "family": rec.family,
                "similarity": round(score, 4),
                "matched_query_photo": q_best,
                "photo_url": photo_url(info.source, info.source_url, "medium") if info else None,
                "photo_owner": info.owner_login if info else None,
                "photo_withheld": info is None,
                # The record's own page at its source (iNat or Mushroom Observer).
                "source": rec.source,
                "source_id": rec.source_id,
                "record_url": record_url(rec.source, rec.source_id),
                "inat_url": record_url(INAT, rec.source_id) if rec.source == INAT else None,
                "species_url": ORG_SPECIES_URL + quote(rec.species) if rec.species else None,
            })
        return specimens

    def temperature(self, rank: str) -> float:
        """The comparison's fitted temperature, else the method's own: a method whose
        scores are log-probabilities (+prior, +occ) needs one near 2, where the cosine
        default of 0.02 would state ~100% for every answer."""
        if self.calibration:
            return self.calibration["temperatures"].get(rank, self.method_temperature(rank))
        return self.method_temperature(rank)

    def method_temperature(self, rank: str) -> float:
        t = getattr(self.model, "confidence_temperature", None)
        if isinstance(t, dict):
            t = t.get(rank, t.get("species"))
        return float(t) if t else CONFIDENCE_TEMPERATURE

    def confidence_note(self) -> str:
        if not self.calibration and getattr(self.model, "confidence_temperature", None):
            return ("Not calibrated by a comparison yet: confidence uses this method's own "
                    "temperature. Run mv compare for this model to calibrate it.")
        if not self.calibration:
            return ("Not calibrated yet: relative confidence among candidates. Run "
                    "mv compare for this model to calibrate it.")
        n = self.calibration["n"].get("species", 0)
        return (f"Calibrated on {n:,} test records (comparison "
                f"{self.calibration['comparison_id']}), DNA-verified records the model did not "
                "learn from. Approximate, and fitted again when a new model or release is built.")

    def identify(self, backbone_model, images: list, top_k: int = 5, context=None,
                 photo_lookup=None, may_show: Callable[[PhotoInfo], bool] = only_open_licences) -> dict:
        vecs, verdicts = self.embed(backbone_model, images)
        return self.identify_vectors(vecs, top_k, context=context,
                                     photo_lookup=photo_lookup, may_show=may_show, gate=verdicts)

    def embed(self, backbone_model, images: list) -> tuple[np.ndarray, list[dict] | None]:
        """Photos -> (normalised vectors, the gate's verdict per photo or None). Every path
        that answers from photos goes through here, so none can skip the gate."""
        prepare = getattr(backbone_model, "prepare", None) or (lambda im: im)
        vecs, base = self.encode(backbone_model, [prepare(im) for im in images])
        verdicts = self.gate.judge(base) if self.gate is not None and base is not None else None
        return normalise(vecs), verdicts

    def encode(self, backbone_model, items: list) -> tuple[np.ndarray, np.ndarray | None]:
        """(this model's vectors, the gate's base-model vectors or None when the gate
        can't run on this backbone). A fine-tuned BioCLIP 2 gives both from one pass."""
        gate = self.gate
        if gate is not None:
            both = getattr(backbone_model, "encode_with_base", None)
            if both is not None and getattr(backbone_model, "base_name", None) == gate.base:
                pair = both(items)
                if pair is not None:
                    return pair
            if self.backbone == gate.base:
                vecs = backbone_model.encode(items)
                return vecs, vecs
        return backbone_model.encode(items), None


def renumber(result: dict, kept: list[int], n_q: int) -> None:
    """An answer made from the kept photos only, renumbered to the photos sent: specimens
    point at the right photo, and every left-out photo gets an empty per-photo entry."""
    for s in result["specimens"]:
        s["matched_query_photo"] = kept[s["matched_query_photo"]]
    own = {}
    for p in result["per_photo"]:
        p["photo"] = kept[p["photo"]]
        for s in p["specimens"]:
            s["matched_query_photo"] = p["photo"]
        own[p["photo"]] = p
    result["per_photo"] = [own.get(i) or {"photo": i, "ranks": {rank: [] for rank in RANKS},
                                          "overall_top": {rank: None for rank in RANKS},
                                          "specimens": []}
                           for i in range(n_q)]
    result["photos"] = n_q


def withhold_confidence(result: dict) -> None:
    """No stated chance anywhere in the result: the names stay, as look-alikes."""
    result["likely"] = {}
    for ranks in [result["ranks"], *(p["ranks"] for p in result["per_photo"])]:
        for candidates in ranks.values():
            for c in candidates:
                c["confidence"] = None
    for p in result["per_photo"]:
        for place in p["overall_top"].values():
            if place:
                place["confidence"] = None
    result["confidence_withheld"] = True
