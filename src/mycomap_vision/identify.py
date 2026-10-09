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

from . import config, likely
from .embed import normalise
from .serving import map_embeddings
from .evaluate import (METHODS, RANKS, NearestSpecimen, build_index, load_records, rank_scores,
                       with_rows)
from .licenses import sized_url

CONFIDENCE_TEMPERATURE = 0.02
ORG_SPECIES_URL = "https://mycomap.org/species/"
INAT_OBS_URL = "https://www.inaturalist.org/observations/"


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


def only_open_licences(info: PhotoInfo) -> bool:
    """The showing rule when no photographers' answers are at hand: CC photos only."""
    return info.license_class in SHOWN_LICENSES


def photo_info_rows(rows) -> dict[int, PhotoInfo]:
    return {int(pid): PhotoInfo(url, lic, owner, uid) for pid, url, lic, owner, uid in rows}


PHOTO_INFO_SQL = "select photo_id, source_url, license_class, owner_login, owner_user_id from photos"


class Specimen(NamedTuple):
    """What a result needs of a reference record."""
    observation_id: str
    species: str
    genus: str
    family: str


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
        self.taxa = np.array([(pos[r.species], pos[r.genus], pos[r.family]) for r in records],
                             dtype=np.int32).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.rec)

    def __getitem__(self, col: int) -> Specimen:
        r = int(self.rec[col])
        sp, ge, fa = (str(self.names[i]) for i in self.taxa[r])
        return Specimen(str(self.obs[r]), sp, ge, fa)

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
                 layer_root: Path | None = None):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.backbone, self.method = backbone, method
        ids, vecs = map_embeddings(conn, backbone, embeddings_root, layer_root)
        if not len(ids):
            raise ValueError(f"no embeddings for {backbone!r}")
        self.embedded = len(ids)
        row_of = {int(p): i for i, p in enumerate(ids.tolist())}
        by_id = load_records(conn, {int(p): int(p) for p in ids.tolist()})
        records = with_rows(by_id, row_of)
        self.index = build_index(records)
        self.model = METHODS[method]()
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
                         n_photo_specimens: int = 3) -> dict:
        """Query photo vectors (already normalised) -> ranks, specimens, hints.
        `context` (place and date) is used by methods with a range-and-season score.

        The main answer uses all the photos together. `per_photo` gives what the same
        model says from each photo on its own, so a person can see which photo pulled
        the answer which way (an underside may match well while the cap does not).

        A specimen's photo is shown only if `may_show` allows it (permissions.py:
        CC licence, or the owner's grant). `photo_lookup` reads licence and owner
        now rather than when the model was built, so a licence change on iNat or a
        withdrawal on mycomap.org applies to the next identification."""
        n_q = int(query.shape[0])
        sims = self.nearest.photo_sims(query)                 # (q, reference photos)
        if self.model is self.nearest:
            # The nearest-specimen score is the mean of each photo's own scores, so
            # both come from the one similarity matrix.
            per_photo = np.maximum.reduceat(sims, self.index.starts, axis=1)
            scores = per_photo.mean(axis=0)
        else:
            scores = self._species_scores(query, context)
            per_photo = (np.stack([self._species_scores(query[i:i + 1], context)
                                   for i in range(n_q)]) if n_q > 1 else scores[None, :])
        ranks = self._ranks(scores, top_k)
        likely_sets = self._likely(scores)
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
            own = self._ranks(per_photo[i], top_k, with_position_of=top)
            photos.append({
                "photo": i,
                "ranks": {rank: own[rank]["candidates"] for rank in RANKS},
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
        }

    def _species_scores(self, query: np.ndarray, context) -> np.ndarray:
        if self.uses_context:
            return self.model.species_scores(query, context)
        return self.model.species_scores(query)

    def _ranks(self, scores: np.ndarray, top_k: int, with_position_of: dict | None = None) -> dict:
        """Top candidates at every rank. With `with_position_of` ({rank: name}), each rank
        is {"candidates": [...], "position_of": where that name places here, 1-based}."""
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
        return ranks

    def _likely(self, scores: np.ndarray) -> dict:
        """Each rank's likely set (likely.py): the names at or above the rank's fitted
        probability floor (and the top one), which held the right name about `coverage` of
        the time in this model's newest test. Ranks without a fitted set (no comparison
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
                "coverage": fit["coverage"],
                # How often such a set held the truth on records it was not fitted on.
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
                "photo_url": sized_url(info.source_url, "medium") if info else None,
                "photo_owner": info.owner_login if info else None,
                "photo_withheld": info is None,
                "inat_url": INAT_OBS_URL + rec.observation_id,
                "species_url": ORG_SPECIES_URL + quote(rec.species) if rec.species else None,
            })
        return specimens

    def temperature(self, rank: str) -> float:
        if self.calibration:
            return self.calibration["temperatures"].get(rank, CONFIDENCE_TEMPERATURE)
        return CONFIDENCE_TEMPERATURE

    def confidence_note(self) -> str:
        if not self.calibration:
            return ("Not calibrated yet: relative confidence among candidates. Run "
                    "mv compare for this model to calibrate it.")
        n = self.calibration["n"].get("species", 0)
        return (f"Calibrated on {n:,} test records (comparison "
                f"{self.calibration['comparison_id']}). Early and approximate: it improves as "
                "the reference set and weekly tests grow.")

    def identify(self, backbone_model, images: list, top_k: int = 5, context=None,
                 photo_lookup=None, may_show: Callable[[PhotoInfo], bool] = only_open_licences) -> dict:
        prepare = getattr(backbone_model, "prepare", None) or (lambda im: im)
        vecs = normalise(backbone_model.encode([prepare(im) for im in images]))
        return self.identify_vectors(vecs, top_k, context=context, photo_lookup=photo_lookup,
                                     may_show=may_show)
