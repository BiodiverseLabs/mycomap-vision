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

import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .embed import load_embeddings, normalise
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


@dataclass
class PhotoInfo:
    source_url: str
    license_class: str
    owner_login: str | None


class Identifier:
    """One backbone + method over all of its embedded, DNA-verified records."""

    def __init__(self, conn: sqlite3.Connection, backbone: str, method: str,
                 embeddings_root: Path | None = None, calibration: dict | None = None):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.backbone, self.method = backbone, method
        ids, vecs = load_embeddings(conn, backbone, embeddings_root)
        if not len(ids):
            raise ValueError(f"no embeddings for {backbone!r}")
        self.embedded = len(ids)
        row_of = {int(p): i for i, p in enumerate(ids.tolist())}
        by_id = load_records(conn, {int(p): int(p) for p in ids.tolist()})
        records = with_rows(by_id, row_of)
        self.index = build_index(records)
        self.model = METHODS[method]()
        self.uses_context = getattr(self.model, "needs_context", False)
        if self.uses_context:
            self.model.fit(vecs, self.index, records=records)
        else:
            self.model.fit(vecs, self.index)
        self.nearest = self.model if isinstance(self.model, NearestSpecimen) else NearestSpecimen()
        if self.nearest is not self.model:
            self.nearest.fit(vecs, self.index)
        # Per reference column: photo id and record.
        rec_of_row = {row: r for r in records for row in r.photo_rows}
        self.col_photo = ids[self.index.cols]
        self.col_record = [rec_of_row[int(c)] for c in self.index.cols.tolist()]
        # A record's photos are contiguous in index.cols: where each record starts.
        self.rec_starts = np.asarray(
            [i for i, r in enumerate(self.col_record)
             if i == 0 or r is not self.col_record[i - 1]], dtype=np.int64)
        self.calibration = calibration
        self.photos = self._photo_info(conn, set(int(p) for p in self.col_photo.tolist()))
        self.records = len(records)
        self.rank_counts = {rank: Counter() for rank in RANKS}
        for r in records:
            for rank in RANKS:
                self.rank_counts[rank][{"species": r.species, "genus": r.genus,
                                        "family": r.family}[rank]] += 1

    @staticmethod
    def _photo_info(conn: sqlite3.Connection, photo_ids: set[int]) -> dict[int, PhotoInfo]:
        out = {}
        rows = conn.execute("select photo_id, source_url, license_class, owner_login from photos")
        for pid, url, lic, owner in rows:
            if pid in photo_ids:
                out[pid] = PhotoInfo(url, lic, owner)
        return out

    def identify_vectors(self, query: np.ndarray, top_k: int = 5,
                         n_specimens: int = 6, context=None) -> dict:
        """Query photo vectors (already normalised) -> ranks, specimens, hints.
        `context` (place and date) is used by methods with a range-and-season score."""
        if self.uses_context:
            scores = self.model.species_scores(query, context)
        else:
            scores = self.model.species_scores(query)
        ranks: dict[str, list[dict]] = {}
        for rank in RANKS:
            names = self.index.species if rank == "species" else self.index.labels[rank]
            rs = rank_scores(scores, self.index, rank)
            finite = np.isfinite(rs)
            conf = np.zeros_like(rs, dtype=np.float64)
            conf[finite] = softmax_confidence(rs[finite], self.temperature(rank))
            order = np.argsort(-np.where(finite, rs, -np.inf))[:top_k]
            ranks[rank] = [{"name": names[i], "score": round(float(rs[i]), 4),
                            "confidence": round(float(conf[i]), 4),
                            "reference_records": self.rank_counts[rank][names[i]]}
                           for i in order if finite[i]]
        # Records are ranked by the same rule as species: for each of your photos,
        # its best match among the record's photos, averaged over your photos.
        sims = self.nearest.photo_sims(query)                 # (q, reference photos)
        per_record = np.maximum.reduceat(sims, self.rec_starts, axis=1)   # (q, records)
        record_score = per_record.mean(axis=0)
        specimens = []
        for ri in np.argsort(-record_score)[:n_specimens].tolist():
            lo = int(self.rec_starts[ri])
            hi = int(self.rec_starts[ri + 1]) if ri + 1 < len(self.rec_starts) else sims.shape[1]
            block = sims[:, lo:hi]
            q_best, c_best = np.unravel_index(int(block.argmax()), block.shape)
            col = lo + int(c_best)
            rec = self.col_record[col]
            pid = int(self.col_photo[col])
            info = self.photos.get(pid)
            specimens.append({
                "observation_id": rec.observation_id,
                "species": rec.species, "genus": rec.genus, "family": rec.family,
                "similarity": round(float(record_score[ri]), 4),
                "matched_query_photo": int(q_best),
                "photo_url": sized_url(info.source_url, "medium") if info else None,
                "photo_owner": info.owner_login if info else None,
                "inat_url": INAT_OBS_URL + rec.observation_id,
                "species_url": ORG_SPECIES_URL + quote(rec.species),
            })
        top_species = ranks["species"][0]["name"] if ranks["species"] else None
        return {
            "model": {"backbone": self.backbone, "method": self.method},
            "reference": {"records": self.records, "species": len(self.index.species),
                          "photos": self.embedded},
            "photos": int(query.shape[0]),
            "ranks": ranks,
            "specimens": specimens,
            "hints": improvement_hints(int(query.shape[0]), ranks,
                                       self.index.ref_count.get(top_species, 0)),
            "species_url": (ORG_SPECIES_URL + quote(top_species)) if top_species else None,
            "calibration": self.calibration,
            "confidence_note": self.confidence_note(),
        }

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

    def identify(self, backbone_model, images: list, top_k: int = 5, context=None) -> dict:
        prepare = getattr(backbone_model, "prepare", None) or (lambda im: im)
        vecs = normalise(backbone_model.encode([prepare(im) for im in images]))
        return self.identify_vectors(vecs, top_k, context=context)
