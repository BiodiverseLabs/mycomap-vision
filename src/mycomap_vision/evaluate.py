"""Prospective evaluation: identify the newest green records using only older ones.

Test records are those validated in the last `test_days` days before the newest
validation; everything validated earlier is the reference set. No random split:
new weeks are the honest test (see CLAUDE.md).

The phase-0 identifier is the nearest DNA-verified specimen. A species' score for
a query record is the mean, over the query's photos, of each photo's best cosine
similarity to any reference photo of that species. A species with one specimen
competes on equal terms with one that has a thousand. Genus and family scores are
the best species score inside them, so every rank gets its own answer.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

BUCKETS = [(0, 0, "novel (0 refs)"), (1, 1, "1 ref"), (2, 2, "2 refs"),
           (3, 5, "3-5 refs"), (6, 30, "6-30 refs"), (31, 10**9, "31+ refs")]
RANKS = ("family", "genus", "species")


@dataclass
class Record:
    observation_id: str
    species: str
    genus: str
    family: str
    validated_on: str | None
    observer: str | None
    photo_rows: list[int] = field(default_factory=list)   # rows into the vector matrix


def clean(s: str | None) -> str:
    return (s or "").strip()


def load_records(conn: sqlite3.Connection, photo_row: dict[int, int],
                 north_america_only: bool = True) -> list[Record]:
    """Green, unconflicted iNat records with at least one embedded photo."""
    na = "and r.north_america = 1" if north_america_only else ""
    rows = conn.execute(f"""
      select r.observation_id, r.scientific_name, r.genus, r.family, r.validated_on,
             o.user_login, op.photo_id, op.position
      from records r
      join inat_observations o on o.observation_id = r.observation_id and o.status = 'ok'
      join observation_photos op on op.observation_id = r.observation_id
      where r.label_conflict = 0 {na}
      order by r.observation_id, op.position
    """).fetchall()
    recs: dict[str, Record] = {}
    for oid, name, genus, family, vdate, login, pid, _pos in rows:
        if pid not in photo_row or not clean(name):
            continue
        rec = recs.get(oid)
        if rec is None:
            rec = recs[oid] = Record(oid, clean(name), clean(genus) or clean(name).split()[0],
                                     clean(family), vdate, login)
        rec.photo_rows.append(photo_row[pid])
    return list(recs.values())


def split_by_time(records: list[Record], test_days: int) -> tuple[list[Record], list[Record], str]:
    dated = [r.validated_on for r in records if r.validated_on]
    newest = date.fromisoformat(max(dated))
    cutoff = (newest - timedelta(days=test_days)).isoformat()
    test = [r for r in records if r.validated_on and r.validated_on > cutoff]
    ref = [r for r in records if not (r.validated_on and r.validated_on > cutoff)]
    return ref, test, cutoff


@dataclass
class Index:
    """Reference photos grouped by species, for per-species max-similarity."""
    species: list[str]                 # species order
    cols: np.ndarray                   # reference photo rows (into vectors), sorted by species
    starts: np.ndarray                 # start offset of each species in cols
    ref_count: Counter                 # reference records per species
    # For each rank: the label names and, per species, the index of its label (-1 = blank).
    labels: dict[str, list[str]] = field(default_factory=dict)
    label_of: dict[str, np.ndarray] = field(default_factory=dict)


def build_index(ref: list[Record]) -> Index:
    by_species: dict[str, list[int]] = defaultdict(list)
    taxa: dict[str, Record] = {}
    count = Counter()
    for r in ref:
        by_species[r.species].extend(r.photo_rows)
        taxa.setdefault(r.species, r)
        count[r.species] += 1
    species = sorted(by_species)
    cols, starts = [], []
    for s in species:
        starts.append(len(cols))
        cols.extend(by_species[s])
    index = Index(species, np.asarray(cols, dtype=np.int64), np.asarray(starts, dtype=np.int64),
                  count)
    for rank in RANKS:
        names = sorted({truth(taxa[s], rank) for s in species} - {""})
        pos = {n: i for i, n in enumerate(names)}
        index.labels[rank] = names
        index.label_of[rank] = np.asarray([pos.get(truth(taxa[s], rank), -1) for s in species],
                                          dtype=np.int64)
    return index


class Scorer:
    """Cosine similarity of query photos to every reference photo, on the GPU when there is one."""

    def __init__(self, ref_vecs: np.ndarray):
        self.torch = None
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.ref = torch.from_numpy(ref_vecs.astype(np.float16)).cuda()
        except ImportError:
            pass
        if self.torch is None:
            self.ref = ref_vecs.astype(np.float32)

    def sims(self, query: np.ndarray) -> np.ndarray:
        if self.torch is not None:
            q = self.torch.from_numpy(query.astype(np.float16)).cuda()
            return (q @ self.ref.T).float().cpu().numpy()
        return query.astype(np.float32) @ self.ref.T


def species_scores(sims: np.ndarray, index: Index) -> np.ndarray:
    """(n_species,): mean over query photos of each photo's best match within the species."""
    return np.maximum.reduceat(sims, index.starts, axis=1).mean(axis=0)


def rank_scores(scores: np.ndarray, index: Index, rank: str) -> np.ndarray:
    """Best species score inside each label of `rank` (species: the scores themselves)."""
    if rank == "species":
        return scores
    out = np.full(len(index.labels[rank]), -np.inf, dtype=np.float32)
    lab = index.label_of[rank]
    keep = lab >= 0
    np.maximum.at(out, lab[keep], scores[keep])
    return out


def top_labels(scores: np.ndarray, names: list[str], k: int) -> list[str]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return [names[i] for i in idx[np.argsort(-scores[idx])]]


def bucket_of(n: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label
    return BUCKETS[-1][2]


def truth(rec: Record, rank: str) -> str:
    return {"species": rec.species, "genus": rec.genus, "family": rec.family}[rank]


def evaluate(vectors: np.ndarray, ref: list[Record], test: list[Record],
             first_photo_only: bool = False, top_k: int = 5) -> dict:
    index = build_index(ref)
    names = {rank: (index.species if rank == "species" else index.labels[rank]) for rank in RANKS}
    scorer = Scorer(vectors[index.cols])       # species-sorted, so reduceat works directly
    tally = {rank: defaultdict(Counter) for rank in RANKS}
    for rec in test:
        rows = rec.photo_rows[:1] if first_photo_only else rec.photo_rows
        scores = species_scores(scorer.sims(vectors[rows]), index)
        b = bucket_of(index.ref_count.get(rec.species, 0))
        for rank in RANKS:
            t = truth(rec, rank)
            if not t:
                continue
            top = top_labels(rank_scores(scores, index, rank), names[rank], top_k)
            for key in ("all", b):
                c = tally[rank][key]
                c["n"] += 1
                c["top1"] += top[:1] == [t]
                c[f"top{top_k}"] += t in top
    out = {}
    for rank in RANKS:
        out[rank] = {k: {"n": c["n"], "top1": round(c["top1"] / c["n"], 4),
                         f"top{top_k}": round(c[f"top{top_k}"] / c["n"], 4)}
                     for k, c in tally[rank].items() if c["n"]}
    return out


def run(conn: sqlite3.Connection, backbone: str, test_days: int = 28,
        max_test: int | None = None, seed: int = 0) -> dict:
    from .embed import load_embeddings
    ids, vecs = load_embeddings(conn, backbone)
    photo_row = {int(p): i for i, p in enumerate(ids.tolist())}
    records = load_records(conn, photo_row)
    ref, test, cutoff = split_by_time(records, test_days)
    if max_test and len(test) > max_test:
        rng = np.random.default_rng(seed)
        test = [test[i] for i in sorted(rng.choice(len(test), max_test, replace=False))]
    report = {
        "backbone": backbone, "cutoff": cutoff, "reference_records": len(ref),
        "test_records": len(test),
        "test_multi_photo_share": round(sum(len(r.photo_rows) > 1 for r in test)
                                        / max(1, len(test)), 3),
        "all_photos": evaluate(vecs, ref, test),
        "first_photo_only": evaluate(vecs, ref, test, first_photo_only=True),
    }
    return report


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2)
