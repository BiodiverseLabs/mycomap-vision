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

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import config
from .methods import (METHODS, Hybrid, LinearHead, NearestSpecimen,  # noqa: F401
                      Scorer, SpeciesMean, species_scores)
from .permissions import EXCLUDED_FROM_USE_SQL
from .permissions import ensure_schema as ensure_permissions_schema

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
    latitude: float | None = None       # true coordinates: used for scores, never shown
    longitude: float | None = None
    observed_on: str | None = None
    projects: tuple[str, ...] = ()      # the .org projects that marked it green


def context_of(rec: "Record"):
    from .prior import Context
    return Context(rec.latitude, rec.longitude, rec.observed_on)


def clean(s: str | None) -> str:
    return (s or "").strip()


def load_records(conn: sqlite3.Connection, photo_row: dict[int, int],
                 north_america_only: bool = True) -> list[Record]:
    """Green, unconflicted iNat records with at least one embedded photo.

    Every reference set, comparison and training run is built here, so this is
    where photos a photographer has refused us (all rights reserved, permission
    withdrawn on mycomap.org; see permissions.py) are left out."""
    na = "and r.north_america = 1" if north_america_only else ""
    ensure_permissions_schema(conn)
    rows = conn.execute(f"""
      select r.observation_id, r.scientific_name, r.genus, r.family, r.validated_on,
             o.user_login, op.photo_id, op.position, r.latitude, r.longitude,
             coalesce(r.observed_on, o.observed_on), r.green_projects
      from records r
      join inat_observations o on o.observation_id = r.observation_id and o.status = 'ok'
      join observation_photos op on op.observation_id = r.observation_id
      where r.label_conflict = 0 {na} and {EXCLUDED_FROM_USE_SQL}
      order by r.observation_id, op.position
    """).fetchall()
    recs: dict[str, Record] = {}
    for oid, name, genus, family, vdate, login, pid, _pos, lat, lon, observed, projects in rows:
        if pid not in photo_row or not clean(name):
            continue
        rec = recs.get(oid)
        if rec is None:
            rec = recs[oid] = Record(oid, clean(name), clean(genus) or clean(name).split()[0],
                                     clean(family), vdate, login, latitude=lat, longitude=lon,
                                     observed_on=observed,
                                     projects=tuple(json.loads(projects or "[]")))
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


# Methods live in methods.py; re-exported here for callers and tests.



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


# Candidate temperatures for turning scores into confidence (softmax(score / T)).
# Wide enough for cosine scores (~0.01) and log-probability scores (~1-10) alike.
T_GRID = np.geomspace(0.001, 10.0, 90)


def nll_by_temperature(scores: np.ndarray, true_idx: int) -> np.ndarray:
    """Negative log-likelihood of the true label under softmax(scores / T), for each T."""
    z = scores[None, :].astype(np.float64) / T_GRID[:, None]
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]
    return lse - z[:, true_idx]


def bucket_of(n: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label
    return BUCKETS[-1][2]


def truth(rec: Record, rank: str) -> str:
    return {"species": rec.species, "genus": rec.genus, "family": rec.family}[rank]


def summarise_groups(groups: dict, largest: int = 12) -> list[dict]:
    """The largest groups first, each with its test count and genus/species top-1."""
    rows = []
    for key, c in groups.items():
        n = c["species_n"] or c["genus_n"]
        rows.append({"name": key, "n": n,
                     "species_top1": round(c["species_top1"] / c["species_n"], 4)
                     if c["species_n"] else None,
                     "genus_top1": round(c["genus_top1"] / c["genus_n"], 4)
                     if c["genus_n"] else None})
    rows.sort(key=lambda r: (-r["n"], r["name"]))
    return rows[:largest]


def fit_method(vectors: np.ndarray, ref: list[Record], method: str):
    """(index, fitted model) for the reference records; reusable across evaluations."""
    index = build_index(ref)
    model = METHODS[method]()
    if getattr(model, "needs_context", False):
        model.fit(vectors, index, records=ref)
    else:
        model.fit(vectors, index)
    return index, model


def evaluate(vectors: np.ndarray, ref: list[Record], test: list[Record],
             first_photo_only: bool = False, top_k: int = 5, method: str = "nearest",
             fitted=None) -> dict:
    """Score the test records. `fitted` (from fit_method) skips refitting, which matters
    for the trained methods."""
    index, model = fitted or fit_method(vectors, ref, method)
    names = {rank: (index.species if rank == "species" else index.labels[rank]) for rank in RANKS}
    uses_context = getattr(model, "needs_context", False)
    tally = {rank: defaultdict(Counter) for rank in RANKS}
    # Calibration: summed NLL per candidate temperature, over test records whose
    # true label is in the reference set (a novel species has no probability to give).
    position = {rank: {n: i for i, n in enumerate(names[rank])} for rank in RANKS}
    nll = {rank: np.zeros(len(T_GRID)) for rank in RANKS}
    n_cal = Counter()
    # Weekly batches are lumpy (one big project, one prolific observer), so results
    # are also kept per project and per observer at genus and species.
    groups = {"project": defaultdict(Counter), "observer": defaultdict(Counter)}
    for rec in test:
        rows = rec.photo_rows[:1] if first_photo_only else rec.photo_rows
        if uses_context:
            scores = model.species_scores(vectors[rows], context_of(rec))
        else:
            scores = model.species_scores(vectors[rows])
        b = bucket_of(index.ref_count.get(rec.species, 0))
        for rank in RANKS:
            t = truth(rec, rank)
            if not t:
                continue
            rs = rank_scores(scores, index, rank)
            if t in position[rank]:
                nll[rank] += nll_by_temperature(rs, position[rank][t])
                n_cal[rank] += 1
            top = top_labels(rs, names[rank], top_k)
            for key in ("all", b):
                c = tally[rank][key]
                c["n"] += 1
                c["top1"] += top[:1] == [t]
                c[f"top{top_k}"] += t in top
            if rank in ("genus", "species"):
                keys = [("project", p) for p in (rec.projects or ("(none)",))]
                keys.append(("observer", rec.observer or "(unknown)"))
                for kind, key in keys:
                    g = groups[kind][key]
                    g[f"{rank}_n"] += 1
                    g[f"{rank}_top1"] += top[:1] == [t]
    out = {}
    for rank in RANKS:
        out[rank] = {k: {"n": c["n"], "top1": round(c["top1"] / c["n"], 4),
                         f"top{top_k}": round(c[f"top{top_k}"] / c["n"], 4)}
                     for k, c in tally[rank].items() if c["n"]}
    out["groups"] = {kind: summarise_groups(g) for kind, g in groups.items()}
    out["calibration"] = {
        rank: {"temperature": float(T_GRID[int(np.argmin(nll[rank]))]), "n": n_cal[rank],
               "nll": round(float(nll[rank].min() / n_cal[rank]), 4)}
        for rank in RANKS if n_cal[rank]}
    return out


def latest_calibration(conn: sqlite3.Connection, backbone: str, method: str) -> dict | None:
    """The newest comparison's fitted temperatures for this model, with where they came from."""
    conn.executescript(SCOREBOARD_SCHEMA)
    row = conn.execute("select id, comparison_id, report_json from eval_runs "
                       "where backbone = ? and method = ? order by id desc limit 1",
                       (backbone, method)).fetchone()
    if not row:
        return None
    cal = json.loads(row[2]).get("all_photos", {}).get("calibration")
    if not cal:
        return None
    return {"run_id": row[0], "comparison_id": row[1],
            "temperatures": {rank: c["temperature"] for rank, c in cal.items()},
            "n": {rank: c["n"] for rank, c in cal.items()}}


SCOREBOARD_SCHEMA = """
create table if not exists eval_runs (
  id              integer primary key autoincrement,
  comparison_id   text not null,       -- runs in one comparison share test and reference photos
  backbone        text not null,
  method          text not null,
  cutoff          text not null,
  test_days       integer not null,
  n_reference     integer not null,
  n_test          integer not null,
  record_set      text not null,       -- hash of the test and reference records and photos
  species_top1    real, genus_top1 real, family_top1 real,
  species_top1_first_photo real,
  report_json     text not null,
  code_version    text,
  created_at      text not null
);
"""


# Fine-tuned backbones (finetune.py) and the newest validation date they trained on.
FINETUNE_SCHEMA = """
create table if not exists finetunes (
  name            text primary key,
  base            text not null,
  trained_through text not null,
  test_days       integer not null,
  meta_json       text not null,
  created_at      text not null
);
"""


def check_not_trained_on_test(conn: sqlite3.Connection, backbones: list[str],
                              cutoff: str) -> None:
    """Refuse to score a fine-tuned model on records it may have trained on: test records
    are those validated after `cutoff`, so it must have trained on nothing later."""
    conn.executescript(FINETUNE_SCHEMA)
    for b in backbones:
        row = conn.execute("select trained_through from finetunes where name = ?",
                           (b,)).fetchone()
        if row and row[0] > cutoff:
            raise ValueError(
                f"{b} was trained on records validated up to {row[0]}, but this comparison "
                f"tests records validated after {cutoff}; its score would be inflated. "
                "Use a shorter --test-days or a model trained on less.")


def with_rows(records: list[Record], row_of: dict[int, int]) -> list[Record]:
    """Records whose photo_rows hold photo ids -> the same records with one backbone's rows."""
    return [replace(r, photo_rows=[row_of[p] for p in r.photo_rows]) for r in records]


def record_set_hash(ref: list[Record], test: list[Record]) -> str:
    h = hashlib.sha1()
    for tag, group in (("ref", ref), ("test", test)):
        for r in sorted(group, key=lambda r: r.observation_id):
            h.update(f"{tag}:{r.observation_id}:{r.species}:{sorted(r.photo_rows)}".encode())
    return h.hexdigest()[:12]


@dataclass
class SharedSet:
    """Test and reference records every compared backbone can see, with their photos."""
    loaded: dict                 # backbone -> (photo ids, vectors)
    ref: list[Record]            # photo_rows hold photo ids
    test: list[Record]
    cutoff: str
    record_set: str
    shared_photos: int


def shared_records(conn: sqlite3.Connection, backbones: list[str], test_days: int = 28,
                   max_test: int | None = None, seed: int = 0,
                   embeddings_root=None) -> SharedSet:
    """The records a comparison of these backbones uses: only photos all of them embedded."""
    from .embed import load_embeddings
    loaded = {}
    for b in backbones:
        root = embeddings_root / b if embeddings_root else None
        ids, vecs = load_embeddings(conn, b, root)
        if not len(ids):
            raise ValueError(f"no embeddings for {b!r}; run mv embed --backbone {b}")
        loaded[b] = (ids, vecs)
    common = set.intersection(*(set(ids.tolist()) for ids, _ in loaded.values()))
    records = load_records(conn, {p: p for p in common})      # photo_rows hold photo ids
    ref, test, cutoff = split_by_time(records, test_days)
    if max_test and len(test) > max_test:
        rng = np.random.default_rng(seed)
        test = [test[i] for i in sorted(rng.choice(len(test), max_test, replace=False))]
    return SharedSet(loaded, ref, test, cutoff, record_set_hash(ref, test), len(common))


def top1(res: dict, rank: str):
    return res.get(rank, {}).get("all", {}).get("top1")


def save_run(conn: sqlite3.Connection, comparison_id: str, backbone: str, method: str,
             shared: SharedSet, test_days: int, all_photos: dict, first_photo: dict,
             extra: dict | None = None) -> dict:
    """One scoreboard row. `first_photo` may be {} for models that have no such variant."""
    conn.executescript(SCOREBOARD_SCHEMA)
    report = {"backbone": backbone, "method": method, "all_photos": all_photos,
              "first_photo_only": first_photo, **(extra or {})}
    with conn:
        conn.execute(
            "insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, "
            "n_reference, n_test, record_set, species_top1, genus_top1, family_top1, "
            "species_top1_first_photo, report_json, code_version, created_at) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (comparison_id, backbone, method, shared.cutoff, test_days, len(shared.ref),
             len(shared.test), shared.record_set, top1(all_photos, "species"),
             top1(all_photos, "genus"), top1(all_photos, "family"),
             top1(first_photo, "species"), json.dumps(report), config.code_version(),
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return report


def compare(conn: sqlite3.Connection, backbones: list[str], methods: list[str] | None = None,
            test_days: int = 28, max_test: int | None = None, seed: int = 0,
            embeddings_root=None, log=print) -> dict:
    """Evaluate every backbone x method on the same records and photos; save to the scoreboard.

    Only photos embedded by every backbone count, so no model is judged on photos
    another could not see.
    """
    methods = methods or ["nearest"]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}: {', '.join(METHODS)}")
    shared = shared_records(conn, backbones, test_days, max_test, seed, embeddings_root)
    check_not_trained_on_test(conn, list(shared.loaded), shared.cutoff)
    ref, test = shared.ref, shared.test
    comparison_id = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-"
                     + shared.record_set[:6])
    runs = []
    for b, (ids, vecs) in shared.loaded.items():
        row_of = {int(p): i for i, p in enumerate(ids.tolist())}
        ref_b, test_b = with_rows(ref, row_of), with_rows(test, row_of)
        for m in methods:
            log(f"  {b} / {m}: {len(test):,} test records against {len(ref):,}")
            fitted = fit_method(vecs, ref_b, m)
            all_photos = evaluate(vecs, ref_b, test_b, method=m, fitted=fitted)
            first = evaluate(vecs, ref_b, test_b, method=m, first_photo_only=True, fitted=fitted)
            runs.append(save_run(conn, comparison_id, b, m, shared, test_days, all_photos, first))
    return {"comparison_id": comparison_id, "cutoff": shared.cutoff, "test_days": test_days,
            "reference_records": len(ref), "test_records": len(test),
            "test_multi_photo_share": round(sum(len(r.photo_rows) > 1 for r in test)
                                            / max(1, len(test)), 3),
            "shared_photos": shared.shared_photos, "record_set": shared.record_set,
            "backbones": list(shared.loaded), "runs": runs}


def comparison_backbones(conn: sqlite3.Connection, comparison_id: str) -> list[str]:
    """The local backbones a saved comparison used (external baselines excluded)."""
    conn.executescript(SCOREBOARD_SCHEMA)
    rows = conn.execute("select distinct backbone from eval_runs where comparison_id = ? "
                        "and backbone not like 'external:%'", (comparison_id,)).fetchall()
    return [r[0] for r in rows]


SCOREBOARD_COLUMNS = ["id", "comparison_id", "backbone", "method", "cutoff", "test_days",
                      "n_reference", "n_test", "record_set", "species_top1", "genus_top1",
                      "family_top1", "species_top1_first_photo", "code_version", "created_at"]


def scoreboard(conn: sqlite3.Connection, comparison_id: str | None = None) -> list[dict]:
    """Saved runs, newest comparison first, best species top-1 first within it."""
    conn.executescript(SCOREBOARD_SCHEMA)
    where, params = "", ()
    if comparison_id:
        where, params = "where comparison_id = ?", (comparison_id,)
    rows = conn.execute(
        f"select {', '.join(SCOREBOARD_COLUMNS)} from eval_runs {where} "
        "order by comparison_id desc, species_top1 desc", params).fetchall()
    return [dict(zip(SCOREBOARD_COLUMNS, r)) for r in rows]


def run_report(conn: sqlite3.Connection, run_id: int) -> dict | None:
    conn.executescript(SCOREBOARD_SCHEMA)
    row = conn.execute("select report_json from eval_runs where id = ?", (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2)
