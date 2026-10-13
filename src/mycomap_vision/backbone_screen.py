"""The frozen-backbone screen on a dataset release (`mv backbone-screen`): which image
model, before any fine-tuning, names fungi best by nearest verified specimen.

Embedding every photo of a release with every candidate costs 3-13 h per backbone on one
GPU, so the screen runs on a subset drawn once, seeded, from the release itself:

- **References: a sample of the world, not of the answers.** Train species are shuffled
  (seeded) and taken whole, every train record of each (capped at `cap` per species, the
  same cap for every species), until about `reference_records` records. Depth then follows
  the release's own distribution, and a query's competitors are as well stocked as its
  true species. (Topping up the queries' own species would make every backbone look
  better than it is.)
- **Queries: val records whose true species was sampled**, in two halves scored apart:
  `queries` records from val excluding the former held-out benchmark (the half that
  decides, docs/retrain-v1.md) and `queries` from the former held-out, each stratified by
  the standard depth band of its species' capped reference count (largest remainder).
- Records in `errata` (a release's errata list: records that are not what the release
  says) are left out of both, and the file's sha256 is recorded.

The subset (`subset.json`: parameters, species list and its sha256, record keys, photo
ids) is written once and every later step reads it, so every backbone is scored on
exactly the same records and photos. `embed` embeds only the subset's photos that a
backbone lacks; `score` runs `evaluate.compare` per half (Steve's standard tables via
`mv standard-summary --comparison <id>`). Nothing here reads a test release.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from . import config

SEED = 20261013
REFERENCE_RECORDS = 16_000
CAP = 50
QUERIES = 2_000
HALVES = ("val-excl-former-heldout", "val-former-heldout")


def results_dir(rel_id: str, root: Path | None = None) -> Path:
    return (root or config.DATA_DIR / "research-results") / rel_id / "backbone-screen"


def band_of(n: int) -> str:
    from .evaluate import standard_band
    return standard_band(n)


def read_errata(path: str | Path | None) -> tuple[set[str], str | None]:
    """Record keys listed in a release's errata TSV and the file's sha256; (set(), None)
    without a file. The keys are read from the `key` column (`mv dataset errata`'s
    review-list format: kind, key, reason, note) or a `record_key` column; a file with
    neither header is refused rather than guessed."""
    if not path:
        return set(), None
    body = Path(path).read_bytes()
    rows = list(csv.reader(body.decode("utf-8-sig").splitlines(), delimiter="\t"))
    head = [h.strip() for h in rows[0]] if rows else []
    col = next((head.index(c) for c in ("key", "record_key") if c in head), None)
    if col is None:
        raise ValueError(f"{path}: no 'key' or 'record_key' column in {head}: not an "
                         "errata list")
    kind = head.index("kind") if "kind" in head else None
    keys = {r[col].strip() for r in rows[1:] if len(r) > col and r[col].strip()
            and (kind is None or r[kind].strip() == "record")}
    return keys, hashlib.sha256(body).hexdigest()


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation of `total` over strata of `sizes` (largest remainder,
    ties broken by stratum name), never more than a stratum holds."""
    pool = sum(sizes.values())
    if pool <= total:
        return dict(sizes)
    raw = {k: total * n / pool for k, n in sizes.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:left]:
        out[k] += 1
    return out


def draw(records: list[dict], seed: int = SEED, reference_records: int = REFERENCE_RECORDS,
         cap: int = CAP, queries: int = QUERIES, errata: set[str] | None = None) -> dict:
    """The subset from a release's included records (dicts with record_key, split, species,
    benchmark). Deterministic for the same records and parameters."""
    errata = errata or set()
    rng = random.Random(seed)
    train: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["split"] == "train" and r.get("species") and r["record_key"] not in errata:
            train[r["species"]].append(str(r["record_key"]))
    species = sorted(train)
    rng.shuffle(species)
    chosen, refs, depth = [], [], {}
    for s in species:
        if len(refs) >= reference_records:
            break
        keys = sorted(train[s])
        if len(keys) > cap:
            keys = sorted(rng.sample(keys, cap))
        chosen.append(s)
        refs += keys
        depth[s] = len(keys)
    halves = {}
    for half in HALVES:
        want_held = half == "val-former-heldout"
        pool: dict[str, list[str]] = defaultdict(list)
        for r in records:
            if (r["split"] == "val" and r.get("species") in depth
                    and r["record_key"] not in errata
                    and bool(r.get("benchmark")) == want_held):
                pool[band_of(depth[r["species"]])].append(str(r["record_key"]))
        take = allocate({b: len(v) for b, v in pool.items()}, queries)
        keys = []
        for b in sorted(pool):
            keys += sorted(rng.sample(sorted(pool[b]), take[b]))
        halves[half] = {"records": sorted(keys), "by_band": {b: take[b] for b in sorted(take)},
                        "pool": {b: len(v) for b, v in sorted(pool.items())}}
    species_list = sorted(chosen)
    return {"params": {"seed": seed, "reference_records": reference_records, "cap": cap,
                       "queries_per_half": queries},
            "species": species_list,
            "species_sha256": hashlib.sha256("\n".join(species_list).encode()).hexdigest(),
            "references": sorted(refs), "queries": halves,
            "errata_records_left_out": len(errata & {str(r["record_key"]) for r in records})}


def make_subset(rel, out: Path, errata_path=None, **params) -> dict:
    """Draw the subset for release `rel` and write subset.json under `out` (refused if one
    is there with other parameters or another release: the subset is drawn once)."""
    from .replications.fungitastic import dataset
    errata, errata_sha = read_errata(errata_path)
    rows = [dict(r) for r in rel.records(("train", "val"))]
    doc = draw(rows, errata=errata, **params)
    keys = set(doc["references"]) | {k for h in doc["queries"].values() for k in h["records"]}
    photos = sorted(int(r[0]) for r in rel.db.execute(
        "select manifest_photo_id, record_key from photos where included = 1")
        if str(r[1]) in keys)
    doc.update(dataset_release=dataset.cite(rel), errata={"file": str(errata_path) if
                                                          errata_path else None,
                                                          "sha256": errata_sha},
               photos=photos)
    path = out / "subset.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        same = (old.get("params") == doc["params"]
                and old.get("dataset_release", {}).get("release_hash")
                == doc["dataset_release"]["release_hash"]
                and old.get("errata", {}).get("sha256") == errata_sha)
        if not same:
            raise ValueError(f"{path} holds a subset drawn with other parameters, release or "
                             "errata; a screen's subset is drawn once (move it aside first)")
        return old
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def load_subset(rel_id: str, root: Path | None = None) -> dict:
    path = results_dir(rel_id, root) / "subset.json"
    if not path.is_file():
        raise ValueError(f"no subset at {path}: run mv backbone-screen subset first")
    return json.loads(path.read_text(encoding="utf-8"))


def photos_to_embed(conn, backbone: str, photo_ids: list[int]) -> dict[str, list]:
    """The subset's photos `backbone` has no vector for, grouped by the local store that
    holds a large copy: {store location: [(photo id, path)]}. A photo with no local copy
    is refused (this step never reads S3: copy the photos first)."""
    have = {r[0] for r in conn.execute("select photo_id from embeddings where backbone = ?",
                                       (backbone,))} if conn.execute(
        "select 1 from sqlite_master where name = 'embeddings'").fetchone() else set()
    out: dict[str, list] = defaultdict(list)
    missing = []
    for pid in photo_ids:
        if pid in have:
            continue
        row = conn.execute("select store, path from photo_copies where photo_id = ? and "
                           "size = 'large' and store not like 's3:%' order by store limit 1",
                           (pid,)).fetchone()
        if row is None:
            missing.append(pid)
        else:
            out[row[0]].append((pid, row[1]))
    if missing:
        raise ValueError(f"{len(missing):,} subset photos have no large copy on this machine "
                         f"(e.g. {missing[:5]}): copy them from S3 first")
    return dict(out)


def embed(conn, subset: dict, backbones: list[str], batch_size: int = 16, readers: int = 8,
          loader=None, log=print) -> dict:
    """Embed the subset's photos for each backbone (only those it lacks)."""
    from . import models
    from .embed import embed_photos
    from .storage import open_store
    loader = loader or models.load_backbone
    done = {}
    for spec in backbones:
        name = models.storage_name(spec)
        todo = photos_to_embed(conn, name, subset["photos"])
        n = sum(len(v) for v in todo.values())
        log(f"[{name}] {n:,} of {len(subset['photos']):,} subset photos to embed")
        stats = {"embedded": 0, "skipped": 0, "seconds": 0.0}
        if n:
            backbone = loader(spec)
            for where, items in todo.items():
                s = embed_photos(conn, open_store(where, config.DATA_DIR), backbone, items,
                                 config.DATA_DIR / "embeddings" / name, batch_size=batch_size,
                                 readers=readers, log=lambda m: log(f"[{name}] {m.strip()}"))
                stats["embedded"] += s.embedded
                stats["skipped"] += len(s.skipped)
                stats["seconds"] += getattr(s, "seconds", 0.0) or 0.0
            del backbone
        done[name] = stats
    return done


def score(conn, rel, subset: dict, backbones: list[str], methods: list[str],
          log=print) -> dict:
    """Each half's queries against the subset's references, every backbone on the photos
    all of them embedded (evaluate.compare; name scores on, for the standard tables)."""
    from . import evaluate
    from .embed import load_embeddings
    from .models import storage_name
    from .replications.fungitastic import dataset
    if subset["dataset_release"]["release_hash"] != rel.release_hash:
        raise ValueError("the subset was drawn from another release")
    rel.check_photos_match(conn)
    wanted = set(subset["photos"])
    loaded = {}
    for b in backbones:
        ids, vecs = load_embeddings(conn, storage_name(b))
        keep = [i for i, p in enumerate(ids.tolist()) if int(p) in wanted]
        loaded[storage_name(b)] = (ids[keep], vecs[keep])
    common = set.intersection(*(set(int(p) for p in ids.tolist())
                                for ids, _ in loaded.values()))
    same = {p: p for p in common}
    refs = set(subset["references"])
    ref = [r for r in rel.as_records("train", same) if r.observation_id in refs]
    out = {"species_sha256": subset["species_sha256"], "photos_scored": len(common),
           "photos_in_subset": len(wanted), "references": len(ref), "halves": {}}
    for half, q in subset["queries"].items():
        keys = set(q["records"])
        test = [r for r in rel.as_records("val", same) if r.observation_id in keys]
        cutoff = max((r.validated_on for r in ref if r.validated_on), default="")
        shared = evaluate.SharedSet(loaded, ref, test, cutoff,
                                    evaluate.record_set_hash(ref, test), len(common))
        res = evaluate.compare(conn, list(loaded), methods, test_days=0, shared=shared,
                               name_scores=True, sets=False,
                               extra={"dataset_release": dataset.cite(rel),
                                      "backbone_screen": {"half": half,
                                                          "species_sha256":
                                                              subset["species_sha256"]}},
                               log=log)
        out["halves"][half] = {"comparison_id": res["comparison_id"], "queries": len(test)}
        log(f"{half}: comparison {res['comparison_id']} ({len(test):,} queries, "
            f"{len(ref):,} references)")
    return out
