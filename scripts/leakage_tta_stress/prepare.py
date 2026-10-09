"""Step 1: everything the leakage / TTA / stress scoring needs, from a manifest snapshot.

Run from the repo root with PYTHONPATH=src (no torch is imported):

    python scripts/leakage_tta_stress/prepare.py --manifest data/stress/manifest-snapshot.sqlite \
        --embeddings <main data>/embeddings/bioclip-2-ft-20261007-165400 \
        --bench-index data/stress/bench-index.sqlite \
        --bench-vectors <main data>/benchmarks/heldout-2026-10-08/embeddings/large/bioclip-2-ft-20261007-165400 \
        --sources <main data>/audits/record-sources-2026-10-09/org-sources-live.tsv --out data/stress

The snapshot is a copy of the shared manifest (sqlite backup), with the taxonomy cache
copied beside it: labels depend on both, and the shared manifest drifts during the day.
The reference is what identify.Identifier serves from that snapshot (load_records,
build_index), and its hash is computed as heldout.reference_summary does. Outputs stay
in --out (git-ignored data/): reference vectors in species order, per-record facts
(observer, true place and date, state, source), the dev queries' photo vectors, the
place-and-date prior fitted on the reference. They hold true coordinates: keep private.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from mycomap_vision import heldout
from mycomap_vision.embed import load_embeddings
from mycomap_vision.evaluate import build_index, load_records, with_rows
from mycomap_vision.identify import Specimens
from mycomap_vision.prior import Context, RangeSeasonPrior, haversine_km
from mycomap_vision.serving import map_embeddings

BACKBONE = "bioclip-2-ft-20261007-165400"
BENCH = "heldout-2026-10-08"


def ordinal(day: str | None) -> int:
    try:
        return date.fromisoformat((day or "")[:10]).toordinal()
    except ValueError:
        return -1


def doy(day: str | None) -> int:
    d = Context(observed_on=day).day_of_year
    return -1 if d is None else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", "--snapshot", dest="snapshot", required=True, type=Path,
                    help="a manifest copy (sqlite backup of the shared one) or a release's "
                         "manifest; never the live shared manifest, which drifts")
    ap.add_argument("--embeddings", required=True, type=Path)
    ap.add_argument("--bench-index", required=True, type=Path)
    ap.add_argument("--bench-vectors", required=True, type=Path)
    ap.add_argument("--sources", required=True, type=Path)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()
    conn = sqlite3.connect(a.snapshot)

    # --- the reference, as identify.Identifier builds it
    ids, vecs = map_embeddings(conn, BACKBONE, a.embeddings)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    records = with_rows(load_records(conn, {int(p): int(p) for p in ids.tolist()}), row_of)
    index = build_index(records)
    print(f"reference: {len(records):,} records, {len(index.cols):,} photos, "
          f"{len(index.species):,} groups ({time.time() - t0:.0f} s)")
    spec = Specimens(records, index.cols.tolist())
    lines = sorted(f"{o}\t" + "\t".join(str(spec.names[i]) for i in spec.taxa[r])
                   for r, o in enumerate(spec.obs.tolist()))
    h = hashlib.sha1(f"{BACKBONE}|".encode())
    h.update("\n".join(lines).encode())
    ref_hash = h.hexdigest()[:12]
    print("reference hash", ref_hash)

    ref = np.empty((len(index.cols), vecs.shape[1]), dtype=np.float16)
    for s in range(0, len(index.cols), 65536):
        ref[s:s + 65536] = vecs[index.cols[s:s + 65536]]
    np.save(a.out / "ref_vecs.npy", ref)
    counts = np.diff(np.append(index.starts, len(index.cols)))
    col_group = np.repeat(np.arange(len(index.species)), counts).astype(np.int32)
    col_photo = ids[index.cols]
    col_rec = spec.rec

    # --- per-record facts
    users = {oid: uid for oid, uid in conn.execute(
        "select observation_id, user_id from inat_observations where status = 'ok'")}
    place = {oid: (st or "", co or "") for oid, st, co in conn.execute(
        "select observation_id, state, country from records")}
    sources = {}
    with open(a.sources, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2 and row[0] != "observation_id":
                sources[row[0]] = row[1]
    group_pos = {s: i for i, s in enumerate(index.species)}
    rec = {
        "obs": np.array([r.observation_id for r in records]),
        "user": np.array([users.get(r.observation_id) or -1 for r in records], dtype=np.int64),
        "lat": np.array([np.nan if r.latitude is None else r.latitude for r in records]),
        "lon": np.array([np.nan if r.longitude is None else r.longitude for r in records]),
        "day": np.array([ordinal(r.observed_on) for r in records], dtype=np.int64),
        "doy": np.array([doy(r.observed_on) for r in records], dtype=np.int64),
        "group": np.array([group_pos[r.unit] for r in records], dtype=np.int32),
        "state": np.array([place.get(r.observation_id, ("", ""))[0] for r in records]),
        "country": np.array([place.get(r.observation_id, ("", ""))[1] for r in records]),
        "source": np.array([sources.get(r.observation_id, "unknown") for r in records]),
    }
    non_inat = rec["source"] != "iNaturalist"
    print("reference records by .org source:", dict(Counter(rec["source"].tolist())))
    print(f"non-iNat references (wrong-photo variant excludes them): {int(non_inat.sum()):,}")

    prior = RangeSeasonPrior()
    prior.fit(records, index.species)

    # --- the dev queries, as heldout.predict answers them
    split_ids = [r[0] for r in conn.execute(
        "select observation_id from heldout_records where benchmark = ? and split = ?",
        (BENCH, a.split))]
    first_validated = dict(conn.execute(
        "select observation_id, first_validated from heldout_records where benchmark = ?",
        (BENCH,)))
    bench = [r for r in heldout.load_benchmark(conn, BENCH, split_ids, "large") if r.photos]
    edb = sqlite3.connect(a.bench_index)
    qids, qvecs = load_embeddings(edb, BACKBONE, a.bench_vectors)
    edb.close()
    qrow = {int(p): i for i, p in enumerate(qids.tolist())}
    ref_obs = set(rec["obs"].tolist())
    ref_photos = set(int(p) for p in col_photo.tolist())
    labeller = heldout.Labeller(conn, extra=[r.truth_name for r in bench if r.truth_name])
    skipped = Counter()
    queries = []
    have = ~np.isnan(rec["lat"])
    for r in bench:
        if r.observation_id in ref_obs:
            skipped["already a reference record"] += 1
            continue
        pids = [p for p, _s, _p in r.photos]
        if ref_photos & set(pids):
            skipped["photo also a reference photo"] += 1
            continue
        rows = [qrow[p] for p in pids if p in qrow]
        if not rows:
            skipped["no vectors"] += 1
            continue
        lat = r.org_latitude if r.org_latitude is not None else r.inat_latitude
        lon = r.org_longitude if r.org_longitude is not None else r.inat_longitude
        region = ("", "")
        if lat is not None and lon is not None:
            d = haversine_km(lat, lon, rec["lat"][have], rec["lon"][have])
            j = np.flatnonzero(have)[int(np.argmin(d))]
            region = (str(rec["state"][j]), str(rec["country"][j]))
        truth = labeller.truth(r.truth_name) if r.truth_name else None
        queries.append({
            "oid": r.observation_id, "truth_name": r.truth_name, "truth": truth,
            "vecs": np.asarray(qvecs[rows], dtype=np.float16),
            "photo_ids": [p for p in pids if p in qrow],
            "user": r.user_id if r.user_id is not None else -1,
            "org_lat": r.org_latitude, "org_lon": r.org_longitude,
            "lat": lat, "lon": lon, "observed_on": r.observed_on,
            "day": ordinal(r.observed_on), "doy": doy(r.observed_on),
            "first_validated": first_validated.get(r.observation_id),
            "region": region, "was_reference": r.was_reference,
        })
    print(f"queries: {len(queries):,} of {len(split_ids):,} {a.split} records; skipped "
          f"{dict(skipped)}")

    np.savez(a.out / "ref_meta.npz", col_group=col_group, col_rec=col_rec,
             col_photo=col_photo, starts=index.starts, non_inat=non_inat, **rec)
    with open(a.out / "prep.pkl", "wb") as f:
        pickle.dump({"species": index.species, "labels": index.labels,
                     "label_of": index.label_of, "ref_count": dict(index.ref_count),
                     "prior": prior, "queries": queries, "ref_hash": ref_hash,
                     "split": a.split, "skipped": dict(skipped)}, f)
    (a.out / "snapshot.json").write_text(json.dumps({
        "reference_hash": ref_hash, "backbone": BACKBONE, "records": len(records),
        "photos": int(len(index.cols)), "groups": len(index.species),
        "non_inat_references": int(non_inat.sum()),
        "sources": dict(Counter(rec["source"].tolist())),
        "queries": len(queries), "skipped": dict(skipped),
        "code_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
        "reproducibility": "exploratory-pre-freeze",
        "manifest_newest_export": conn.execute("select max(exported_at) from records"
                                               ).fetchone()[0]}, indent=2))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
