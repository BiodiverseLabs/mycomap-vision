"""Re-score a held-out benchmark split with edited reference labels (label audit, 2026-10-09).

One command, CPU only (numpy; torch is never imported, which avoids the OpenBLAS crash seen when
both are loaded). It reads a manifest COPY and never writes to it or to the benchmark folder:

    python tools/label_audit_rescore.py --manifest <copy of manifest.sqlite> \
        --cache <scratch folder> [--exclude ids.txt] [--relabel oid_name.tsv] [--merge pairs.tsv]

- First run: for every photo of the split, its --top-k nearest reference photos (the backbone's
  stored vectors), saved in --cache. Later runs reuse it (a few seconds to load).
- --exclude: reference records to drop (first column: an observation id; a header is skipped).
- --relabel: reference records to give another name (observation id, name). The name is labelled
  the way the benchmark labels its answers (heldout.Labeller).
- --merge: label pairs (keep, other): `other` is scored as `keep`, in answers and in the key.
  Exact for nearest (the best match of the union is the better of the two); merging can only add
  hits, so compare it with a null (random merges of as many labels).
Prints Steve's standard tables (top 1/3/5/10; species strict / s.l. / complex, genus strict /
s.l.; species top-1 and top-5 by the true species' reference records) for the baseline and the
edited labels, both from this scorer, with the records fixed and broken.

The scorer reproduces the official development run within 0.3 points (species top-1 48.4 vs 48.3
nearest, 53.0 vs 52.8 nearest + species average). Exploratory, before the dataset freeze.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mycomap_vision import evaluate, heldout, name_equiv

KS = (1, 3, 5, 10)
BANDS = ("0", "1-4", "5-19", "20-99", "100+")


def band(n: int) -> str:
    return "0" if n == 0 else "1-4" if n < 5 else "5-19" if n < 20 else "20-99" if n < 100 else "100+"


def load_shards(db: sqlite3.Connection, root: Path, backbone: str):
    shards = sorted({r[0] for r in db.execute(
        "select distinct shard from embeddings where backbone = ?", (backbone,))})
    ids = [np.load(root / f"shard-{s:05d}.ids.npy") for s in shards]
    vecs = [np.load(root / f"shard-{s:05d}.npy") for s in shards]
    return np.concatenate(ids), np.concatenate(vecs)


def build_cache(conn, data: Path, bench: str, split: str, backbone: str, size: str,
                top_k: int, cache: Path) -> None:
    t = time.time()
    rid, rvec = load_shards(conn, data / "embeddings" / backbone, backbone)
    recs = evaluate.load_records(conn, {int(p): i for i, p in enumerate(rid.tolist())})
    rows = np.concatenate([np.array(r.photo_rows) for r in recs])
    rec_of = np.concatenate([np.full(len(r.photo_rows), i) for i, r in enumerate(recs)])
    R = rvec[rows].astype(np.float32)
    bdir = data / "benchmarks" / bench / "embeddings" / size
    bdb = sqlite3.connect(f"file:{(bdir / 'index.sqlite').as_posix()}?mode=ro", uri=True)
    qid, qvec = load_shards(bdb, bdir / backbone, backbone)
    qmap = {int(p): i for i, p in enumerate(qid.tolist())}
    qp = []
    for (oid,) in conn.execute("select observation_id from heldout_records where benchmark = ? "
                               "and split = ? order by observation_id", (bench, split)):
        for (p,) in conn.execute("select photo_id from heldout_photos where benchmark = ? and "
                                 "observation_id = ? and status = 'done' order by position",
                                 (bench, oid)):
            if p in qmap:
                qp.append((oid, p))
    Q = np.stack([qvec[qmap[p]] for _, p in qp]).astype(np.float32)
    TI = np.empty((len(Q), top_k), np.int32)
    TS = np.empty((len(Q), top_k), np.float16)
    for s in range(0, len(Q), 256):
        S = Q[s:s + 256] @ R.T
        idx = np.argpartition(-S, top_k, axis=1)[:, :top_k]
        sv = np.take_along_axis(S, idx, axis=1)
        o = np.argsort(-sv, axis=1)
        TI[s:s + 256] = np.take_along_axis(idx, o, axis=1)
        TS[s:s + 256] = np.take_along_axis(sv, o, axis=1)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "topk.npz", TI=TI, TS=TS, rec_of=rec_of,
             q_oid=np.array([o for o, _ in qp]),
             ref_oid=np.array([r.observation_id for r in recs]),
             ref_species=np.array([r.species for r in recs]),
             ref_genus=np.array([r.genus for r in recs]),
             ref_taxon=np.array([r.taxon for r in recs]))
    np.save(cache / "ref.npy", R.astype(np.float16))
    np.save(cache / "query.npy", Q.astype(np.float16))
    print(f"cache: {len(recs):,} reference records, {len(R):,} photos, {len(Q):,} query photos "
          f"({time.time() - t:.0f} s)", file=sys.stderr)


class Scorer:
    def __init__(self, cache: Path, conn, bench: str, split: str):
        z = np.load(cache / "topk.npz")
        self.TI, self.TS, self.rec_of = z["TI"], z["TS"].astype(np.float32), z["rec_of"]
        self.ref_oid = [str(o) for o in z["ref_oid"]]
        self.sp = z["ref_species"].astype(object)
        self.ge = z["ref_genus"].astype(object)
        self.tx = z["ref_taxon"].astype(object)
        self.R = np.load(cache / "ref.npy", mmap_mode="r")
        self.Q = np.load(cache / "query.npy").astype(np.float32)
        self.photos = defaultdict(list)
        for i, o in enumerate(z["q_oid"]):
            self.photos[str(o)].append(i)
        key = {o: t for o, t in conn.execute(
            "select observation_id, truth_name from heldout_records where benchmark = ? and "
            "split = ?", (bench, split))}
        self.labeller = heldout.Labeller(conn, extra=[t for t in key.values() if t])
        self.truth = {o: self.labeller.truth(t) for o, t in key.items() if t}

    def index(self, exclude=frozenset(), relabel=None):
        sp, ge, tx = self.sp.copy(), self.ge.copy(), self.tx.copy()
        for i, (s, g, t) in (relabel or {}).items():
            sp[i], ge[i], tx[i] = s, g, t
        unit = np.where(sp != "", sp, tx)
        keep = np.array([i not in exclude and u != "" for i, u in enumerate(unit)])
        names = sorted(set(unit[keep]))
        pos = {n: k for k, n in enumerate(names)}
        uid = np.array([pos[u] if k else -1 for u, k in zip(unit, keep)])
        is_sp = np.zeros(len(names), bool)
        u_ge = np.full(len(names), "", object)
        cnt = Counter()
        for i in np.nonzero(keep)[0]:
            cnt[uid[i]] += 1
            is_sp[uid[i]] |= bool(sp[i])
            u_ge[uid[i]] = ge[i] or u_ge[uid[i]]
        ph = uid[self.rec_of]
        protos = np.zeros((len(names), self.R.shape[1]), np.float32)
        ok = np.nonzero(ph >= 0)[0]
        for s in range(0, len(ok), 60000):
            rows = ok[s:s + 60000]
            np.add.at(protos, ph[rows], np.asarray(self.R[rows], dtype=np.float32))
        protos /= np.linalg.norm(protos, axis=1, keepdims=True).clip(1e-12)
        pc = Counter(ph[ph >= 0].tolist())
        genera = sorted({g for g in u_ge if g})
        gpos = {g: k for k, g in enumerate(genera)}
        return dict(names=np.array(names, object), ph=ph, is_sp=is_sp, protos=protos,
                    gid=np.array([gpos.get(g, -1) for g in u_ge]), genera=genera,
                    one_photo=np.array([pc[i] == 1 for i in range(len(names))]),
                    count={names[u]: c for u, c in cnt.items()})

    def scores(self, ix, photos, method):
        n = len(ix["names"])
        near = np.zeros(n, np.float32)
        top2 = np.zeros(n, np.float32)
        for p in photos:
            u = ix["ph"][self.TI[p]]
            s = self.TS[p]
            u, s = u[u >= 0], s[u >= 0]
            floor = s[-1]
            uu, first = np.unique(u, return_index=True)
            m = np.full(n, floor, np.float32)
            m[uu] = s[first]
            near += m
            if method == "nearest+mean":
                rest = np.ones(len(u), bool)
                rest[first] = False
                u2, f2 = np.unique(u[rest], return_index=True)
                sec = np.full(n, floor, np.float32)
                sec[u2] = s[rest][f2]
                top2 += (m + sec) / 2
        near /= len(photos)
        if method == "nearest":
            return near
        top2 /= len(photos)
        mean = (self.Q[photos] @ ix["protos"].T).mean(axis=0)
        return 0.6 * np.where(ix["one_photo"], near, top2) + 0.4 * mean

    def evaluate(self, ix, method, merge=None, depth_from=None):
        M = (lambda x: merge.get(x, x)) if merge else (lambda x: x)
        depth_from = depth_from or ix["count"]
        names, sp_idx = ix["names"], np.nonzero(ix["is_sp"])[0]
        hits, depth, ids = defaultdict(list), [], []
        for oid in sorted(self.photos):
            t = self.truth.get(oid)
            if t is None:
                continue
            sc = self.scores(ix, self.photos[oid], method)
            if t.species:
                preds = []
                for u in sp_idx[np.argsort(-sc[sp_idx])[:10]]:
                    if M(names[u]) not in preds:
                        preds.append(M(names[u]))
                for lv in ("strict", "sl", "complex"):
                    h = [name_equiv.species_match(p, M(t.species))[lv] for p in preds]
                    hits["species " + lv].append([any(h[:k]) for k in KS])
                depth.append(band(depth_from.get(t.species, 0)))
                ids.append(oid)
            if t.genus:
                g = np.full(len(ix["genera"]), -np.inf, np.float32)
                ok = ix["gid"] >= 0
                np.maximum.at(g, ix["gid"][ok], sc[ok])
                top = [ix["genera"][k] for k in np.argsort(-g)[:10]]
                for lv in ("strict", "sl"):
                    h = [name_equiv.genus_match(p, t.genus)[lv] for p in top]
                    hits["genus " + lv].append([any(h[:k]) for k in KS])
        out = {k: np.array(v) for k, v in hits.items()}
        out["depth"], out["ids"] = np.array(depth), ids
        return out


def table(out, title: str) -> str:
    lines = [title, f"  {'':16s}  top 1  top 3  top 5 top 10      n"]
    for k in ("species strict", "species sl", "species complex", "genus strict", "genus sl"):
        a = out[k]
        lines.append(f"  {k:16s} " + " ".join(f"{100 * a[:, i].mean():6.1f}" for i in range(4))
                     + f" {len(a):6d}")
    lines.append("  species strict by the true species' reference records (top 1 / top 5):")
    a, d = out["species strict"], out["depth"]
    for b in BANDS:
        m = d == b
        if m.sum():
            lines.append(f"    {b:6s} {100 * a[m, 0].mean():5.1f} / {100 * a[m, 2].mean():5.1f}"
                         f"  (n={m.sum()})")
    return "\n".join(lines)


def first_column(path: str) -> set[str]:
    rows = [l.split("\t")[0].strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
    return {r for r in rows if r.isdigit()}


def pairs(path: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [(r[0], r[1]) for r in csv.reader(f, delimiter="\t") if len(r) >= 2]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="a COPY of manifest.sqlite (opened read-only)")
    ap.add_argument("--data", default=None, help="the Vision data folder (default: beside the "
                    "real manifest, i.e. MV_DATA_DIR)")
    ap.add_argument("--cache", required=True, help="scratch folder for the neighbour cache")
    ap.add_argument("--benchmark", default="heldout-2026-10-08")
    ap.add_argument("--split", default="dev", choices=["dev"],
                    help="dev only: label choices must never be made on test predictions")
    ap.add_argument("--backbone", default="bioclip-2-ft-20261007-165400")
    ap.add_argument("--size", default="large")
    ap.add_argument("--top-k", type=int, default=3000)
    ap.add_argument("--methods", default="nearest,nearest+mean")
    ap.add_argument("--exclude")
    ap.add_argument("--relabel")
    ap.add_argument("--merge")
    a = ap.parse_args(argv)
    from mycomap_vision import config
    data = Path(a.data) if a.data else config.DATA_DIR
    conn = sqlite3.connect(f"file:{Path(a.manifest).as_posix()}?mode=ro", uri=True)
    cache = Path(a.cache)
    if not (cache / "topk.npz").exists():
        build_cache(conn, data, a.benchmark, a.split, a.backbone, a.size, a.top_k, cache)
    sc = Scorer(cache, conn, a.benchmark, a.split)
    pos = {o: i for i, o in enumerate(sc.ref_oid)}
    exclude = frozenset(pos[o] for o in first_column(a.exclude) if o in pos) if a.exclude else frozenset()
    relabel = {}
    for o, name in pairs(a.relabel) if a.relabel else []:
        if o in pos:
            t = sc.labeller.truth(name)
            relabel[pos[o]] = (t.species, t.genus, "" if t.species else t.label)
    merge = {}
    for keep, other in pairs(a.merge) if a.merge else []:
        if keep != other:
            merge[other] = merge.get(keep, keep)
    base = sc.index()
    edited = sc.index(exclude, relabel) if (exclude or relabel) else base
    print(json.dumps({"manifest_sha256_16": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()[:16],
                      "reference_records": len(sc.ref_oid), "excluded": len(exclude),
                      "relabelled": len(relabel), "merged_labels": len(merge),
                      "reproducibility": "exploratory-pre-freeze"}))
    for m in a.methods.split(","):
        b = sc.evaluate(base, m)
        o = sc.evaluate(edited, m, merge or None, depth_from=base["count"])
        print(table(b, f"\nBASELINE {m}"))
        print(table(o, f"\nEDITED {m}"))
        bb, oo = b["species strict"][:, 0], o["species strict"][:, 0]
        print(f"  species top-1: {int((oo & ~bb).sum())} fixed, {int((bb & ~oo).sum())} broken")


if __name__ == "__main__":
    main()
