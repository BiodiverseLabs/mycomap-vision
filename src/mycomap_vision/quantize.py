"""Would int8 reference vectors cost accuracy? An offline check, before any serving change.

The box holds every reference photo's vector (float16, 768 dimensions: 1,536 bytes) and
v1 pushes its working set over the 3.5 GB cap (docs/retrain-v1.md §5). Stored as int8 with
one float32 scale per photo, a vector takes 772 bytes: about half. This module measures what
that does to the answers, on a dataset release's records, before anyone builds the serving
side:

- quantize_int8: per vector, scale = max |component| / 127, q = round(v / scale);
  dequantize gives q * scale (what a serving search would effectively compare against);
- check: for each method, the same reference and test records of the release scored twice,
  with the float16 reference vectors and with their int8 round trip (the query photos stay
  as the model gives them, as on the box, which embeds them fresh), then top-1 and top-5 at species and genus for both and the
  paired count of records only one of the two gets right (a sign test says whether the
  difference is more than chance).

Pre-declared reading (2026-10-12): int8 is "no loss" when species top-1 differs by less than
0.3 points and the paired sign test is not significant (p >= 0.05) for nearest+mean, the
served default. Anything else goes to Steve with the numbers.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np

from . import evaluate

NO_LOSS_POINTS = 0.3
SIGNIFICANCE = 0.05


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(int8 codes, float32 scale per vector). A zero vector keeps scale 0."""
    v = np.asarray(vectors, dtype=np.float32)
    peak = np.abs(v).max(axis=1)
    scale = (peak / 127.0).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.0)[:, None]
    q = np.clip(np.rint(v / safe), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * scale[:, None]).astype(np.float32)


def bytes_per_vector(dim: int) -> dict:
    return {"float16": 2 * dim, "int8": dim + 4}


def sign_test_p(only_a: int, only_b: int) -> float:
    """Two-sided exact sign test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def hits(ref_vectors: np.ndarray, query_vectors: np.ndarray, ref, test, method: str,
         backbone: str | None = None) -> dict:
    """Per test record: (species top-1, species top-5, genus top-1, genus top-5) hits, with
    the reference index built from `ref_vectors` and the test photos read from
    `query_vectors` (same rows)."""
    index, model = evaluate.fit_method(ref_vectors, ref, method, backbone=backbone)
    out = {}
    for rec in test:
        scores = model.species_scores(query_vectors[rec.photo_rows])
        row = []
        for rank in ("species", "genus"):
            t = evaluate.truth(rec, rank)
            if not t:
                row += [None, None]
                continue
            top = evaluate.top_labels(evaluate.rank_scores(scores, index, rank),
                                      index.labels[rank], 5)
            row += [top[:1] == [t], t in top]
        out[rec.observation_id] = tuple(row)
    return out


def summarise(a: dict, b: dict) -> dict:
    """float16 (a) vs int8 (b) on the same records."""
    out = {}
    for i, name in enumerate(("species_top1", "species_top5", "genus_top1", "genus_top5")):
        keys = [k for k in a if a[k][i] is not None]
        n = len(keys)
        fa = sum(a[k][i] for k in keys)
        fb = sum(b[k][i] for k in keys)
        only_a = sum(a[k][i] and not b[k][i] for k in keys)
        only_b = sum(b[k][i] and not a[k][i] for k in keys)
        out[name] = {"n": n, "float16": round(fa / max(n, 1), 4), "int8": round(fb / max(n, 1), 4),
                     "points": round(100 * (fb - fa) / max(n, 1), 2),
                     "only_float16": only_a, "only_int8": only_b,
                     "sign_test_p": round(sign_test_p(only_a, only_b), 4)}
    return out


def check(conn: sqlite3.Connection, backbone: str, release, methods: list[str],
          embeddings_root=None, log=print) -> dict:
    shared = evaluate.shared_records(conn, [backbone], embeddings_root=embeddings_root,
                                     release=release)
    ids, vecs = shared.loaded[backbone]
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    ref = evaluate.with_rows(shared.ref, row_of)
    test = evaluate.with_rows(shared.test, row_of)
    q, scale = quantize_int8(vecs)
    rt = dequantize(q, scale).astype(vecs.dtype)
    err = float(np.abs(rt.astype(np.float32) - vecs.astype(np.float32)).max())
    results = {}
    for m in methods:
        if not evaluate.method_applies(m, backbone):
            continue
        log(f"  {backbone} / {m}: {len(test):,} records, float16 then int8")
        results[m] = summarise(hits(vecs, vecs, ref, test, m, backbone),
                               hits(rt, vecs, ref, test, m, backbone))
    default = results.get("nearest+mean", {}).get("species_top1")
    verdict = None
    if default:
        verdict = ("no loss" if abs(default["points"]) < NO_LOSS_POINTS
                   and default["sign_test_p"] >= SIGNIFICANCE else "a difference: for Steve")
    dim = int(vecs.shape[1])
    return {"backbone": backbone, "dataset_release": release.cite() if release else None,
            "test_records": len(test), "reference_photos": int(len(ids)),
            "max_abs_roundtrip_error": round(err, 5), "bytes_per_vector": bytes_per_vector(dim),
            "vectors_mb": {"float16": round(len(ids) * 2 * dim / 1e6),
                           "int8": round(len(ids) * (dim + 4) / 1e6)},
            "methods": results, "verdict_nearest_mean": verdict}


def format_check(r: dict) -> str:
    rel = (r.get("dataset_release") or {}).get("dataset_release", "live manifest")
    lines = [f"int8 reference vectors vs float16: {r['backbone']} on {rel}, "
             f"{r['test_records']:,} val records, {r['reference_photos']:,} photos; vectors "
             f"{r['vectors_mb']['float16']:,} MB -> {r['vectors_mb']['int8']:,} MB "
             f"(round-trip error at most {r['max_abs_roundtrip_error']})"]
    for m, res in r["methods"].items():
        lines.append(f"  {m}")
        for name, c in res.items():
            lines.append(f"    {name:<13} {100 * c['float16']:6.2f}% -> {100 * c['int8']:6.2f}% "
                         f"({c['points']:+.2f} pts; only float16 {c['only_float16']}, only "
                         f"int8 {c['only_int8']}, p {c['sign_test_p']}; n {c['n']:,})")
    lines.append(f"verdict (nearest+mean species top-1, pre-declared): "
                 f"{r['verdict_nearest_mean'] or 'not measured'}")
    return "\n".join(lines)


# --- int8 shards for a serving release (Steve, 2026-10-12: v1 stays on the 4 GB box) ----

SCALE_SUFFIX = ".scale.npy"          # serving.SCALE_SUFFIX
CONVERT_ROWS = 65_536


def write_int8_shards(src: Path, dest: Path) -> dict:
    """A backbone's float embedding folder as int8 shards for a release: each
    shard-NNNNN.npy becomes int8 codes with shard-NNNNN.scale.npy beside it (one float32
    per vector); the photo-id files and anything else are copied as they are. Each shard
    is checked after conversion: every vector's round trip keeps cosine >= MIN_COSINE with
    the original. Returns counts. `dest` must not exist."""
    import shutil
    src, dest = Path(src), Path(dest)
    if dest.exists():
        raise ValueError(f"{dest} exists")
    dest.mkdir(parents=True)
    shards = rows = 0
    worst = 1.0
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith("shard-") and name.endswith(".npy")) or \
                name.endswith((".ids.npy", SCALE_SUFFIX)):
            shutil.copyfile(p, dest / name)
            continue
        v = np.load(p, mmap_mode="r")
        if v.ndim != 2 or v.dtype not in (np.float16, np.float32):
            raise ValueError(f"{p}: not a float vector shard ({v.dtype}, {v.shape})")
        codes = np.empty(v.shape, dtype=np.int8)
        scale = np.empty(len(v), dtype=np.float32)
        for s in range(0, len(v), CONVERT_ROWS):
            block = np.asarray(v[s:s + CONVERT_ROWS], dtype=np.float32)
            q, sc = quantize_int8(block)
            codes[s:s + CONVERT_ROWS], scale[s:s + CONVERT_ROWS] = q, sc
            back = dequantize(q, sc)
            num = (back * block).sum(1)
            den = np.linalg.norm(back, axis=1) * np.linalg.norm(block, axis=1)
            ok = den > 0
            if ok.any():
                worst = min(worst, float((num[ok] / den[ok]).min()))
        if worst < MIN_COSINE:
            raise ValueError(f"{p}: an int8 round trip fell to cosine {worst:.5f}")
        np.save(dest / name, codes)
        np.save(dest / (name[:-len(".npy")] + SCALE_SUFFIX), scale)
        shards += 1
        rows += len(v)
    return {"shards": shards, "vectors": rows, "worst_cosine": round(worst, 6)}


MIN_COSINE = 0.999
