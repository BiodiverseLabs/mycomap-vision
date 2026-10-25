"""A model-free near-copy check for a sealed test release (paper plan §4, 2026-10-13).

A test release must not leak copies of its base release's photos. Byte-identical copies are
dropped by the build (dataset_release.separate_from_base); a re-saved, resized or lightly
cropped copy is not byte-identical, so the build also FLAGS each test photo whose difference
hash (dHash) is within Hamming distance 4 of any included base photo. A flag drops nothing:
the test set stays fixed, and scoring reports near-duplicate finds apart. No model is used,
so the check can run before the seal is broken and does not depend on the model under test.

dHash (64 bits): decode, apply the EXIF orientation, grey, resize to 9 x 8 (Lanczos), and set
bit i where a pixel is brighter than its right-hand neighbour. JPEGs are decoded with
draft("L", (64, 64)) (the decoder's own 1/2-1/8 DCT scaling: several times faster, and the
same input always gives the same hash with the same Pillow).

The base side is an INDEX file computed once per base release (`build_index`, on an
instance in the bucket's region for a large release: `mv aws-launch-dhash-index`): every
included photo's original_sha256 and dHash, with the release_hash, the algorithm and the
Pillow version it was made with. A test build refuses an index made for another release, by
another algorithm or another Pillow, or one that misses any included base photo.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .dataset_release import ReleaseError

ALGORITHM = "dhash64-draft64-lanczos-v1"
NEAR = 4                                # Hamming distance at or under which a photo is flagged
INDEX_FILE = "dhash-index.npz"


class DhashError(ReleaseError):
    pass


def index_path(rid: str) -> Path:
    """Where a release's dHash index lives on this machine (beside the releases, never in
    one: a release is never edited)."""
    from . import config
    return config.DATA_DIR / "research-release-extras" / rid / INDEX_FILE


def pillow_version() -> str:
    import PIL
    return PIL.__version__


def dhash64(body: bytes) -> int:
    """The 64-bit difference hash of an image file's bytes."""
    from PIL import Image, ImageOps
    from .guards import JPEG_FAMILY, photo_formats
    img = Image.open(io.BytesIO(body), formats=photo_formats())
    if img.format in JPEG_FAMILY:
        img.draft("L", (64, 64))
    img = ImageOps.exif_transpose(img).convert("L")
    px = np.asarray(img.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (px[:, :-1] > px[:, 1:]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def to_hex(h: int) -> str:
    return f"{h:016x}"


def nearest(queries: np.ndarray, base: np.ndarray, max_distance: int = NEAR,
            chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For each query hash (uint64), the index of its nearest base hash and the Hamming
    distance, or (-1, -1) when none is within max_distance."""
    queries = np.asarray(queries, dtype=np.uint64)
    base = np.asarray(base, dtype=np.uint64)
    idx = np.full(len(queries), -1, dtype=np.int64)
    dist = np.full(len(queries), -1, dtype=np.int64)
    if not len(base) or not len(queries):
        return idx, dist
    for s in range(0, len(queries), chunk):
        q = queries[s:s + chunk]
        d = np.bitwise_count(q[:, None] ^ base[None, :])          # (chunk, base), uint8
        best = d.argmin(axis=1)
        bd = d[np.arange(len(q)), best].astype(np.int64)
        ok = bd <= max_distance
        idx[s:s + chunk] = np.where(ok, best, -1)
        dist[s:s + chunk] = np.where(ok, bd, -1)
    return idx, dist


# --- the base release's index ---------------------------------------------------------------

def build_index(rel, store_reader, *, workers: int = 16, log=print) -> dict:
    """dHash every included photo of a release (needs its private part: the stores). Each
    original is checked against its sha256 before it is hashed. Returns the index dict that
    save_index writes."""
    from concurrent.futures import ThreadPoolExecutor

    from .dataset_release import sha256_bytes
    stores = dict(rel.private.execute(
        "select ps.photo_key, s.location from photo_stores ps join stores s using (store_id)"))
    rows = rel.db.execute("select photo_key, original_path, original_sha256 from photos "
                          "where included = 1 order by photo_key").fetchall()

    def one(row):
        key, path, sha = row
        body = store_reader(stores[key], path)
        if sha256_bytes(body) != sha:
            raise DhashError(f"{key}: the original's sha256 differs from the release")
        return dhash64(body)

    hashes = np.zeros(len(rows), dtype=np.uint64)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, h in enumerate(ex.map(one, rows)):
            hashes[i] = h
            if (i + 1) % 20000 == 0:
                log(f"  dhash {i + 1:,} / {len(rows):,} photos")
    return {"release": rel.id, "release_hash": rel.release_hash, "algorithm": ALGORITHM,
            "pillow": pillow_version(), "photos": len(rows),
            "photo_key": [r[0] for r in rows], "sha256": [r[2] for r in rows],
            "dhash": hashes}


def save_index(index: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: index[k] for k in ("release", "release_hash", "algorithm", "pillow", "photos")}
    with open(path, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta, sort_keys=True)),
                            photo_key=np.array(index["photo_key"], dtype=str),
                            sha256=np.array(index["sha256"], dtype=str),
                            dhash=np.asarray(index["dhash"], dtype=np.uint64))
    return path


def load_index(path: Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        out = {**meta, "photo_key": [str(x) for x in z["photo_key"]],
               "sha256": [str(x) for x in z["sha256"]], "dhash": z["dhash"].astype(np.uint64)}
    if not (len(out["photo_key"]) == len(out["sha256"]) == len(out["dhash"]) == out["photos"]):
        raise DhashError(f"{path}: the index's arrays disagree with its photo count")
    return out


def check_index(index: dict, base) -> None:
    """Refuse an index that is not this base release's, at this algorithm and Pillow, over
    every included photo."""
    if index["release"] != base.id or index["release_hash"] != base.release_hash:
        raise DhashError(f"the dHash index is for {index['release']} "
                         f"({index['release_hash'][:12]}), not {base.id} "
                         f"({base.release_hash[:12]})")
    if index["algorithm"] != ALGORITHM:
        raise DhashError(f"the dHash index was made with {index['algorithm']}, not {ALGORITHM}")
    if index["pillow"] != pillow_version():
        raise DhashError(f"the dHash index was made with Pillow {index['pillow']}, this is "
                         f"{pillow_version()}: hashes could differ")
    missing = base.photo_shas() - set(index["sha256"])
    if missing:
        raise DhashError(f"the dHash index misses {len(missing):,} included photos of {base.id}")
