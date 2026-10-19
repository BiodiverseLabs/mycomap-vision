"""The freeze spec of `mv paper-score` (paper PLAN §8, "What the freeze fixes"): the whole
predictor, part by part, by hash, so the sealed test is scored by exactly the predictor the
tag names and the report says which one that was.

Parts (each a sha256 unless said otherwise):

- `release`: the base release's id, release_hash and reference_hash (the reference index
  is v1's train + val with the erratum; never the live index);
- `errata`: the erratum file;
- `models`: per backbone a comparator uses, every file of it in data/models (weights,
  heads, metadata) and the base checkpoint it was built on (the hub's safetensors);
- `vectors`: per backbone, the stored vectors of every included photo of the base release
  (ids in order, then the float16 rows): the reference index as scored;
- `taxonomy`: iNat's taxonomy cache (the answer key's canonical step; mistake distance, E29);
- `occurrence`: the iNat occurrence store, when a comparator uses place;
- `place_settings`: per named place setting, its values (canonical JSON);
- `gate`: the not-a-fungus gate's fingerprint and its files;
- `calibration`: per comparator, the temperatures and list floors (values, not hashes);
- `crosswalk`: per published comparator, the crosswalk cache and review;
- `attrs`, `deadly_taxa`: the pinned files;
- `twins`: the cross-post twin groups files (validation, sealed_test), when a comparator
  scores by reference record;
- `code_commit`: the paper-score commit the tag freezes.

`compute` reads all of them; `check` compares them with the config's `freeze` block. The
sealed test refuses any part that differs or that the block leaves out; val and draft runs
report the differences and go on (a val rehearsal scores against train only, so its
release part is the same but the comparators may still be in flux).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FREEZE_KEYS = ("release", "errata", "models", "vectors", "taxonomy", "occurrence",
               "place_settings", "gate", "calibration", "crosswalk", "attrs", "deadly_taxa",
               "answer_key", "answer_key_taxonomy", "twins", "code_commit")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def cached_hub_file(model_name: str):
    """The hub checkpoint already in the local cache (no network call), or None."""
    from . import models
    name = models.split_options(model_name)[0]
    if not name.startswith("hf-hub:"):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        got = try_to_load_from_cache(name[len("hf-hub:"):], "open_clip_model.safetensors")
    except Exception:
        return None
    return got if isinstance(got, str) else None


def model_files(backbone: str, root: Path) -> dict[str, str]:
    """{file name: sha256} of the backbone's own files in data/models (`<name>.<suffix>`),
    and `base:<file>` for the hub checkpoint a fine-tune or alias was built from."""
    from . import models
    out = {p.name: sha256_file(p) for p in sorted(root.glob(f"{backbone}.*")) if p.is_file()}
    spec = None
    meta = root / f"{backbone}.json"
    if meta.is_file():
        spec = json.loads(meta.read_text(encoding="utf-8")).get("base_spec")
    else:
        try:
            spec = models.resolve_spec(backbone)
        except ValueError:
            spec = None
    if spec and spec.startswith("open_clip:"):
        base = cached_hub_file(spec[len("open_clip:"):]) or \
            models.open_clip_weights_file(spec[len("open_clip:"):])
        if base:
            out[f"base:{Path(base).name}"] = sha256_file(base)
    return out


def vectors_sha(ids: np.ndarray, vecs: np.ndarray, wanted: list[int]) -> dict:
    """sha256 over the wanted photos' ids (sorted, int64) then their rows (float16), and
    how many of them have no vector."""
    row = {int(p): i for i, p in enumerate(np.asarray(ids).tolist())}
    have = sorted(p for p in set(wanted) if p in row)
    h = hashlib.sha256(np.asarray(have, dtype=np.int64).tobytes())
    if have:
        h.update(np.ascontiguousarray(
            np.asarray(vecs)[[row[p] for p in have]], dtype=np.float16).tobytes())
    return {"sha256": h.hexdigest(), "photos": len(have),
            "missing": len(set(wanted)) - len(have)}


def compute(config: dict, base, conn, resolve, models_root: Path | None = None,
            load_vectors=None, taxonomy_path=None) -> dict:
    """Every part of the predictor `config` names, read now."""
    from .replications.fungitastic import dataset
    if models_root is None:
        from .finetune import models_dir
        models_root = models_dir()
    if load_vectors is None:
        from .embed import load_embeddings

        def load_vectors(b):
            return load_embeddings(conn, b)
    comps = config.get("comparators", [])
    vision = [c for c in comps if c.get("kind") == "vision"]
    cite = dataset.cite(base)
    spec: dict = {"release": {k: cite.get(k) for k in ("dataset_release", "release_hash",
                                                       "reference_hash")}}
    if config.get("errata"):
        spec["errata"] = sha256_file(resolve(config["errata"]["file"]))
    backbones = sorted({c["backbone"] for c in vision if not str(c["backbone"]).startswith("TBD")})
    spec["models"] = {b: model_files(b, models_root) for b in backbones}
    wanted = [int(p) for (p,) in base.db.execute(
        "select manifest_photo_id from photos where included = 1")]
    spec["vectors"] = {}
    for b in backbones:
        ids, vecs = load_vectors(b)
        spec["vectors"][b] = vectors_sha(ids, vecs, wanted)
    from . import taxonomy
    tax = Path(taxonomy_path) if taxonomy_path else taxonomy.cache_path()
    spec["taxonomy"] = sha256_file(tax) if tax.is_file() else None
    if any(c.get("place") for c in comps):
        from .occurrence import default_store_path
        store = Path(default_store_path())
        spec["occurrence"] = sha256_file(store) if store.is_file() else None
    names = sorted({c["place_setting"] for c in comps if c.get("place_setting")})
    if names:
        from . import placeprior
        spec["place_settings"] = {n: canonical_sha(placeprior.place_setting(n)[1])
                                  for n in names}
    if config.get("gate"):
        gdir = resolve(config["gate"]["dir"])
        from . import fungusgate
        gate = fungusgate.FungusGate.load(gdir)
        spec["gate"] = {"fingerprint": gate.fingerprint() if gate is not None else None,
                        "files": {p.name: sha256_file(p) for p in sorted(Path(gdir).iterdir())
                                  if p.is_file()} if Path(gdir).is_dir() else {}}
    cal = {c["id"]: c["calibration"] for c in vision if c.get("calibration")}
    if cal:
        spec["calibration"] = cal
    cw = {c["id"]: {"cache": sha256_file(resolve(c["crosswalk"]["cache"])),
                    **({"review": sha256_file(resolve(c["crosswalk"]["review"]))}
                       if c["crosswalk"].get("review") else {})}
          for c in comps if c.get("kind") == "published" and c.get("crosswalk")}
    if cw:
        spec["crosswalk"] = cw
    for key in ("attrs", "deadly_taxa"):
        block = config.get(key) or {}
        f = block.get("file")
        if f and not str(f).startswith("TBD"):
            p = resolve(f)
            spec[key] = sha256_file(p) if p.is_file() else None
    from . import twins as twins_mod
    if config.get("twins") and any(twins_mod.is_grouped(c["method"]) for c in vision):
        spec["twins"] = {}
        for which, entry in sorted(config["twins"].items()):
            f = entry.get("file") if isinstance(entry, dict) else None
            if f and not str(f).startswith("TBD"):
                p = resolve(f)
                spec["twins"][which] = sha256_file(p) if p.is_file() else None
    spec["answer_key"] = config.get("answer_key")
    akt = config.get("answer_key_taxonomy") or {}
    if akt:
        spec["answer_key_taxonomy"] = {
            k: sha256_file(resolve(akt[k]["file"]))
            for k in ("snapshot", "successors", "label_vocabulary") if akt.get(k)}
        from .paper_score import extras_files
        files = extras_files(akt)
        if files:
            spec["answer_key_taxonomy"]["extras"] = [sha256_file(resolve(x["file"]))
                                                     for x in files]
    spec["code_commit"] = config.get("code_commit")
    return spec


def _diff(want, got, path: str, out: list[str]) -> None:
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(want) | set(got)):
            if k not in want:
                out.append(f"{path}.{k}: not in the freeze block")
            elif k not in got:
                out.append(f"{path}.{k}: frozen but not found now")
            else:
                _diff(want[k], got[k], f"{path}.{k}", out)
    elif want != got:
        out.append(f"{path}: frozen {str(want)[:16]}, now {str(got)[:16]}")


def check(block: dict | None, actual: dict) -> list[str]:
    """The parts where the config's `freeze` block and the predictor read now differ
    (empty: the same predictor). A part the block leaves out counts as a difference."""
    if not block:
        return ["no freeze block in the config"]
    out: list[str] = []
    for k in sorted(set(block) | set(actual)):
        if k.startswith("_"):
            continue
        if k not in block:
            out.append(f"{k}: not in the freeze block")
        elif k not in actual:
            out.append(f"{k}: frozen but not part of this config's predictor")
        else:
            _diff(block[k], actual[k], k, out)
    return out
