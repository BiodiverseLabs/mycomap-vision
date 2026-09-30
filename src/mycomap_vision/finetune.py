"""Fine-tune a backbone on DNA-verified labels, so its sense of "looks alike" follows DNA.

Only the last few transformer blocks (and what comes after them) are trained; the
rest stays frozen, so the model shifts towards our labels without forgetting the
fungi it knows and we don't. Training uses the reference records of the
comparison it will be judged in, never the test weeks: the date it was trained
through is saved, and evaluate.compare refuses to score it on records up to that
date.

The loss is a cosine classifier at species, genus and family, each initialised
from the frozen model's class averages (so training starts where the species-mean
method already is). The output vector stays the model's embedding, so the
fine-tuned model is used like any other backbone: embed, then nearest specimen,
species average or a trained head on top. Photos are drawn with probability
1 / (species photos) ** sampling_power: 0 is the natural mix, 1 fully balanced;
the default 0.5 keeps common species from drowning rare ones without pretending
they're rare (balanced training lost badly on the sample, see methods.LinearHead).

A fine-tuned model is registered as `<base>-ft-<tag>`: data/models/<name>.json
(what it was trained on) and <name>.pt (only the trained weights; the rest are
the base model's). models.load_backbone finds it by that name.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, evaluate

RANKS = ("species", "genus", "family")


@dataclass
class FinetuneConfig:
    epochs: float = 2.0
    blocks: int = 4                 # last transformer blocks trained; earlier ones frozen
    batch_size: int = 64
    lr_backbone: float = 1e-5
    lr_heads: float = 1e-3
    weight_decay: float = 1e-4
    scale: float = 20.0             # cosine-classifier sharpness
    sampling_power: float = 0.5     # 0 natural mix, 1 balanced
    rank_weights: tuple = (1.0, 0.5, 0.25)   # species, genus, family
    warmup: float = 0.05            # share of steps with a rising learning rate
    workers: int = 6
    seed: int = 0
    max_steps: int | None = None    # cap, for smoke tests


@dataclass
class TrainSet:
    items: list[tuple[int, str]]            # (photo_id, path in the store)
    labels: np.ndarray                      # (n, 3): species, genus, family index; -1 unknown
    names: dict[str, list[str]]
    base_rows: np.ndarray                   # row of each item in the base embeddings
    trained_through: str                    # the comparison cutoff: newest date trained on
    records: int
    vectors: np.ndarray = field(repr=False, default=None)   # the base model's embeddings
    # Per item: its record's group (a species, or a one-word name, which has no species
    # label but still trains at genus and family). Photos are drawn by group.
    units: np.ndarray = field(repr=False, default=None)


def build_trainset(conn: sqlite3.Connection, base: str, store_location: str, size: str,
                   test_days: int = 28, embeddings_root=None) -> TrainSet:
    """Every photo of the reference records a comparison of `base` would use, with its
    labels. The base must be embedded: its vectors seed the classifier."""
    shared = evaluate.shared_records(conn, [base], test_days, embeddings_root=embeddings_root)
    ids, vecs = shared.loaded[base]
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    paths = dict(conn.execute("select photo_id, path from photo_copies where store = ? and "
                              "size = ?", (store_location, size)).fetchall())
    # A one-word name ("Russula") has no species label: -1 there, trained at genus and
    # family only, and it adds no species class.
    names = {rank: sorted({evaluate.truth(r, rank) for r in shared.ref} - {""}) for rank in RANKS}
    pos = {rank: {n: i for i, n in enumerate(names[rank])} for rank in RANKS}
    unit_pos = {u: i for i, u in enumerate(sorted({r.unit for r in shared.ref}))}
    items, labels, rows, units = [], [], [], []
    for rec in shared.ref:
        lab = [pos[rank].get(evaluate.truth(rec, rank), -1) for rank in RANKS]
        for pid in rec.photo_rows:              # photo ids here (shared_records)
            if pid in paths:
                items.append((pid, paths[pid]))
                labels.append(lab)
                rows.append(row_of[pid])
                units.append(unit_pos[rec.unit])
    if not items:
        raise ValueError(f"no {size} photos of reference records in {store_location}")
    return TrainSet(items, np.asarray(labels, dtype=np.int64).reshape(-1, 3), names,
                    np.asarray(rows, dtype=np.int64), shared.cutoff, len(shared.ref), vecs,
                    np.asarray(units, dtype=np.int64))


def class_prototypes(vectors: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Unit-length average vector per class (zeros for a class with no photos)."""
    v = vectors.astype(np.float32)
    keep = labels >= 0
    sums = np.zeros((n_classes, v.shape[1]), dtype=np.float32)
    np.add.at(sums, labels[keep], v[keep])
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return np.where(norms > 0, sums / np.maximum(norms, 1e-12), 0).astype(np.float32)


def sampling_weights(species: np.ndarray, power: float) -> np.ndarray:
    """Per photo: 1 / (photos of its species) ** power, summing to 1."""
    counts = np.bincount(species[species >= 0], minlength=max(int(species.max()) + 1, 1))
    w = np.where(species >= 0, 1.0 / np.maximum(counts[np.clip(species, 0, None)], 1) ** power,
                 0.0)
    return w / w.sum()


_BLOCK = re.compile(r"(?:^|\.)(?:resblocks|blocks)\.(\d+)\.")
# Parameters that come after the last block: final norms, projection, pooling, head.
_AFTER_BLOCKS = {"ln_post", "proj", "norm", "fc_norm", "head", "attn_pool"}


def block_count(names: list[str]) -> int:
    found = {int(m.group(1)) for n in names if (m := _BLOCK.search(n))}
    return max(found) + 1 if found else 0


def is_trainable(name: str, n_blocks: int, tail: int) -> bool:
    """True for the last `tail` blocks and everything after them; embeddings, the
    patch projection and earlier blocks stay frozen."""
    m = _BLOCK.search(name)
    if m:
        return int(m.group(1)) >= n_blocks - tail
    top = name.removeprefix("trunk.").split(".")[0]
    return top in _AFTER_BLOCKS


# --- the finetune registry (read by evaluate's leak guard) ----------------------

def register(conn: sqlite3.Connection, meta: dict) -> None:
    conn.executescript(evaluate.FINETUNE_SCHEMA)
    with conn:
        conn.execute("insert or replace into finetunes (name, base, trained_through, test_days, "
                     "meta_json, created_at) values (?, ?, ?, ?, ?, ?)",
                     (meta["name"], meta["base"], meta["trained_through"], meta["test_days"],
                      json.dumps(meta), meta["created_at"]))


def models_dir() -> Path:
    return config.DATA_DIR / "models"


def read_meta(name: str, root: Path | None = None) -> dict:
    path = (root or models_dir()) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no fine-tuned model {name!r} in {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


# --- training (needs torch) -----------------------------------------------------

class PhotoDataset:
    """(image tensor, labels) per training photo, read from a store by location, so it
    pickles into DataLoader workers (each opens its own S3 client)."""

    def __init__(self, location: str, items, labels, transform, draft: int | None = None):
        self.location, self.items, self.labels = location, items, labels
        self.transform, self.draft = transform, draft
        self._store = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image, ImageOps

        from .storage import open_store
        if self._store is None:
            self._store = open_store(self.location, Path(self.location))
        try:
            img = Image.open(io.BytesIO(self._store.get(self.items[i][1])))
            if self.draft and img.format == "JPEG":
                img.draft("RGB", (self.draft, self.draft))     # decode big JPEGs smaller, fast
            img = ImageOps.exif_transpose(img).convert("RGB")
            return self.transform(img), self.labels[i]
        except Exception:
            return None


def _collate(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return (torch.stack([b[0] for b in batch]),
            torch.as_tensor(np.stack([b[1] for b in batch])))


def train_transform(eval_transform):
    """Augmentation matching the model's own input size and normalisation. No hue or
    saturation changes: colour is how many fungi are told apart."""
    from torchvision import transforms as T
    size, normalize = 224, None
    for t in getattr(eval_transform, "transforms", []):
        if isinstance(t, T.CenterCrop):
            size = t.size[0] if isinstance(t.size, (tuple, list)) else t.size
        if isinstance(t, T.Normalize):
            normalize = t
    if normalize is None:
        raise ValueError("the model's transform has no Normalize step")
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(normalize.mean, normalize.std),
    ]), size


def finetune(conn: sqlite3.Connection, base: str, store, size: str, name: str,
             cfg: FinetuneConfig | None = None, test_days: int = 28, embeddings_root=None,
             out_dir: Path | None = None, loader=None, log=print, should_stop=None) -> dict:
    """Train, save <name>.pt and <name>.json in out_dir, register; returns the metadata.
    `should_stop()` is asked before each step; when it says stop, screening.Stopped is
    raised and nothing is saved (a half-trained model is not the model asked for)."""
    import torch
    import torch.nn.functional as F

    from . import models
    from .screening import Stopped
    cfg = cfg or FinetuneConfig()
    out_dir = out_dir or models_dir()
    started = time.monotonic()
    ts = build_trainset(conn, base, store.location, size, test_days, embeddings_root)
    log(f"[{name}] {len(ts.items):,} photos of {ts.records:,} records, "
        f"{len(ts.names['species']):,} species, trained through {ts.trained_through}")
    backbone = (loader or models.load_backbone)(base)
    module = backbone.image_module()
    dev = backbone.device
    torch.manual_seed(cfg.seed)
    param_names = [n for n, _ in module.named_parameters()]
    n_blocks = block_count(param_names)
    if cfg.blocks > n_blocks:
        raise ValueError(f"{base} has {n_blocks} blocks; can't train the last {cfg.blocks}")
    tail = []
    for n, p in module.named_parameters():
        p.requires_grad = is_trainable(n, n_blocks, cfg.blocks)
        if p.requires_grad:
            tail.append((n, p))
    module.eval()          # ViTs have no batch norm; keep any dropout off
    base_vecs = ts.vectors[ts.base_rows]
    heads = torch.nn.ParameterList([
        torch.nn.Parameter(torch.from_numpy(class_prototypes(base_vecs, ts.labels[:, k],
                                                             len(ts.names[rank]))).to(dev))
        for k, rank in enumerate(RANKS)])
    opt = torch.optim.AdamW([{"params": [p for _, p in tail], "lr": cfg.lr_backbone},
                             {"params": list(heads), "lr": cfg.lr_heads}],
                            weight_decay=cfg.weight_decay)
    per_epoch = len(ts.items)
    steps = max(1, int(cfg.epochs * per_epoch) // cfg.batch_size)
    if cfg.max_steps:
        steps = min(steps, cfg.max_steps)
    warm = max(1, int(cfg.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s / steps, 1))))
    transform, crop = train_transform(backbone.transform)
    # By group, not species label: a one-word record has none and would never be drawn.
    weights = sampling_weights(ts.units if ts.units is not None else ts.labels[:, 0],
                               cfg.sampling_power)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=steps * cfg.batch_size, replacement=True,
        generator=torch.Generator().manual_seed(cfg.seed))
    data = torch.utils.data.DataLoader(
        PhotoDataset(store.location, ts.items, ts.labels, transform, draft=2 * crop),
        batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.workers,
        collate_fn=_collate, drop_last=True, persistent_workers=cfg.workers > 0,
        pin_memory=dev == "cuda")
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    step, seen, t0, losses = 0, 0, time.monotonic(), []
    for batch in data:
        if should_stop is not None and should_stop():
            raise Stopped(f"stopped at step {step:,} of {steps:,} (time limit); not saved")
        if batch is None:
            continue
        x, y = batch[0].to(dev, non_blocking=True), batch[1].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            f = backbone.forward_features(x)
        f = F.normalize(f.float(), dim=1)
        loss = sum(w * F.cross_entropy(cfg.scale * f @ F.normalize(h, dim=1).T, y[:, k],
                                       ignore_index=-1)
                   for k, (w, h) in enumerate(zip(cfg.rank_weights, heads)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in tail] + list(heads), 1.0)
        opt.step()
        sched.step()
        step += 1
        seen += len(x)
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == steps:
            log(f"[{name}] step {step:,}/{steps:,}, loss {np.mean(losses[-100:]):.3f}, "
                f"{seen / (time.monotonic() - t0):.0f} photos/s")
        if step >= steps:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in tail}, out_dir / f"{name}.pt")
    meta = {"name": name, "base": base,
            "base_spec": getattr(backbone, "spec", None) or models.resolve_spec(base),
            "weights": f"{name}.pt", "trained_through": ts.trained_through,
            "test_days": test_days, "records": ts.records, "photos": len(ts.items),
            "species": len(ts.names["species"]), "steps": step, "photos_seen": seen,
            "final_loss": round(float(np.mean(losses[-100:])), 4) if losses else None,
            "blocks_trained": cfg.blocks, "of_blocks": n_blocks, "size": size,
            "config": asdict(cfg), "minutes": round((time.monotonic() - started) / 60, 1),
            "code_version": config.code_version(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    register(conn, meta)
    log(f"[{name}] saved to {out_dir}, {meta['minutes']} min")
    return meta
