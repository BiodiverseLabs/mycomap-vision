"""A small trained set head on the frozen model (experiment, exp/observation-sets, Step 2).

Gated attention MIL pooling (Ilse et al. 2018) from an observation's photo vectors to
one observation vector:

    a_i = w . (tanh(V x_i) * sigmoid(U x_i))       one weight per photo
    p   = sum_i softmax(a)_i x_i                    attention-weighted mean
    z   = normalise(p + W p)                        W starts at zero

At the start (w = 0, W = 0) it is the plain mean of the photos, so training can only
move it away from that baseline (reported as `set-mean`). Trained with a proxy loss at
observation level (normalised softmax over one proxy per reference group, proxies
started at the group's mean photo), with random photo dropout so 1-photo records are
seen in training. Answers are NOT the proxies (a closed classifier): every reference
observation is embedded by the head, a record is compared with them, and species score
by their best observation or the mean of their two best, so a species with 1-4
reference records still works.

Training data: the current reference records only, split by time: records validated
up to `TRAIN_UNTIL` (the fine-tune's own training cut, 2026-09-07) train; newer ones
are the validation slice for early stopping (the fine-tune never saw them either, so
they look like new records). Every id of the held-out benchmark is removed before
anything is trained (`training_split` refuses otherwise), dev is used only for the
comparison afterwards.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

from .obsets import Reference

TRAIN_UNTIL = "2026-09-07"


class HeldOutInTraining(RuntimeError):
    """A held-out benchmark record reached the head's training or validation set."""


@dataclass
class Split:
    train: np.ndarray        # record numbers (Reference order)
    val: np.ndarray


def training_split(ref: Reference, held_out: set[str], until: str = TRAIN_UNTIL) -> Split:
    """Train on records validated up to `until`, validate on the newer ones. Records of
    the held-out benchmark are dropped, and a split holding one raises."""
    keep = ~np.isin(ref.rec_obs, np.array(sorted(held_out), dtype=ref.rec_obs.dtype))
    dated = ref.rec_validated != ""
    old = dated & (ref.rec_validated <= until)
    split = Split(np.flatnonzero(keep & (old | ~dated)), np.flatnonzero(keep & dated & ~old))
    check_no_held_out(ref, split, held_out)
    return split


def check_no_held_out(ref: Reference, split: Split, held_out: set[str]) -> None:
    for part in (split.train, split.val):
        leaked = set(ref.rec_obs[part].tolist()) & held_out
        if leaked:
            raise HeldOutInTraining(f"{len(leaked):,} held-out records in the head's data "
                                    f"(first: {sorted(leaked)[:3]})")


def _nn():
    import torch
    return torch, torch.nn


def make_head(dim: int, hidden: int = 256, project: bool = True):
    """project=False: attention pooling only (W stays zero and is not trained)."""
    torch, nn = _nn()

    class GatedAttentionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.V = nn.Linear(dim, hidden)
            self.U = nn.Linear(dim, hidden)
            self.w = nn.Linear(hidden, 1)
            self.W = nn.Linear(dim, dim)
            for layer in (self.w, self.W):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            if not project:
                self.W.requires_grad_(False)

        def attention(self, x, mask):
            """x (B, n, d), mask (B, n) True for a real photo -> (B, n) weights summing to 1."""
            a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
            return torch.softmax(a.masked_fill(~mask, float("-inf")), dim=1)

        def forward(self, x, mask):
            alpha = self.attention(x, mask)
            p = (alpha.unsqueeze(-1) * x).sum(dim=1)
            return torch.nn.functional.normalize(p + self.W(p), dim=-1), alpha

    return GatedAttentionHead()


@dataclass
class TrainConfig:
    epochs: int = 30
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.05
    max_photos: int = 12            # photos sampled per record per step
    keep: float = 0.6               # photo dropout: each photo kept with this chance (>= 1 kept)
    single: float = 0.25            # share of records cut to one photo
    patience: int = 4
    seed: int = 0
    project: bool = True            # False: attention pooling only, no learned projection


class Batcher:
    """Padded photo batches for records, from the reference vectors on the device."""

    def __init__(self, ref: Reference, device: str):
        torch, _ = _nn()
        self.torch = torch
        self.vecs = torch.from_numpy(np.asarray(ref.vectors, dtype=np.float16)).to(device)
        counts = np.bincount(ref.col_rec, minlength=ref.records)
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.counts = counts.astype(np.int64)
        self.device = device

    def batch(self, recs: np.ndarray, rng: np.random.Generator | None, cfg: TrainConfig):
        """(x (B, n, d) float32, mask (B, n)) for records `recs`; with `rng`, photos are
        dropped at random (training), else every photo (up to max_photos) is used."""
        cols = []
        for r in recs.tolist():
            c = np.arange(self.starts[r], self.starts[r] + self.counts[r])
            if rng is not None:
                c = c[rng.random(len(c)) < cfg.keep] if len(c) > 1 else c
                if not len(c):
                    c = np.array([self.starts[r] + rng.integers(self.counts[r])])
                if len(c) > 1 and rng.random() < cfg.single:
                    c = c[rng.integers(len(c)):][:1]
                c = rng.permutation(c)[:cfg.max_photos]
            else:
                c = c[:cfg.max_photos]
            cols.append(c)
        return self.pad(cols)

    def pad(self, cols: list[np.ndarray]):
        torch = self.torch
        n = max(len(c) for c in cols)
        idx = np.zeros((len(cols), n), dtype=np.int64)
        mask = np.zeros((len(cols), n), dtype=bool)
        for i, c in enumerate(cols):
            idx[i, :len(c)] = c
            mask[i, :len(c)] = True
        idx_t = torch.from_numpy(idx).to(self.device)
        return self.vecs[idx_t].float(), torch.from_numpy(mask).to(self.device)


def embed_records(head, batcher: Batcher, recs: np.ndarray, cfg: TrainConfig,
                  chunk: int = 4096):
    """(len(recs), d) head vectors of reference records, all photos."""
    torch = batcher.torch
    out = []
    with torch.no_grad():
        for s in range(0, len(recs), chunk):
            x, m = batcher.batch(recs[s:s + chunk], None, cfg)
            out.append(head(x, m)[0])
    return torch.cat(out) if out else torch.zeros((0, batcher.vecs.shape[1]),
                                                  device=batcher.device)


def embed_query(head, query: np.ndarray, device: str):
    """One record's head vector and its photos' attention weights."""
    torch, _ = _nn()
    x = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(device)[None]
    m = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        z, alpha = head(x, m)
    return z[0], alpha[0].cpu().numpy()


def val_accuracy(head, batcher: Batcher, ref: Reference, split: Split, cfg: TrainConfig,
                 n: int = 3000, seed: int = 1) -> float:
    """Group top-1 of validation records against the training records' head vectors (best
    observation per group), on up to `n` validation records whose group has training
    records."""
    torch = batcher.torch
    in_train = np.zeros(len(ref.index.species), dtype=bool)
    in_train[ref.rec_unit[split.train]] = True
    val = split.val[in_train[ref.rec_unit[split.val]]]
    if len(val) > n:
        val = np.random.default_rng(seed).choice(val, n, replace=False)
    head.eval()
    ref_z = embed_records(head, batcher, split.train, cfg)
    q_z = embed_records(head, batcher, val, cfg)
    units = torch.from_numpy(ref.rec_unit[split.train]).to(batcher.device)
    right = 0
    for s in range(0, len(val), 256):
        sims = q_z[s:s + 256] @ ref_z.T
        best = torch.full((sims.shape[0], len(ref.index.species)), float("-inf"),
                          device=batcher.device)
        best.scatter_reduce_(1, units.expand(sims.shape[0], -1), sims, "amax")
        pred = best.argmax(dim=1).cpu().numpy()
        right += int((pred == ref.rec_unit[val[s:s + 256]]).sum())
    head.train()
    return right / max(len(val), 1)


def train_head(ref: Reference, split: Split, cfg: TrainConfig | None = None,
               device: str | None = None, log=print) -> tuple[object, dict]:
    """Train the head on split.train; keep the epoch with the best validation accuracy."""
    torch, _ = _nn()
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    batcher = Batcher(ref, device)
    dim = batcher.vecs.shape[1]
    head = make_head(dim, project=cfg.project).to(device)
    n_units = len(ref.index.species)
    # Proxies start at each group's mean training photo (zero for groups only in val).
    units = ref.rec_unit[split.train]
    proxy0 = torch.zeros((n_units, dim), device=device)
    cols = np.flatnonzero(np.isin(ref.col_rec, split.train))
    for s in range(0, len(cols), 65536):
        c = cols[s:s + 65536]
        proxy0.index_add_(0, torch.from_numpy(ref.rec_unit[ref.col_rec[c]]).to(device),
                          batcher.vecs[torch.from_numpy(c).to(device)].float())
    proxies = torch.nn.Parameter(torch.nn.functional.normalize(proxy0, dim=-1))
    opt = torch.optim.AdamW([*(p for p in head.parameters() if p.requires_grad), proxies], lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    y_all = torch.from_numpy(units).to(device)
    history = []
    best = (val_accuracy(head, batcher, ref, split, cfg), -1, copy.deepcopy(head.state_dict()))
    log(f"  epoch 0 (the plain mean): val group top-1 {best[0]:.4f}")
    history.append({"epoch": 0, "val_top1": best[0]})
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(split.train))
        total = 0.0
        for s in range(0, len(order), cfg.batch):
            pick = order[s:s + cfg.batch]
            x, m = batcher.batch(split.train[pick], rng, cfg)
            z, _ = head(x, m)
            logits = z @ torch.nn.functional.normalize(proxies, dim=-1).T / cfg.temperature
            loss = torch.nn.functional.cross_entropy(logits, y_all[torch.from_numpy(pick).to(device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(pick)
        acc = val_accuracy(head, batcher, ref, split, cfg)
        history.append({"epoch": epoch, "loss": total / len(order), "val_top1": acc,
                        "seconds": round(time.time() - t0, 1)})
        log(f"  epoch {epoch}: loss {total / len(order):.4f}, val group top-1 {acc:.4f} "
            f"({time.time() - t0:.0f} s)")
        if acc > best[0]:
            best = (acc, epoch, copy.deepcopy(head.state_dict()))
        elif epoch - best[1] >= cfg.patience:
            break
    head.load_state_dict(best[2])
    head.eval()
    return head, {"best_epoch": best[1], "best_val_top1": best[0], "history": history,
                  "train_records": int(len(split.train)), "val_records": int(len(split.val)),
                  "train_until": TRAIN_UNTIL, "config": cfg.__dict__}


# --- Step 2 run ------------------------------------------------------------------------

def run_step2(conn, name: str, split_name: str = "dev", cfg: TrainConfig | None = None,
              out_dir=None, stem: str | None = None, weights=(0.25, 0.5, 0.75),
              save_head=None, exclude: set[str] | None = None, log=print) -> dict:
    """Train the head (reference records only, time split), then answer `split_name` of
    the benchmark with it, its untrained start (`set-mean`) and blends with nearest+mean,
    judged like Step 1. Returns the judged output plus training notes and every scored
    record's attention weights ({observation id: [weight per photo, in photo order]})."""
    import torch

    from . import heldout
    from .obsets import (BACKBONE, SetEngine, benchmark_queries, blend, judge_all,
                         load_reference, model_key, ranked, stored_answers, write_report)
    t0 = time.time()
    cfg = cfg or TrainConfig()
    ref = load_reference(conn, BACKBONE, exclude)
    held_out = set(heldout.benchmark_ids(conn, name))
    split = training_split(ref, held_out)
    log(f"  head data: {len(split.train):,} train records (validated <= {TRAIN_UNTIL}), "
        f"{len(split.val):,} validation; {len(held_out):,} benchmark ids kept out")
    queries = benchmark_queries(conn, name, split_name, ref)
    leaked = {r.observation_id for r, _ in queries} & set(ref.rec_obs[split.train].tolist())
    if leaked:
        raise HeldOutInTraining(f"{len(leaked)} {split_name} records are training records")
    t_train = time.time()
    head, notes = train_head(ref, split, cfg, log=log)
    notes["train_seconds"] = round(time.time() - t_train, 1)
    if save_head is not None:
        torch.save(head.state_dict(), save_head)
    notes["time_slice"] = head_time_slice_check(ref, split, head, cfg)
    log(f"  time slice: {notes['time_slice']}")
    engine = SetEngine(ref)
    batcher = Batcher(ref, engine.device)
    every = np.arange(ref.records)
    heads = {"set-head": head, "set-mean": make_head(batcher.vecs.shape[1]).to(engine.device).eval()}
    ref_z = {k: embed_records(h, batcher, every, cfg) for k, h in heads.items()}
    results: dict[tuple, dict] = {}
    attention: dict[str, list[float]] = {}

    def put(method, oid, scores):
        results.setdefault(model_key(method), {})[oid] = ranked(scores, ref.index)
    for i, (rec, q) in enumerate(queries):
        oid = rec.observation_id
        nm = engine.scores(q)["nearest+mean"]
        put("nearest+mean", oid, nm)
        for k, h in heads.items():
            z, alpha = embed_query(h, q, engine.device)
            sims = ref_z[k] @ z
            for how in ("max", "top2"):
                s = engine.per_unit(sims, how).cpu().numpy().astype(np.float32)
                put(f"{k}-{how}", oid, s)
                if k == "set-head":
                    for w in weights:
                        put(f"blend:{k}-{how}@{w}", oid, blend(nm, s, w))
            if k == "set-head":
                attention[oid] = [round(float(a), 5) for a in alpha]
        if (i + 1) % 500 == 0:
            log(f"  {i + 1:,} records ({time.time() - t0:.0f} s)")
    ids = {r.observation_id for r, _ in queries}
    if not exclude:                 # the stored answers were made against the full reference
        for (b, method, place, size), by in stored_answers(conn, name, ids).items():
            results[(b, method, place, size)] = by
    out = judge_all(conn, name, [r for r, _ in queries], results,
                    baseline=model_key("nearest+mean"), ref=ref)
    out["reference"] = {"records": ref.records, "photos": int(len(ref.photo_ids)),
                        "excluded_ids": len(exclude or ())}
    out["training"] = notes
    out["attention"] = attention
    out["seconds"] = round(time.time() - t0, 1)
    if out_dir is not None:
        stem = stem or f"obsets-step2-{split_name}"
        out["file"] = str(write_report(out, out_dir, stem,
                                       {"training": notes, "seconds": out["seconds"]}))
    return out


def head_time_slice_check(ref: Reference, split: Split, head, cfg: TrainConfig,
                          weights=(0.5, 0.75, 0.9)) -> dict:
    """The head's blend with nearest+mean on the validation slice (newer records) against the
    training records, by Vision's labels (strict). Only partly independent of the head:
    its epoch was chosen on this slice; the blend weights were not."""
    from .obsets import SetEngine, blend, subset_reference
    from .heldout_report import mcnemar
    base = subset_reference(ref, split.train)
    engine = SetEngine(base, device=next(head.parameters()).device.type)
    batcher = Batcher(base, engine.device)
    base_z = embed_records(head, batcher, np.arange(base.records), cfg)
    qbatch = Batcher(ref, engine.device)
    unit_of = {s: i for i, s in enumerate(base.index.species)}
    sp_label = base.index.label_of["species"]
    rows_of = np.split(np.arange(len(ref.col_rec)), np.flatnonzero(np.diff(ref.col_rec)) + 1)
    hits: dict[str, list] = {}
    for r in split.val.tolist():
        truth = ref.index.species[ref.rec_unit[r]]
        if ref.index.label_of["species"][ref.rec_unit[r]] < 0:
            continue
        nm = engine.scores(ref.vectors[rows_of[r]])["nearest+mean"]
        z = embed_records(head, qbatch, np.array([r]), cfg)[0]
        h = engine.per_unit(base_z @ z, "top2").cpu().numpy()
        s = {"nearest+mean": nm, "set-head-top2": h}
        for w in weights:
            s[f"blend:set-head-top2@{w}"] = blend(nm, h, w)
        for m, v in s.items():
            v = np.where(sp_label >= 0, v, -np.inf)
            hits.setdefault(m, []).append(unit_of.get(truth, -1) == int(np.argmax(v)))
    nm_h = np.array(hits["nearest+mean"])
    out = {"records": len(nm_h), "methods": {}}
    for m, h in hits.items():
        h = np.array(h)
        fixed, broken = int((h & ~nm_h).sum()), int((~h & nm_h).sum())
        out["methods"][m] = {"species_top1": round(float(h.mean()), 4), "fixed": fixed,
                             "broken": broken, "mcnemar_p": round(mcnemar(fixed, broken), 6)}
    return out
