"""Where Vision looked (Steve, 2026-10-26): on an answer, each photo's map of where its
evidence for the answer's name was, made when someone asks, and kept beside the run.

**What a map is.** Vision names a find by retrieval, so a photo's evidence for the name X is
its served score for X (the record-balanced nearest + mean, photos only):

    T_X(q) = w * mean(k best records of X, each by its best photo cosine) + (1 - w) * cos(q, m_X)

with w and k the served method's own (0.6 and 2) and m_X X's species average. The map is the
gradient-weighted attention rollout of T_X (Chefer et al. 2021: R += mean over heads of
(dT/dA * A)+ R, every block), read on the 24 x 24 patches of the centre square the model sees.
The model keeps a few "register" patches with very large token norms (Darcet et al. 2023);
attention maps light them up as specks that hiding them shows don't matter, so they are
replaced by their neighbours' median, and the map is lightly smoothed. The research look (v1
val, 314 photos; research repo experiments/2026-10-26-where-vision-looked.md) chose this
method: the most faithful of the affordable ones (hiding its brightest tenth costs about six
times what a random tenth does), ~2.5 forward passes a photo (~2.7 s on the box's CPU).

**What a match is.** Beside each photo, the DNA-verified specimen photo the run link already
shows for it: patches of the two photos whose last-block tokens are each other's nearest
(mutual), among the photo's strongest-evidence tenth, kept apart and away from the edges.
Similar patterns, not proof: they can match on colour or light as well as on the fungus.

**Cost and the queue.** One worker makes maps, one photo at a time, and only while the model
queue has nothing waiting or running; it holds the model lock for that one photo, so an
identification that arrives meanwhile waits for at most one photo's map. Reading the photos
(from the run's copies, or the observation's source through the polite fetcher) happens
outside the lock. A request is refused (Busy) while too many photos are already waiting.

**People first** (Excluded photos preload, 2026-10-27): a batch job (Job.batch, the preload's
one photo at a time) never counts toward that limit and always waits behind every person's
ask; it is never stopped mid-map, so a person's ask waits at most for the one batch photo
being made. Each map's time is logged ("[looked] <id>: map 4.1 s, read 0.3 s").
"""

from __future__ import annotations

import io
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

VERSION = 1
LOOKED = "looked.json"                      # kept beside the run (runs.py)
REGISTER_FACTOR = 2.0                       # token norm over this x the photo's median
MATCH_PAIRS = 3
MATCH_SHARE = 0.1                           # the photo's strongest share of patches matched
MAP_LEVELS = 255


class Busy(RuntimeError):
    def __init__(self, wait_seconds: float):
        super().__init__(f"about {wait_seconds:.0f} s of maps are ahead")
        self.wait_seconds = wait_seconds


class NoMap(ValueError):
    """This run's answer can't be mapped (no species answer, a model no longer served)."""


class NeverMap(NoMap):
    """No map can be made from what is served now, however often it is tried: the name is not
    in the reference set, or no other DNA-verified record of it is there to compare with. Only
    a new release (a new reference set) or the record's new name can change that."""


# --- the model ---------------------------------------------------------------------------

def vit_of(loaded) -> tuple[object, Callable] | None:
    """The open_clip ViT image tower and its preprocessing inside a loaded backbone
    (models.FinetunedBackbone keeps the trained tower as base.model; models.OpenClipBackbone
    as model), or None (an ONNX tower, a timm model, a test stand-in)."""
    for vit in (getattr(getattr(loaded, "base", None), "model", None),
                getattr(loaded, "model", None)):
        if (vit is not None and hasattr(vit, "conv1") and hasattr(vit, "transformer")
                and hasattr(vit, "positional_embedding")):
            transform = getattr(loaded, "transform", None) or getattr(
                getattr(loaded, "base", None), "transform", None)
            if transform is not None:
                return vit, transform
    return None


class Tower:
    """A ViT tower run block by block, so its attention and tokens can be read: the same
    vector as the tower's own forward (tests), with attention kept for the gradient."""

    def __init__(self, vit, transform):
        import torch
        self.torch, self.vit, self.transform = torch, vit, transform
        if not getattr(vit.transformer, "batch_first", True):
            raise ValueError("the tower's transformer is not batch-first")
        self.blocks = list(vit.transformer.resblocks)
        g = vit.grid_size
        self.grid = int(g[0] if isinstance(g, (tuple, list)) else g)
        size = vit.image_size
        self.side = int(size[0] if isinstance(size, (tuple, list)) else size)
        # v1 (24 blocks): registers read after block 20, matches on the last block's tokens.
        self.register_layer = max(0, len(self.blocks) - 4)
        self.match_layer = len(self.blocks) - 1
        self.device = next(vit.parameters()).device

    def prepare(self, body: bytes) -> tuple[object, dict]:
        """(input tensor, the centre square the model sees as fractions of the photo)."""
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
        return self.transform(img), crop_box(*img.size)

    def run(self, x, attention: bool, layers: tuple[int, ...] = ()):
        """(unit vectors, per-block attention probabilities or [], {layer: tokens})."""
        torch, vit = self.torch, self.vit
        import torch.nn.functional as F
        h = vit._embeds(x.to(self.device))
        atts, taps = [], {}
        for i, blk in enumerate(self.blocks):
            h = self._block(blk, h, atts if attention else None)
            if i in layers:
                taps[i] = h
        pooled, _ = vit._pool(h)
        if vit.proj is not None:
            pooled = pooled @ vit.proj
        return F.normalize(pooled.float(), dim=-1), atts, taps

    def _block(self, blk, x, atts):
        import torch.nn.functional as F
        a = blk.attn
        y = blk.ln_1(x)
        b, n, d = y.shape
        heads = a.num_heads
        qkv = F.linear(y, a.in_proj_weight, a.in_proj_bias).reshape(b, n, 3, heads, d // heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if atts is None:
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            p = ((q @ k.transpose(-2, -1)) * (d // heads) ** -0.5).float().softmax(-1).to(v.dtype)
            atts.append(p)
            o = p @ v
        x = x + blk.ls_1(a.out_proj(o.transpose(1, 2).reshape(b, n, d)))
        return x + blk.ls_2(blk.mlp(blk.ln_2(x)))

    def tokens(self, x, layer: int):
        """(patches, dim) centred, unit tokens after block `layer`, for matching."""
        torch = self.torch
        with torch.inference_mode():
            _, _, taps = self.run(x[None], attention=False, layers=(layer,))
        t = taps[layer][0, 1:].float()
        t = t - t.mean(0, keepdim=True)
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def crop_box(width: int, height: int) -> dict:
    """The centre square a resize-then-centre-crop keeps, as fractions of the photo."""
    side = min(width, height)
    return {"x": (width - side) / 2 / width, "y": (height - side) / 2 / height,
            "w": side / width, "h": side / height}


# --- the name's score -----------------------------------------------------------------------

@dataclass
class Target:
    """T_X for one name: its reference photo vectors, which record each is of, its average."""
    name: str
    vectors: np.ndarray                     # (photos, dim) unit
    record: np.ndarray                      # (photos,) 0..records-1
    mean: np.ndarray                        # (dim,) unit
    weight: float = 0.6
    k: int = 2

    def score(self, q):
        """(batch,) the served per-photo score, differentiable in q (a torch tensor)."""
        import torch
        v = torch.as_tensor(self.vectors, dtype=q.dtype, device=q.device)
        rec = torch.as_tensor(self.record, dtype=torch.long, device=q.device)
        n_rec = int(self.record.max()) + 1
        s = q @ v.T
        best = torch.full((q.shape[0], n_rec), -2.0, dtype=s.dtype, device=q.device)
        best = best.scatter_reduce(1, rec.expand(q.shape[0], -1), s, "amax")
        near = best.topk(min(self.k, n_rec), dim=1).values.mean(1)
        m = torch.as_tensor(self.mean, dtype=q.dtype, device=q.device)
        return self.weight * near + (1 - self.weight) * (q @ m)


def target_for(ident, name: str, hidden: np.ndarray | None = None) -> Target:
    """The served index's T_X for `name` (identify.Identifier), or NoMap. `hidden`: reference
    columns left out, as if never there (Identifier.left_out: a reference record's own photo
    must not match its own record); X's average is then made again from the records left."""
    from .methods import rows_at
    index = ident.index
    try:
        u = index.species.index(name)
    except ValueError:
        raise NeverMap(f"{name} is not in the reference set this model answers from") from None
    lo = int(index.starts[u])
    hi = int(index.starts[u + 1]) if u + 1 < len(index.starts) else len(index.cols)
    cols = np.arange(lo, hi, dtype=np.int64)
    if index.rec_starts is None:
        record = np.arange(len(cols))
    else:
        bounds = index.rec_starts[(index.rec_starts >= lo) & (index.rec_starts < hi)]
        record = np.searchsorted(bounds, cols, side="right") - 1
    left_out = hidden is not None and bool(np.isin(cols, hidden).any())
    if left_out:
        keep = ~np.isin(cols, hidden)
        if not keep.any():
            raise NeverMap(f"no other DNA-verified record of {name} to compare with")
        cols = cols[keep]
        record = np.unique(record[keep], return_inverse=True)[1]
    vectors = np.asarray(ident.reference_vectors(cols), dtype=np.float32)
    scoring = getattr(ident.model, "base", ident.model)
    means = getattr(scoring, "means", None)
    if left_out:
        # Record-balanced, as the method's own average: each record's unit sum, then their mean.
        per = np.zeros((int(record.max()) + 1, vectors.shape[1]), dtype=np.float64)
        np.add.at(per, record, vectors)
        per /= np.maximum(np.linalg.norm(per, axis=1, keepdims=True), 1e-12)
        mean = per.mean(axis=0)
    elif means is not None:
        mean = rows_at(means.ref, [u])[0]
    else:
        mean = vectors.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
    return Target(name, vectors, record.astype(np.int64), mean.astype(np.float32),
                  float(getattr(scoring, "weight", 0.6)), int(getattr(scoring, "k", 2)))


# --- maps and matches -------------------------------------------------------------------------

def registers(tokens) -> np.ndarray:
    """(grid, grid) bool: the patches whose token norm is over REGISTER_FACTOR x the median."""
    n = tokens[0, 1:].float().norm(dim=-1).detach().cpu().numpy()
    g = int(round(len(n) ** 0.5))
    return (n > REGISTER_FACTOR * np.median(n)).reshape(g, g)


def clean(m: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """Each register patch gets the median of its non-register neighbours."""
    out = np.array(m, dtype=np.float64)
    for y, x in zip(*np.nonzero(reg)):
        ys, xs = slice(max(0, y - 1), y + 2), slice(max(0, x - 1), x + 2)
        near = m[ys, xs][~reg[ys, xs]]
        out[y, x] = float(np.median(near)) if near.size else float(np.median(m))
    return out


def smooth(m: np.ndarray) -> np.ndarray:
    """3 x 3 average, centre weight 4 (edges repeated)."""
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    g = m.shape[0]
    p = np.pad(m, 1, mode="edge")
    out = sum(k[dy, dx] * p[dy:dy + g, dx:dx + g] for dy in range(3) for dx in range(3))
    return out / k.sum()


def evidence(tower: Tower, x, target: Target):
    """(map, registers, last-block tokens) for one photo: the cleaned, smoothed
    gradient-weighted rollout of T_X over its patches."""
    torch = tower.torch
    with torch.enable_grad():
        xi = x[None].clone().requires_grad_(True)
        q, atts, taps = tower.run(xi, attention=True,
                                  layers=(tower.register_layer, tower.match_layer))
        grads = torch.autograd.grad(target.score(q).sum(), atts)
    with torch.no_grad():
        n = atts[0].shape[-1]
        rel = torch.eye(n, device=atts[0].device)
        for a, g in zip(atts, grads):
            rel = rel + (g[0].float() * a[0].float()).clamp(min=0).mean(0) @ rel
        m = rel[0, 1:].cpu().numpy().reshape(tower.grid, tower.grid)
        reg = registers(taps[tower.register_layer])
        t = taps[tower.match_layer][0, 1:].float()
        t = t - t.mean(0, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return smooth(clean(m, reg)), reg, t


def match_pairs(tq, tr, qmap: np.ndarray, skip_q: np.ndarray, skip_r: np.ndarray,
                k: int = MATCH_PAIRS) -> list[list[float]]:
    """Up to k pairs [qx, qy, rx, ry] (cell centres, fractions of each crop square): mutual
    nearest tokens among the photo's strongest MATCH_SHARE of patches, skipping registers,
    a border ring, and pairs too close to one already kept on either photo."""
    import torch
    g = qmap.shape[0]
    edge = max(0, round(g * 2 / 24))
    apart = max(1, round(g * 5 / 24))
    border = np.ones((g, g), bool)
    border[edge:g - edge, edge:g - edge] = False
    sq, sr = (skip_q | border).ravel(), (skip_r | border).ravel()
    s = (tq @ tr.T).clone()
    s[torch.from_numpy(sq).to(s.device)] = -2.0
    s[:, torch.from_numpy(sr).to(s.device)] = -2.0
    q_best, r_best = s.argmax(1).cpu().numpy(), s.argmax(0).cpu().numpy()
    strength = np.where(sq, -np.inf, qmap.ravel())
    top = np.argsort(-strength)[:max(1, int(MATCH_SHARE * g * g))]
    pairs = [(float(s[i, q_best[i]]), int(i), int(q_best[i])) for i in top
             if np.isfinite(strength[i]) and r_best[q_best[i]] == i and not sr[q_best[i]]]
    kept: list[tuple[int, int, int, int]] = []
    for _, i, j in sorted(pairs, reverse=True):
        qy, qx = divmod(i, g)
        ry, rx = divmod(j, g)
        if all(max(abs(qx - a), abs(qy - b)) >= apart and max(abs(rx - c), abs(ry - d)) >= apart
               for a, b, c, d in kept):
            kept.append((qx, qy, rx, ry))
        if len(kept) == k:
            break
    at = lambda v: round((v + 0.5) / g, 4)
    return [[at(a), at(b), at(c), at(d)] for a, b, c, d in kept]


def quantise(m: np.ndarray) -> list[int]:
    """The map as 0..MAP_LEVELS (its 99th percentile = the top), row by row."""
    v = np.clip(np.asarray(m, dtype=np.float64), 0, None)
    top = float(np.percentile(v, 99)) or float(v.max()) or 1.0
    return np.clip(np.round(v / top * MAP_LEVELS), 0, MAP_LEVELS).astype(int).ravel().tolist()


def photo_vector(tower: Tower, body: bytes) -> np.ndarray:
    """(dim,) the photo's unit vector from the same forward pass the map reads."""
    x, _ = tower.prepare(body)
    with tower.torch.inference_mode():
        q, _, _ = tower.run(x[None], attention=False)
    return q[0].cpu().numpy().astype(np.float32)


def photo_map(tower: Tower, body: bytes, target: Target, specimen: bytes | None) -> dict:
    """One photo's entry: its map and crop, and with a specimen photo the matched pairs and
    the specimen's own crop (pairs None without one)."""
    x, crop = tower.prepare(body)
    m, reg, tq = evidence(tower, x, target)
    out = {"crop": {k: round(v, 4) for k, v in crop.items()}, "grid": tower.grid,
           "map": quantise(m), "pairs": None}
    if specimen is not None:
        xr, crop_r = tower.prepare(specimen)
        out["specimen_crop"] = {k: round(v, 4) for k, v in crop_r.items()}
        tr = tower.tokens(xr, tower.match_layer)
        with tower.torch.no_grad():
            _, _, taps = tower.run(xr[None], attention=False, layers=(tower.register_layer,))
            reg_r = registers(taps[tower.register_layer])
        out["pairs"] = match_pairs(tq, tr, m, reg, reg_r)
    return out


# --- the worker -------------------------------------------------------------------------------

@dataclass
class Task:
    """One photo: `fetch` reads what it needs (outside the model lock), `make` maps it."""
    photo: int
    fetch: Callable[[], object]
    make: Callable[[object], dict]


@dataclass
class Job:
    rid: str
    tasks: list[Task]
    finish: Callable[[list[dict]], None]          # keeps the finished doc
    done: int = 0
    started: float = 0.0
    error: str | None = None
    finished: threading.Event = field(default_factory=threading.Event)
    batch: bool = False                           # the preload's: behind every person's ask
    on_error: Callable[[Exception], None] | None = None  # told (the exception) when it fails


class Worker:
    """One map at a time, behind the identifications (module docstring). `idle()` says the
    model queue has nothing waiting or running; `lock` is the model lock. With
    background=False a job runs in the caller's thread (tests)."""

    def __init__(self, idle: Callable[[], bool], lock, max_waiting_photos: int = 30,
                 seconds_per_photo: float = 3.0, poll: float = 0.2, background: bool = True,
                 failed_for: float = 600.0, clock=time.monotonic, sleep=time.sleep, note=print):
        self.idle, self.lock = idle, lock
        self.max_waiting_photos, self.seconds_per_photo = max_waiting_photos, seconds_per_photo
        self.poll, self.background, self.failed_for = poll, background, failed_for
        self.clock, self.sleep, self.note = clock, sleep, note
        self._cond = threading.Condition()
        self._queue: list[Job] = []
        self._current: Job | None = None
        self._failed: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        self._thread: threading.Thread | None = None
        self.photos_made = 0                        # for health and tests
        self.last_seconds: float | None = None      # the last map's time holding the lock

    def status(self, rid: str) -> dict | None:
        """{"status": "working", ...} while queued or making, {"status": "failed", ...} for a
        while after a failure, else None."""
        with self._cond:
            for pos, job in enumerate(([self._current] if self._current else []) + self._queue):
                if job.rid == rid:
                    return {"status": "working", "photos": len(job.tasks), "done": job.done,
                            "ahead": pos}
            hit = self._failed.get(rid)
            if hit and self.clock() - hit[0] < self.failed_for:
                return {"status": "failed", "detail": hit[1]}
        return None

    def waiting_photos(self) -> int:
        """Photos people asked for that are not made yet (batch photos don't count)."""
        with self._cond:
            left = sum(len(j.tasks) for j in self._queue if not j.batch)
            if self._current and not self._current.batch:
                left += len(self._current.tasks) - self._current.done
            return left

    def batch_pending(self) -> bool:
        """A batch job is queued or being made."""
        with self._cond:
            return any(j.batch for j in ([self._current] if self._current else []) + self._queue)

    def submit(self, job: Job) -> dict:
        """Queue the job (or join the one already queued for its run); Busy when too many
        photos are waiting. Without background, makes it now."""
        with self._cond:
            if any(j.rid == job.rid for j in ([self._current] if self._current else []) + self._queue):
                pass
            else:
                people = [j for j in self._queue if not j.batch]
                waiting = sum(len(j.tasks) for j in people) + (
                    len(self._current.tasks) - self._current.done
                    if self._current and not self._current.batch else 0)
                if (not job.batch and waiting
                        and waiting + len(job.tasks) > self.max_waiting_photos):
                    raise Busy(waiting * self.seconds_per_photo)
                self._failed.pop(job.rid, None)
                # A person's ask goes ahead of every queued batch job; a batch job goes last.
                at = len(self._queue) if job.batch else next(
                    (i for i, j in enumerate(self._queue) if j.batch), len(self._queue))
                self._queue.insert(at, job)
                self._cond.notify()
                if self.background and self._thread is None:
                    self._thread = threading.Thread(target=self._work, name="where-looked",
                                                    daemon=True)
                    self._thread.start()
        if not self.background:
            self._run_next()
        return self.status(job.rid) or {"status": "ready"}

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
            self._run_next()

    def _run_next(self) -> None:
        with self._cond:
            if not self._queue:
                return
            job = self._queue.pop(0)
            self._current = job
            job.started = self.clock()
        entries: list[dict] = []
        try:
            for task in job.tasks:
                read_at = self.clock()
                got = task.fetch()                      # the network, outside the lock
                read = self.clock() - read_at
                while not self.idle():                  # identifications first
                    self.sleep(self.poll)
                with self.lock:
                    start = self.clock()
                    entry = task.make(got)
                    took = self.clock() - start
                entries.append({"photo": task.photo, **entry})
                self.photos_made += 1
                self.last_seconds = took
                self.seconds_per_photo += 0.3 * (took - self.seconds_per_photo)
                self.note(f"[looked] {job.rid}: map {took:.1f} s, read {read:.1f} s"
                          + (" (preload)" if job.batch else ""))
                with self._cond:
                    job.done += 1
            job.finish(entries)
        except Exception as e:  # noqa: BLE001 - this run's maps fail alone
            self.note(f"[looked] {job.rid}: {type(e).__name__}: {e}")
            job.error = "the maps could not be made; try again later"
            with self._cond:
                self._failed[job.rid] = (self.clock(), job.error)
                while len(self._failed) > 500:
                    self._failed.popitem(last=False)
            if job.on_error is not None:
                job.on_error(e)
        finally:
            with self._cond:
                self._current = None
            job.finished.set()
