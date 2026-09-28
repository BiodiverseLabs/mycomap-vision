"""Identification methods: how photo vectors become species scores.

Every method has fit(vectors, index) over the reference records only, and
species_scores(query) -> one score per species (higher is better) for a record's
photos. `index` is an evaluate.Index: reference photo rows grouped by species.

- nearest       the best-matching DNA-verified specimen (no training)
- species-mean  the species' average vector (no training)
- linear        a trained linear classifier (balanced softmax available, off by default)
- hybrid        linear + nearest: the classifier where data is rich, the specimen
                lookup keeping single-record species in play
"""

from __future__ import annotations

import numpy as np


class Scorer:
    """Cosine similarity of query photos to every reference photo, on the GPU when there is one."""

    def __init__(self, ref_vecs: np.ndarray):
        self.torch = None
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.ref = torch.from_numpy(ref_vecs.astype(np.float16)).cuda()
        except ImportError:
            pass
        if self.torch is None:
            self.ref = ref_vecs.astype(np.float32)

    def sims(self, query: np.ndarray) -> np.ndarray:
        if self.torch is not None:
            q = self.torch.from_numpy(query.astype(np.float16)).cuda()
            return (q @ self.ref.T).float().cpu().numpy()
        return query.astype(np.float32) @ self.ref.T


def species_scores(sims: np.ndarray, index) -> np.ndarray:
    """(n_species,): mean over query photos of each photo's best match within the species."""
    return np.maximum.reduceat(sims, index.starts, axis=1).mean(axis=0)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def _cuda_torch():
    """torch when a CUDA GPU is available, else None (numpy is used instead)."""
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def photos_per_species(index) -> np.ndarray:
    return np.diff(np.append(index.starts, len(index.cols)))


class NearestSpecimen:
    """Score = mean over query photos of the best match among the species' reference photos."""
    name = "nearest"

    def fit(self, vectors: np.ndarray, index) -> None:
        self.index = index
        self.scorer = Scorer(vectors[index.cols])   # species-sorted, so reduceat works

    def species_scores(self, query: np.ndarray) -> np.ndarray:
        return species_scores(self.scorer.sims(query), self.index)

    def photo_sims(self, query: np.ndarray) -> np.ndarray:
        """(q, reference photos) similarities, columns in index.cols order."""
        return self.scorer.sims(query)


class SpeciesMean:
    """Score = mean over query photos of the similarity to the species' average vector."""
    name = "species-mean"

    def fit(self, vectors: np.ndarray, index) -> None:
        self.index = index
        ref = vectors[index.cols].astype(np.float32)
        sums = np.add.reduceat(ref, index.starts, axis=0)
        protos = sums / np.linalg.norm(sums, axis=1, keepdims=True).clip(1e-12)
        self.scorer = Scorer(protos.astype(np.float16))

    def species_scores(self, query: np.ndarray) -> np.ndarray:
        return self.scorer.sims(query).mean(axis=0)


class LinearHead:
    """A linear classifier on the frozen photo vectors, trained with balanced softmax.

    Balanced softmax adds each species' log share of the training photos to its
    logit while training, and leaves it out when predicting. The classifier then
    learns what separates species without learning that common ones are common,
    so a species with a handful of photos isn't pushed down by one with thousands.
    A record's score for a species is the mean of its photos' log-probabilities.

    Measured on 300 photos of one species against 3 of another, at a photo exactly
    between them, balanced softmax cut the common species' lead in log-probability
    from 11 to 2.

    But on the real 8,000-record sample (4,317 species, median 4 photos each) it
    cost far more than it gave: species top-1 3% balanced against 13% plain, and 0%
    against 36% for species with 6-30 records, because test records arrive at real
    frequencies. So plain training is the default; the nearest-specimen lookup is
    what keeps rare species in play. Retest on the full data (docs/PLAN.md).
    """
    name = "linear"
    scale = 16.0          # inputs are unit vectors; this lets logits get decisive
    epochs = 40
    batch = 2048
    lr = 0.05
    weight_decay = 1e-4
    seed = 0
    balanced = False      # True = balanced softmax (see the class notes for why it's off)
    device = "auto"       # "auto": the GPU when there is one, else numpy on the CPU

    def fit(self, vectors: np.ndarray, index, state: dict | None = None) -> None:
        """Train; or, given a saved `state` (from state()), restore it instead."""
        if state is not None:
            self.w, self.b = state["w"], state["b"]
            return
        x = vectors[index.cols].astype(np.float32) * self.scale
        counts = photos_per_species(index)
        y = np.repeat(np.arange(len(counts)), counts)
        log_prior = (np.log(counts / counts.sum()) if self.balanced
                     else np.zeros(len(counts))).astype(np.float32)
        torch = _cuda_torch() if self.device == "auto" else None
        if torch is not None:
            self.w, self.b = self._fit_torch(torch, x, y, log_prior)
            return
        n, d = x.shape
        c = len(counts)
        w = np.zeros((d, c), dtype=np.float32)
        b = np.zeros(c, dtype=np.float32)
        adam = {k: np.zeros_like(v) for k, v in (("mw", w), ("vw", w), ("mb", b), ("vb", b))}
        rng = np.random.default_rng(self.seed)
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                idx = order[start:start + self.batch]
                xb, yb = x[idx], y[idx]
                z = xb @ w + b + log_prior
                p = np.exp(log_softmax(z))
                p[np.arange(len(idx)), yb] -= 1.0
                g = p / len(idx)
                gw = xb.T @ g + self.weight_decay * w
                gb = g.sum(axis=0)
                step += 1
                w, b = self._adam(w, b, gw, gb, adam, step)
        self.w, self.b = w, b

    def _fit_torch(self, torch, x, y, log_prior):
        """The same training on the GPU (Adam, same batches, weight decay as an L2 term)."""
        dev = "cuda"
        xt = torch.from_numpy(x).to(dev)
        yt = torch.from_numpy(y).to(dev)
        prior = torch.from_numpy(log_prior).to(dev)
        w = torch.zeros((x.shape[1], len(log_prior)), device=dev, requires_grad=True)
        b = torch.zeros(len(log_prior), device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=self.lr)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        n = len(y)
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=gen).to(dev)
            for start in range(0, n, self.batch):
                idx = order[start:start + self.batch]
                z = xt[idx] @ w + b + prior
                l2 = 0.5 * self.weight_decay * (w * w).sum()
                loss = torch.nn.functional.cross_entropy(z, yt[idx]) + l2
                opt.zero_grad()
                loss.backward()
                opt.step()
        return w.detach().cpu().numpy(), b.detach().cpu().numpy()

    def _adam(self, w, b, gw, gb, s, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for key, g in (("w", gw), ("b", gb)):
            s["m" + key] = beta1 * s["m" + key] + (1 - beta1) * g
            s["v" + key] = beta2 * s["v" + key] + (1 - beta2) * g * g
        mhat_w = s["mw"] / (1 - beta1 ** t)
        vhat_w = s["vw"] / (1 - beta2 ** t)
        mhat_b = s["mb"] / (1 - beta1 ** t)
        vhat_b = s["vb"] / (1 - beta2 ** t)
        return (w - self.lr * mhat_w / (np.sqrt(vhat_w) + eps),
                b - self.lr * mhat_b / (np.sqrt(vhat_b) + eps))

    def state(self) -> dict:
        return {"w": self.w, "b": self.b}

    def photo_logits(self, query: np.ndarray) -> np.ndarray:
        """Per-photo logits with the frequency prior left out."""
        return (query.astype(np.float32) * self.scale) @ self.w + self.b

    def species_scores(self, query: np.ndarray) -> np.ndarray:
        return log_softmax(self.photo_logits(query), axis=1).mean(axis=0)


class Hybrid:
    """The linear classifier's log-probabilities plus the nearest-specimen ones.

    Where a species has plenty of photos the classifier is the sharper judge; where
    it has one record, the specimen lookup still gives a close match its due.
    """
    name = "hybrid"
    nn_temperature = 0.02

    def fit(self, vectors: np.ndarray, index, state: dict | None = None) -> None:
        self.nearest = NearestSpecimen()
        self.nearest.fit(vectors, index)
        self.head = LinearHead()
        self.head.fit(vectors, index, state=state)

    def state(self) -> dict:
        return self.head.state()

    def species_scores(self, query: np.ndarray) -> np.ndarray:
        nn = log_softmax(self.nearest.species_scores(query) / self.nn_temperature)
        return nn + self.head.species_scores(query)

    def photo_sims(self, query: np.ndarray) -> np.ndarray:
        return self.nearest.photo_sims(query)


METHODS = {m.name: m for m in (NearestSpecimen, SpeciesMean, LinearHead, Hybrid)}

# Log-probability methods also come with the range-and-season score (prior.py).
from functools import partial  # noqa: E402

from .prior import WithPrior  # noqa: E402

for _base in (LinearHead, Hybrid):
    METHODS[f"{_base.name}+prior"] = partial(WithPrior, _base)
