"""Identification methods: how photo vectors become species scores.

Every method has fit(vectors, index) over the reference records only, and
species_scores(query) -> one score per species (higher is better) for a record's
photos. `index` is an evaluate.Index: reference photo rows grouped by species.

- nearest       the best-matching DNA-verified specimen (no training)
- species-mean  the species' average vector (no training)
- linear        a trained linear classifier, frequency-neutral (balanced softmax)
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
    from 11 to 2. It doesn't reach parity with so few photos; the hybrid's
    nearest-specimen half covers the rest.
    """
    name = "linear"
    scale = 16.0          # inputs are unit vectors; this lets logits get decisive
    epochs = 40
    batch = 2048
    lr = 0.05
    weight_decay = 1e-4
    seed = 0
    balanced = True       # False = plain softmax (kept for comparison and tests)

    def fit(self, vectors: np.ndarray, index) -> None:
        x = vectors[index.cols].astype(np.float32) * self.scale
        counts = photos_per_species(index)
        y = np.repeat(np.arange(len(counts)), counts)
        log_prior = (np.log(counts / counts.sum()) if self.balanced
                     else np.zeros(len(counts))).astype(np.float32)
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

    def fit(self, vectors: np.ndarray, index) -> None:
        self.nearest = NearestSpecimen()
        self.nearest.fit(vectors, index)
        self.head = LinearHead()
        self.head.fit(vectors, index)

    def species_scores(self, query: np.ndarray) -> np.ndarray:
        nn = log_softmax(self.nearest.species_scores(query) / self.nn_temperature)
        return nn + self.head.species_scores(query)

    def photo_sims(self, query: np.ndarray) -> np.ndarray:
        return self.nearest.photo_sims(query)


METHODS = {m.name: m for m in (NearestSpecimen, SpeciesMean, LinearHead, Hybrid)}
