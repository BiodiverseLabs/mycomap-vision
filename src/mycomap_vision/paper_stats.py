"""The paper's statistics, fixed before the sealed test is opened (paper analysis plan,
"Statistics"). Numpy only, every function pure and seeded:

- a rate's interval: bootstrap over observers (all of an observer's finds go together),
  percentile, `reps` resamples;
- a paired difference (two methods on the same finds): its interval by the same paired
  cluster bootstrap, and McNemar's exact test on the discordant finds, one- or two-sided;
- Holm's step-down adjustment within a family of hypotheses;
- non-inferiority (the place-scoring band): the one-sided lower bound of the paired
  difference against a margin, with a bootstrap p-value, combined with the overall
  superiority test as one intersection-union test (the larger p);
- the depth trend: logistic regression of a right answer on log2(1 + reference records)
  and covariates, fitted as a GEE with an independence working correlation and an
  observer-clustered (sandwich) variance, with a one-sided Wald test on the slope.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

REPS = 10_000
LEVEL = 0.95
CHUNK = 500                     # bootstrap resamples drawn at a time (memory)


# --- intervals ---------------------------------------------------------------------------------

def _groups(values: np.ndarray, clusters) -> tuple[np.ndarray, np.ndarray]:
    """Per cluster: (sum of values, number of finds), in a fixed order."""
    labels, inv = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    sums = np.bincount(inv, weights=np.asarray(values, dtype=np.float64), minlength=len(labels))
    counts = np.bincount(inv, minlength=len(labels)).astype(np.float64)
    return sums, counts


def _resampled_means(sums: np.ndarray, counts: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """`reps` cluster-bootstrap ratio estimates sum(values) / sum(finds)."""
    rng = np.random.default_rng(seed)
    g = len(sums)
    out = np.empty(reps)
    for at in range(0, reps, CHUNK):
        n = min(CHUNK, reps - at)
        pick = rng.integers(0, g, size=(n, g))
        out[at:at + n] = sums[pick].sum(axis=1) / np.maximum(counts[pick].sum(axis=1), 1)
    return out


def cluster_ci(hits, clusters, reps: int = REPS, seed: int = 0, level: float = LEVEL
               ) -> tuple[float, float] | None:
    """Percentile interval of a rate, resampling whole clusters (observers)."""
    hits = np.asarray(hits, dtype=np.float64)
    if not len(hits):
        return None
    sums, counts = _groups(hits, clusters)
    est = _resampled_means(sums, counts, reps, seed)
    a = (1 - level) / 2
    lo, hi = np.percentile(est, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def paired_boot(a, b, clusters, reps: int = REPS, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of mean(a) - mean(b) on the same finds, resampling clusters."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sums, counts = _groups(d, clusters)
    return _resampled_means(sums, counts, reps, seed)


def paired_ci(a, b, clusters, reps: int = REPS, seed: int = 0, level: float = LEVEL
              ) -> tuple[float, float] | None:
    if not len(a):
        return None
    est = paired_boot(a, b, clusters, reps, seed)
    q = (1 - level) / 2
    lo, hi = np.percentile(est, [100 * q, 100 * (1 - q)])
    return float(lo), float(hi)


# --- tests -------------------------------------------------------------------------------------

def _binom_tail_ge(k: int, n: int) -> Fraction:
    """P(X >= k) for X ~ Binomial(n, 1/2), exactly."""
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    return Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n)


def mcnemar_exact(b: int, c: int, alternative: str = "two-sided") -> float:
    """Exact McNemar test on the discordant finds: `b` where only the first method is right,
    `c` where only the second is. "greater": the first is better; "less": worse;
    "two-sided": twice the smaller tail, at most 1."""
    n = b + c
    if n == 0:
        return 1.0
    if alternative == "greater":
        return float(_binom_tail_ge(b, n))
    if alternative == "less":
        return float(_binom_tail_ge(c, n))
    if alternative != "two-sided":
        raise ValueError(f"alternative is greater, less or two-sided, not {alternative!r}")
    return float(min(Fraction(1), 2 * min(_binom_tail_ge(b, n), _binom_tail_ge(c, n))))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm's step-down adjustment: {id: {p, p_holm, reject}}. Ties keep input order."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(order)
    out, running, stop = {}, 0.0, False
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * pvalues[k])
        running = max(running, adj)               # monotone
        stop = stop or pvalues[k] > alpha / (m - i)
        out[k] = {"p": pvalues[k], "p_holm": running, "reject": not stop}
    return {k: out[k] for k in pvalues}


def noninferiority(a, b, clusters, margin: float, reps: int = REPS, seed: int = 0,
                   level: float = LEVEL) -> dict:
    """a not worse than b by more than `margin` (absolute): the one-sided `level` lower bound
    of mean(a) - mean(b) from the paired cluster bootstrap, and its p-value (the bootstrap
    share at or below -margin, +1 smoothed)."""
    if not len(a):
        return {"n": 0, "lower_bound": None, "p": 1.0, "margin": margin}
    est = paired_boot(a, b, clusters, reps, seed)
    lower = float(np.percentile(est, 100 * (1 - level)))
    p = (int((est <= -margin).sum()) + 1) / (reps + 1)
    return {"n": int(len(a)), "difference": float(np.mean(np.asarray(a, float)
                                                          - np.asarray(b, float))),
            "lower_bound": lower, "margin": margin, "p": p, "holds": lower > -margin}


def intersection_union(*pvalues: float) -> float:
    """An intersection-union test rejects only if every part does: its p is the largest."""
    return max(pvalues)


def normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# --- the depth trend (GEE) ---------------------------------------------------------------------

def gee_logistic(y, X, clusters, names: list[str] | None = None, max_iter: int = 100,
                 tol: float = 1e-10, small_sample: bool = True) -> dict:
    """Logistic regression of y (0/1) on X (an intercept is added), as a GEE with an
    independence working correlation: the estimates are the maximum-likelihood ones, the
    variance the observer-clustered sandwich A^-1 B A^-1 (A = X'WX, B = sum over clusters of
    the cluster's score outer product), times G / (G - 1) with `small_sample`. Returns each
    coefficient with its robust standard error and z, and the model-based SE beside."""
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=np.float64).reshape(len(y), -1)])
    names = ["intercept"] + (names or [f"x{i}" for i in range(1, X.shape[1])])
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        step = np.linalg.solve((X * w[:, None]).T @ X, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    A = (X * (mu * (1 - mu))[:, None]).T @ X
    Ainv = np.linalg.inv(A)
    labels, inv = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    scores = np.zeros((len(labels), X.shape[1]))
    np.add.at(scores, inv, X * (y - mu)[:, None])
    B = scores.T @ scores
    g = len(labels)
    V = Ainv @ B @ Ainv * (g / (g - 1) if small_sample and g > 1 else 1.0)
    se = np.sqrt(np.diag(V))
    return {"n": int(len(y)), "clusters": g, "converged": bool(np.max(np.abs(step)) < tol),
            "coefficients": {n: {"estimate": float(b), "se_robust": float(s),
                                 "z": float(b / s) if s > 0 else None,
                                 "se_model": float(m)}
                             for n, b, s, m in zip(names, beta, se, np.sqrt(np.diag(Ainv)))}}


def one_sided_wald(coef: dict, alternative: str = "greater") -> float:
    """p of a one-sided Wald test on a gee_logistic coefficient."""
    z = coef["z"]
    if z is None:
        return 1.0
    return normal_sf(z) if alternative == "greater" else normal_sf(-z)
