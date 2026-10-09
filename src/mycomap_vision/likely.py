"""The likely set: the names that could be the answer, with an honest chance of holding it.

One headline answer hides how uncertain an identification is: on the newest test records
the first species is right about a third of the time, and when it is wrong the truth is
often second (a look-alike). So each rank also gets a likely set, the names worth
considering, with how often such a set held the right one ("the right genus is in this
list about 9 times in 10").

Split-conformal prediction with a probability floor (LAC; Sadinle, Lei & Wasserman 2019):

- A test record's score is 1 minus the calibrated probability of its true name. A record
  whose true name no set could hold gets +inf and counts as a miss, so the stated
  coverage includes them: a name the reference set lacks, or one ranked below the
  `cap` a set is cut at.
- The floor is 1 minus the finite-sample quantile ceil((n + 1) * coverage) / n of those
  scores, fitted on records the model never saw (a comparison's test records).
- A set is every name at or above the floor, most probable first, and always the top
  name (which can only add coverage).

Why a floor rather than adding names until their total reaches a threshold (adaptive
prediction sets): the calibrated probabilities spread thinly over many names, and on
comparison 20261008-012435-4ef7b0 the adaptive sets needed 10-12 genera for 90% where
the floor needs 3.7 (and 1.8 families).

The fit aims for COVERAGE and steps down 5% at a time to the first target that is
reachable and keeps sets short (MAX_MEAN_SIZE names on average, measured on held-back
halves). It records what it chose; the page states that coverage.
"""

from __future__ import annotations

import math

import numpy as np

COVERAGE = 0.90           # what a likely set aims to hold
STEP = 0.05               # lower targets tried in turn
LOWEST = 0.50             # below this a set is not offered
MAX_SET = 15              # never list more names than this
MAX_MEAN_SIZE = 5.0       # a target whose sets average more names than this is too vague


def probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(scores / temperature) over the finite scores (identify.softmax_confidence's
    formula); a name with no score gets 0."""
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    out = np.zeros_like(s)
    if finite.any():
        z = (s[finite] - s[finite].max()) / temperature
        e = np.exp(z)
        out[finite] = e / e.sum()
    return out


# --- compact rows: what a fit needs from one record, without its full row of scores --
#
# A fit needs each record's calibrated probabilities only for the names a set could
# list (the top MAX_SET, plus one to tell when a set is cut) and the true name's place
# among them; a true name further down already counts as a miss (score). The softmax
# normaliser at each candidate temperature keeps those probabilities exact. At species
# that is ~110 numbers per record instead of one score per species (~18,000).

KEEP = MAX_SET + 1
PAST = -1                 # a compact row's true position when the name is known but kept out


def logsumexp_by_temperature(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """log(sum(exp(score / T))) over the finite scores, for each T in `grid`."""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    m = s.max()
    return m / grid + np.log(np.exp((s[None, :] - m) / grid[:, None]).sum(axis=1))


def compact(scores: np.ndarray, true_idx: int | None, grid: np.ndarray, keep: int = KEEP,
            lse: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int | None]:
    """(the `keep` highest finite scores, highest first; the normaliser at each T in
    `grid`, or `lse` when already made; the true name's position among them: None when
    the reference set lacks it, PAST when it has it further down)."""
    s = np.asarray(scores, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(s))
    order = finite[np.argsort(-s[finite], kind="stable")][:keep]
    pos = None
    if true_idx is not None and np.isfinite(s[true_idx]):
        hit = np.flatnonzero(order == true_idx)
        pos = int(hit[0]) if len(hit) else PAST
    if lse is None:
        lse = logsumexp_by_temperature(s, grid)
    return s[order].astype(np.float32), lse, pos


def expand(row: tuple[np.ndarray, np.ndarray, int | None], t_index: int,
           temperature: float) -> tuple[np.ndarray, int | None]:
    """A compact row as (probabilities of its kept names, true position) at one of the
    grid's temperatures: what score(), likely_set() and set_metrics() take."""
    top, lse, pos = row
    return np.exp(top.astype(np.float64) / temperature - lse[t_index]), pos


def score(probs: np.ndarray, true_idx: int | None, cap: int = MAX_SET) -> float:
    """1 - the probability of the true name; +inf when no set could hold it (including a
    compact row's PAST: known, but further down than any set reaches)."""
    if true_idx is None or true_idx < 0 or not np.isfinite(probs[true_idx]):
        return math.inf
    p = np.where(np.isfinite(probs), probs, 0.0)
    if int((p > p[true_idx]).sum()) >= cap:
        return math.inf
    return float(1.0 - p[true_idx])


def quantile_index(n: int, coverage: float) -> int:
    """Index into n sorted scores of the split-conformal quantile for this coverage."""
    return min(n - 1, max(0, math.ceil((n + 1) * coverage) - 1))


def fit_floor(scores: list[float], coverage: float) -> float | None:
    """The probability floor for exactly this coverage, or None when it is unreachable."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(s):
        return None
    q = s[quantile_index(len(s), coverage)]
    return float(max(0.0, 1.0 - q)) if np.isfinite(q) else None


def likely_set(probs: np.ndarray, floor: float, cap: int = MAX_SET) -> tuple[list[int], bool]:
    """(indices of the names at or above `floor`, and the top name, most probable first;
    whether the set was cut at `cap`)."""
    p = np.where(np.isfinite(probs), probs, -1.0)
    above = np.flatnonzero(p >= floor - 1e-12)
    top = int(np.argmax(p))
    if p[top] >= 0 and top not in above:
        above = np.append(above, top)
    order = above[np.argsort(-p[above], kind="stable")]
    return [int(i) for i in order[:cap]], len(order) > cap


def set_metrics(rows: list[tuple[np.ndarray, int | None]], floor: float,
                cap: int = MAX_SET) -> dict:
    """Coverage and mean size of the likely sets over (probabilities, true index) rows.
    `coverage` counts a true name the reference set lacks as a miss (the stated figure);
    `conditional_coverage` is over the rows whose true name it has."""
    hits, sizes = 0, []
    for probs, true_idx in rows:
        chosen, _ = likely_set(probs, floor, cap)
        sizes.append(len(chosen))
        hits += true_idx is not None and true_idx >= 0 and true_idx in chosen
    n = len(rows)
    known = sum(t is not None for _, t in rows)
    return {"n": n, "coverage": round(hits / n, 4) if n else None,
            "conditional_coverage": round(hits / known, 4) if known else None,
            "n_known": known,
            "mean_size": round(float(np.mean(sizes)), 2) if n else None}


def crosscheck(rows: list[tuple[np.ndarray, int | None]], coverage: float,
               cap: int = MAX_SET) -> dict | None:
    """Fit on each half and measure on the other: coverage and size on records the floor
    was not fitted on."""
    halves = (rows[0::2], rows[1::2])
    checks = []
    for a, b in (halves, halves[::-1]):
        floor = fit_floor([score(p, t, cap) for p, t in a], coverage)
        if floor is None or not b:
            return None
        checks.append(set_metrics(b, floor, cap))
    n = sum(c["n"] for c in checks)
    known = sum(c["n_known"] for c in checks)
    return {"n": n,
            "coverage": round(sum(c["coverage"] * c["n"] for c in checks) / n, 4),
            "conditional_coverage": round(sum((c["conditional_coverage"] or 0) * c["n_known"]
                                              for c in checks) / known, 4) if known else None,
            "n_known": known,
            "mean_size": round(sum(c["mean_size"] * c["n"] for c in checks) / n, 2)}


def fit_and_check(rows: list[tuple[np.ndarray, int | None]], coverage: float = COVERAGE,
                  cap: int = MAX_SET, max_mean_size: float = MAX_MEAN_SIZE) -> dict | None:
    """The highest target from `coverage` down that is reachable and keeps sets short,
    with its floor fitted on all rows and its coverage checked on held-back halves. None
    when no target from LOWEST up qualifies."""
    scores = [score(p, t, cap) for p, t in rows]
    unlistable = float(np.isinf(scores).mean()) if scores else 0.0
    target = coverage
    while target >= LOWEST - 1e-9:
        floor = fit_floor(scores, target)
        check = crosscheck(rows, target, cap) if floor is not None else None
        if check and check["mean_size"] <= max_mean_size:
            return {"method": "lac", "requested": coverage, "coverage": round(target, 2),
                    "floor": round(floor, 6), "n": len(rows),
                    "unlistable_share": round(unlistable, 4), "crosscheck": check,
                    "in_sample": set_metrics(rows, floor, cap)}
        target = round(target - STEP, 2)
    return None
