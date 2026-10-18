"""The review's measurement fixes on v1 val (Task B, 2026-10-18): pure functions over per-find
arrays, so `mv paper-score` can call the same code on the sealed test (its wiring is the
paper lane's; the threshold is always passed in there, never read from serving).

Inputs are plain arrays aligned by find. A find with no answer is dropped by the caller.

- confident_answers: how often the top species answer reaches a confidence (apiv1's 0.70
  rule by default) and is wrong, over every find and over the finds whose true species has
  no reference record (E11's "new species" row: a confident answer there is always wrong);
- selective_accuracy: accuracy and coverage of the answers at or above each threshold;
- genus_backoff: genus top-1 for the finds whose species has no reference;
- unseen_observer / recent_cohort / near_train_flags: the generalisation splits (an observer
  with no training records; a find photographed within the last N months of the cutoff; a
  photo within dHash distance 4 of a training photo).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .apiv1 import CONFIDENT_AT


def _arr(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


def confident_answers(conf, right, depth, threshold: float = CONFIDENT_AT) -> dict:
    """{"all": ..., "unknown": ...}: per group, n finds, how many top answers reach
    `threshold` (confident, share), how many of those are wrong (wrong, and wrong_share =
    confident-wrong over all the group's finds), and the precision of the confident ones.
    `depth`: the true species' reference records (0 = no reference: "unknown")."""
    conf, right, depth = _arr(conf), _arr(right, bool), _arr(depth, int)
    out = {}
    for name, keep in (("all", np.ones(len(conf), bool)), ("unknown", depth == 0)):
        c = conf[keep] >= threshold
        r = right[keep]
        n, k = int(keep.sum()), int(c.sum())
        wrong = int((c & ~r).sum())
        out[name] = {"n": n, "confident": k, "share": _share(k, n), "wrong": wrong,
                     "wrong_share": _share(wrong, n),
                     "precision": _share(k - wrong, k)}
    out["threshold"] = threshold
    return out


def selective_accuracy(conf, right, thresholds=(0.3, 0.5, 0.7, 0.9)) -> list[dict]:
    """Per threshold: of the finds whose top answer reaches it, how many (n), their share
    of all finds (coverage) and the share right (accuracy; None when none reach it)."""
    conf, right = _arr(conf), _arr(right, bool)
    out = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        out.append({"threshold": float(t), "n": n, "coverage": _share(n, len(conf)),
                    "accuracy": float(right[keep].mean()) if n else None})
    return out


def genus_backoff(genus_right, depth) -> dict:
    """Genus top-1 over the finds whose true species has no reference record: what the
    identifier can still get right for a species it has never seen."""
    g, depth = _arr(genus_right, bool), _arr(depth, int)
    keep = depth == 0
    n = int(keep.sum())
    return {"n": n, "rate": float(g[keep].mean()) if n else None}


def unseen_observer(observers, reference_observers) -> np.ndarray:
    """True for a find whose observer has no record in the reference (a new observer's
    camera, habits and places: the generalisation that counts for a new user). A find with
    no observer is never 'unseen' (nothing to tell)."""
    ref = {o for o in reference_observers if o}
    return np.array([bool(o) and o not in ref for o in observers], dtype=bool)


def _months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    m += 1
    days = [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31][m - 1]
    return date(y, m, min(d.day, days))


def recent_cohort(observed_on, cutoff: str, months: int | None = None) -> np.ndarray:
    """True for a find photographed recently: observed after `cutoff` minus `months`
    calendar months (with months=None, observed after the cutoff itself). A find with no
    date (None, '', or a bad string) is never recent. Dates are ISO (YYYY-MM-DD...)."""
    start = date.fromisoformat(cutoff[:10])
    if months:
        start = _months_before(start, int(months))
    out = []
    for d in observed_on:
        try:
            out.append(date.fromisoformat(str(d)[:10]) > start)
        except (TypeError, ValueError):
            out.append(False)
    return np.array(out, dtype=bool)


def near_train_flags(val_hashes: dict, train_hashes, max_distance: int | None = None,
                     chunk: int = 64) -> dict:
    """{photo_key: bool} for val photos: True when a training photo lies within
    `max_distance` (dhash.NEAR, 4) Hamming bits of its dHash (a re-saved, resized or
    lightly cropped copy). `val_hashes`: {photo_key: uint64 dHash}; `train_hashes`: the
    training photos' hashes. Small chunks keep the comparison matrix to ~chunk x train."""
    from . import dhash
    keys = list(val_hashes)
    q = np.asarray([val_hashes[k] for k in keys], dtype=np.uint64)
    idx, _ = dhash.nearest(q, np.asarray(train_hashes, dtype=np.uint64),
                           dhash.NEAR if max_distance is None else max_distance, chunk)
    return {k: bool(i >= 0) for k, i in zip(keys, idx.tolist())}
