"""Small scoring helpers shared by comparisons and the held-out report."""

from __future__ import annotations

from typing import Hashable, Iterable


def macro_f1(truth: Iterable[Hashable], predicted: Iterable[Hashable]) -> float | None:
    """Macro-averaged F1 over every label in the truth or the predictions, as
    scikit-learn's f1_score(average="macro") computes it by default (and the Picek group's
    fgvc library reports): a label predicted but never true scores 0 and counts, so a
    model is charged for scattering answers over many species. None with no records."""
    truth, predicted = list(truth), list(predicted)
    if len(truth) != len(predicted):
        raise ValueError("truth and predictions differ in length")
    if not truth:
        return None
    tp: dict = {}
    fp: dict = {}
    fn: dict = {}
    for t, p in zip(truth, predicted):
        if t == p:
            tp[t] = tp.get(t, 0) + 1
        else:
            fn[t] = fn.get(t, 0) + 1
            fp[p] = fp.get(p, 0) + 1
    labels = set(truth) | set(predicted)
    total = 0.0
    for c in labels:
        denom = 2 * tp.get(c, 0) + fp.get(c, 0) + fn.get(c, 0)
        total += 2 * tp.get(c, 0) / denom if denom else 0.0
    return total / len(labels)
