"""Order, class and phylum candidates for an identification (docs/api.md §4).

The index scores species; genus and family are the best score among their species
(evaluate.rank_scores). The ranks above family are scored the same way, one step up:
each order, class and phylum gets the best score among its genera. A genus's order,
class and phylum come from iNat's taxonomy (taxonomy.py, the cache beside the
manifest), the same answers that set its family. A genus that cache doesn't settle has
none, so it adds to no higher name.

No comparison has fitted a temperature at these ranks yet, so their confidence is
None: names and scores only, never `confident`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

HIGHER_RANKS = ("order", "class", "phylum")


class HigherRanks:
    def __init__(self, genera: list[str], lineage: dict[str, dict[str, str]],
                 genus_records: Counter):
        """`genera`: the index's genus labels, in its order; `lineage`: genus ->
        {order, class, phylum, ...}; `genus_records`: reference records per genus."""
        self.labels: dict[str, list[str]] = {}
        self.label_of: dict[str, np.ndarray] = {}
        self.records: dict[str, Counter] = {}
        for rank in HIGHER_RANKS:
            of = [(lineage.get(g) or {}).get(rank) or "" for g in genera]
            names = sorted(set(of) - {""})
            pos = {n: i for i, n in enumerate(names)}
            self.labels[rank] = names
            self.label_of[rank] = np.asarray([pos.get(n, -1) for n in of], dtype=np.int64)
            counts = Counter()
            for g, n in zip(genera, of):
                if n:
                    counts[n] += genus_records.get(g, 0)
            self.records[rank] = counts

    @classmethod
    def from_taxonomy(cls, tax, genera: list[str], genus_records: Counter) -> "HigherRanks | None":
        if tax is None or not getattr(tax, "genera", None):
            return None
        return cls(genera, tax.genera, genus_records)

    def candidates(self, genus_scores: np.ndarray, top_k: int) -> dict[str, list[dict]]:
        """{rank: the top `top_k` names}, from the genus scores (in genus label order)."""
        out = {}
        gs = np.asarray(genus_scores, dtype=np.float32)
        for rank in HIGHER_RANKS:
            lab = self.label_of[rank]
            keep = (lab >= 0) & np.isfinite(gs)
            scores = np.full(len(self.labels[rank]), -np.inf, dtype=np.float32)
            np.maximum.at(scores, lab[keep], gs[keep])
            order = np.argsort(-scores, kind="stable")[:top_k]
            out[rank] = [{"name": self.labels[rank][i], "score": round(float(scores[i]), 4),
                          "confidence": None,
                          "reference_records": int(self.records[rank][self.labels[rank][i]]),
                          "place": None}
                         for i in order if np.isfinite(scores[i])]
        return out
