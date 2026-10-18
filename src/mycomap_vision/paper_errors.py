"""What the mistakes are, for `mv paper-score` (paper E29, E21), from each comparator's top
species answer per find (descriptive: counts and rates, no tests):

- confused pairs: the most frequent (true species, answered species) pairs among wrong
  answers, with the share of that species' finds they take;
- how far a wrong answer lands: same genus, same family, same order, or further (genus to
  family and order from iNat's taxonomy cache, else the release's own families);
- deadly taxa (a config list, `genera` and `species`; matched strict or s.l., so a later
  split does not drop a find): per listed taxon, its finds and how often each was answered
  with a deadly name (the right one or another) or a harmless one ("predicted-from"), and
  how many finds of harmless species were answered with a deadly name ("predicted-as").
  The paper says plainly: not for deciding what is safe to eat.
"""

from __future__ import annotations

from collections import Counter, defaultdict

DISTANCES = ("same genus", "same family", "same order", "further", "unknown")


def genus_of(name: str) -> str:
    return (name or "").split(" ")[0]


def _exact(answer: str, truth: str) -> bool:
    return answer == truth


def confused_pairs(finds, top1: dict[str, str], n: int = 20, right=_exact) -> dict:
    """The `n` most frequent wrong (truth, answer) species pairs (`right`: the answer key's
    species credit; an answer it credits is not a mistake)."""
    per_truth = Counter(f.species for f in finds if f.species)
    pairs = Counter()
    wrong = 0
    for f in finds:
        a = top1.get(f.key)
        if f.species and a and not right(a, f.species):
            pairs[(f.species, a)] += 1
            wrong += 1
    top = [{"truth": t, "answer": a, "finds": c, "share_of_truth": c / per_truth[t],
            "same_genus": genus_of(t) == genus_of(a)}
           for (t, a), c in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
    return {"wrong_answers": wrong, "distinct_pairs": len(pairs), "top": top}


def distance(truth_genus: str, answer_genus: str, ranks: dict[str, dict]) -> str:
    """How far an answer's genus lands from the truth's (`ranks`: genus -> {family, order})."""
    if truth_genus == answer_genus:
        return "same genus"
    a, b = ranks.get(truth_genus), ranks.get(answer_genus)
    if not a or not b or not a.get("family") or not b.get("family"):
        return "unknown"
    if a["family"] == b["family"]:
        return "same family"
    if a.get("order") and a.get("order") == b.get("order"):
        return "same order"
    return "further" if a.get("order") and b.get("order") else "unknown"


def mistake_distance(finds, top1: dict[str, str], ranks: dict[str, dict],
                     right=_exact) -> dict:
    counts = Counter()
    for f in finds:
        a = top1.get(f.key)
        if f.species and a and not right(a, f.species):
            counts[distance(f.genus or genus_of(f.species), genus_of(a), ranks)] += 1
    total = sum(counts.values())
    return {"wrong_answers": total,
            **{d: {"finds": counts[d], "share": counts[d] / total if total else None}
               for d in DISTANCES}}


def read_deadly(spec: dict) -> dict:
    """The deadly list from its pinned file ({"file", "sha256", "match"}): a CSV with a
    `scientific_name` column; a one-word name is a genus. Refused when the file's sha256 is
    not the one pinned. An inline {"species": [...], "genera": [...]} is taken as given."""
    import csv
    import hashlib
    from pathlib import Path
    if "file" not in spec:
        return {"species": list(spec.get("species", [])), "genera": list(spec.get("genera", [])),
                "match": list(spec.get("match", ["strict", "sl"]))}
    body = Path(spec["file"]).read_bytes()
    got = hashlib.sha256(body).hexdigest()
    if got != spec.get("sha256"):
        raise ValueError(f"deadly list {spec['file']}: sha256 {got[:12]}, the config pins "
                         f"{str(spec.get('sha256'))[:12]}: not this list")
    names = [r["scientific_name"].strip() for r in csv.DictReader(
        body.decode("utf-8-sig").splitlines()) if (r.get("scientific_name") or "").strip()]
    return {"species": [n for n in names if " " in n], "genera": [n for n in names if " " not in n],
            "match": list(spec.get("match", ["strict", "sl"])), "file_sha256": got}


def deadly_match(name: str, deadly: dict, right=_exact) -> str | None:
    """The listed deadly taxon `name` falls in (a listed species, by the answer key's
    credit `right`, or s.l. when the list's `match` includes it; else a listed genus), or
    None."""
    from . import name_equiv
    if not name:
        return None
    loose = "sl" in deadly.get("match", ["strict", "sl"])
    for sp in deadly.get("species", []):
        if right(name, sp) or (loose and name_equiv.species_match(name, sp).get("sl")):
            return sp
    g = genus_of(name)
    return g if g in set(deadly.get("genera", [])) else None


def deadly_taxa(finds, top1: dict[str, str], deadly: dict, right=_exact) -> dict:
    """Predicted-from (each deadly taxon's finds and what they were called) and
    predicted-as (harmless finds called by a deadly name)."""
    per = defaultdict(lambda: {"finds": 0, "same_taxon": 0, "other_deadly": 0, "harmless": 0,
                               "no_answer": 0})
    harmless_finds, harmless_as_deadly = 0, Counter()
    for f in finds:
        if not f.species:
            continue
        truth = deadly_match(f.species, deadly, right)
        answer = top1.get(f.key)
        called = deadly_match(answer, deadly, right) if answer else None
        if truth:
            row = per[truth]
            row["finds"] += 1
            if not answer:
                row["no_answer"] += 1
            elif called == truth:
                row["same_taxon"] += 1
            elif called:
                row["other_deadly"] += 1
            else:
                row["harmless"] += 1               # the dangerous direction
        else:
            harmless_finds += 1
            if called:
                harmless_as_deadly[called] += 1
    return {"list": deadly, "note": "descriptive; not for deciding what is safe to eat",
            "predicted_from": dict(sorted(per.items())),
            "predicted_as": {"harmless_finds": harmless_finds,
                             "called_deadly": sum(harmless_as_deadly.values()),
                             "by_taxon": dict(sorted(harmless_as_deadly.items()))}}
