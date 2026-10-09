"""The standard summary of a benchmark report (Steve, 2026-10-09), printed first.

For each Vision model (a backbone and method, with its place and photo size):

- a table of top 1 / 3 / 5 / 10 by rows species strict, species s.l., species
  complex (beta), genus strict, genus s.l. Strict is the report's own judgement
  (Vision's labels; iNat by taxon id where the answer was found on iNat). s.l. and
  complex come from name_equiv (feat/name-equivalence): until it is merged those
  rows say so and stay empty. A right answer at a stricter level counts at every
  looser one. Top 10 needs answers stored ten deep (predict since this branch);
  answers stored five deep say "not stored" there;
- species top-1 and top-5 by how many reference records the true species has
  (0, 1-4, 5-19, 20-99, 100+);

then iNat's computer vision on the records it and every Vision model answered, with
the Vision models on the same records, or "not run". Every table gives its n.
"""

from __future__ import annotations

from collections import Counter

from .heldout import TOP_K
from .heldout_report import DEPTH_BUCKETS, INAT, model_name, rate

KS = (1, 3, 5, 10)
LEVELS = (("species", "strict", "species strict"), ("species", "sl", "species s.l."),
          ("species", "complex", "species complex (beta)"), ("genus", "strict", "genus strict"),
          ("genus", "sl", "genus s.l."))
EQUIV_MISSING = "needs name_equiv (feat/name-equivalence, not merged)"
OLD_DEPTH = 5           # answers stored before the summary format were kept five deep


def name_equiv():
    """The name_equiv module once feat/name-equivalence is merged, else None."""
    try:
        from . import name_equiv as module
    except ImportError:
        return None
    return module


def first_hits(result: dict, truth, labeller, by_taxon_id: bool, equiv) -> dict:
    """{(rank, level): 0-based position of the first right candidate, None for none}, for
    the ranks the answer key has. A stricter hit counts at every looser level."""
    taxa = result.get("truth_taxa") or {}
    out = {}
    for rank in ("species", "genus"):
        want = getattr(truth, rank)
        if not want:
            continue
        top = result.get(rank) or []
        if by_taxon_id and taxa.get(rank) is not None:
            strict = [c.get("taxon_id") == taxa[rank] for c in top]
        else:
            strict = [labeller.same(c["name"], want) for c in top]
        levels = {"strict": strict}
        if equiv is not None:
            match = equiv.species_match if rank == "species" else equiv.genus_match
            got = [match(c.get("name") or "", want) for c in top]
            levels["sl"] = [s or bool(g.get("sl")) for s, g in zip(strict, got)]
            if rank == "species":
                levels["complex"] = [s or bool(g.get("complex"))
                                     for s, g in zip(levels["sl"], got)]
        for level, hits in levels.items():
            out[(rank, level)] = next((i for i, h in enumerate(hits) if h), None)
    return out


def depth_of(result: dict) -> int:
    return int(result.get("top_k") or OLD_DEPTH)


def ladder(results: dict[str, dict], ids: set[str], truths: dict, labeller,
           by_taxon_id: bool, equiv) -> dict:
    """The top 1 / 3 / 5 / 10 table on `ids` (records with an answer key). A column
    deeper than some record's stored answer is None ("not stored")."""
    use = {oid: results[oid] for oid in ids if oid in results and oid in truths}
    depth = min((depth_of(r) for r in use.values()), default=0)
    tally = {(rank, level): Counter() for rank, level, _ in LEVELS}
    for oid, res in use.items():
        hits = first_hits(res, truths[oid], labeller, by_taxon_id, equiv)
        for rank, level, _ in LEVELS:
            if (rank, "strict") not in hits:          # no answer at this rank
                continue
            c = tally[(rank, level)]
            c["n"] += 1
            pos = hits.get((rank, level))
            for k in KS:
                c[k] += pos is not None and pos < k
    rows = {}
    for rank, level, label in LEVELS:
        c = tally[(rank, level)]
        if level != "strict" and equiv is None:
            rows[label] = {"n": c["n"], "note": EQUIV_MISSING}
            continue
        rows[label] = {"n": c["n"], **{f"top{k}": rate(c[k], c["n"]) if k <= depth else None
                                       for k in KS}}
    return {"records": len(use), "stored_depth": depth, "rows": rows}


def by_reference_depth(judged: dict[str, dict], feats: dict[str, dict]) -> dict:
    """Species top-1 and top-5 by the true species' reference records."""
    tally = {label: Counter() for _lo, _hi, label in DEPTH_BUCKETS}
    for oid, ranks in judged.items():
        bucket = feats[oid]["species reference records"]
        if "species" not in ranks or bucket not in tally:
            continue
        hit1, hit5, _name, _conf = ranks["species"]
        tally[bucket]["n"] += 1
        tally[bucket]["top1"] += hit1
        tally[bucket]["top5"] += hit5
    return {label: {"n": c["n"], "top1": rate(c["top1"], c["n"]),
                    "top5": rate(c["top5"], c["n"])} for label, c in tally.items()}


def standard_summary(head: dict, preds: dict, judged: dict, truths: dict, feats: dict,
                     labeller, deeper: dict | None = None) -> dict:
    """The summary for one report. `preds` / `judged`: the report's {model: {id: ...}};
    `deeper`: {model: {id: full identify result}} from answers.sqlite, whose ranks are
    used where they go deeper than the stored top answers."""
    equiv = name_equiv()
    vision = sorted(m for m in judged if m[0] != INAT)
    inat = sorted(m for m in judged if m[0] == INAT)

    def results_of(m):
        out = dict(preds.get(m, {}))
        for oid, full in (deeper or {}).get(m, {}).items():
            ranks = full.get("ranks") or {}
            if oid in out and min(len(ranks.get(r) or []) for r in ("species", "genus")) \
                    > min(len(out[oid].get(r) or []) for r in ("species", "genus")):
                out[oid] = {**ranks, "top_k": TOP_K}
        return out
    results = {m: results_of(m) for m in judged}
    out = {**head, "name_equivalence": "merged" if equiv else EQUIV_MISSING, "models": {}}
    for m in vision:
        out["models"][model_name(m)] = {
            **ladder(results[m], set(judged[m]), truths, labeller, False, equiv),
            "species_by_true_species_reference_records": by_reference_depth(judged[m], feats)}
    if not inat:
        out["inat_same_records"] = {"status": "not run"}
        return out
    same = set.intersection(*(set(judged[m]) for m in vision + inat))
    out["inat_same_records"] = {
        "status": "run", "records": len(same),
        "models": {model_name(m): ladder(results[m], same, truths, labeller, m[0] == INAT, equiv)
                   for m in vision + inat}}
    return out


def _cell(r) -> str:
    return f"{'-':>8}" if not r or r.get("rate") is None else f"{100 * r['rate']:7.1f}%"


def _table(lines: list[str], table: dict, indent: str = "    ") -> None:
    lines.append(f"{indent}{'':<24}" + "".join(f"{'top ' + str(k):>8}" for k in KS) + f"{'n':>9}")
    for label, row in table["rows"].items():
        if "note" in row:
            lines.append(f"{indent}{label:<24}  {row['note']}")
            continue
        cells = "".join(_cell(row.get(f"top{k}")) if row.get(f"top{k}") is not None
                        else f"{'n.s.':>8}" for k in KS)
        lines.append(f"{indent}{label:<24}{cells}{row['n']:>9,}")
    if table["stored_depth"] and table["stored_depth"] < max(KS):
        lines.append(f"{indent}(n.s.: answers stored {table['stored_depth']} deep; "
                     f"predict again for top {max(KS)})")


def format_summary(s: dict) -> str:
    """The summary as plain text, every table with its n."""
    state = "sealed" if s.get("sealed") else "development, not sealed"
    lines = [f"{s['benchmark']} / {s.get('split') or 'all'} ({state}): {s['records']:,} records, "
             f"{s['scored_records']:,} with an answer key"]
    for model, m in s["models"].items():
        lines.append("")
        lines.append(f"{model}  (n = {m['records']:,} records)")
        _table(lines, m)
        lines.append("    species, by the true species' reference records")
        lines.append(f"    {'':<24}{'top 1':>8}{'top 5':>8}{'n':>9}")
        for bucket, row in m["species_by_true_species_reference_records"].items():
            lines.append(f"    {bucket + ' records':<24}{_cell(row['top1'])}{_cell(row['top5'])}"
                         f"{row['n']:>9,}")
    lines.append("")
    same = s["inat_same_records"]
    if same["status"] != "run":
        lines.append("iNat CV: not run")
    else:
        lines.append(f"iNat CV and Vision on the same records  (n = {same['records']:,} records)")
        for model, table in same["models"].items():
            lines.append(f"  {model}")
            _table(lines, table)
    return "\n".join(lines)
