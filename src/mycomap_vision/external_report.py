"""The agreed protocol for scoring outside models that have no temporary codes (external.py).

A model trained on Danish records can't name a North American provisional species, and
many of our formal names are not in its vocabulary. So beside the standard summary
(`mv heldout report`) it is scored four ways, every table with its n:

  (iv) coverage: of a benchmark split, and of every North American record Vision holds,
       the share whose true name is (a) a temporary code, (b) a formal species in the
       model's vocabulary, (c) a formal species not in it (and of (c) those it has once
       name_equiv's s.l. joins recently split genera), (d) one word, no species; and the
       same at genus (in the vocabulary strict, s.l. only, not, no genus);
  (i)  genus (strict, s.l.) and family on all records;
  (ii) species (strict, s.l., complex) on records whose true name is a formal species,
       also by the true species' reference records in Vision's reference (0 ... 100+);
  (iii) same vocabulary: on records whose true name is in model C's vocabulary, C, Vision
       as served, and Vision restricted to C's vocabulary (Vision's species list with only
       C's names kept, in Vision's order).

Vision's answers are read as stored (heldout_predictions and the benchmark's
answers.sqlite: ten deep since 10/9, five before); Vision is never predicted again here.
Restricting a stored list can leave fewer than k names, so restricted top-k is a lower
bound where that happens: each restricted table says on how many records.

External answers come from heldout_predictions once imported (`mv heldout import-
external`), or straight from `mv external predict`'s JSONL, so the report can run before
anything is written to the manifest. A model's vocabulary is its classes in Vision's
names (external.vision_names), compared with writing set aside (heldout.name_key).

Photos are never shown and no coordinate leaves here. On a sealed benchmark the test
split is refused: its looks are recorded by `mv heldout report` only.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from . import external, heldout, heldout_report, names
from .heldout_report import DEPTH_BUCKETS, bucket, rate
from .heldout_summary import KS, by_reference_depth, ladder

SERVED = "bioclip-2-ft-20261007-165400"     # the fine-tune the site serves (10/7)
SERVED_METHODS = ("nearest", "nearest+prior", "nearest+mean")

FORMAL, PROVISIONAL, ONE_WORD = "formal", "provisional (temporary code)", "one word, no species"
IN_VOCAB, SL_ONLY, NOT_IN = "in vocabulary", "in vocabulary s.l. only", "not in vocabulary"
NO_GENUS = "no genus"


def equiv():
    from . import name_equiv
    return name_equiv


# --- the vocabulary -------------------------------------------------------------------------

class Vocabulary:
    """A model's species and genera as Vision names them."""

    def __init__(self, cn: external.ClassNames):
        eq = equiv()
        self.species = {heldout.name_key(n) for n in cn.species if n and len(n.split()) > 1}
        self.species_sl = set()
        for n in cn.species:
            p = eq.parts(n)
            if n and p.folded and not p.provisional:
                self.species_sl.add((eq.genus_group(p.genus).lower(), p.folded))
        self.genera = {g.lower() for g in cn.genus if g}
        self.genus_groups = {eq.genus_group(g).lower() for g in cn.genus if g}

    def has(self, label: str) -> bool:
        return bool(label) and heldout.name_key(label) in self.species

    def has_sl(self, label: str) -> bool:
        eq = equiv()
        p = eq.parts(label)
        return (not p.provisional and bool(p.folded)
                and (eq.genus_group(p.genus).lower(), p.folded) in self.species_sl)

    def genus_kind(self, genus: str) -> str:
        if not genus:
            return NO_GENUS
        if genus.lower() in self.genera:
            return IN_VOCAB
        return SL_ONLY if equiv().genus_group(genus).lower() in self.genus_groups else NOT_IN


def name_kind(truth: heldout.Truth) -> str:
    if not truth.species:
        return ONE_WORD
    return PROVISIONAL if names.parse_name(truth.label).code else FORMAL


def species_kind(truth: heldout.Truth, vocab: Vocabulary) -> str:
    kind = name_kind(truth)
    if kind != FORMAL:
        return kind
    if vocab.has(truth.label):
        return f"{FORMAL}, {IN_VOCAB}"
    return f"{FORMAL}, {SL_ONLY}" if vocab.has_sl(truth.label) else f"{FORMAL}, {NOT_IN}"


SPECIES_KINDS = (PROVISIONAL, f"{FORMAL}, {IN_VOCAB}", f"{FORMAL}, {SL_ONLY}",
                 f"{FORMAL}, {NOT_IN}", ONE_WORD)
GENUS_KINDS = (IN_VOCAB, SL_ONLY, NOT_IN, NO_GENUS)


def coverage(truths: Iterable[tuple[heldout.Truth, int]], vocab: Vocabulary) -> dict:
    """Records (each truth with its record count) by what the model can name: species and
    genus kinds, with shares, and the distinct species of each species kind."""
    sp, ge, distinct = Counter(), Counter(), defaultdict(set)
    total = 0
    for t, n in truths:
        total += n
        k = species_kind(t, vocab)
        sp[k] += n
        if t.species:
            distinct[k].add(heldout.name_key(t.label))
        ge[vocab.genus_kind(t.genus)] += n

    def table(counts, kinds):
        return {k: {"records": counts[k], "share": round(counts[k] / total, 4) if total else None}
                for k in kinds}
    return {"records": total, "species": table(sp, SPECIES_KINDS),
            "genus": table(ge, GENUS_KINDS),
            "distinct_species": {k: len(distinct[k]) for k in SPECIES_KINDS if k != ONE_WORD}}


def reference_truths(conn: sqlite3.Connection, labeller: heldout.Labeller) -> list:
    """Every North American record Vision holds (green, no label conflict), as (truth,
    records) per name; guests (not the fungus in the photo) left out."""
    out = []
    for name, n in conn.execute(
            "select scientific_name, count(*) from records where north_america = 1 and "
            "label_conflict = 0 and coalesce(scientific_name, '') <> '' group by 1"):
        t = labeller.truth(name)
        if t.guest or not (t.species or t.genus):
            continue
        out.append((t, n))
    return out


def coverage_report(conn: sqlite3.Connection, bench: str, split: str = "dev",
                    models: Iterable[str] | None = None, with_reference: bool = True) -> dict:
    """(iv) alone, needing no answers: per model, coverage of the split's answer keys and of
    Vision's North American records, and how its classes took Vision's names."""
    ids = heldout.benchmark_ids(conn, bench, split=split)
    records = heldout.load_benchmark(conn, bench, ids)
    labeller = heldout.Labeller(conn, extra=[r.truth_name for r in records if r.truth_name])
    truths = [t for t in (labeller.truth(r.truth_name) for r in records if r.truth_name)
              if not t.guest]
    ref = reference_truths(conn, labeller) if with_reference else []
    out = {"benchmark": bench, "split": split, "models": {}}
    for short in models or external.MODELS:
        model = external.model_for(short)
        cn = external.vision_names(external.read_labels(model), labeller)
        vocab = Vocabulary(cn)
        m = {"classes": model.num_classes, "class_names": dict(cn.how),
             "split": coverage(((t, 1) for t in truths), vocab)}
        if with_reference:
            m["north_american_records"] = coverage(ref, vocab)
        out["models"][model.backbone] = m
    return out


# --- answers ----------------------------------------------------------------------------------

def read_jsonl(path: Path, ids: set[str]) -> tuple[str, dict[str, dict]]:
    """(backbone, {id: result}) from `mv external predict`'s file, for `ids` only."""
    backbone, out = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            backbone = backbone or r["backbone"]
            if r["backbone"] != backbone:
                raise ValueError(f"{path} mixes {backbone} and {r['backbone']}")
            if r["observation_id"] in ids:
                out[r["observation_id"]] = r["result"]
    if backbone is None:
        raise ValueError(f"{path} holds no answers")
    return backbone, out


def stored_answers(conn: sqlite3.Connection, bench: str, ids: set[str]) -> tuple[dict, dict]:
    """({model key: {id: result}} for every model in heldout_predictions on its chosen
    reference, {key: reference hash})."""
    chosen = heldout_report.chosen_references(conn, bench, ids)
    preds, _other, _near = heldout_report.load_predictions(conn, bench, ids, chosen)
    return preds, chosen


def deepest(result: dict, full: dict | None) -> dict:
    """A stored answer, with the full identify result's ranks where they go deeper."""
    if not full:
        return result
    ranks = full.get("ranks") or {}
    if len(ranks.get("species") or []) > len(result.get("species") or []):
        return {**result, **{r: ranks.get(r) or [] for r in heldout.RANKS},
                "top_k": max(len(ranks.get("species") or []), 1)}
    return result


def restrict(result: dict, vocab: Vocabulary, depth: int = max(KS)) -> tuple[dict, int]:
    """Vision's answer with only the model's species kept, in Vision's order (genus and
    family as they were), and how many species were left (fewer than `depth`: the
    restricted top-k is a lower bound)."""
    kept = [c for c in result.get("species") or [] if vocab.has(c.get("name") or "")]
    return {**result, "species": kept[:depth], "top_k": max(KS)}, len(kept)


def family_ladder(results: dict, ids: set[str], truths: dict, labeller) -> dict:
    c = Counter()
    for oid in ids:
        t, res = truths.get(oid), results.get(oid)
        if not t or not t.family or res is None:
            continue
        c["n"] += 1
        top = [x.get("name") for x in res.get("family") or []]
        pos = next((i for i, n in enumerate(top) if labeller.same(n, t.family)), None)
        for k in KS:
            c[k] += pos is not None and pos < k
    return {"n": c["n"], **{f"top{k}": rate(c[k], c["n"]) for k in KS}}


def per_image(results: dict, ids: set[str], truths: dict, labeller) -> dict | None:
    """The authors' per-image metric: each photo's own top-1 species, over photos."""
    k = n = 0
    for oid in ids:
        res, t = results.get(oid), truths.get(oid)
        if not res or not t or not t.species or "per_image_top1" not in res:
            continue
        for name in res["per_image_top1"]:
            n += 1
            k += labeller.same(name, t.species)
    return rate(k, n) if n else None


def pick(rows: dict, keys: tuple) -> dict:
    return {k: rows[k] for k in keys if k in rows}


SPECIES_ROWS = ("species strict", "species s.l.", "species complex (beta)")
GENUS_ROWS = ("genus strict", "genus s.l.")


def report(conn: sqlite3.Connection, bench: str, split: str = "dev",
           results_files: Iterable[Path] = (), vision_backbone: str = SERVED,
           models: Iterable[str] | None = None, with_reference: bool = True,
           out_path: Path | None = None) -> dict:
    """The protocol tables (module doc) for each external model, from the manifest
    (read-only) and any `mv external predict` JSONL files."""
    if split == "test" and heldout.set_state(conn, bench)["sealed"]:
        raise ValueError("a sealed benchmark's test split is scored only by mv heldout report, "
                         "which records every look")
    ids = set(heldout.benchmark_ids(conn, bench, split=split))
    records = heldout.load_benchmark(conn, bench, sorted(ids))
    stored, chosen = stored_answers(conn, bench, ids)
    ext: dict[str, dict] = {k[0]: v for k, v in stored.items() if external.is_external(k[0])
                            and k[0] != heldout_report.INAT}
    for path in results_files:
        backbone, got = read_jsonl(Path(path), ids)
        ext[backbone] = got                       # a file is newer than what was imported
    if models:
        want = {external.model_for(m).backbone for m in models}
        ext = {b: v for b, v in ext.items() if b in want}
    vision = {k: v for k, v in stored.items() if k[0] == vision_backbone}
    for key in list(vision):
        full = heldout_report.load_answers(conn, bench, key, chosen[key], set(vision[key]))
        vision[key] = {oid: deepest(res, full.get(oid)) for oid, res in vision[key].items()}
    vision = dict(sorted(vision.items(), key=lambda kv: (
        SERVED_METHODS.index(kv[0][1]) if kv[0][1] in SERVED_METHODS else 9, kv[0])))

    extra = [r.truth_name for r in records if r.truth_name]
    extra += [c["name"] for by in [*ext.values(), *vision.values()] for res in by.values()
              for rank in heldout.RANKS for c in res.get(rank) or []]
    labeller = heldout.Labeller(conn, extra=extra)
    truths = {r.observation_id: labeller.truth(r.truth_name) for r in records if r.truth_name}
    truths = {o: t for o, t in truths.items() if not t.guest}
    ref = heldout_report.reference_run(conn, bench, vision_backbone, chosen)
    counts = ((ref or {}).get("rank_counts") or {}).get("species", {})
    feats = {o: {"species reference records": bucket(counts.get(t.species, 0), DEPTH_BUCKETS)
                 if ref and t.species else "n/a"} for o, t in truths.items()}
    eq = equiv()
    all_ids = set(truths)
    formal = {o for o, t in truths.items() if name_kind(t) == FORMAL}
    ref_truths = reference_truths(conn, labeller) if with_reference else []

    def vision_name(key):
        return heldout_report.model_name(key)

    out = {"benchmark": bench, "split": split, "records": len(records),
           "scored_records": len(truths), "formal_species_records": len(formal),
           "vision": {vision_name(k): {"reference_hash": chosen.get(k), "records": len(v)}
                      for k, v in vision.items()},
           "photo_only": "external models: photos only, temperature 1, mean of logits",
           "models": {}}
    for backbone, results in sorted(ext.items()):
        model = external.model_for(backbone)
        cn = external.vision_names(external.read_labels(model), labeller)
        vocab = Vocabulary(cn)
        mine = set(results) & all_ids
        in_vocab = {o for o in formal & mine if vocab.has(truths[o].label)}
        judged = heldout_report.score({(backbone,): results}, truths, labeller)[(backbone,)]
        m: dict = {
            "hf_id": model.hf_id, "classes": model.num_classes, "licence": external.LICENCE,
            "class_names": dict(cn.how), "answered": len(mine),
            "coverage": {"split": coverage(((truths[o], 1) for o in all_ids), vocab)},
            "all_records": {
                **pick(ladder(results, mine, truths, labeller, False, eq)["rows"], GENUS_ROWS),
                "family strict": family_ladder(results, mine, truths, labeller)},
            "formal_species": {
                **ladder(results, formal & mine, truths, labeller, False, eq),
                "per_image_top1": per_image(results, formal & mine, truths, labeller),
                "by_true_species_reference_records": by_reference_depth(
                    {o: judged[o] for o in formal & mine if o in judged}, feats)},
            "same_vocabulary": {"records": len(in_vocab), "models": {}},
        }
        if with_reference:
            m["coverage"]["north_american_records"] = coverage(ref_truths, vocab)
        same = m["same_vocabulary"]["models"]
        same[backbone] = {**ladder(results, in_vocab, truths, labeller, False, eq),
                          "per_image_top1": per_image(results, in_vocab, truths, labeller)}
        for key, vres in vision.items():
            both = in_vocab & set(vres)
            same[vision_name(key)] = ladder(vres, both, truths, labeller, False, eq)
            restricted, short = {}, Counter()
            for oid in both:
                restricted[oid], left = restrict(vres[oid], vocab)
                for k in KS:
                    short[k] += left < k
            table = ladder(restricted, both, truths, labeller, False, eq)
            table["rows"] = pick(table["rows"], SPECIES_ROWS)
            table["fewer_names_than_k_after_restricting"] = {f"top{k}": short[k] for k in KS}
            same[f"{vision_name(key)} restricted to {model.short}"] = table
        out["models"][backbone] = m
    out["vision_models"] = {}
    for key, vres in vision.items():
        mine = set(vres) & all_ids
        judged = heldout_report.score({key: vres}, truths, labeller)[key]
        out["vision_models"][vision_name(key)] = {
            "all_records": {
                **pick(ladder(vres, mine, truths, labeller, False, eq)["rows"], GENUS_ROWS),
                "family strict": family_ladder(vres, mine, truths, labeller)},
            "formal_species": {
                **ladder(vres, formal & mine, truths, labeller, False, eq),
                "by_true_species_reference_records": by_reference_depth(
                    {o: judged[o] for o in formal & mine if o in judged}, feats)}}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        out["file"] = str(out_path)
    return out


# --- as text ----------------------------------------------------------------------------------

def _pct(r) -> str:
    return f"{'-':>8}" if not r or r.get("rate") is None else f"{100 * r['rate']:7.1f}%"


def _rows(lines: list[str], rows: dict, indent: str = "    ") -> None:
    lines.append(f"{indent}{'':<26}" + "".join(f"{'top ' + str(k):>8}" for k in KS) + f"{'n':>8}")
    for label, row in rows.items():
        if not isinstance(row, dict) or "n" not in row:
            continue
        if "note" in row:
            lines.append(f"{indent}{label:<26}  {row['note']}")
            continue
        cells = "".join(_pct(row.get(f"top{k}")) if row.get(f"top{k}") is not None
                        else f"{'n.s.':>8}" for k in KS)
        lines.append(f"{indent}{label:<26}{cells}{row['n']:>8,}")


def _coverage(lines: list[str], title: str, cov: dict) -> None:
    lines.append(f"    {title} (n = {cov['records']:,} records)")
    for rank in ("species", "genus"):
        for kind, v in cov[rank].items():
            share = "-" if v["share"] is None else f"{100 * v['share']:5.1f}%"
            lines.append(f"      {rank:<8}{kind:<40}{share:>7}{v['records']:>9,}")


def format_report(r: dict) -> str:
    lines = [f"{r['benchmark']} / {r['split']}: {r['records']:,} records, "
             f"{r['scored_records']:,} scored, {r['formal_species_records']:,} with a formal "
             "species name. External models: photos only."]
    for name, m in r["vision_models"].items():
        lines += ["", f"{name}  (Vision)"]
        lines.append("  (i) all records")
        _rows(lines, m["all_records"])
        lines.append("  (ii) formal species")
        _rows(lines, m["formal_species"]["rows"])
    for backbone, m in r["models"].items():
        lines += ["", f"{backbone}  ({m['hf_id']}, {m['classes']:,} classes, {m['licence']}; "
                      f"answered {m['answered']:,})"]
        lines.append("  (iv) coverage")
        for where, cov in m["coverage"].items():
            _coverage(lines, where.replace("_", " "), cov)
        lines.append("  (i) all records")
        _rows(lines, m["all_records"])
        lines.append("  (ii) formal species")
        _rows(lines, m["formal_species"]["rows"])
        pi = m["formal_species"].get("per_image_top1")
        if pi:
            lines.append(f"    per-image species top 1: {_pct(pi).strip()} of {pi['n']:,} photos")
        lines.append("    species by the true species' reference records (top 1, top 5, n)")
        for band, row in m["formal_species"]["by_true_species_reference_records"].items():
            lines.append(f"      {band + ' records':<16}{_pct(row['top1'])}{_pct(row['top5'])}"
                         f"{row['n']:>8,}")
        sv = m["same_vocabulary"]
        lines.append(f"  (iii) same vocabulary: true name in the model's vocabulary "
                     f"(n = {sv['records']:,} records)")
        for name, table in sv["models"].items():
            lines.append(f"    {name}  (n = {table['records']:,})")
            _rows(lines, table["rows"], indent="      ")
            short = table.get("fewer_names_than_k_after_restricting")
            if short and any(short.values()):
                lines.append("      restricted list shorter than k (top k a lower bound) on: "
                             + ", ".join(f"{k} {v:,}" for k, v in short.items()))
    return "\n".join(lines)
