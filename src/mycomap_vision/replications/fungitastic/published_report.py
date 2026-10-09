"""The agreed protocol for scoring outside models that have no temporary codes (published.py).

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

Every table is given twice: by exact names (Vision's labels, writing set aside) and with
the GBIF synonym crosswalk (crosswalk.py: names sharing an accepted key in the GBIF Backbone
or the Catalogue of Life count as one). One judge (Scorer) is used for every model in a
table, Vision included, so the crosswalk can make a Vision answer right as well as a
Danish one. Scoring only: it never touches Vision's labels.

Vision's answers are read as stored (heldout_predictions and the benchmark's
answers.sqlite: ten deep since 10/9, five before); Vision is never predicted again here.
Restricting a stored list can leave fewer than k names, so restricted top-k is a lower
bound where that happens: each restricted table says on how many records.

External answers come from heldout_predictions once imported (`mv heldout import-
external`), or straight from `mv external predict`'s JSONL, so the report can run before
anything is written to the manifest. A model's vocabulary is its classes in Vision's
names (external.vision_names), compared with writing set aside (heldout.name_key).

The report makes no network request (the crosswalk is read from its cache) and writes
nothing to the manifest. Photos are never shown and no coordinate leaves here. On a
sealed benchmark the test split is refused: its looks are recorded by `mv heldout
report` only.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from . import published as external
from ... import heldout, heldout_report, names
from ...heldout_report import DEPTH_BUCKETS, bucket, rate
from ...heldout_summary import KS, by_reference_depth, ladder

SERVED = "bioclip-2-ft-20261007-165400"     # the fine-tune the site serves (10/7)
SERVED_METHODS = ("nearest", "nearest+prior", "nearest+mean")

FORMAL, PROVISIONAL, ONE_WORD = "formal", "provisional (temporary code)", "one word, no species"
IN_VOCAB, SL_ONLY, NOT_IN = "in vocabulary", "in vocabulary s.l. only", "not in vocabulary"
NO_GENUS = "no genus"
EXACT, CROSSWALK = "exact names", "with the GBIF crosswalk"


def equiv():
    from ... import name_equiv
    return name_equiv


# --- one judge for every model ----------------------------------------------------------------

class Scorer:
    """Vision's labels (heldout.Labeller), and with a crosswalk (crosswalk.py) also GBIF's
    synonymy: two names are one when their labels agree or they share an accepted key.
    The same Scorer judges Vision and the outside models in every table, so a synonym can
    make a Vision answer right as well as theirs. Scoring only: never a training label."""

    def __init__(self, labeller: heldout.Labeller, crosswalk=None):
        self.labeller, self.crosswalk = labeller, crosswalk

    def __getattr__(self, name):
        return getattr(self.labeller, name)

    def same(self, a: str | None, b: str | None) -> bool:
        if self.labeller.same(a, b):
            return True
        return (self.crosswalk is not None
                and self.crosswalk.same(self.labeller.label(a), self.labeller.label(b)))


# --- the vocabulary -------------------------------------------------------------------------

class Vocabulary:
    """A model's species and genera as Vision names them; with a crosswalk, also every
    accepted key of its classes (by Vision label, own name and GBIF's accepted name)."""

    def __init__(self, cn: external.ClassNames, crosswalk=None):
        eq = equiv()
        self.crosswalk = crosswalk
        self.species = {heldout.name_key(n) for n in cn.species if n and len(n.split()) > 1}
        self.keys: set[str] = set()
        if crosswalk is not None:
            for n in [*cn.species, *cn.model_names, *cn.accepted]:
                self.keys |= crosswalk.of(n)
        self.species_sl = set()
        for n in cn.species:
            p = eq.parts(n)
            if n and p.folded and not p.provisional:
                self.species_sl.add((eq.genus_group(p.genus).lower(), p.folded))
        self.genera = {g.lower() for g in cn.genus if g}
        self.genus_groups = {eq.genus_group(g).lower() for g in cn.genus if g}

    def has(self, label: str) -> bool:
        if not label:
            return False
        if heldout.name_key(label) in self.species:
            return True
        return self.crosswalk is not None and bool(self.crosswalk.of(label) & self.keys)

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
    """A stored answer, or the full identify result's ranks when it was kept (answers.sqlite,
    the same reference): they were asked TOP_K deep, so a shorter list there is every
    candidate the method had (a range prior can leave fewer), not a shallow store."""
    ranks = (full or {}).get("ranks")
    if not ranks:
        return result
    return {**result, **{r: ranks.get(r) or [] for r in heldout.RANKS}, "top_k": heldout.TOP_K}


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


def macro_f1(results: dict, ids: set[str], truths: dict, scorer) -> dict | None:
    """Species macro-F1 of the top-1 answers on `ids`, averaged over the true species there
    (each judged by `scorer`, so with the crosswalk synonyms are one class). A record's
    answer is a false positive for every class it names that is not its own."""
    use = [o for o in ids if o in results and truths.get(o) and truths[o].species]
    if not use:
        return None
    classes: list[str] = []
    for o in use:
        t = truths[o].species
        if not any(scorer.same(t, c) for c in classes):
            classes.append(t)
    f1s = []
    for c in classes:
        tp = fp = fn = 0
        for o in use:
            top = (results[o].get("species") or [{}])[0].get("name")
            mine, said = scorer.same(truths[o].species, c), bool(top) and scorer.same(top, c)
            tp += mine and said
            fp += said and not mine
            fn += mine and not said
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"n": len(use), "classes": len(classes), "macro_f1": round(sum(f1s) / len(f1s), 4)}


def pick(rows: dict, keys: tuple) -> dict:
    return {k: rows[k] for k in keys if k in rows}


SPECIES_ROWS = ("species strict", "species s.l.", "species complex (beta)")
GENUS_ROWS = ("genus strict", "genus s.l.")


# --- loading, and the names a report compares ---------------------------------------------------

def load(conn: sqlite3.Connection, bench: str, split: str = "dev",
         results_files: Iterable[Path] = (), vision_backbone: str = SERVED,
         models: Iterable[str] | None = None) -> dict:
    """What the report and the crosswalk read: the split's records and answer keys, Vision's
    stored answers (ten deep where the full answer was kept), the outside models'
    answers, and the labeller."""
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
    want = sorted({external.model_for(m).backbone for m in models} if models
                  else {m.backbone for m in external.MODELS.values()       # those set up
                        if external.labels_path(m).is_file()})
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
    return {"bench": bench, "split": split, "records": records, "chosen": chosen,
            "vision": vision, "vision_backbone": vision_backbone,
            "ext": {b: v for b, v in ext.items() if b in want}, "want": want,
            "labeller": labeller,
            "truths": {o: t for o, t in truths.items() if not t.guest}}


def class_names(ctx: dict, backbone: str) -> external.ClassNames:
    cache = ctx.setdefault("_class_names", {})
    if backbone not in cache:
        cache[backbone] = external.vision_names(
            external.read_labels(external.model_for(backbone)), ctx["labeller"])
    return cache[backbone]


def formal_label(labeller, name: str | None) -> str | None:
    """A name's Vision label when it is a formal species (no temporary code), else None."""
    if not name:
        return None
    t = labeller.truth(name)
    return t.label if t.species and not names.parse_name(t.label).code else None


def crosswalk_names(conn: sqlite3.Connection, ctx: dict, with_reference: bool = True) -> set:
    """Every formal species name a report compares: answer keys of the split (and of
    Vision's North American records), the answers' names, and each model's classes (Vision
    label, own name and GBIF's accepted name). Temporary codes are never sent."""
    lab = ctx["labeller"]
    out = {t.label for t in ctx["truths"].values() if formal_label(lab, t.label)}
    if with_reference:
        out |= {t.label for t, _n in reference_truths(conn, lab) if formal_label(lab, t.label)}
    for by in [*ctx["vision"].values(), *ctx["ext"].values()]:
        for res in by.values():
            for c in res.get("species") or []:
                label = formal_label(lab, c.get("name"))
                if label:
                    out.add(label)
    for backbone in ctx["want"]:
        cn = class_names(ctx, backbone)
        out |= {n for n in [*cn.species, *cn.model_names, *cn.accepted]
                if n and len(n.split()) > 1 and not names.parse_name(n).code}
    return out


def fill_crosswalk(conn: sqlite3.Connection, bench: str, split: str = "dev",
                   results_files: Iterable[Path] = (), vision_backbone: str = SERVED,
                   models: Iterable[str] | None = None, with_reference: bool = True,
                   matcher=None, log=print) -> dict:
    """Ask GBIF (gbif.Matcher: cached, paced) for every name the report will compare."""
    from . import crosswalk as gbif
    ctx = load(conn, bench, split, results_files, vision_backbone, models)
    wanted = crosswalk_names(conn, ctx, with_reference)
    own = matcher is None
    matcher = matcher or gbif.Matcher()
    try:
        log(f"  {len(wanted):,} names to match in {', '.join(gbif.SOURCES)}")
        stats = matcher.fill(wanted, log=log)
        cw = gbif.Crosswalk.from_cache(matcher, wanted)
    finally:
        if own:
            matcher.close()
    return {"names": len(wanted), **stats, **cw.summary(wanted)}


def cached_crosswalk(conn: sqlite3.Connection, ctx: dict, with_reference: bool = True):
    """The crosswalk from the GBIF cache, or None when there is no cache. No request."""
    from . import crosswalk as gbif
    if not gbif.cache_path().is_file():
        return None
    matcher = gbif.Matcher()
    try:
        return gbif.Crosswalk.from_cache(matcher, crosswalk_names(conn, ctx, with_reference))
    finally:
        matcher.close()


def coverage_report(conn: sqlite3.Connection, bench: str, split: str = "dev",
                    models: Iterable[str] | None = None, with_reference: bool = True,
                    crosswalk: bool | object = True) -> dict:
    """(iv) alone, needing no answers: per model, coverage of the split's answer keys and of
    Vision's North American records, by exact names and (from the GBIF cache) with the
    crosswalk, and how its classes took Vision's names."""
    ctx = load(conn, bench, split, (), SERVED, models)
    cw = cached_crosswalk(conn, ctx, with_reference) if crosswalk is True else (crosswalk or None)
    truths = list(ctx["truths"].values())
    ref = reference_truths(conn, ctx["labeller"]) if with_reference else []
    out = {"benchmark": bench, "split": split, "models": {}}
    for backbone in ctx["want"]:
        model = external.model_for(backbone)
        cn = class_names(ctx, backbone)
        m = {"classes": model.num_classes, "class_names": dict(cn.how)}
        for label, x in ((EXACT, None), (CROSSWALK, cw)):
            if label == CROSSWALK and cw is None:
                m[label] = "not used: fill the cache with mv external crosswalk"
                continue
            vocab = Vocabulary(cn, x)
            m[label] = {"split": coverage(((t, 1) for t in truths), vocab)}
            if with_reference:
                m[label]["north_american_records"] = coverage(ref, vocab)
        out["models"][backbone] = m
    return out


# --- the tables -------------------------------------------------------------------------------

def tables(ctx: dict, scorer: Scorer, backbone: str | None, ref_truths: list,
           feats: dict) -> dict:
    """The protocol's tables for one outside model, or ({name: tables}) for each Vision
    model when `backbone` is None, all judged by `scorer`."""
    eq = equiv()
    truths, vision = ctx["truths"], ctx["vision"]
    all_ids = set(truths)
    formal = {o for o, t in truths.items() if name_kind(t) == FORMAL}

    def standard(res, key, extra_formal=None):
        mine = set(res) & all_ids
        judged = heldout_report.score({key: res}, truths, scorer)[key]
        return {"all_records": {
                    **pick(ladder(res, mine, truths, scorer, False, eq)["rows"], GENUS_ROWS),
                    "family strict": family_ladder(res, mine, truths, scorer)},
                "formal_species": {
                    **ladder(res, formal & mine, truths, scorer, False, eq),
                    **(extra_formal or {}),
                    "by_true_species_reference_records": by_reference_depth(
                        {o: judged[o] for o in formal & mine if o in judged}, feats)}}

    if backbone is None:
        return {heldout_report.model_name(k): standard(v, k) for k, v in vision.items()}
    results = ctx["ext"][backbone]
    model = external.model_for(backbone)
    vocab = Vocabulary(class_names(ctx, backbone), scorer.crosswalk)
    mine = set(results) & all_ids
    in_vocab = {o for o in formal & mine if vocab.has(truths[o].label)}
    m = {"coverage": {"split": coverage(((truths[o], 1) for o in all_ids), vocab)},
         **standard(results, (backbone,), {"per_image_top1": per_image(
             results, formal & mine, truths, scorer)}),
         "same_vocabulary": {"records": len(in_vocab), "models": {}}}
    if ref_truths:
        m["coverage"]["north_american_records"] = coverage(ref_truths, vocab)
    same = m["same_vocabulary"]["models"]
    same[backbone] = {**ladder(results, in_vocab, truths, scorer, False, eq),
                      "per_image_top1": per_image(results, in_vocab, truths, scorer),
                      "macro_f1": macro_f1(results, in_vocab, truths, scorer)}
    for key, vres in vision.items():
        name = heldout_report.model_name(key)
        both = in_vocab & set(vres)
        same[name] = {**ladder(vres, both, truths, scorer, False, eq),
                      "macro_f1": macro_f1(vres, both, truths, scorer)}
        restricted, short = {}, Counter()
        for oid in both:
            restricted[oid], left = restrict(vres[oid], vocab)
            for k in KS:
                short[k] += left < k
        table = ladder(restricted, both, truths, scorer, False, eq)
        table["rows"] = pick(table["rows"], SPECIES_ROWS)
        table["fewer_names_than_k_after_restricting"] = {f"top{k}": short[k] for k in KS}
        table["macro_f1"] = macro_f1(restricted, both, truths, scorer)
        same[f"{name} restricted to {model.short}"] = table
    return m


def report(conn: sqlite3.Connection, bench: str, split: str = "dev",
           results_files: Iterable[Path] = (), vision_backbone: str = SERVED,
           models: Iterable[str] | None = None, with_reference: bool = True,
           crosswalk: bool | object = True, out_path: Path | None = None) -> dict:
    """The protocol tables (module doc) for each external model and for Vision, from the
    manifest (read-only) and any `mv external predict` JSONL files: once by exact names
    (Vision's labels) and, with a crosswalk, once more with GBIF's synonymy, the same
    judge for every model. `crosswalk`: True reads the GBIF cache (no request; fill it
    with mv external crosswalk), False skips it, or a gbif.Crosswalk to use."""
    from . import crosswalk as gbif
    ctx = load(conn, bench, split, results_files, vision_backbone, models)
    labeller, truths = ctx["labeller"], ctx["truths"]
    cw = cached_crosswalk(conn, ctx, with_reference) if crosswalk is True else (crosswalk or None)
    ref = heldout_report.reference_run(conn, bench, vision_backbone, ctx["chosen"])
    counts = ((ref or {}).get("rank_counts") or {}).get("species", {})
    feats = {o: {"species reference records": bucket(counts.get(t.species, 0), DEPTH_BUCKETS)
                 if ref and t.species else "n/a"} for o, t in truths.items()}
    ref_truths = reference_truths(conn, labeller) if with_reference else []
    formal = {o for o, t in truths.items() if name_kind(t) == FORMAL}
    scorers = {EXACT: Scorer(labeller)}
    if cw is not None:
        scorers[CROSSWALK] = Scorer(labeller, cw)
    out = {"benchmark": bench, "split": split, "records": len(ctx["records"]),
           "scored_records": len(truths), "formal_species_records": len(formal),
           "vision": {heldout_report.model_name(k): {"reference_hash": ctx["chosen"].get(k),
                                                     "records": len(v)}
                      for k, v in ctx["vision"].items()},
           "photo_only": "external models: photos only, temperature 1, mean of logits",
           "scorers": list(scorers), "models": {}, "vision_models": {}}
    if cw is not None:
        used = crosswalk_names(conn, ctx, with_reference)
        joined = [g for g in gbif.merged_groups(cw, used)
                  if len({heldout.name_key(n) for n in g}) > 1]
        out["crosswalk"] = {**cw.summary(used), "groups_joining_different_names": len(joined),
                            "sample": [list(g) for g in joined[:40]]}
    else:
        out["crosswalk"] = {"status": "not used (fill it with mv external crosswalk)"}
    for backbone in ctx["want"]:
        if backbone not in ctx["ext"]:
            continue
        model = external.model_for(backbone)
        out["models"][backbone] = {
            "hf_id": model.hf_id, "classes": model.num_classes, "licence": external.LICENCE,
            "class_names": dict(class_names(ctx, backbone).how),
            "answered": len(set(ctx["ext"][backbone]) & set(truths)),
            **{label: tables(ctx, sc, backbone, ref_truths, feats)
               for label, sc in scorers.items()}}
    for label, sc in scorers.items():
        for name, t in tables(ctx, sc, None, ref_truths, feats).items():
            out["vision_models"].setdefault(name, {})[label] = t
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


def _coverage(lines: list[str], title: str, cov: dict, indent: str = "    ") -> None:
    lines.append(f"{indent}{title} (n = {cov['records']:,} records)")
    for rank in ("species", "genus"):
        for kind, v in cov[rank].items():
            share = "-" if v["share"] is None else f"{100 * v['share']:5.1f}%"
            lines.append(f"{indent}  {rank:<8}{kind:<40}{share:>7}{v['records']:>9,}")


def _standard(lines: list[str], t: dict, indent: str = "  ") -> None:
    lines.append(f"{indent}(i) all records")
    _rows(lines, t["all_records"], indent + "  ")
    lines.append(f"{indent}(ii) formal species")
    _rows(lines, t["formal_species"]["rows"], indent + "  ")
    pi = t["formal_species"].get("per_image_top1")
    if pi:
        lines.append(f"{indent}  per-image species top 1: {_pct(pi).strip()} of "
                     f"{pi['n']:,} photos")
    lines.append(f"{indent}  species by the true species' reference records (top 1, top 5, n)")
    for band, row in t["formal_species"]["by_true_species_reference_records"].items():
        lines.append(f"{indent}    {band + ' records':<16}{_pct(row['top1'])}"
                     f"{_pct(row['top5'])}{row['n']:>8,}")


def format_report(r: dict) -> str:
    lines = [f"{r['benchmark']} / {r['split']}: {r['records']:,} records, "
             f"{r['scored_records']:,} scored, {r['formal_species_records']:,} with a formal "
             "species name. External models: photos only."]
    cw = r.get("crosswalk") or {}
    if "names" in cw:
        lines.append(f"GBIF crosswalk: {cw.get('matched', 0):,} of {cw['names']:,} names "
                     f"matched exactly (Backbone {cw.get('matched in gbif', 0):,}, Catalogue "
                     f"of Life {cw.get('matched in col', 0):,}); "
                     f"{cw['groups_joining_different_names']:,} groups join differently "
                     "written names")
    else:
        lines.append(f"GBIF crosswalk: {cw.get('status')}")
    for name, by in r["vision_models"].items():
        for label, t in by.items():
            lines += ["", f"{name}  (Vision; {label})"]
            _standard(lines, t)
    for backbone, m in r["models"].items():
        for label in r["scorers"]:
            t = m[label]
            lines += ["", f"{backbone}  ({m['hf_id']}, {m['classes']:,} classes, "
                          f"{m['licence']}; answered {m['answered']:,}; {label})"]
            lines.append("  (iv) coverage")
            for where, cov in t["coverage"].items():
                _coverage(lines, where.replace("_", " "), cov)
            _standard(lines, t)
            sv = t["same_vocabulary"]
            lines.append(f"  (iii) same vocabulary: true name in the model's vocabulary "
                         f"(n = {sv['records']:,} records)")
            for name, table in sv["models"].items():
                lines.append(f"    {name}  (n = {table['records']:,})")
                _rows(lines, table["rows"], indent="      ")
                f1 = table.get("macro_f1")
                if f1:
                    lines.append(f"      species macro-F1 (top 1) {100 * f1['macro_f1']:.1f}% "
                                 f"over {f1['classes']:,} species, n = {f1['n']:,}")
                short = table.get("fewer_names_than_k_after_restricting")
                if short and any(short.values()):
                    lines.append("      restricted list shorter than k (top k a lower bound) "
                                 "on: " + ", ".join(f"{k} {v:,}" for k, v in short.items()))
    return "\n".join(lines)
