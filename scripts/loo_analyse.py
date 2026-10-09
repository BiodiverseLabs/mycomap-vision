"""Flag and rank the leave-one-out scan's records (exp/loo-mislabel-scan).

    python scripts/loo_analyse.py --scan DIR/scan --audits data/audits --out DIR

Reads the scan's chunks (scripts/loo_scan.py), applies loo.classify with loo.Rules, tags
records other lanes already handle, and writes review lists into --out (under data/,
never committed): review-a-wrong-label.csv, review-b-name-pairs.csv (pairs) and
review-b-records.csv, review-c-wrong-photos.csv, review-d-look-alikes.csv (pairs),
all-records.csv.gz, the pre-registered removal lists, and summary.json.
No coordinates, no observer names.
"""

from __future__ import annotations

import sys

sys.modules["torch"] = None      # noqa: E402

import argparse
import csv
import gzip
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mycomap_vision import config, loo, name_equiv
from mycomap_vision.heldout import name_key

KS = {"top1pct": 0.01, "top2pct": 0.02}


def load_scan(scan: Path, partial: bool = False):
    with open(scan / "layout.pkl", "rb") as f:
        state = pickle.load(f)
    parts = defaultdict(list)
    for p in sorted((scan / "chunks").glob("chunk-*.npz")):
        with np.load(p) as z:
            for k in z.files:
                parts[k].append(z[k])
    arr = {k: np.concatenate(v) for k, v in parts.items()}
    n = state["layout"].n_records
    order = np.argsort(arr["rec"])
    complete = len(order) == n and np.array_equal(arr["rec"][order], np.arange(n))
    if not complete and not partial:
        raise SystemExit(f"scan incomplete: {len(order):,} of {n:,} records")
    rec = {}
    for k, v in arr.items():
        if k.startswith("photo_"):
            continue
        full = np.zeros((n,) + v.shape[1:], dtype=v.dtype)
        if k in ("top_unit", "nb_rec", "gen_top"):
            full[:] = 0
        full[arr["rec"]] = v
        rec[k] = full
    rec["scored"] = np.zeros(n, dtype=bool)
    rec["scored"][arr["rec"]] = True
    photo = {k[6:]: v for k, v in arr.items() if k.startswith("photo_")}
    return state, rec, photo


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (missing: {path})")
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ","))


def tags_from(audits: Path) -> tuple[dict, dict, dict]:
    """({oid: [tags]}, {oid: live .com title}, {frozenset(name pair): kind}) from the lists
    other lanes made on 2026-10-09."""
    tags: dict[str, list[str]] = defaultdict(list)
    for r in read_tsv(audits / "non-inat-reference-records-2026-10-09.tsv"):
        tags[r["observation_id"]].append("non-iNat id: wrong iNat photos (fix lane)")
        SOURCE_OF[r["observation_id"]] = source_key(r.get("reason", ""))
    live = {}
    for r in read_tsv(audits / "label-audit-2026-10-09" / "ref_vs_live_title.tsv"):
        if r["category"] not in ("same", "missing on .com"):
            live[r["oid"]] = r["live_com_title"]
            tags[r["oid"]].append(f"label audit: live .com title differs ({r['category']})")
    for r in read_tsv(audits / "label-audit-2026-10-09" / "com_title_vs_number_name.tsv"):
        # .org lags because .com's index number didn't follow the title: fixed on .com
        # (number follows title) and a re-pull, not by relabelling Vision (label audit).
        if r.get("set") == "ref" and r.get("diff") not in ("", "same", None):
            tags[r["oid"]].append("label audit: .com number names another name -> .com fix "
                                  "+ re-pull")
    for r in read_tsv(audits / "label-audit-2026-10-09" / "identical-photo-different-label.tsv"):
        tags[r["oid"]].append("label audit: identical photo under another label")
    for r in read_tsv(audits / "label-audit-2026-10-09" / "ref_dna_genus_disagree.tsv"):
        tags[r["oid"]].append("label audit: DNA sequence names another genus")
    best = defaultdict(float)
    for r in read_tsv(audits / "non-fungus-scan" / "review-list-2026-10-09.csv"):
        if r.get("mislinked_record") == "True":       # the record-source fix lane's
            continue
        try:
            best[r["observation_id"]] = max(best[r["observation_id"]], float(r["p_not_fungus"]))
        except (KeyError, ValueError):
            pass
    for oid, p in best.items():
        tags[oid].append(f"non-fungus scan: photo flagged (p {p:.2f})")
    pairs = {}
    for r in read_tsv(audits / "label-audit-2026-10-09" / "label-pairs.tsv"):
        a, b = r.get("label_a (keep / live)") or r.get("label_a"), r.get("label_b")
        if a and b:
            pairs[frozenset((a, b))] = r["kind"]
    for r in read_tsv(audits / "label-audit-2026-10-09" / "identical-its-label-pairs.tsv"):
        pairs.setdefault(frozenset((r["main_label"], r["other_label"])), "identical ITS")
    for r in read_tsv(audits / "label-audit-2026-10-09" / "hard-synonym-labels.tsv"):
        a = r.get("reference_label (hard synonym on .com)")
        b = (r.get("accepted name") or "").strip('"').replace('""', '"')
        if a and b:
            pairs.setdefault(frozenset((a, b)), "hard synonym on .com")
    return tags, live, pairs


SOURCE_OF: dict[str, str] = {}
# The record-source fix lane's records.source names, from the .org source in its list.
SOURCES = (("mo observations", "mo"), ("mycoportal", "mycoportal"), ("sequences", "com_sequence"),
           ("genbank", "genbank"), ("inaturalist", "inat"))
# Who fixes a record flagged on several lists (agreed across the three review lists):
# first match wins; every list also names the others.
OWNERS = (("non-iNat id", "record-source fix"), ("non-fungus scan", "non-fungus scan"),
          ("label audit", "label audit"), ("STALE LABEL", "label audit"))


def source_key(reason: str) -> str:
    low = reason.lower()
    return next((key for word, key in SOURCES if word in low), "unknown")


def record_key(oid: str) -> str:
    """'<source>:<id>' as the record-source fix lane names sources; the snapshot stores
    every id bare and treats it as iNat."""
    return f"{SOURCE_OF.get(oid, 'inat')}:{oid}"


def owner_of(tags: list[str], category: str) -> tuple[str, str]:
    """(owner lane, the other lanes that flag it). The non-fungus scan's flags on genuine
    iNat records are candidates (~14% hand-checked precision): that lane owns a record
    only when this scan also calls its photos wrong (c); otherwise it is listed as a
    candidate (agreed with that lane, 2026-10-09)."""
    lanes = []
    for t in tags:
        lane = next((lane for prefix, lane in OWNERS if t.startswith(prefix)), None)
        if lane and lane not in lanes:
            lanes.append(lane)
    order = [lane for _p, lane in OWNERS]
    lanes.sort(key=order.index)
    candidate = "non-fungus scan" in lanes and category != "c"
    if candidate:
        lanes.remove("non-fungus scan")
    owner = lanes[0] if lanes else "loo scan"
    also = lanes[1:] + (["non-fungus scan (candidate)"] if candidate else [])
    return owner, "; ".join(also)


def same_name(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return name_key(a) == name_key(b) or name_equiv.species_match(a, b)["sl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--audits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--partial", action="store_true", help="smoke test on the chunks so far")
    a = ap.parse_args()
    state, R, P = load_scan(a.scan, a.partial)
    scored = R["scored"]
    layout, recs, labels, meta = state["layout"], state["records"], state["labels"], state["meta"]
    rules = loo.Rules()
    N = layout.n_records
    units = layout.units
    unit_species = np.asarray(labels["unit_species"])
    unit_genus = np.asarray(labels["unit_genus"])
    genera = labels["genus"]
    oid = recs["observation_id"]
    lab_unit = layout.rec_unit
    label = np.where((unit_species[lab_unit] >= 0) & scored, lab_unit, -1)
    pred = np.where(scored, R["top_unit"][:, 0], -1)
    # Records per unit, and per record the label's records outside its observer-day group.
    support = Counter(lab_unit.tolist())
    in_group = Counter(zip(layout.rec_group.tolist(), lab_unit.tolist()))
    label_left = np.asarray([support[u] - in_group[(g, u)]
                             for g, u in zip(layout.rec_group.tolist(), lab_unit.tolist())])
    rates, counts = loo.pair_rates(label, pred)
    sp = label >= 0
    n_scored = Counter(label[sp].tolist())
    self_rate = {u: c / n_scored[u] for u, c in Counter(label[sp & (pred == label)].tolist()).items()}
    for u in set(label[sp].tolist()) - set(self_rate):
        self_rate[u] = 0.0
    nb_units = lab_unit[R["nb_rec"]]                          # (N, 10)
    nb_pred = (nb_units == pred[:, None]).sum(axis=1)
    nb_label = (nb_units == lab_unit[:, None]).sum(axis=1)
    # Photos: distinct top genera among photos that match something well (descriptive), and
    # the best match of any photo whose best record is in another genus (rule c).
    photo_genera = np.zeros(N, dtype=int)
    good = P["best_sim"] >= rules.no_match
    seen = defaultdict(set)
    for r, g, ok in zip(P["rec"].tolist(), P["top_genus"].tolist(), good.tolist()):
        if ok and g >= 0:
            seen[r].add(g)
    for r, s in seen.items():
        photo_genera[r] = len(s)
    photo_genera = np.maximum(photo_genera, 1)
    rec_genus = unit_genus[lab_unit]
    pg_label, pg_best = rec_genus[P["rec"]], rec_genus[P["best_rec"]]
    other = (pg_label >= 0) & (pg_best >= 0) & (pg_label != pg_best)
    dup_other = np.zeros(N)
    np.maximum.at(dup_other, P["rec"][other], P["best_sim"][other])
    dup_with = {}
    for r, b, sim in zip(P["rec"][other].tolist(), P["best_rec"][other].tolist(),
                         P["best_sim"][other].tolist()):
        if sim >= rules.duplicate and sim >= dup_other[r] - 1e-6:
            dup_with[r] = units[lab_unit[b]]
    tags, live, audit_pairs = tags_from(a.audits)
    relation_cache: dict[tuple, str] = {}

    def relation(u, v):
        key = (u, v)
        if key not in relation_cache:
            relation_cache[key] = loo.related_names(units[u], units[v], name_key, name_equiv)
        return relation_cache[key]

    rows = []
    cat = np.full(N, "", dtype=object)
    stren = np.zeros(N)
    for i in np.flatnonzero(scored).tolist():
        u, v = int(label[i]), int(pred[i])
        margin = (float(R["top_score"][i, 0] - R["own_score"][i])
                  if np.isfinite(R["own_score"][i]) else float("inf"))
        rd = {"label": u, "pred": v, "conf": float(R["top_conf"][i, 0]), "margin": margin,
              "nb_pred": int(nb_pred[i]), "best_match": float(R["best_match"][i]),
              "dup_other_genus": float(dup_other[i]), "label_left": int(label_left[i])}
        rel = relation(u, v) if u >= 0 and v >= 0 and u != v else ""
        c = loo.classify(rd, rules, rates, counts, self_rate, support, rel)
        cat[i] = c
        if c == "a":
            stren[i] = loo.strength(rd, rates, self_rate)
        elif c == "c":
            stren[i] = loo.photo_strength(rd, rules)
        lab_name = units[lab_unit[i]]
        sug = units[v] if v >= 0 else ""
        lt = live.get(oid[i], "")
        t = list(tags.get(oid[i], []))
        if lt and sug and same_name(lt, sug):
            t.append("STALE LABEL: live .com title already = suggestion")
        pk = audit_pairs.get(frozenset((lab_name, sug)))
        if pk:
            t.append(f"label audit pair: {pk}")
        owner, also = owner_of(t, c) if c else ("", "")
        rows.append({
            "record_key": record_key(oid[i]), "observation_id": oid[i],
            "owner_lane": owner, "also_flagged_by": also, "category": c, "strength": round(stren[i], 4),
            "current_label": lab_name, "suggested_name": sug,
            "confidence": round(float(R["top_conf"][i, 0]), 4),
            "margin": round(margin, 4) if np.isfinite(margin) else "label has no other records",
            "label_rank": int(R["own_rank"][i]), "label_confidence": round(float(R["own_conf"][i]), 4),
            "nearest10_suggested": int(nb_pred[i]), "nearest10_label": int(nb_label[i]),
            "label_records_elsewhere": int(label_left[i]),
            "label_recognised_rate": round(self_rate.get(u, 0.0), 3) if u >= 0 else "",
            "suggested_records": support.get(v, 0) if v >= 0 else 0,
            "suggested_taken_for_label_rate": round(rates.get((v, u), 0.0), 3),
            "label_taken_for_suggested_rate": round(rates.get((u, v), 0.0), 3),
            "name_relation": relation(u, v) if u >= 0 and v >= 0 and u != v else "",
            "best_match": round(float(R["best_match"][i]), 4),
            "worst_photo_match": round(float(R["worst_match"][i]), 4),
            "photos": int(R["n_photos"][i]), "photo_genera": int(photo_genera[i]),
            "photo_duplicate_in_other_genus": round(float(dup_other[i]), 4)
            if dup_other[i] >= rules.duplicate else "",
            "duplicate_of_record_named": dup_with.get(i, ""),
            "wrong_photo_reason": ("" if c != "c" else "duplicate photo in another genus"
                                   if dup_other[i] >= rules.duplicate else "matches nothing"),
            "label_genus_rank": int(R["own_gen_rank"][i]),
            "top_genus": genera[R["gen_top"][i, 0]],
            "alternatives": "; ".join(f"{units[x]} {R['top_conf'][i, j]:.2f}"
                                      for j, x in enumerate(R["top_unit"][i, 1:5].tolist(), 1)
                                      if x >= 0),
            "live_com_title": lt, "known_issue": " | ".join(t),
        })
    a.out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with gzip.open(a.out / "all-records.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

    def write(name, items, keys=fields):
        with open(a.out / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)

    by = defaultdict(list)
    for r in rows:
        if r["category"]:
            by[r["category"]].append(r)
    for c in by:
        by[c].sort(key=lambda r: -r["strength"] if c in "ac" else -r["confidence"])
    write("review-a-wrong-label.csv", by["a"])
    write("review-c-wrong-photos.csv", by["c"])
    write("review-b-records.csv", by["b"])
    write("review-d-records.csv", by["d"])
    # Pairs: label A -> suggested B, with both directions' rates.
    def pairs_of(c):
        agg = defaultdict(list)
        for r in by[c]:
            agg[(r["current_label"], r["suggested_name"])].append(r)
        out = []
        pos = {u: i for i, u in enumerate(units)}
        for (la, sb), rs in agg.items():
            ua, ub = pos[la], pos[sb]
            out.append({"label_a": la, "suggested_b": sb, "flagged_records": len(rs),
                        "records_a": support[ua], "records_b": support[ub],
                        "a_taken_for_b": round(rates.get((ua, ub), 0.0), 3),
                        "b_taken_for_a": round(rates.get((ub, ua), 0.0), 3),
                        "a_recognised": round(self_rate.get(ua, 0.0), 3),
                        "b_recognised": round(self_rate.get(ub, 0.0), 3),
                        "name_relation": rs[0]["name_relation"],
                        "label_audit_pair": audit_pairs.get(frozenset((la, sb)), ""),
                        "stale_title_records": sum("STALE LABEL" in r["known_issue"] for r in rs)})
        out.sort(key=lambda p: (-p["flagged_records"], p["label_a"]))
        return out
    pb, pd = pairs_of("b"), pairs_of("d")
    write("review-b-name-pairs.csv", pb, list(pb[0]) if pb else ["label_a"])
    write("review-d-look-alikes.csv", pd, list(pd[0]) if pd else ["label_a"])
    # Photo pairs: a photo whose best match (>= rules.duplicate) is a record of another genus.
    pos_in_rec = np.zeros(len(P["rec"]), dtype=int)
    starts = np.r_[0, np.flatnonzero(np.diff(P["rec"])) + 1]
    for s0, s1 in zip(starts, np.r_[starts[1:], len(P["rec"])]):
        pos_in_rec[s0:s1] = np.arange(s1 - s0)
    dup_rows = []
    for k in np.flatnonzero(other & (P["best_sim"] >= rules.duplicate)).tolist():
        r, b = int(P["rec"][k]), int(P["best_rec"][k])
        dup_rows.append({"record_key": record_key(oid[r]),
                         "photo_id": recs["photo_ids"][r][pos_in_rec[k]],
                         "label": units[lab_unit[r]], "other_record_key": record_key(oid[b]),
                         "other_label": units[lab_unit[b]],
                         "cosine": round(float(P["best_sim"][k]), 4)})
    dup_rows.sort(key=lambda d: -d["cosine"])
    write("duplicate-photos-other-genus.csv", dup_rows,
          ["record_key", "photo_id", "label", "other_record_key", "other_label", "cosine"])
    # Pre-registered removal lists: (a) + (c) by strength, top 1% / 2% of reference records.
    ranked = sorted((r for r in rows if r["category"] in ("a", "c")),
                    key=lambda r: -r["strength"])
    removal = {}
    for name, share in KS.items():
        k = int(round(share * N))
        removal[name] = [r["observation_id"] for r in ranked[:k]]
        clean = [r for r in ranked if "non-iNat id" not in r["known_issue"]]
        removal[name + "-skip-noninat"] = [r["observation_id"] for r in clean[:k]]
    for name, ids in removal.items():
        (a.out / f"remove-{name}.txt").write_text("\n".join(ids) + "\n")

    def tag_counts(items):
        c = Counter()
        for r in items:
            for t in (r["known_issue"].split(" | ") if r["known_issue"] else ["(no known issue)"]):
                c[t.split(" (")[0] if not t.startswith("label audit pair") else
                  "label audit pair"] += 1
        return dict(c.most_common())
    summary = {
        "meta": meta, "rules": loo.Rules().__dict__,
        "code_version_analysis": config.code_version(),
        "reproducibility": "exploratory-pre-freeze",
        "owners": {c: dict(Counter(r["owner_lane"] for r in by[c]).most_common()) for c in "abcd"},
        "records": N, "scored": int(scored.sum()), "species_labelled": int(sp.sum()),
        "loo_species_top1_agrees": int((sp & (pred == label)).sum()),
        "categories": {c: len(by[c]) for c in "abcd"},
        "category_tags": {c: tag_counts(by[c]) for c in "abcd"},
        "pairs": {"b": len(pb), "d": len(pd)},
        "top_b_pairs": pb[:25], "top_d_pairs": pd[:15],
        "removal_sizes": {k: len(v) for k, v in removal.items()},
        "a_strength_quantiles": (np.quantile([r["strength"] for r in by["a"]],
                                             [0.1, 0.5, 0.9]).round(3).tolist()
                                 if by["a"] else []),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("records", "species_labelled",
                                              "loo_species_top1_agrees", "categories",
                                              "pairs", "removal_sizes",
                                              "category_tags")}, indent=2))


if __name__ == "__main__":
    main()
