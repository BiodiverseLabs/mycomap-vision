"""Steve's standard summary (2026-10-09) for a saved comparison (mv compare, a trainer run).

The held-out benchmark has its own (heldout_summary.py); this is the same layout for a
comparison on a dataset release, the first official numbers of a model trained on v1:

- one table per model (backbone and method), grouped "photos only" and "photos + place
  and date" (a method that reads the record's place and date): top 1 / 3 / 5 / 10 by rows
  species strict, species s.l., species complex (beta), genus strict, genus s.l.;
- species top-1 and top-5 by the TRUE species' reference records (0, 1-4, 5-19, 20-99,
  100+);
- iNat's computer vision on the same records, or "not run";

every table with its n, and the dataset release the comparison was measured on. s.l. and
complex need the comparison run with --name-scores (evaluate's standard_equivalence);
without it those rows say so rather than showing nothing.
"""

from __future__ import annotations

import json
import sqlite3

from . import evaluate

KS = evaluate.STANDARD_K
ROWS = (("species", "strict", "species strict"), ("species", "sl", "species s.l."),
        ("species", "complex", "species complex (beta)"), ("genus", "strict", "genus strict"),
        ("genus", "sl", "genus s.l."))
NEEDS_NAME_SCORES = "not measured: run the comparison with --name-scores"


def uses_place(method: str) -> bool:
    """True for a method that scores with the record's place and date."""
    m = evaluate.METHODS.get(method)
    return bool(getattr(getattr(m, "func", m), "needs_context", False))


def model_table(report: dict) -> dict:
    """The top-k table and the depth table of one scoreboard row's report."""
    allp = report["all_photos"]
    std = allp.get("standard") or {}
    eq = allp.get("standard_equivalence")
    rows = {}
    for rank, level, label in ROWS:
        if level == "strict":
            c = (std.get(rank) or {}).get("all")
            rows[label] = ({"n": c["n"], **{f"top{k}": c.get(f"top{k}") for k in KS}}
                           if c else {"n": 0, "note": "not scored"})
        elif eq and rank in eq:
            rows[label] = {"n": eq[rank]["n"], **eq[rank][level]}
        else:
            rows[label] = {"n": 0, "note": NEEDS_NAME_SCORES}
    depth = {}
    for band in std.get("bands", []):
        c = (std.get("species") or {}).get(band)
        if c:
            depth[band] = {"n": c["n"], "top1": c.get("top1"), "top5": c.get("top5")}
    return {"rows": rows, "species_by_true_species_reference_records": depth}


def summary(conn: sqlite3.Connection, comparison_id: str) -> dict:
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    got = conn.execute("select backbone, method, n_test, n_reference, cutoff, report_json "
                       "from eval_runs where comparison_id = ? order by backbone, method",
                       (comparison_id,)).fetchall()
    if not got:
        raise ValueError(f"no comparison {comparison_id} on the scoreboard")
    reports = [(b, m, n_test, n_ref, cutoff, json.loads(r)) for b, m, n_test, n_ref, cutoff, r
               in got]
    releases = {json.dumps(r.get("dataset_release"), sort_keys=True) for *_, r in reports}
    if len(releases) > 1:
        raise ValueError(f"comparison {comparison_id} mixes dataset releases: {releases}")
    out = {"comparison_id": comparison_id, "dataset_release": reports[0][5].get("dataset_release"),
           "test_records": reports[0][2], "reference_records": reports[0][3],
           "cutoff": reports[0][4], "photos_only": {}, "photos_and_place": {},
           "inat_same_records": {"status": "not run"}}
    for b, m, *_rest, rep in reports:
        if b.startswith("external:inat"):
            out["inat_same_records"] = {"status": "run", "model": f"{b} / {m}",
                                        **model_table(rep)}
            continue
        group = "photos_and_place" if uses_place(m) else "photos_only"
        out[group][f"{b} / {m}"] = model_table(rep)
    return out


def _pct(v) -> str:
    return f"{'-':>8}" if v is None else f"{100 * v:7.1f}%"


def format_summary(s: dict) -> str:
    rel = s.get("dataset_release") or {}
    head = (f"comparison {s['comparison_id']}: {s['test_records']:,} test records against "
            f"{s['reference_records']:,} reference records (cutoff {s['cutoff']})")
    if rel:
        head += (f"\ndataset release {rel['dataset_release']}"
                 f"{' (DRAFT: exploratory only)' if rel.get('draft') else ''}, release_hash "
                 f"{rel['release_hash'][:12]}, reference_hash {rel['reference_hash'][:12]}")
    else:
        head += "\nNOT on a dataset release (live manifest): exploratory only"
    lines = [head]
    for group, title in (("photos_only", "Photos only"),
                         ("photos_and_place", "Photos + place and date")):
        lines += ["", f"== {title} =="]
        if not s[group]:
            lines.append("  (no method of this kind in the comparison)")
        for model, t in s[group].items():
            lines += ["", f"{model}"]
            lines.append(f"    {'':<24}" + "".join(f"{'top ' + str(k):>8}" for k in KS)
                         + f"{'n':>9}")
            for label, row in t["rows"].items():
                if "note" in row:
                    lines.append(f"    {label:<24}  {row['note']}")
                else:
                    lines.append(f"    {label:<24}" + "".join(_pct(row.get(f"top{k}"))
                                                           for k in KS) + f"{row['n']:>9,}")
            lines.append("    species, by the true species' reference records")
            lines.append(f"    {'':<24}{'top 1':>8}{'top 5':>8}{'n':>9}")
            for band, row in t["species_by_true_species_reference_records"].items():
                lines.append(f"    {band + ' records':<24}{_pct(row['top1'])}{_pct(row['top5'])}"
                             f"{row['n']:>9,}")
    lines.append("")
    inat = s["inat_same_records"]
    lines.append("iNat CV on the same records: " + ("not run" if inat["status"] != "run"
                                                    else inat["model"]))
    return "\n".join(lines)
