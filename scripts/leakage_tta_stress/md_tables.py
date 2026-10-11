"""Markdown tables in Steve's standard format from report.py JSON files (for the write-up).

    python md_tables.py report.json[,report2.json] model[,model...]
"""

import json
import sys

LEVELS = ("species strict", "species s.l.", "species complex (beta)", "genus strict",
          "genus s.l.")
BANDS = ("0", "1-4", "5-19", "20-99", "100+")


def label(m):
    tag, rest = m.split(":", 1)
    return rest if tag in ("main", "test-leak", "ctrl") else f"{tag} {rest}"


def p(x):
    return "-" if x is None else f"{100 * x:.1f}"


models = {}
for path in sys.argv[1].split(","):
    models.update(json.load(open(path))["models"])
want = sys.argv[2].split(",")
print("| condition / method | " + " | ".join(f"{lv} @1/3/5/10" for lv in LEVELS) + " |")
print("|---|" + "---|" * len(LEVELS))
for m in want:
    rows = models[m]["ladder"]["rows"]
    cells = []
    for lv in LEVELS:
        r = rows[lv]
        cells.append(" / ".join(p((r.get(f"top{k}") or {}).get("rate")) for k in (1, 3, 5, 10)))
    print(f"| {label(m)} | " + " | ".join(cells) + " |")
n = {lv: models[want[0]]["ladder"]["rows"][lv]["n"] for lv in ("species strict", "genus strict")}
print(f"\n(n = {models[want[0]]['ladder']['records']:,} records; species rows "
      f"{n['species strict']:,}, genus rows {n['genus strict']:,})\n")
d0 = models[want[0]]["depth"]
print("| condition / method | " + " | ".join(f"{b} refs (n {d0[b]['n']:,}) @1 / @5" for b in BANDS)
      + " |")
print("|---|" + "---|" * len(BANDS))
for m in want:
    d = models[m]["depth"]
    print(f"| {label(m)} | " + " | ".join(
        f"{p(d[b]['top1']['rate'])} / {p(d[b]['top5']['rate'])}" for b in BANDS) + " |")
