"""The review list: every reference photo whose zero-shot top class is not a fungus (or a
kept class: slip, microscope, habitat), with its record, label, score, the record's
.org source and whether it is the record's only photo. A list for people to review, not a
filter. Written to data/audits/non-fungus-scan/ (private, never committed); prints the
counts the experiment entry reports.
Usage: python review_list.py [manifest] [org-sources.tsv]
"""
import csv
import json
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mycomap_vision import config
from mycomap_vision.heldout_report import DEPTH_BUCKETS, bucket

from srcmap import load_sources, source_of
from zs_score import OUT

KINGDOMS = ((47170, "Fungi"), (47686, "Protozoa"), (3, "Aves"), (40151, "Mammalia"),
            (47158, "Insecta"), (1, "other animals"), (47126, "Plantae"), (48222, "Chromista"))


def main() -> None:
    F = pickle.load(open(OUT / "frame.pkl", "rb"))
    C, rows = F["classes"], F["rows"]
    P = F["bioclip-2"]
    J = [C.index(c) for c in F["junk"]]
    top = P.argmax(1)
    pj = P[:, J].sum(1)
    flagged = np.isin(top, J)
    manifest = sys.argv[1] if len(sys.argv) > 1 else config.DATA_DIR / "manifest.sqlite"
    src = load_sources(sys.argv[2] if len(sys.argv) > 2 else OUT / "org-green-source.tsv")
    conn = sqlite3.connect(Path(manifest).resolve().as_uri() + "?mode=ro", uri=True)
    kingdom = {}
    for oid, anc, tid in conn.execute("select observation_id, taxon_ancestor_ids, taxon_id "
                                      "from inat_observations where status = 'ok'"):
        ids = set(json.loads(anc)) if anc else set()
        if tid:
            ids.add(tid)
        kingdom[oid] = next((n for k, n in KINGDOMS if k in ids), "other" if ids else "none")
    by_rec = defaultdict(list)
    for i, r in enumerate(rows):
        by_rec[r[0]].append(i)
    n_flag = {o: int(flagged[ix].sum()) for o, ix in by_rec.items()}
    from rescore import mislinked as agreed
    mislinked = agreed() & set(by_rec)

    path = OUT / "review-list-2026-10-09.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "observation_id", "label", "genus", "label_reference_records",
                    "depth_band", "top_class", "top_class_score", "p_not_fungus",
                    "record_photos", "record_flagged_photos", "only_photo",
                    "record_loses_all_photos", "org_source", "mislinked_record",
                    "inat_observation_kingdom", "license_class"])
        order = sorted(np.flatnonzero(flagged), key=lambda i: (rows[i][0] not in mislinked, -pj[i]))
        for i in order:
            oid, pid, _k, nph, sp, ge, _fa, unit, cnt, _pos, lic = rows[i]
            w.writerow([pid, oid, unit, ge, cnt, bucket(cnt, DEPTH_BUCKETS), C[top[i]],
                        round(float(P[i, top[i]]), 4), round(float(pj[i]), 4), nph, n_flag[oid],
                        nph == 1, n_flag[oid] == nph, source_of(src, oid), oid in mislinked,
                        kingdom.get(oid, "none"), lic])
    print(f"wrote {path}")

    def tally(mask, name):
        ix = np.flatnonzero(mask)
        recs = {rows[i][0] for i in ix}
        lose = {o for o in recs if n_flag[o] == len(by_rec[o])}
        print(f"\n== {name}: {len(ix):,} photos in {len(recs):,} records; "
              f"{len(lose):,} records would lose ALL their photos")
        print("  by class:", Counter(C[top[i]] for i in ix).most_common())
        band = Counter(bucket(rows[i][8], DEPTH_BUCKETS) for i in ix)
        print("  by label depth band (photos):", [(b, band[b]) for *_x, b in DEPTH_BUCKETS])
        rb = Counter(bucket(rows[by_rec[o][0]][8], DEPTH_BUCKETS) for o in lose)
        print("  records losing all photos, by band:", [(b, rb[b]) for *_x, b in DEPTH_BUCKETS])
        print("  only photo of its record:", sum(rows[i][3] == 1 for i in ix))
    is_mis = np.array([r[0] in mislinked for r in rows])
    tally(flagged, "all flagged")
    tally(flagged & ~is_mis, "flagged, genuine iNat records")
    tally(flagged & is_mis, "flagged, mislinked (non-iNat) records")
    mrecs = {o for o in mislinked}
    print(f"\n== mislinked records: {len(mrecs):,} records, {int(is_mis.sum()):,} photos; "
          "by .org source:", Counter(source_of(src, o) for o in mrecs).most_common())
    print("  iNat observation kingdom:", Counter(kingdom.get(o, "none") for o in mrecs).most_common())
    band = Counter(bucket(rows[by_rec[o][0]][8], DEPTH_BUCKETS) for o in mrecs)
    print("  by label depth band (records):", [(b, band[b]) for *_x, b in DEPTH_BUCKETS])
    units = Counter(rows[by_rec[o][0]][7] for o in mrecs)
    gone = [u for u, n in units.items()
            if n == sum(1 for o in by_rec if rows[by_rec[o][0]][7] == u)]
    print(f"  labels whose every reference record is mislinked: {len(gone):,}")


if __name__ == "__main__":
    main()
