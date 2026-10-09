"""The reference records that are not iNat records, by rule, from a read-only export of .org's
`observations` (observation_id <TAB> source per row, every row, not only green ones):

- every .org row with the record's id has a source other than 'iNaturalist' (Mushroom
  Observer, MycoPortal / MyCoPortal, .com Sequences): Vision fetched the iNat observation
  that happens to share the number; or
- no .org row has the id any more, and the iNat observation's date differs from the
  record's (a re-keyed legacy record whose iNat look-alike is unrelated).

Writes <data>/audits/non-fungus-scan/non-inat-reference-records-rule-<date>.tsv
(observation_id, reason). It never touches the shared list the label audit, the
record-sources fix and the other lanes read (<data>/audits/non-inat-reference-records-
2026-10-09.tsv, whose sha256 they record); on the 2026-10-09 snapshot the rule gives that
list's ids less one record that isn't in the snapshot's references.
Usage: python mislinked_list.py <manifest> <org-sources.tsv> [date]
"""
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from zs_score import OUT


def build(manifest: Path, org_sources: Path) -> dict[str, str]:
    rows = pickle.load(open(OUT / "frame.pkl", "rb"))["rows"]
    recs = {r[0] for r in rows}
    live = defaultdict(set)
    for line in org_sources.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0]:
            live[parts[0]].add(parts[1])
    conn = sqlite3.connect(Path(manifest).resolve().as_uri() + "?mode=ro", uri=True)
    same_day = {o: bool(a and b and a[:10] == b[:10]) for o, a, b in conn.execute(
        "select r.observation_id, r.observed_on, o.observed_on from records r "
        "left join inat_observations o on o.observation_id = r.observation_id")}
    out = {}
    for o in recs:
        s = live.get(o)
        if s and "iNaturalist" not in s:
            out[o] = "org source: " + "+".join(sorted(s))
        elif not s and not same_day.get(o, False):
            out[o] = "not on .org now; iNat date differs"
    return out


def main() -> None:
    manifest, org_sources = Path(sys.argv[1]), Path(sys.argv[2])
    date = sys.argv[3] if len(sys.argv) > 3 else "2026-10-09"
    out = build(manifest, org_sources)
    path = OUT / f"non-inat-reference-records-rule-{date}.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("observation_id\treason\n")
        for o in sorted(out, key=lambda x: (len(x), x)):
            f.write(f"{o}\t{out[o]}\n")
    print(f"{len(out):,} records -> {path}", Counter(out.values()).most_common())


if __name__ == "__main__":
    main()
