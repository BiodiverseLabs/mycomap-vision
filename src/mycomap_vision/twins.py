"""Cross-post twins as one specimen (experiment record-scoring, E32; Steve 2026-10-18).

One find posted to both Mushroom Observer and iNaturalist is two reference records. The
record-grouped scoring that serves (methods.NearestAndMeanRecordsBalanced, arm c) counts a
specimen once among a species' best records and in its average, so the two records must be
grouped: evaluate.build_index(ref, specimen_of) puts them in one run of columns.

The pairs are errata.crosspost_twin_pairs: an MO and an iNat record with the same release
label and a photo within errata.TWIN_MAX_BITS bits of each other in the release's dHash
index (templates and an exclusion list of micrographs never pair). Groups are the connected
components over those pairs. They are computed once from a dataset release and kept as a
GROUPS FILE, a TSV with the header `observation_id<TAB>specimen` (observation_id = the
release's record_key, which is the manifest's observation id), since the served reference
and the research commands have no dHash index of their own. Every command that scores a
grouped method names the file it used and its sha256.

- `build(release, ...)`: the pairs, restricted to records in `splits`, as groups;
- `write(groups, path)` / `read(path)`: the file (LF, sorted: same groups, same sha256);
- `specimen_of(groups, records)`: the groups among these reference records only (a twin
  whose other side is not in the reference is a record like any other).
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

HEADER = ("observation_id", "specimen")


def groups_from_pairs(pairs, keep=None) -> dict[str, str]:
    """{record: specimen key} for the connected components over `pairs` ((a, b) record
    keys), keeping only pairs whose two records are in `keep` (None: all). The key is
    'twin-' + the component's smallest record key, so the same pairs give the same keys."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if keep is not None and (a not in keep or b not in keep):
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: f"twin-{find(x)}" for x in parent}


def build(release, index: dict, exclude_photos=frozenset(), skip=frozenset(),
          splits=("train",)) -> dict[str, str]:
    """The release's cross-post twin groups among its records in `splits` (errata.
    crosspost_twin_pairs: `index` its dHash index, `exclude_photos` the micrograph list,
    `skip` records already in an erratum)."""
    from . import errata
    pairs = errata.crosspost_twin_pairs(release, index, exclude_photos, skip)
    keep = {k for (k, split) in release.db.execute(
        "select record_key, split from records where included = 1") if split in set(splits)}
    return groups_from_pairs(((p.mo_record, p.inat_record) for p in pairs), keep - set(skip))


def write(groups: dict[str, str], path) -> dict:
    """Write the groups file (LF line ends, sorted), refusing to overwrite one; returns
    {file, sha256, records, groups}."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a groups file is never rewritten")
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(HEADER)
    for rec, spec in sorted(groups.items()):
        w.writerow((rec, spec))
    body = buf.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"file": path.name, "sha256": hashlib.sha256(body).hexdigest(),
            "records": len(groups), "groups": len(set(groups.values()))}


def read(path) -> dict[str, str]:
    """{observation_id: specimen} from a groups file; refuses one without both columns."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if rows and not set(HEADER) <= set(rows[0]):
        raise ValueError(f"{Path(path).name}: a twin groups file has columns "
                         f"{' and '.join(HEADER)}")
    return {r["observation_id"]: r["specimen"] for r in rows}


def cite(path) -> dict:
    """{file, sha256, records, groups} of a groups file, for a report."""
    body = Path(path).read_bytes()
    groups = read(path)
    return {"file": Path(path).name, "sha256": hashlib.sha256(body).hexdigest(),
            "records": len(groups), "groups": len(set(groups.values()))}


def specimen_of(groups: dict[str, str], records) -> dict[str, str]:
    """The groups among these reference records only: a specimen key kept for a record
    when at least one other record of its group is in the reference too (a lone twin is
    an ordinary record)."""
    present = [r.observation_id for r in records if r.observation_id in groups]
    counts: dict[str, int] = {}
    for oid in present:
        counts[groups[oid]] = counts.get(groups[oid], 0) + 1
    return {oid: groups[oid] for oid in present if counts[groups[oid]] > 1}


def is_grouped(method: str) -> bool:
    """Whether `method` scores by reference record, so twin grouping changes it (the
    record-grouped arms and their +place twins)."""
    from .methods import RECORD_METHODS
    return any(method == m.name or method == f"{m.name}+place" for m in RECORD_METHODS)
