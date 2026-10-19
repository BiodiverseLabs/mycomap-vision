"""Cross-post twins at serve time: one find posted to both iNaturalist and Mushroom Observer,
held by Vision as two reference records (2026-10-18).

Identifying an observation from a link leaves its own reference record out (identify.
Identifier.left_out). Its cross-post under the other source stayed in and matched itself:
iNat 84404214 came back as MO 444349 "100% alike" on every photo, the same four pictures.
The paper's rule finds such pairs (errata.match_twins: an MO and an iNat record with the same
label and a photo within errata.TWIN_MAX_BITS dHash bits, template hashes left out; erratum r4
and the record-grouped scoring use it). This module applies the same rule to the served
reference, so a link leaves out every twin of the observation, not just its own record:

- `build` (offline, `mv twins`): every pair among the manifest's records whose photos the
  dHash index holds, grouped into finds (a find posted three times is one group), with each
  photo's dHash and the template hashes. Saved as FILE beside the manifest; `mv release`
  ships it, and the server reads it from the served release (or MV_TWINS).
- `ReferenceTwins` (served): those groups and hashes laid over an index's columns.
  `mates` gives a record's twins; `same_picture` the records holding a photo within
  TWIN_MAX_BITS of a given dHash (a fetched observation's photos, or a reference record's own),
  whatever its label.

No photo is read here at serve time; dHashes of fetched photos come from dhash.dhash64.

The second part (below) is the record-grouped scoring's SPECIMEN GROUPING: the groups file a
dataset release gives (release_groups, `mv twin-groups`), which E32 passed under and the
served nearest+mean-records-balanced is calibrated with. Leaving twins out (above) and
grouping specimens (below) are separate jobs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import errata

FILE = "twins.npz"
RULE = "crosspost-twins-v1"     # what answers_version names: change it when the rule changes


def photo_id_of_key(key: str) -> int | None:
    """A dHash index photo key ('inat:<photo id>', 'mo:<image id>') as the manifest numbers
    the photo (mo.photo_id_of for MO), or None for another source."""
    from .mo import photo_id_of
    source, _, number = key.partition(":")
    if not number.isdigit():
        return None
    if source == "inat":
        return int(number)
    if source == "mo":
        return photo_id_of(int(number))
    return None


def _groups(pairs) -> dict[str, int]:
    """{record: group id}: records joined by any pair are one find (union-find)."""
    parent: dict[str, str] = {}

    def root(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({root(x) for x in parent})
    gid = {r: i for i, r in enumerate(roots)}
    return {x: gid[root(x)] for x in sorted(parent)}


def build(conn, dhash_photo_keys, dhash_values, *, links=(),
          source_note: dict | None = None) -> dict:
    """The twin table for a manifest's records (rule: errata.match_twins), from a dHash index's
    photo keys and hashes. `links`: (iNat number, MO number) pairs the sites link as one find
    (iNat's "Mushroom Observer URL" field, MO's link to iNat), joined into the same finds
    when both are records here: they catch a cross-post whose pictures differ. Returns what
    `save` writes."""
    hash_of: dict[int, int] = {}
    for key, h in zip(dhash_photo_keys, dhash_values):
        pid = photo_id_of_key(str(key))
        if pid is not None:
            hash_of[pid] = int(h)
    label_source = {obs: (name or "", source) for obs, name, source in conn.execute(
        "select observation_id, scientific_name, source from records")}
    photos, hash_records = [], []
    for obs, pid in conn.execute("select observation_id, photo_id from observation_photos"):
        h = hash_of.get(int(pid))
        if h is None or obs not in label_source:
            continue
        label, source = label_source[obs]
        photos.append((int(pid), obs, label, source, h))
        hash_records.append((h, obs))
    templates = errata.template_hashes(hash_records)
    del hash_records
    pairs = sorted(errata.match_twins(photos, templates))
    linked = sorted({(f"mo:{m}", str(i)) for i, m in links
                     if f"mo:{m}" in label_source and str(i) in label_source})
    groups = _groups(pairs + linked)
    held = sorted({p[0] for p in photos})
    return {"meta": {"rule": RULE, "max_bits": errata.TWIN_MAX_BITS,
                     "template_records": errata.TEMPLATE_RECORDS, "pairs": len(pairs),
                     "link_pairs": len(linked),
                     "link_only_pairs": len(set(linked) - set(pairs)),
                     "records": len(groups), "finds": len(set(groups.values())),
                     "photos": len(held), "templates": len(templates),
                     "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     **(source_note or {})},
            "photo_id": np.array(held, dtype=np.int64),
            "dhash": np.array([hash_of[p] for p in held], dtype=np.uint64),
            "templates": np.array(sorted(templates), dtype=np.uint64),
            "group_record": np.array(list(groups), dtype=str),
            "group_id": np.array(list(groups.values()), dtype=np.int64),
            "pairs": pairs}


def save(table: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(table["meta"], sort_keys=True)),
                            photo_id=table["photo_id"], dhash=table["dhash"],
                            templates=table["templates"], group_record=table["group_record"],
                            group_id=table["group_id"])
    return path


def version(path: Path | None) -> tuple:
    """What a served answer depends on here: the rule, and which table (cheap: a stat)."""
    if path is None or not Path(path).is_file():
        return (RULE, None)
    st = Path(path).stat()
    return (RULE, st.st_size, st.st_mtime_ns)


class ReferenceTwins:
    """A twin table laid over one index's columns (col_photo: each column's photo id;
    col_record: identify.Specimens, rec per column and obs per record)."""

    def __init__(self, path: Path, col_photo: np.ndarray, col_record):
        with np.load(Path(path), allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            photo_id, dhash = z["photo_id"], z["dhash"].astype(np.uint64)
            templates = z["templates"].astype(np.uint64)
            group_of = dict(zip(z["group_record"].tolist(), z["group_id"].tolist()))
        self.templates = templates
        self.col_rec = np.asarray(col_record.rec)
        col_photo = np.asarray(col_photo, dtype=np.int64)
        at = np.searchsorted(photo_id, col_photo).clip(0, max(len(photo_id) - 1, 0))
        known = (photo_id[at] == col_photo) if len(photo_id) else np.zeros(len(col_photo), bool)
        self.col_hash = np.where(known, dhash[at] if len(dhash) else 0, 0).astype(np.uint64)
        # A column whose photo is unknown or a template never matches.
        self.col_usable = known & ~np.isin(self.col_hash, templates)
        self.rec_group = np.array([group_of.get(str(o), -1) for o in col_record.obs],
                                  dtype=np.int64)

    def mates(self, recs) -> np.ndarray:
        """Every record in the same find as any of `recs` (themselves included)."""
        recs = np.asarray(recs, dtype=np.int64)
        groups = self.rec_group[recs] if len(recs) else np.empty(0, np.int64)
        groups = groups[groups >= 0]
        if not len(groups):
            return recs
        return np.union1d(recs, np.flatnonzero(np.isin(self.rec_group, groups)))

    def hashes(self, cols) -> np.ndarray:
        """The usable dHashes of these columns."""
        cols = np.asarray(cols, dtype=np.int64)
        return self.col_hash[cols[self.col_usable[cols]]] if len(cols) else np.empty(0, np.uint64)

    def same_picture(self, hashes) -> np.ndarray:
        """The records holding a photo within TWIN_MAX_BITS of any of these dHashes (one
        picture, whatever the record's label), template hashes on either side left out."""
        q = np.asarray([int(h) for h in hashes], dtype=np.uint64)
        q = q[~np.isin(q, self.templates)]
        if not len(q):
            return np.empty(0, np.int64)
        cols = np.flatnonzero(self.col_usable)
        near = np.zeros(len(cols), dtype=bool)
        for h in q:                         # a link has at most a handful of photos
            near |= np.bitwise_count(self.col_hash[cols] ^ h) <= errata.TWIN_MAX_BITS
        return np.unique(self.col_rec[cols[near]]).astype(np.int64)


def table_counts(path: Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


# --- specimen grouping (record-grouped scoring, E32) --------------------------------------
#
# Cross-post twins as one specimen (experiment record-scoring, E32; Steve 2026-10-18).
#
# One find posted to both Mushroom Observer and iNaturalist is two reference records. The
# record-grouped scoring that serves (methods.NearestAndMeanRecordsBalanced, arm c) counts a
# specimen once among a species' best records and in its average, so the two records must be
# grouped: evaluate.build_index(ref, specimen_of) puts them in one run of columns.
#
# The pairs are errata.crosspost_twin_pairs: an MO and an iNat record with the same release
# label and a photo within errata.TWIN_MAX_BITS bits of each other in the release's dHash
# index (templates and an exclusion list of micrographs never pair). Groups are the connected
# components over those pairs. They are computed once from a dataset release and kept as a
# GROUPS FILE, a TSV with the header `observation_id<TAB>specimen` (observation_id = the
# release's record_key, which is the manifest's observation id), since the served reference
# and the research commands have no dHash index of their own. Every command that scores a
# grouped method names the file it used and its sha256. The groups file is what GROUPS
# specimens (the definition E32 passed under and (c) is calibrated with); leaving a linked
# observation's twins out of its own answer is a separate job (Identifier.left_out).
#
# - `release_groups(release, ...)`: the pairs, restricted to records in `splits`, as groups;
# - `write(groups, path)` / `read(path)`: the file (LF, sorted: same groups, same sha256);
# - `specimen_of(groups, records)`: the groups among these reference records only (a twin
#   whose other side is not in the reference is a record like any other).

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


def release_groups(release, index: dict, exclude_photos=frozenset(), skip=frozenset(),
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


def partition(groups: dict[str, str], records) -> set[frozenset]:
    """The specimens `groups` makes among these reference records (specimen_of), as a set
    of record sets: two groups files define the same served method over a reference when
    their partitions are equal. Checked 2026-10-18 for the served file twin-groups built
    with r3 against E32's train file over v1 train after r4: identical, 2,092 specimens
    of 4,190 records."""
    out: dict[str, set] = {}
    for rec, key in specimen_of(groups, records).items():
        out.setdefault(key, set()).add(rec)
    return {frozenset(v) for v in out.values()}
