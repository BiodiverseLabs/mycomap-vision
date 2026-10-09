"""Held-out benchmarks: DNA-validated records Vision has not seen, scored honestly.

The first is heldout-2026-10-08: 13,145 North American iNat records green on legacy
mycomap.com that a broken .com -> .org sync kept out of Vision, so the full-run model
(bioclip-2-ft-20261007-165400, trained through 2026-09-07) and its reference index
never saw them. Steve (2026-10-08): it is a development benchmark, not the paper's
test set. Its records will join training and the reference index after a relabel
and retrain; it measures Vision before that (and anything tuned on dev). The paper
gets a fresh, smaller set, frozen as a sealed benchmark (`--holdout`).

    mv heldout freeze   store a set from frozen CSVs: ids, the answer key, place, the
                        dev/test split, the inputs' hashes; with --holdout, also hold
                        every id out of training and the reference index (holdouts.py)
    mv heldout fetch    iNat details, and photos at LARGE (the size the full-run model
                        trained and indexed on) into the benchmark's own folder
    mv heldout predict  embed them and identify each record with Vision's index
    mv heldout inat     iNat's computer vision on a subsample (inat_cv.InatClient)
    mv heldout report   scores, CIs, paired tests, calibration, breakdowns, label audit
                        (heldout_report.py)

The answer key is the observation's name on .com: the record title's name
(cms_custom_database_43.field_483, "<iNat id> - <name>"). pool.csv's com_name is
.com's index name, which lags the title on some records (373882777: index
"Polyporales", title "Trametes gibbosa"), so where a titles export says the title
differs, the title is the answer and the index name is kept for the audit. Names are
compared by Vision's labels with writing differences (provisional-code formats, a
missing `var.`) set aside. A title above species (one word) is not scored at species
and is flagged in the label audit as possibly stale, for a .com refresh; `freeze`
runs again on a new snapshot of the same ids and takes the new names (each freeze is
logged in heldout_freezes).

The split: dev (3,000: the 100-record pilot and 2,900 at random) to tune and explore,
and test (10,145). On a sealed benchmark (held out), `mv heldout report --split test`
is the paper's number: it says so loudly and every such report is recorded
(heldout_test_looks). On a development benchmark test is not sealed.

Leakage guards:
- a sealed benchmark: freeze refuses an id the records table holds, and every id is
  held out from then on (holdouts.py: never stored, loaded, downloaded, embedded,
  released, shipped to a trainer or added at night);
- the benchmark's photos, vectors and answers live in its own tables and folder
  (<manifest folder>/benchmarks/<name>/), never in photo_copies or embeddings. Its
  iNat details (uuid, public place, date, photo list) also go where every record's
  do (inat_observations, inat.save_batch): inert without a records row, they let an
  occurrence store leave each benchmark observation out by uuid, and a development
  benchmark's records, once exported, need no second fetch;
- predict refuses a reference index holding a record of a sealed benchmark, and
  skips a record of a development benchmark that is already a reference record, or
  whose photo is a reference photo;
- each answer keeps the reference it was made against (a hash of the reference
  records and their labels), so a "before" run stays apart from an "after" one and
  the report scores one reference per model.

Photos are used for scoring only, never shown. All-rights-reserved photos of a
photographer who withdrew permission on mycomap.org are neither downloaded nor used
(permissions.py). No answer, report or CSV carries a coordinate.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import config, guests, holdouts, names, taxonomy
from .dates import real_date
from .ratelimit import MinInterval

RANKS = ("family", "genus", "species")
SCHEMA = """
create table if not exists heldout_sets (
  name          text primary key,
  records       integer not null,
  ids_sha256    text not null,      -- sha256 of the newline-joined ids, sorted as numbers
  inputs_json   text not null,      -- the latest freeze's input files: path, bytes, sha256
  truth_rule    text not null,
  frozen_at     text not null,      -- the first freeze
  refrozen_at   text,               -- the latest freeze on a new snapshot
  code_version  text
);
-- Every freeze of a set (the first, and each new snapshot of the same ids).
create table if not exists heldout_freezes (
  benchmark     text not null,
  frozen_at     text not null,
  inputs_json   text not null,
  names_changed integer not null,
  code_version  text
);
create table if not exists heldout_records (
  benchmark       text not null,
  observation_id  text not null,
  truth_name      text,             -- the answer key: the .com record title's name
  truth_status    text not null,    -- label-audit category (TRUTH_* below)
  index_name      text,             -- .com's index name (pool.csv com_name), for the audit
  name_source     text not null,    -- where the answer came from (SOURCE_* below)
  org_name        text,
  latitude real, longitude real,    -- .org's coordinates: true ones, never shown
  country text, first_validated text, projects text,
  split           text,             -- 'dev' (tune and explore) | 'test'; null: none
  primary key (benchmark, observation_id)
);
-- Every report on the test split of a sealed benchmark, so how often it was looked at
-- can be shown.
create table if not exists heldout_test_looks (
  benchmark     text not null,
  looked_at     text not null,
  code_version  text,
  models_json   text not null,      -- the backbones and methods scored
  records       integer not null,
  report_path   text
);
create table if not exists heldout_observations (
  benchmark       text not null,
  observation_id  text not null,
  status          text not null,    -- 'ok' | 'missing'
  uuid text, observed_on text, inat_latitude real, inat_longitude real, obscured integer,
  geoprivacy text, user_id integer, user_login text, photo_count integer,
  fetched_at      text not null,
  primary key (benchmark, observation_id)
);
create table if not exists heldout_photos (
  benchmark       text not null,
  observation_id  text not null,
  photo_id        integer not null,
  position        integer,
  license_code    text,
  license_class   text not null,
  attribution     text,
  owner_user_id   integer,
  owner_login     text,
  source_url      text not null,
  host            text,
  status          text not null default 'pending',  -- pending | done | missing | error
  store text, size text, path text, bytes integer, sha256 text,
  attempts        integer not null default 0,
  error text, downloaded_at text,
  primary key (benchmark, observation_id, photo_id)
);
create table if not exists heldout_runs (
  benchmark       text not null,
  backbone        text not null,
  method          text not null,
  reference_hash  text not null,    -- the reference records and their labels (see predict)
  created_at      text not null,
  code_version    text,
  reference_json  text not null,    -- sizes, hashes, records per name, observer-days
  primary key (benchmark, backbone, method, reference_hash)
);
create table if not exists heldout_predictions (
  benchmark       text not null,
  observation_id  text not null,
  backbone        text not null,    -- or external:inat-cv
  method          text not null,
  reference_hash  text not null,    -- '' for iNat
  predicted_at    text not null,
  code_version    text,
  reference_records integer,
  photos          integer,
  place           text,             -- the place and date given: inat | org | none
  result_json     text not null,    -- per rank: the top 5 with confidence (the full
                                    -- answer is in the benchmark's answers.sqlite)
  primary key (benchmark, observation_id, backbone, method, reference_hash)
);
"""

# Photos of a photographer who withdrew permission on mycomap.org: all rights reserved
# ones are not used (the rule of evaluate.load_records, for the benchmark's photos).
USABLE_SQL = """
  not exists (select 1 from photo_permissions xpp where xpp.inat_user_id = hp.owner_user_id
              and hp.license_class = 'arr' and xpp.status = 'withdrawn')
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    from .permissions import ensure_schema as ensure_permissions_schema
    conn.executescript(SCHEMA)
    holdouts.ensure_schema(conn)
    ensure_permissions_schema(conn)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(s) -> str:
    return (s or "").strip() if isinstance(s, str) or s is None else str(s).strip()


def bench_dir(conn: sqlite3.Connection, name: str) -> Path:
    """<the manifest's folder>/benchmarks/<name>: the benchmark's photos, vectors, raw
    answers and reports, never mixed with the reference set's."""
    holdouts.check_name(name)
    main = next((r[2] for r in conn.execute("pragma database_list") if r[1] == "main"), "")
    return (Path(main).parent if main else config.DATA_DIR) / "benchmarks" / name


# --- the answer key and comparing names -------------------------------------------------

TRUTH_SPECIES = "names a species"
TRUTH_ONE_WORD = "one word, above species: name may be stale, needs a .com refresh"
TRUTH_NONE = "no name (no answer)"
TRUTH_STATUSES = (TRUTH_SPECIES, TRUTH_ONE_WORD, TRUTH_NONE)
NO_ANSWER = frozenset({TRUTH_NONE})
TRUTH_RULE = ("the .com record title's name (else the index name, where the title is the "
              "same), compared by Vision's labels with writing differences set aside; a "
              "one-word name is scored above species only")

SOURCE_INDEX = "index name (the title is the same)"
SOURCE_FORMAT = "title: the index name differs in writing only"
SOURCE_INDEX_ONE_WORD = "title: the index name is one word, the title names a species"
SOURCE_OTHER_SPECIES = "title: the index name is another species of the genus"
SOURCE_OTHER_GENUS = "title: the index name is in another genus"
SOURCE_TITLE_ONE_WORD = "title: one word, the index name names a species"
SOURCE_OTHER = "title: the index name differs otherwise"
SOURCES = (SOURCE_INDEX, SOURCE_FORMAT, SOURCE_INDEX_ONE_WORD, SOURCE_OTHER_SPECIES,
           SOURCE_OTHER_GENUS, SOURCE_TITLE_ONE_WORD, SOURCE_OTHER)
_ID_PREFIX = re.compile(r"^\s*\d+\s+-\s+")

RANK_WORDS = {"var.": "var.", "var": "var.", "subsp.": "subsp.", "subsp": "subsp.",
              "ssp.": "subsp.", "ssp": "subsp.", "f.": "f.", "forma": "f."}
_SP_HYPHEN = re.compile(r"(?<= )sp-(?=\S)", re.IGNORECASE)


def name_key(name) -> str:
    """Spellings of one name share this key: .org's rule (names.variant_key: quotes,
    `sp.`, spacing, capitals, temporary-code formats), plus `sp-CODE` read as
    `sp. CODE` and an infraspecific name with or without its rank word ('Hygrocybe
    glutinipes rubra' is 'Hygrocybe glutinipes var. rubra')."""
    folded = _SP_HYPHEN.sub("sp. ", names.fold_name(name))
    parts = names.parse_name(folded)
    if parts.code:
        return parts.key
    words = parts.key.split()
    if len(words) == 4 and words[2] in RANK_WORDS:
        del words[2]
    return " ".join(words)


def truth_status(name: str | None) -> str:
    name = clean(name)
    if not name:
        return TRUTH_NONE
    return TRUTH_ONE_WORD if len(name.split()) == 1 else TRUTH_SPECIES


def title_name(title: str | None) -> str:
    """The name in a .com record title: '373882777 - Trametes gibbosa' -> 'Trametes gibbosa'
    (an export may already give the name alone)."""
    return clean(_ID_PREFIX.sub("", clean(title)))


@dataclass(frozen=True)
class AnswerKey:
    truth: str | None             # the name the record is scored on; None: no answer
    status: str                   # TRUTH_*
    source: str                   # SOURCE_*


def answer_key(index_name: str | None, title: str | None) -> AnswerKey:
    """The answer for one record: the title's name where a titles export has one (it
    differs from the index name), else the index name."""
    index, name = clean(index_name), title_name(title)
    if not name:
        return AnswerKey(index or None, truth_status(index), SOURCE_INDEX)
    if not index or name == index:
        source = SOURCE_INDEX
    elif name_key(name) == name_key(index):
        source = SOURCE_FORMAT
    elif len(name.split()) == 1:
        source = SOURCE_TITLE_ONE_WORD
    elif len(index.split()) == 1:
        source = SOURCE_INDEX_ONE_WORD
    elif name.split()[0].lower() != index.split()[0].lower():
        source = SOURCE_OTHER_GENUS
    elif len(name.split()) >= 2 and len(index.split()) >= 2:
        source = SOURCE_OTHER_SPECIES
    else:
        source = SOURCE_OTHER
    return AnswerKey(name, truth_status(name), source)


def titles_by_observation(titles_tsv: Path, links_csv: Path, ids: Iterable[str]) -> dict[str, str]:
    """{iNat id: title name} from a titles export (record_id, index_name, title: records
    whose index name differs from the title) and the links (record_id, external_id = the
    iNat id). An observation linked to two records with different titles gets none."""
    want = set(ids)
    records_of: dict[str, set[str]] = defaultdict(set)
    for r in read_csv(links_csv):
        oid = clean(r.get("external_id"))
        if oid in want:
            records_of[oid].add(clean(r.get("record_id")))
    with open(titles_tsv, encoding="utf-8-sig", newline="") as f:
        title_of = {clean(r.get("record_id")): title_name(r.get("title"))
                    for r in csv.DictReader(f, delimiter="\t")}
    out = {}
    for oid, rids in records_of.items():
        found = {title_of[r] for r in rids if title_of.get(r)}
        if len(found) == 1:
            out[oid] = found.pop()
    return out


# --- freeze -----------------------------------------------------------------------------

POOL_COLUMNS = ("observation_id", "com_name")
SPLITS = ("dev", "test")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def file_info(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": h.hexdigest()}


def ids_sha256(ids: Iterable[str]) -> str:
    """sha256 of the newline-joined ids, sorted as numbers (as the set was frozen)."""
    ids = list(ids)
    key = (lambda i: (0, int(i), i)) if all(i.isdigit() for i in ids) else (lambda i: (0, 0, i))
    return hashlib.sha256("\n".join(sorted(ids, key=key)).encode()).hexdigest()


def split_sha256(ids: Iterable[str]) -> str:
    """sha256 of the newline-joined ids sorted as text: how split.json records a split
    (the pool's own hash sorts them as numbers, ids_sha256)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _num(v) -> float | None:
    try:
        return float(v) if clean(v) else None
    except ValueError:
        return None


def read_splits(ids: list[str], splits: dict[str, Path], split_json: Path | None) -> dict[str, str]:
    """{id: split} from one CSV per split. The splits must cover the pool exactly, each
    id once; with split.json, each split's count and id hash must match it."""
    if set(splits) != set(SPLITS):
        raise ValueError(f"the splits are {' and '.join(SPLITS)}, each from its own CSV")
    of: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for split, path in splits.items():
        members[split] = holdouts.read_ids_csv(path)
        for i in members[split]:
            if i in of:
                raise ValueError(f"{i} is in both {of[i]} and {split}")
            of[i] = split
    pool = set(ids)
    outside, left = sorted(set(of) - pool), sorted(pool - set(of))
    if outside or left:
        raise ValueError(f"the splits must cover the pool exactly: {len(outside)} ids not in "
                         f"the pool (first: {outside[:3]}), {len(left)} pool ids in no split "
                         f"(first: {left[:3]})")
    if split_json:
        info = json.loads(Path(split_json).read_text(encoding="utf-8"))
        for split, got in members.items():
            want = info.get(split) or {}
            if want.get("records") != len(got) or want.get("sha256_ids") != split_sha256(got):
                raise ValueError(f"the {split} CSV does not match {Path(split_json).name} "
                                 f"({len(got):,} records, ids hash {split_sha256(got)[:16]})")
    return of


def _in_records(conn: sqlite3.Connection, ids: list[str]) -> list[str]:
    conn.execute("create temp table if not exists freeze_ids (id text primary key)")
    conn.execute("delete from freeze_ids")
    conn.executemany("insert into freeze_ids values (?)", [(i,) for i in ids])
    found = [r[0] for r in conn.execute(
        "select r.observation_id from records r join freeze_ids f on f.id = r.observation_id "
        "order by 1")]
    conn.execute("delete from freeze_ids")
    conn.commit()
    return found


def freeze(conn: sqlite3.Connection, name: str, pool_csv: Path,
           titles_tsv: Path | None = None, links_csv: Path | None = None,
           snapshot_csv: Path | None = None, expect_sha: str | None = None,
           splits: dict[str, Path] | None = None, split_json: Path | None = None,
           holdout: bool = False, at: str | None = None) -> dict:
    """Store a set with its answer key (answer_key: the title's name where a titles
    export with its links gives one, else pool.csv's com_name) and dev/test split.

    With `holdout` (a sealed set), ids the records table holds are refused and every id
    is held out (holdouts.py); without, the set is a development benchmark whose
    records may be or become reference records.

    Freezing a set again takes a new snapshot of the same ids: their names, places and
    the like are updated (an answer refreshed on .com), the split must not change, and
    the freeze is logged. Other ids under the same name are refused."""
    holdouts.check_name(name)
    ensure_schema(conn)
    rows = read_csv(pool_csv)
    missing = [c for c in POOL_COLUMNS if c not in (rows[0] if rows else {})]
    if not rows or missing:
        raise ValueError(f"{pool_csv} needs the columns {', '.join(POOL_COLUMNS)}")
    ids = [clean(r["observation_id"]) for r in rows]
    if not all(ids):
        raise ValueError(f"{pool_csv} has a row with no observation_id")
    repeated = [i for i, n in Counter(ids).items() if n > 1]
    if repeated:
        raise ValueError(f"{pool_csv} repeats {len(repeated)} ids (first: {repeated[:5]})")
    sha = ids_sha256(ids)
    if expect_sha and not sha.startswith(expect_sha.strip().lower()):
        raise ValueError(f"the ids hash to {sha[:16]}, not {expect_sha}: not the frozen set")
    old = conn.execute("select ids_sha256, records, inputs_json from heldout_sets "
                       "where name = ?", (name,)).fetchone()
    if old and old[0] != sha:
        raise ValueError(f"{name} is already frozen with other ids ({old[1]:,} records); "
                         "a set's records never change: freeze under a new name")
    known = _in_records(conn, ids)
    if holdout and known:
        raise ValueError(f"refused: {len(known):,} of these records are already reference or "
                         f"training records (first: {', '.join(known[:5])}); a held-out set "
                         "must be records Vision has never seen")
    if bool(titles_tsv) != bool(links_csv):
        raise ValueError("a titles export needs its links (record_id -> iNat id), and back")
    split_of = read_splits(ids, splits, split_json) if splits else {}
    titles = titles_by_observation(titles_tsv, links_csv, ids) if titles_tsv else {}
    inputs = {"pool": file_info(pool_csv)}
    if titles_tsv:
        inputs["titles"] = file_info(titles_tsv)
        inputs["links"] = file_info(links_csv)
    if snapshot_csv:
        inputs["snapshot"] = file_info(snapshot_csv)
    for split, path in (splits or {}).items():
        inputs[f"split_{split}"] = file_info(path)
    if split_json:
        inputs["split_json"] = file_info(split_json)
    at = at or now_iso()
    status, source = Counter(), Counter()
    new_rows = {}
    for r in rows:
        oid = clean(r["observation_id"])
        key = answer_key(r.get("com_name"), titles.get(oid))
        status[key.status] += 1
        source[key.source] += 1
        new_rows[oid] = (key.truth, key.status, clean(r.get("com_name")) or None, key.source,
                         clean(r.get("org_name")) or None, _num(r.get("lat")),
                         _num(r.get("lng")), clean(r.get("country")) or None,
                         clean(r.get("first_validated")) or None,
                         clean(r.get("projects")) or None)
    out = {"benchmark": name, "records": len(ids), "ids_sha256": sha,
           "answer_key": {s: status[s] for s in TRUTH_STATUSES if status[s]},
           "answer_source": {s: source[s] for s in SOURCES if source[s]},
           "already_reference_records": len(known)}
    if old:
        stored = dict(conn.execute("select observation_id, split from heldout_records "
                                   "where benchmark = ?", (name,)).fetchall())
        if split_of and split_of != stored:
            raise ValueError(f"{name}'s split never changes; this snapshot's split differs")
        before = dict(conn.execute("select observation_id, truth_name from heldout_records "
                                   "where benchmark = ?", (name,)).fetchall())
        changed = sorted(o for o, row in new_rows.items() if row[0] != before.get(o))
        same_inputs = all(json.loads(old[2]).get(k) == inputs.get(k)
                          for k in ("pool", "titles", "links"))
        if same_inputs and not holdout:
            return {**out, "already_frozen": True, "names_changed": 0}
        with conn:
            conn.executemany(
                "update heldout_records set truth_name = ?, truth_status = ?, index_name = ?, "
                "name_source = ?, org_name = ?, latitude = ?, longitude = ?, country = ?, "
                "first_validated = ?, projects = ? where benchmark = ? and observation_id = ?",
                [(*row, name, oid) for oid, row in new_rows.items()])
            conn.execute("update heldout_sets set inputs_json = ?, refrozen_at = ? where name = ?",
                         (json.dumps(inputs), at, name))
            conn.execute("insert into heldout_freezes values (?, ?, ?, ?, ?)",
                         (name, at, json.dumps(inputs), len(changed), config.code_version()))
            held = holdouts.insert(conn, name, ids, at) if holdout else 0
        return {**out, "refrozen": True, "names_changed": len(changed),
                "names_changed_first": changed[:10], "held_out_added": held}
    with conn:
        conn.execute("insert into heldout_sets values (?, ?, ?, ?, ?, ?, ?, ?)",
                     (name, len(ids), sha, json.dumps(inputs), TRUTH_RULE, at, None,
                      config.code_version()))
        conn.execute("insert into heldout_freezes values (?, ?, ?, ?, ?)",
                     (name, at, json.dumps(inputs), 0, config.code_version()))
        conn.executemany("insert into heldout_records values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         [(name, oid, *row, split_of.get(oid)) for oid, row in new_rows.items()])
        held = holdouts.insert(conn, name, ids, at) if holdout else 0
    return {**out, "splits": dict(Counter(split_of.values())), "held_out": bool(holdout),
            "held_out_added": held, "inputs": inputs}


def benchmark_ids(conn: sqlite3.Connection, name: str, subset: Iterable[str] | None = None,
                  limit: int | None = None, split: str | None = None) -> list[str]:
    """The set's ids (in number order): all, those of one split, and/or those of `subset`
    in it. Refuses an unknown set or split."""
    ensure_schema(conn)
    if not conn.execute("select 1 from heldout_sets where name = ?", (name,)).fetchone():
        raise ValueError(f"no frozen benchmark {name!r}: run mv heldout freeze first")
    if split is not None and split not in splits_of(conn, name):
        raise ValueError(f"{name} has no {split!r} split (it has: "
                         f"{', '.join(splits_of(conn, name)) or 'none'})")
    ids = [r[0] for r in conn.execute(
        "select observation_id from heldout_records where benchmark = ? "
        "and (? is null or split = ?) "
        "order by cast(observation_id as integer), observation_id", (name, split, split))]
    if subset is not None:
        want = set(holdouts.clean_ids(subset))
        ids = [i for i in ids if i in want]
    return ids[:limit] if limit is not None else ids


def splits_of(conn: sqlite3.Connection, name: str) -> list[str]:
    return [r[0] for r in conn.execute("select distinct split from heldout_records where "
                                       "benchmark = ? and split is not null order by 1", (name,))]


def sealed(conn: sqlite3.Connection, name: str) -> bool:
    """A sealed benchmark (the paper's): its records are held out, its test split is
    looked at only for the paper's number."""
    return holdouts.is_held_out(conn, name)


# --- fetch: iNat details and photos ----------------------------------------------------

def save_details(conn: sqlite3.Connection, name: str, requested: list[str], results: list[dict],
                 fetched_at: str) -> dict:
    """One iNat answer into the benchmark's tables, and as every record's iNat details
    (inat.save_batch: inat_observations with the uuid, photos and licence history; no
    photo copy). Requested ids iNat didn't return are marked missing."""
    from .inat import parse_observation, save_batch
    save_batch(conn, requested, results, fetched_at)
    stats = Counter()
    seen = set()
    with conn:
        for obs in results:
            row, photos = parse_observation(obs)
            oid = row["observation_id"]
            seen.add(oid)
            conn.execute(
                "insert or replace into heldout_observations values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (name, oid, "ok", row["uuid"], row["observed_on"], row["inat_latitude"],
                 row["inat_longitude"], row["obscured"], row["geoprivacy"], row["user_id"],
                 row["user_login"], row["photo_count"], fetched_at))
            keep = []
            for p in photos:
                keep.append(p["photo_id"])
                conn.execute(
                    "insert into heldout_photos (benchmark, observation_id, photo_id, position, "
                    "license_code, license_class, attribution, owner_user_id, owner_login, "
                    "source_url, host) values (?,?,?,?,?,?,?,?,?,?,?) "
                    "on conflict(benchmark, observation_id, photo_id) do update set "
                    "position = excluded.position, license_code = excluded.license_code, "
                    "license_class = excluded.license_class, attribution = excluded.attribution, "
                    "owner_user_id = excluded.owner_user_id, owner_login = excluded.owner_login, "
                    "source_url = excluded.source_url, host = excluded.host",
                    (name, oid, p["photo_id"], p["position"], p["license_code"],
                     p["license_class"], p["attribution"], p["owner_user_id"], p["owner_login"],
                     p["source_url"], p["host"]))
            # A photo its owner has since removed from the observation is no longer used.
            conn.execute(f"delete from heldout_photos where benchmark = ? and observation_id = ? "
                         f"and photo_id not in ({','.join('?' * len(keep)) or 'null'})",
                         (name, oid, *keep))
            stats["ok"] += 1
            stats["photos"] += len(photos)
        for oid in requested:
            if oid not in seen:
                conn.execute("insert or replace into heldout_observations (benchmark, "
                             "observation_id, status, fetched_at) values (?, ?, 'missing', ?)",
                             (name, oid, fetched_at))
                stats["missing"] += 1
    return dict(stats)


def fetch_details(conn: sqlite3.Connection, name: str, ids: list[str], refresh: bool = False,
                  session=None, fetch: Callable | None = None, raw_dir: Path | None = None,
                  log=print) -> dict:
    """iNat's details (date, public place, observer, photos and licences) for `ids` not
    fetched before, 200 a request at 1 request a second (inat.fetch_batch). Each answer
    is kept gzipped under the benchmark's raw/ folder."""
    import requests

    from .inat import BATCH, chunks, fetch_batch
    ensure_schema(conn)
    if not refresh:
        have = {r[0] for r in conn.execute(
            "select observation_id from heldout_observations where benchmark = ?", (name,))}
        ids = [i for i in ids if i not in have]
    raw_dir = raw_dir or bench_dir(conn, name) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
    fetch = fetch or fetch_batch
    limiter = MinInterval(1.0)
    totals = Counter(requested=len(ids))
    for batch in chunks(ids, BATCH):
        results = fetch(session, batch, limiter)
        at = now_iso()
        with gzip.open(raw_dir / f"obs-{batch[0]}-{at[:10]}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(results, f)
        totals.update(save_details(conn, name, batch, results, at))
        totals["batches"] += 1
        if totals["batches"] % 10 == 0:
            log(f"  {totals['ok'] + totals['missing']:,}/{len(ids):,} observations")
    return dict(totals)


def pending_photos(conn: sqlite3.Connection, name: str, ids: list[str], size: str,
                   store_location: str, limit: int | None = None) -> list[dict]:
    """The set's photos with no copy at `size` in the store, failed fewest times first;
    missing ones, ones that failed MAX_ATTEMPTS times and ones a photographer withdrew
    are left out."""
    from .photos import MAX_ATTEMPTS
    want = set(ids)
    rows = conn.execute(f"""
      select hp.observation_id, hp.photo_id, hp.source_url, hp.host, hp.attempts
      from heldout_photos hp
      where hp.benchmark = ? and hp.status != 'missing' and hp.attempts < {MAX_ATTEMPTS}
        and not (hp.status = 'done' and hp.size = ? and hp.store = ?) and {USABLE_SQL}
      order by hp.attempts, hp.photo_id
    """, (name, size, store_location)).fetchall()
    out, seen = [], set()
    for oid, pid, url, host, _attempts in rows:
        if oid in want and pid not in seen:
            seen.add(pid)
            out.append({"photo_id": pid, "source_url": url, "host": host})
    return out[:limit] if limit is not None else out


def save_photo(conn: sqlite3.Connection, name: str, r, size: str, now: str,
               store_location: str) -> None:
    """One download (photos.Result) into heldout_photos; a stopped one is not a failure."""
    if r.status == "done":
        conn.execute("update heldout_photos set status = 'done', store = ?, size = ?, path = ?, "
                     "bytes = ?, sha256 = ?, error = null, downloaded_at = ? "
                     "where benchmark = ? and photo_id = ?",
                     (store_location, size, r.local_path, r.bytes, r.sha256, now, name,
                      r.photo_id))
    elif r.error != "stopped":
        conn.execute("update heldout_photos set status = ?, error = ?, attempts = attempts + 1 "
                     "where benchmark = ? and photo_id = ?", (r.status, r.error, name, r.photo_id))


def fetch_photos(conn: sqlite3.Connection, name: str, ids: list[str], store, size: str = "large",
                 policies=None, max_hours: float | None = None, poll: float = 30,
                 log=print) -> dict:
    """Download the set's photos into `store` (by default the benchmark's own folder;
    s3://... works too) within iNat's limits, shared with the reference downloader
    (photos.run_downloads, photos.seed_budgets). Resumable: a rerun fetches only what is
    still missing, and Ctrl+C keeps what is done."""
    from .photos import run_downloads
    rows = pending_photos(conn, name, ids, size, store.location)
    log(f"  {len(rows):,} {size} photos to download into {store.location}")
    return run_downloads(conn, rows, store, size,
                         lambda r, now: save_photo(conn, name, r, size, now, store.location),
                         policies=policies, max_hours=max_hours, poll=poll, log=log)


# --- records as the benchmark sees them ------------------------------------------------

@dataclass
class HeldOutRecord:
    observation_id: str
    truth_name: str | None
    truth_status: str
    index_name: str | None            # .com's index name, when the title differs: the audit's
    name_source: str
    org_latitude: float | None        # true coordinates: for scores, never shown
    org_longitude: float | None
    observed_on: str | None           # iNat's date
    inat_latitude: float | None       # iNat's public (possibly obscured) place
    inat_longitude: float | None
    user_id: int | None
    photos: list[tuple[int, str, str]]   # (photo id, store, path) of usable copies, in order
    split: str | None = None
    uuid: str | None = None              # iNat's observation uuid


def load_benchmark(conn: sqlite3.Connection, name: str, ids: list[str] | None = None,
                   size: str | None = None) -> list[HeldOutRecord]:
    """The set's records with their answer, iNat details and usable photo copies (at
    `size` when given). They are read from the benchmark's tables, so a held-out record
    loads here although no reference path will load it."""
    ensure_schema(conn)
    want = set(ids) if ids is not None else None
    photos: dict[str, list] = defaultdict(list)
    for oid, pid, store, path in conn.execute(f"""
          select hp.observation_id, hp.photo_id, hp.store, hp.path from heldout_photos hp
          where hp.benchmark = ? and hp.status = 'done' and (? is null or hp.size = ?)
            and {USABLE_SQL}
          order by hp.observation_id, hp.position, hp.photo_id""", (name, size, size)):
        photos[oid].append((int(pid), store, path))
    out = []
    for row in conn.execute("""
          select r.observation_id, r.truth_name, r.truth_status, r.index_name, r.name_source,
                 r.latitude, r.longitude, o.observed_on, o.inat_latitude, o.inat_longitude,
                 o.user_id, r.split, o.uuid
          from heldout_records r left join heldout_observations o
            on o.benchmark = r.benchmark and o.observation_id = r.observation_id
            and o.status = 'ok'
          where r.benchmark = ? order by cast(r.observation_id as integer)""", (name,)):
        if want is not None and row[0] not in want:
            continue
        out.append(HeldOutRecord(row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                                 real_date(row[7]), row[8], row[9], row[10],
                                 photos.get(row[0], []), row[11], row[12]))
    return out


def open_location(location: str):
    from .storage import LocalStore, S3Store
    return S3Store(location) if location.startswith("s3://") else LocalStore(Path(location))


def context_for(rec: HeldOutRecord, place: str):
    """The place and date a method with a range-and-season score is given: iNat's public
    place (what a user sharing the observation shows), .org's true one, or none."""
    from .prior import Context
    if place == "none":
        return None
    if place == "org":
        return Context(rec.org_latitude, rec.org_longitude, rec.observed_on)
    return Context(rec.inat_latitude, rec.inat_longitude, rec.observed_on)


# --- truth labels (Vision's label rules) ----------------------------------------------------

@dataclass(frozen=True)
class Truth:
    species: str
    genus: str
    family: str
    label: str                 # the answer as Vision labels it
    normalised: bool           # matched a known label only once writing was set aside
    guest: str | None          # guests.excluded: not the fungus in the photo


class Labeller:
    """Vision's labels for the answer key and for answers: names.manifest_labels for
    spellings, taxonomy.labels_for for genus, family and one-word names, guests.excluded.
    `extra` are names met outside the manifest (answers, the key) labelled by the same
    rule. A name Vision doesn't know that matches one known label once writing
    differences are set aside (name_key: provisional-code formats, a missing `var.`)
    takes that label. Family falls back to the one Vision's records give the genus."""

    def __init__(self, conn: sqlite3.Connection, extra: Iterable[str] = ()):
        manifest_names = names.counted(names.name_counts(conn))
        self.labels = names.manifest_labels(conn, extra=list(extra))
        self.known = {clean(self.labels.get(n, n)) for n in manifest_names}
        by_key: dict[str, set[str]] = defaultdict(set)
        for lab in self.known:
            by_key[name_key(lab)].add(lab)
        self.known_by_key = {k: next(iter(v)) for k, v in by_key.items() if len(v) == 1}
        self.tax = taxonomy.for_manifest(conn)
        families: dict[str, Counter] = defaultdict(Counter)
        for name, genus, family in conn.execute(
                "select scientific_name, genus, family from records "
                "where coalesce(scientific_name, '') <> '' and label_conflict = 0"):
            label = clean(self.labels.get(name, name))
            lab = taxonomy.labels_for(label, genus, family, label != clean(name), self.tax)
            if lab.genus and lab.family:
                families[lab.genus][lab.family] += 1
        self.genus_family = {g: c.most_common(1)[0][0] for g, c in families.items()}
        self.known_genera = ({lab.split()[0] for lab in self.known if lab}
                             | set(self.genus_family)
                             | (self.tax.genus_words if self.tax else set()))

    def label(self, name: str | None) -> str:
        label = clean(self.labels.get(name, name))
        if label and label not in self.known:
            label = self.known_by_key.get(name_key(label), label)
        return label

    def same(self, a: str | None, b: str | None) -> bool:
        """Two names are one answer: the same label, or the same once writing is set aside."""
        la, lb = self.label(a), self.label(b)
        return bool(la) and (la == lb or name_key(la) == name_key(lb))

    def truth(self, name: str) -> Truth:
        label = self.label(name)
        words = label.split()
        lab = taxonomy.labels_for(label, words[0] if words else "", "",
                                  label != clean(name), self.tax)
        family = lab.family or self.genus_family.get(lab.genus, "")
        normalised = label != clean(self.labels.get(name, name))
        return Truth(lab.species, lab.genus, family, label, normalised,
                     guests.excluded(lab.genus))


# --- predict ---------------------------------------------------------------------------

def embeddings_db(conn: sqlite3.Connection, name: str) -> sqlite3.Connection:
    """The benchmark's own embedding index (embed.SCHEMA in its own file): its vectors
    never enter the manifest's embeddings table, so nothing can load them as references.
    Each photo's vector is kept, so a later method can score the set without fetching or
    embedding anything again."""
    path = bench_dir(conn, name) / "embeddings" / "index.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    edb = sqlite3.connect(path)
    from .embed import SCHEMA as EMBED_SCHEMA
    edb.executescript(EMBED_SCHEMA)
    return edb


def embed_benchmark(conn: sqlite3.Connection, name: str, backbone: str,
                    records: list[HeldOutRecord], load_model: Callable[[], object],
                    batch_size: int = 32, log=print) -> dict:
    """One vector per usable photo of `records` with `backbone`, into the benchmark's own
    folder and index (embed.embed_photos). Photos already embedded are skipped."""
    from .embed import embed_photos
    edb = embeddings_db(conn, name)
    try:
        done = {r[0] for r in edb.execute("select photo_id from embeddings where backbone = ?",
                                          (backbone,))}
        by_store: dict[str, list[tuple[int, str]]] = defaultdict(list)
        queued = set()
        for rec in records:
            for pid, store, path in rec.photos:
                if pid not in done and pid not in queued:
                    queued.add(pid)
                    by_store[store].append((pid, path))
        stats = {"embedded": 0, "skipped": 0, "already": len(done)}
        if not queued:
            return stats
        model = load_model()
        out_dir = bench_dir(conn, name) / "embeddings" / backbone
        for location, todo in by_store.items():
            s = embed_photos(edb, open_location(location), model, todo, out_dir,
                             batch_size=batch_size, log=log)
            stats["embedded"] += s.embedded
            stats["skipped"] += len(s.skipped)
        return stats
    finally:
        edb.close()


def load_vectors(conn: sqlite3.Connection, name: str, backbone: str) -> dict[int, np.ndarray]:
    from .embed import load_embeddings
    edb = embeddings_db(conn, name)
    try:
        ids, vecs = load_embeddings(edb, backbone, bench_dir(conn, name) / "embeddings" / backbone)
    finally:
        edb.close()
    return {int(p): vecs[i] for i, p in enumerate(ids.tolist())}


def reference_summary(conn: sqlite3.Connection, identifier) -> dict:
    """Which reference an Identifier answers from, so a run can be told apart from one on
    another manifest state (before and after a relabel): the hash of its records and
    their labels (`hash`), of its records alone, their number, and for the report the
    records per name at each rank and the (iNat observer, date) of each record."""
    col = identifier.col_record
    obs = [str(o) for o in col.obs.tolist()]
    lines = sorted(f"{o}\t" + "\t".join(str(col.names[i]) for i in col.taxa[r])
                   for r, o in enumerate(obs))
    h = hashlib.sha1(f"{identifier.backbone}|".encode())
    h.update("\n".join(lines).encode())
    want = set(obs)
    days = sorted({f"{uid}|{day}" for oid, uid, day in conn.execute(
        "select observation_id, user_id, observed_on from inat_observations where status = 'ok'")
        if oid in want and uid is not None and real_date(day)})
    newest_export = conn.execute("select max(exported_at) from records").fetchone()[0]
    return {"hash": h.hexdigest()[:12], "obs": want,
            "json": {"records": identifier.records, "photos": identifier.embedded,
                     "species": len(identifier.index.labels["species"]),
                     "records_hash": hashlib.sha1("\n".join(sorted(obs)).encode()
                                                  ).hexdigest()[:12],
                     "manifest_newest_export": newest_export,
                     "rank_counts": {rank: dict(identifier.rank_counts[rank]) for rank in RANKS},
                     "observer_days": days, "calibration": identifier.calibration}}


def summarise(ranks: dict, top: int = 5) -> dict:
    return {rank: [{"name": c["name"], "confidence": c["confidence"], "score": c["score"],
                    "reference_records": c["reference_records"]} for c in ranks[rank][:top]]
            for rank in RANKS}


def answers_db(conn: sqlite3.Connection, name: str) -> sqlite3.Connection:
    """The benchmark's full answers (each identify result, gzipped JSON), beside its photos
    rather than in the manifest, which is shipped to releases and trainers."""
    path = bench_dir(conn, name) / "answers.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    adb = sqlite3.connect(path)
    adb.execute("create table if not exists answers (observation_id text not null, "
                "backbone text not null, method text not null, reference_hash text not null, "
                "result_gz blob not null, "
                "primary key (observation_id, backbone, method, reference_hash))")
    return adb


def full_answer(conn: sqlite3.Connection, name: str, oid: str, backbone: str, method: str,
                reference_hash: str) -> dict | None:
    adb = answers_db(conn, name)
    try:
        row = adb.execute("select result_gz from answers where observation_id = ? and "
                          "backbone = ? and method = ? and reference_hash = ?",
                          (oid, backbone, method, reference_hash)).fetchone()
    finally:
        adb.close()
    return json.loads(gzip.decompress(row[0])) if row else None


def save_scores(path: Path, rows: list[tuple], ident, truths: dict, place: str) -> int:
    """Per-record photo scores in the arrays occtune.save_scored_set writes (branch
    feat/occurrence-prior; written here without importing it): observation_id, species
    (the index's groups), scores (records x groups: mean over photos of each photo's
    best cosine within the group, "similarity"), kind, truth / truth_genus /
    truth_family (Vision's labels; '' where the answer has none), latitude, longitude
    and observed_on (the place and date `place` gives), uuid, group_genus,
    group_family, group_is_species. `mv tune-occurrence --records dev.csv --scores
    <path>` reads it."""
    index = ident.index
    nan = float("nan")

    def strs(values):
        return np.array(["" if v is None else str(v) for v in values], dtype=str)

    def group_label(rank):
        names_ = index.labels[rank]
        return strs(names_[i] if i >= 0 else "" for i in index.label_of[rank].tolist())
    recs = [r for r, _ in rows]
    t = [truths.get(r.observation_id) for r in recs]
    ctx = [context_for(r, place) for r in recs]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, observation_id=strs(r.observation_id for r in recs),
        species=strs(index.species),
        scores=np.stack([s for _, s in rows]).astype(np.float32) if rows
        else np.zeros((0, len(index.species)), np.float32),
        kind=np.array("similarity"),
        truth=strs(x.species if x else "" for x in t),
        truth_genus=strs(x.genus if x else "" for x in t),
        truth_family=strs(x.family if x else "" for x in t),
        latitude=np.array([c.latitude if c and c.latitude is not None else nan for c in ctx],
                          dtype=np.float64),
        longitude=np.array([c.longitude if c and c.longitude is not None else nan for c in ctx],
                           dtype=np.float64),
        observed_on=strs(c.observed_on if c else "" for c in ctx),
        uuid=strs(r.uuid for r in recs),
        group_genus=group_label("genus"), group_family=group_label("family"),
        group_is_species=(index.label_of["species"] >= 0))
    return len(rows)


def predict(conn: sqlite3.Connection, name: str, backbone: str, methods: list[str],
            ids: list[str], load_model: Callable[[], object], place: str = "inat",
            size: str = "large", batch_size: int = 32, redo: bool = False,
            make_identifier: Callable | None = None, scores_out: Path | None = None,
            log=print) -> dict:
    """Embed the set's photos with `backbone` and identify each record with each method,
    against the manifest's reference index for that backbone (Identifier: every record
    Vision would serve). Saves the top 5 per rank for the report, and the whole identify
    result in the benchmark's answers.sqlite. A run against another reference (a relabel,
    new records) is kept beside the earlier one, not over it. `scores_out`: also write
    every answered record's photo scores against the first method's index (save_scores),
    for tuning a place prior on dev."""
    from . import evaluate
    from .identify import Identifier
    if place not in ("inat", "org", "none"):
        raise ValueError("place is inat, org or none")
    for m in methods:
        if m not in evaluate.METHODS:
            raise ValueError(f"unknown method {m!r}: {', '.join(evaluate.METHODS)}")
    ensure_schema(conn)
    records = [r for r in load_benchmark(conn, name, ids, size) if r.photos]
    log(f"  {len(records):,} of {len(ids):,} records have {size} photos to answer from")
    out = {"records": len(records), "embedding": embed_benchmark(
        conn, name, backbone, records, load_model, batch_size, log)}
    vectors = load_vectors(conn, name, backbone)
    members = set(benchmark_ids(conn, name))
    is_sealed = sealed(conn, name)
    make_identifier = make_identifier or (lambda m: Identifier(
        conn, backbone, m, calibration=evaluate.latest_calibration(conn, backbone, m),
        photo_info=False))
    version = config.code_version()
    scored: list[tuple] = []
    for method in methods:
        ident = make_identifier(method)
        want_scores = scores_out is not None and method == methods[0]
        ref = reference_summary(conn, ident)
        inside = members & ref["obs"]
        if inside and is_sealed:
            raise holdouts.HeldOutLeak(
                f"the {backbone} reference index holds {len(inside):,} records of the sealed "
                f"{name} (first: {', '.join(sorted(inside)[:5])}); nothing was predicted")
        ref_photos = set(int(p) for p in ident.col_photo.tolist())
        with conn:
            conn.execute("insert or replace into heldout_runs values (?, ?, ?, ?, ?, ?, ?)",
                         (name, backbone, method, ref["hash"], now_iso(), version,
                          json.dumps(ref["json"])))
        done = set() if redo else {r[0] for r in conn.execute(
            "select observation_id from heldout_predictions where benchmark = ? and "
            "backbone = ? and method = ? and reference_hash = ?",
            (name, backbone, method, ref["hash"]))}
        stats = Counter()
        rows, full = [], []
        adb = answers_db(conn, name)
        try:
            for rec in records:
                if rec.observation_id in done and not want_scores:
                    stats["already"] += 1
                    continue
                if rec.observation_id in inside:
                    stats["already a reference record (skipped)"] += 1
                    continue
                pids = [pid for pid, _s, _p in rec.photos]
                if ref_photos & set(pids):
                    stats["photo also a reference photo (skipped)"] += 1
                    continue
                vecs = [vectors[p] for p in pids if p in vectors]
                if not vecs:
                    stats["no vectors"] += 1
                    continue
                if want_scores:
                    sims = ident.nearest.photo_sims(np.stack(vecs))
                    scored.append((rec, np.maximum.reduceat(sims, ident.index.starts,
                                                            axis=1).mean(axis=0)))
                if rec.observation_id in done:
                    stats["already"] += 1
                    continue
                result = ident.identify_vectors(np.stack(vecs), top_k=5,
                                                context=context_for(rec, place))
                rows.append((name, rec.observation_id, backbone, method, ref["hash"], now_iso(),
                             version, ident.records, len(vecs), place,
                             json.dumps(summarise(result["ranks"]))))
                full.append((rec.observation_id, backbone, method, ref["hash"],
                             gzip.compress(json.dumps(result).encode())))
                stats["predicted"] += 1
                if len(rows) >= 200:
                    _save_predictions(conn, adb, rows, full)
                    log(f"  {backbone} / {method}: {stats['predicted']:,} predicted")
            _save_predictions(conn, adb, rows, full)
        finally:
            adb.close()
        out[method] = {"reference_hash": ref["hash"], "reference_records": ident.records,
                       **dict(stats)}
        if want_scores:
            labeller = Labeller(conn, extra=[r.truth_name for r, _ in scored if r.truth_name])
            truths = {r.observation_id: labeller.truth(r.truth_name)
                      for r, _ in scored if r.truth_name}
            out["scores_out"] = {"path": str(scores_out),
                                 "records": save_scores(scores_out, scored, ident, truths, place)}
        del ident
    return out


def _save_predictions(conn: sqlite3.Connection, adb: sqlite3.Connection, rows: list,
                      full: list) -> None:
    with adb:
        adb.executemany("insert or replace into answers values (?, ?, ?, ?, ?)", full)
    with conn:
        conn.executemany("insert or replace into heldout_predictions values "
                         "(?,?,?,?,?,?,?,?,?,?,?)", rows)
    rows.clear()
    full.clear()


# --- iNat's computer vision ----------------------------------------------------------------

def shrink(body: bytes, max_side: int = 500) -> bytes:
    """A stored copy cut down to iNat's medium size (500 px on the long side) before it
    is sent: iNat scales photos down anyway, and this keeps each upload small."""
    from PIL import Image

    from .embed import decode
    img = decode(body)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def inat_answers(photo_scores: list[dict], score: str, top: int = 5) -> dict:
    """A record's top taxa per rank, best score on any photo first (inat_cv.best_per_rank)."""
    from .inat_cv import best_per_rank
    names_of = {rank: {tid: t["name"] for s in photo_scores for tid, t in s[rank].items()}
                for rank in RANKS}
    best = best_per_rank(photo_scores, score)
    return {rank: [{"name": names_of[rank][tid], "taxon_id": tid,
                    "confidence": round(s / 100, 4)} for tid, s in best[rank][:top]]
            for rank in RANKS}


def inat_cv(conn: sqlite3.Connection, name: str, ids: list[str], client, size: str = "large",
            redo: bool = False, log=print) -> dict:
    """iNat's computer vision on `ids` (a named subsample): each usable photo is scored
    with iNat's public place (inat_cv.InatClient: cached, 1 request a second, a 401 stops
    the run), and the record's answer per rank is the best score on any photo, photo only
    (vision-max) and with iNat's place model (combined-max). The answer key is looked up
    on iNat (inat_cv.resolve_truth), so the report judges iNat by taxon id."""
    from .evaluate import Record
    from .inat_cv import BACKBONE, parse_aggregated, resolve_truth
    ensure_schema(conn)
    records = load_benchmark(conn, name, ids, size)
    labeller = Labeller(conn, extra=[r.truth_name for r in records if r.truth_name])
    done = set() if redo else {r[0] for r in conn.execute(
        "select observation_id from heldout_predictions where benchmark = ? and backbone = ? "
        "and method = 'combined-max'", (name, BACKBONE))}
    stores: dict[str, object] = {}
    stats = Counter()
    version = config.code_version()
    for i, rec in enumerate(records, 1):
        if rec.observation_id in done:
            stats["already"] += 1
            continue
        if not rec.truth_name:
            stats["no answer key (not sent)"] += 1
            continue
        if not rec.photos:
            stats["no photos"] += 1
            continue
        truth = labeller.truth(rec.truth_name)
        scored = []
        for pid, location, path in rec.photos:
            store = stores.setdefault(location, open_location(location))
            scored.append(parse_aggregated(client.score_image(
                pid, lambda s=store, p=path: shrink(s.get(p)), rec.inat_latitude,
                rec.inat_longitude)))
        as_record = Record(rec.observation_id, truth.species, truth.genus, truth.family, None,
                           None, stored_name=clean(rec.truth_name)
                           if truth.label != clean(rec.truth_name) else "",
                           taxon="" if truth.species else truth.label)
        t = resolve_truth(client, as_record)
        truth_taxa = {"species": t.species, "genus": t.genus, "family": t.family,
                      "species_known": t.species_known}
        with conn:
            for method, score in (("vision-max", "vision"), ("combined-max", "combined")):
                result = {**inat_answers(scored, score), "truth_taxa": truth_taxa}
                conn.execute("insert or replace into heldout_predictions values "
                             "(?,?,?,?,?,?,?,?,?,?,?)",
                             (name, rec.observation_id, BACKBONE, method, "", now_iso(),
                              version, None, len(scored), "inat", json.dumps(result)))
        stats["scored"] += 1
        if i % 10 == 0:
            log(f"  {i}/{len(records)} records, {client.calls} iNat calls")
    return {"records": len(records), "inat_calls": client.calls, **dict(stats)}
