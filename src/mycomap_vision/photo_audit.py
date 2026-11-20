"""The photo review's rescoring audit (Steve, 2026-11-20): every photo a person decided, scored
by the photo-type tagger with a model that never trained on it, so each tab's Decided list can
show the decisions the tagger disagrees with most.

The scores come from the research repo's experiments/photo-type-tagger/r12_audit.py: 5 folds by
record, R12's probe per tab fit on 4 folds and scoring the 5th (an out-of-fold probability,
`p`), with the served tagger's p beside it (`comparator_p`). Its TSV, one row per decided photo
and tab, and its manifest are loaded here with `mv feedback load-audit-scores` (load).

Kept apart from the queue: the scores live in their own tables and are read on their own
(latest_run, scores). Loading them changes no item, decision, flag, band, suggestion or
open/decided state, and adds no seed; load checks that inside every write (fingerprint) and
undoes a chunk that would. So the queue's kept lists (feedback_lists.ItemLists, its Warmer) and
the API's ViewCache stay exactly as a fresh read would give them: the load is a write from
another connection, which ItemLists answers with a whole read of unchanged rows.

Disagreement for tab T (photo_review.audit_disagreement), on a decided photo with audit score p_T:
  left out for T's reason (photo_review.TAB_OF_REASON)  ->  1 - p_T  (the tagger says it is not T)
  kept, or left out for another reason                  ->  p_T      (the tagger says it is T)
The Decided list of a tab sorts by it ("Strongest disagreement first"). The tab's list holds the
photos flagged for it plus those left out for its reason; the photos most likely MISSED from it
(decided elsewhere, no flag for T, but a high p_T) are the outcome filter "High score, not in this
tab" (HIGH_NOT_IN_TAB), cut at the run's own High point for T: where the out-of-fold precision
first reaches 85% (r12_audit's cut_points), else HIGH_FALLBACK.

What load refuses (nothing is written): a TSV whose sha256 is not the one given, or not the
manifest's; a manifest of another format; a TSV already loaded; a bad header, value, tab or
order (rows sorted by item id, then tab: no row twice); an item the store does not have, or whose
photo is not the row's; a failing `mv feedback seal-check`; an item on a held record (the E17
hold list, by the seal gate: never named); an item on a v1-test record (a pinned key list); and,
with no v1-test list, any item whose record is not a v1 train/val record (--only-v1 skips those
instead).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Callable

from . import feedback as fb
from . import photo_review as pr

FORMAT = "mycomap-vision photo-audit-scores 1"
FIELDS = ("item_id", "photo_key", "record_key", "group", "tab", "label", "p", "p_image",
          "comparator", "comparator_p", "fold", "model")
# The outcome filter's code for the photos a tab may have missed.
HIGH_NOT_IN_TAB = pr.HIGH_NOT_IN_TAB
# The High cut when a run gives none for a tab.
HIGH_FALLBACK = 0.5
# Rows written per transaction: short writes, so the review keeps saving during a load.
CHUNK = 50_000

SCHEMA = """
create table if not exists photo_audit_runs (
  id integer primary key, run_id text not null, tsv_sha256 text not null unique,
  manifest_sha256 text not null, manifest text not null, rows integer not null default 0,
  items integer not null default 0, tabs text not null default '[]',
  high text not null default '{}', comparators text not null default '{}',
  status text not null default 'loading', loaded_at real not null, loaded_by text
);
create table if not exists photo_audit_scores (
  run integer not null references photo_audit_runs(id), tab text not null,
  item_id integer not null, p real not null, p_image real, comparator_p real,
  primary key (run, tab, item_id)
) without rowid;
"""


class LoadRefused(ValueError):
    pass


def ensure_schema(conn) -> None:
    conn.executescript(SCHEMA)


def _has_tables(conn) -> bool:
    return conn.execute("select count(*) from sqlite_master where type = 'table' and name in "
                        "('photo_audit_runs', 'photo_audit_scores')").fetchone()[0] == 2


def latest_run(conn) -> dict | None:
    """The latest completely loaded run, or None (no run, or the tables not made yet)."""
    if not _has_tables(conn):
        return None
    r = conn.execute("select id, run_id, tsv_sha256, rows, items, tabs, high, comparators, "
                     "loaded_at from photo_audit_runs where status = 'complete' "
                     "order by id desc limit 1").fetchone()
    if r is None:
        return None
    return {"id": r[0], "run_id": r[1], "tsv_sha256": r[2], "rows": r[3], "items": r[4],
            "tabs": json.loads(r[5]), "high": json.loads(r[6]), "comparators": json.loads(r[7]),
            "loaded_at": r[8]}


def latest_run_id(conn) -> int | None:
    if not _has_tables(conn):
        return None
    return conn.execute("select max(id) from photo_audit_runs where status = 'complete'").fetchone()[0]


def scores(conn, run: int, tab: str) -> dict[int, tuple[float, float | None, float | None]]:
    """item id -> (p, p_image, comparator_p) of `tab` in run `run`."""
    return {int(i): (float(p), pi, cp) for i, p, pi, cp in conn.execute(
        "select item_id, p, p_image, comparator_p from photo_audit_scores where run = ? and tab = ?",
        (run, tab))}


# --- loading ----------------------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_key_list(path: str | Path, sha256: str | None) -> frozenset:
    """A pinned record key list (one a line): refused unless its raw bytes have `sha256`."""
    raw = Path(path).read_bytes()
    if not sha256 or hashlib.sha256(raw).hexdigest() != sha256.strip().lower():
        raise LoadRefused("the v1-test key list's sha256 does not match --v1-test-sha256")
    return frozenset(x.strip() for x in raw.decode("utf-8").split("\n") if x.strip())


def fingerprint(conn) -> tuple:
    """What a load must leave as it was: the items (how many, ids, open/decided, rules,
    evidence), the decisions and the seeds. Cheap aggregates, read inside the write."""
    return tuple(conn.execute(
        "select (select count(*) from items), (select coalesce(max(id), 0) from items), "
        "(select total(status = 'open') from items), (select total(length(status) * id) from items), "
        "(select total(length(rule)) from items), (select total(length(evidence)) from items), "
        "(select total(length(coalesce(current, '')) + length(coalesce(proposed, ''))) from items), "
        "(select count(*) from decisions), (select coalesce(max(id), 0) from decisions), "
        "(select total(at) from decisions), "
        "(select total(length(coalesce(value, '')) * id) from decisions), "
        "(select count(*) from seeds), (select max(at) from seeds)").fetchone())


def _num(text: str, what: str, line: int, blank_ok: bool = False) -> float | None:
    if text == "" and blank_ok:
        return None
    try:
        v = float(text)
    except ValueError:
        raise LoadRefused(f"line {line}: {what} {text!r} is not a number") from None
    if not 0.0 <= v <= 1.0:
        raise LoadRefused(f"line {line}: {what} {v} is outside 0..1")
    return v


def _rows(path: Path):
    """(line, item_id, tab, p, p_image, comparator_p, photo_key, label, comparator) per row,
    checked: header, values, tabs and order."""
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        head = next(rd, None)
        if tuple(head or ()) != FIELDS:
            raise LoadRefused(f"the header is not {', '.join(FIELDS)}")
        col = {c: i for i, c in enumerate(FIELDS)}
        prev = None
        for line, r in enumerate(rd, start=2):
            if len(r) != len(FIELDS):
                raise LoadRefused(f"line {line}: {len(r)} columns, not {len(FIELDS)}")
            try:
                item_id = int(r[col["item_id"]])
            except ValueError:
                raise LoadRefused(f"line {line}: item_id {r[col['item_id']]!r}") from None
            tab = r[col["tab"]]
            if tab not in pr.TAB_CODES:
                raise LoadRefused(f"line {line}: unknown tab {tab!r}")
            key = (item_id, tab)
            if prev is not None and key <= prev:
                raise LoadRefused(f"line {line}: rows must be sorted by item_id then tab, "
                                  f"each once ({item_id} {tab})")
            prev = key
            yield (line, item_id, tab, _num(r[col["p"]], "p", line),
                   _num(r[col["p_image"]], "p_image", line, True),
                   _num(r[col["comparator_p"]], "comparator_p", line, True),
                   r[col["photo_key"]], r[col["label"]], r[col["comparator"]])


def load(store: fb.Store, tsv: str | Path, expect_sha256: str, gate: fb.SealGate,
         manifest: str | Path | None = None, v1_test_keys: frozenset | None = None,
         only_v1: bool = False, dry_run: bool = False, loaded_by: str | None = None,
         _in_write: Callable | None = None) -> dict:
    """Load an audit run (module doc: what is refused). Every check runs before anything is
    written; the rows then go in CHUNK at a time, each chunk in its own write that is undone
    (and the load refused) if the items, decisions or seeds read differently at its end than at
    its start. The run counts only once all its rows are in (status 'complete').
    `_in_write(conn)` runs inside each write, for the tests."""
    tsv = Path(tsv)
    manifest = Path(manifest) if manifest else tsv.with_suffix(".json")
    got = sha256_of(tsv)
    if got != (expect_sha256 or "").strip().lower():
        raise LoadRefused(f"the TSV's sha256 is {got[:12]}..., not the one given")
    if not manifest.exists():
        raise LoadRefused(f"no manifest at {manifest}")
    msha = sha256_of(manifest)
    try:
        man = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError:
        raise LoadRefused("the manifest is not JSON") from None
    if man.get("format") != FORMAT:
        raise LoadRefused(f"the manifest's format is not {FORMAT!r}")
    if man.get("tsv_sha256") != got:
        raise LoadRefused("the manifest names another TSV (its tsv_sha256 differs)")
    if _has_tables(store.conn):
        old = store.conn.execute("select id, status from photo_audit_runs where tsv_sha256 = ?",
                                 (got,)).fetchone()
        if old is not None:
            raise LoadRefused(f"this TSV is already loaded (run {old[0]}, {old[1]})")

    # pass 1: the file
    photo_of: dict[int, str] = {}
    label_of: dict[int, str] = {}
    tabs, comparators, rows = set(), {}, 0
    for _, item_id, tab, _, _, _, key, label, comp in _rows(tsv):
        if photo_of.setdefault(item_id, key) != key:
            raise LoadRefused(f"item {item_id} names two photos")
        label_of[item_id] = label
        tabs.add(tab)
        if comp:
            comparators[tab] = comp
        rows += 1
    if not rows:
        raise LoadRefused("the TSV has no rows")

    # the items, the seal and the hold
    items = {}
    for r in store.conn.execute("select * from items where item_type = ?", (pr.ITEM_TYPE,)):
        it = fb.Store._item(r)
        items[it["id"]] = it
    last = {}
    for i, d, v in store.conn.execute("select item_id, decision, value from decisions order by id"):
        last[i] = {"decision": d, "value": v}            # the latest one wins
    unknown = sorted(i for i in photo_of if i not in items)
    if unknown:
        raise LoadRefused(f"{len(unknown)} items are not photo items of this store "
                          f"(first: {', '.join(map(str, unknown[:5]))})")
    wrong = sorted(i for i, k in photo_of.items() if items[i]["subject"] != k)
    if wrong:
        raise LoadRefused(f"{len(wrong)} items are other photos than the TSV says "
                          f"(first: {', '.join(map(str, wrong[:5]))}): a TSV of another store?")
    check = fb.seal_check(store, gate)
    if not check["ok"]:
        raise LoadRefused("mv feedback seal-check fails on this store: fix that first "
                          f"(held {len(check['held'])}, sealed {len(check['sealed'])}, "
                          f"unknown {len(check['unknown'])})")
    skip, held, on_test, not_v1 = set(), 0, [], []
    for i in photo_of:
        it = items[i]
        if gate.judge_row(it)[0] == "held":
            held += 1
            continue
        rec = pr.record_of_item(it) or ""
        if v1_test_keys is not None and rec in v1_test_keys:
            on_test.append(i)
        elif v1_test_keys is None and rec not in gate.v1_keys:
            not_v1.append(i)
    if held:
        raise LoadRefused(f"{held} items are on held records (the E17 hold list)")
    if on_test:
        raise LoadRefused(f"{len(on_test)} items are on v1-test records "
                          f"(first: {', '.join(map(str, sorted(on_test)[:5]))})")
    if not_v1:
        if not only_v1:
            raise LoadRefused(f"{len(not_v1)} items are not on v1 train/val records: give the "
                              "pinned v1-test key list (--v1-test-keys, --v1-test-sha256) to load "
                              "them, or --only-v1 to leave them out")
        skip = set(not_v1)
    high = {t: float(((man.get("tabs") or {}).get(t) or {}).get("cuts", {}).get("high", HIGH_FALLBACK))
            for t in sorted(tabs)}
    changed = sum(1 for i in photo_of
                  if items[i]["status"] == "decided" and _label(last.get(i)) != label_of[i])
    report = {"run_id": man.get("run_id"), "tsv_sha256": got, "manifest_sha256": msha,
              "rows": rows, "items": len(photo_of), "tabs": sorted(tabs), "high": high,
              "comparators": comparators, "skipped_not_v1": len(skip),
              "rows_loaded": sum(len(tabs) for i in photo_of if i not in skip),
              "open_now": sum(1 for i in photo_of if items[i]["status"] == "open"),
              "decision_changed_since_scored": changed, "dry_run": dry_run,
              "seal_check_ok": True}
    if dry_run:
        return report

    # pass 2: the writes
    ensure_schema(store.conn)
    with store.conn:
        run = store.conn.execute(
            "insert into photo_audit_runs (run_id, tsv_sha256, manifest_sha256, manifest, tabs, "
            "high, comparators, loaded_at, loaded_by) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (man.get("run_id") or "", got, msha, json.dumps(man), json.dumps(sorted(tabs)),
             json.dumps(high), json.dumps(comparators), time.time(), loaded_by)).lastrowid

    def write(chunk: list[tuple]) -> None:
        with store.conn:
            store.conn.execute("begin immediate")
            before = fingerprint(store.conn)
            store.conn.executemany(
                "insert into photo_audit_scores (run, tab, item_id, p, p_image, comparator_p) "
                "values (?, ?, ?, ?, ?, ?)", chunk)
            if _in_write is not None:
                _in_write(store.conn)
            if fingerprint(store.conn) != before:
                raise LoadRefused("a write changed the queue's items, decisions or seeds: undone")

    try:
        chunk = []
        for _, item_id, tab, p, pi, cp, *_ in _rows(tsv):
            if item_id in skip:
                continue
            chunk.append((run, tab, item_id, p, pi, cp))
            if len(chunk) >= CHUNK:
                write(chunk)
                chunk = []
        if chunk:
            write(chunk)
    except BaseException:
        with store.conn:
            store.conn.execute("delete from photo_audit_scores where run = ?", (run,))
            store.conn.execute("delete from photo_audit_runs where id = ?", (run,))
        raise
    with store.conn:
        store.conn.execute("update photo_audit_runs set status = 'complete', rows = ?, items = ? "
                           "where id = ?", (report["rows_loaded"], len(photo_of) - len(skip), run))
    return {**report, "run": run}


def _label(last: dict | None) -> str:
    """The research label of a latest decision (r12 label_of): the reason's tab, "other" for a
    reason with none, "specimen" when kept."""
    last = last or {}
    if last.get("decision") == "keep_photo":
        return "specimen"
    if last.get("decision") == "exclude_photo":
        v = last.get("value") or ""
        tab = pr.TAB_OF_REASON.get(v)
        return "other" if tab is None or v in ("other", "blurry", "species_description") else tab
    return ""
