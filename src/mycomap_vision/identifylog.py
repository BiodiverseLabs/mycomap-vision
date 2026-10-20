"""The identification log: one row for every identification Vision answers or serves, kept
append-only (Steve, 2026-10-20).

Why: the paper's sealed test, v1-test, is the records first validated after 2026-10-12
13:17:51 UTC. Its truth-independence check asks, per test observation, whether Vision
had answered it before its first green validation, since a validator might have seen
the answer. Nothing else records that: nginx keeps the observation id only on the v1
path, observation-answers.sqlite keeps only the latest answer per key (a rerun
overwrites answered_at), and run links cover the site's runs only.

One row per answer: when (UTC), the path (PATHS), the source and observation id where
there is one, whether it was answered now or a kept answer was served (`fresh`), the
answer key, answers version, model and release, who asked (`asked_by_*`: the mycomap.org
id and name of a signed-in site user; the credential kind and its key id or pseudonymous
subject for API v1 and the app; the reviewer for Human Feedback), the caller's
client_ref, and the run link's id if any.

Rows are only ever inserted. The table refuses UPDATE and DELETE itself (triggers), this
module has no code that does either, and tests/test_identify_log.py fails if any code in
the package does. Kept until the paper closes; retention is forever for now.

Writing never fails or slows an identification: `record` puts the row on a queue and
returns; one writer thread inserts in batches. A write that fails is retried, then
counted and reported (`state()`, /api/health with the server key), never raised.

Rows recovered from older stores (`import_answers`, `import_runs`) say so: `provenance`
names the store and `at_exact` is 0, meaning Vision first answered at or before `at`.
Nothing is made up: a store that does not say when, or about which observation, gives no
row.

On the box: state/identify-log.sqlite, beside vision-usage.sqlite and
observation-answers.sqlite. Off the box: beside the manifest. Private: who asked is in
it, so it is never served whole; `first_answers` (GET /api/identify-log/first with the
server key or as an admin, or `mv identify-log first`) gives per observation only when
Vision first answered it, and how.
"""

from __future__ import annotations

import queue
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import config

FILE = "identify-log.sqlite"

# Where an answer was given (docs/api.md §11).
SITE_UPLOAD = "site_upload"                  # POST /api/identify (the Identify page, photos)
SITE_OBSERVATION = "site_observation"        # POST /api/identify/observation ("from a link")
V1_IDENTIFY = "v1_identify"                  # POST /api/v1/identify (photos)
V1_OBSERVATION = "v1_observation"            # POST /api/v1/identify/{inat|mo}/{id}
V1_ANSWERS = "v1_answers"                    # GET /api/v1/answers/{inat|mo}/{id} (kept only)
APP_SESSION = "app_session"                  # /api/v1/sessions: each new combined answer
FEEDBACK_RESCORE = "feedback_rescore"        # Human Feedback's reveal and refresh
PATHS = (SITE_UPLOAD, SITE_OBSERVATION, V1_IDENTIFY, V1_OBSERVATION, V1_ANSWERS,
         APP_SESSION, FEEDBACK_RESCORE)

LIVE = "live"
FROM_ANSWERS = "recovered:observation-answers"   # its answered_at: first answered at or before
FROM_RUNS = "recovered:run-links"                # a run's answered_at: at or before
PROVENANCES = (LIVE, FROM_ANSWERS, FROM_RUNS)

SCHEMA = """
create table if not exists identify_log (
    id              integer primary key autoincrement,
    at              real not null,     -- unix seconds, UTC
    at_utc          text not null,     -- the same, ISO 8601
    at_exact        integer not null,  -- 1: answered then; 0: at or before (recovered rows)
    path            text not null,     -- identifylog.PATHS
    source          text,              -- inat | mo, where there is an observation
    observation_id  text,
    observation_from text,             -- request | client_ref | answer_key | run
    fresh           integer not null,  -- 1: scored now; 0: a kept answer served
    answer_key      text,
    answers_version text,
    model           text,              -- backbone/method
    release         text,
    asked_by_kind   text,              -- site | anonymous | server | app | key | reviewer
    asked_by_id     text,              -- mycomap.org user id, key id or pseudonymous subject
    asked_by_name   text,              -- a signed-in person's mycomap.org name
    client_ref      text,
    run_id          text,
    provenance      text not null      -- identifylog.PROVENANCES
);
create index if not exists identify_log_by_observation
    on identify_log (source, observation_id, at);
create trigger if not exists identify_log_no_update before update on identify_log
begin select raise(abort, 'identify_log is append-only: no UPDATE'); end;
create trigger if not exists identify_log_no_delete before delete on identify_log
begin select raise(abort, 'identify_log is append-only: no DELETE'); end;
"""

COLUMNS = ("at", "at_utc", "at_exact", "path", "source", "observation_id", "observation_from",
           "fresh", "answer_key", "answers_version", "model", "release", "asked_by_kind",
           "asked_by_id", "asked_by_name", "client_ref", "run_id", "provenance")
INSERT = (f"insert into identify_log ({', '.join(COLUMNS)}) "
          f"values ({', '.join('?' for _ in COLUMNS)})")

_OBSERVATION_REF = re.compile(r"^(inat|mo):(\d{1,12})$")


def iso_utc(t: float) -> str:
    from .answercache import iso_utc as iso
    return iso(t)


def default_path(manifest_path: Path | None = None) -> Path:
    """state/identify-log.sqlite on the box; beside the manifest elsewhere."""
    if config.RELEASE_ROOT:
        return config.RELEASE_ROOT / "state" / FILE
    return Path(manifest_path or config.MANIFEST_PATH).parent / FILE


@dataclass(frozen=True)
class Asker:
    """Who asked: `kind` site (a signed-in mycomap.org user), anonymous, server
    (mycomap.org's key), app, key (a developer token) or reviewer (Human Feedback)."""
    kind: str
    id: str | None = None
    name: str | None = None


def site_asker(person: dict | None) -> Asker:
    """A site visitor, from the signed session: mycomap.org id and name."""
    if not person or not person.get("sub"):
        return Asker("anonymous")
    return Asker("site", str(person["sub"]), person.get("name"))


def credential_asker(cred) -> Asker:
    """An API v1 caller: mycomap.org's server key, the app (its member's pseudonymous
    subject) or a developer token (its key id)."""
    kind = getattr(cred, "kind", None) or "unknown"
    if kind == "key":
        return Asker("key", getattr(cred, "key_id", None) or getattr(cred, "subject", None))
    return Asker(kind, getattr(cred, "subject", None))


def observation_of_ref(client_ref: str | None) -> tuple[str, str] | None:
    """(source, id) when a caller's client_ref names an observation ("inat:123", "mo:45",
    as by-observation's default), else None."""
    m = _OBSERVATION_REF.match(client_ref or "")
    return (m.group(1), m.group(2)) if m else None


def make_row(path: str, at: float, *, fresh: bool, source: str | None = None,
             observation_id: str | None = None, observation_from: str | None = None,
             answer_key: str | None = None, answers_version: str | None = None,
             model: str | None = None, release: str | None = None,
             asked_by: Asker | None = None, client_ref: str | None = None,
             run_id: str | None = None, provenance: str = LIVE, at_exact: bool = True) -> tuple:
    if path not in PATHS:
        raise ValueError(f"unknown path {path!r}")
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown provenance {provenance!r}")
    if observation_id is None and source is None:
        ref = observation_of_ref(client_ref)
        if ref is not None:
            (source, observation_id), observation_from = ref, "client_ref"
    if observation_id is not None and observation_from is None:
        observation_from = "request"
    asked_by = asked_by or Asker("unknown")

    def text(v):
        return None if v is None else str(v)
    return (float(at), iso_utc(at), int(bool(at_exact)), path, text(source),
            text(observation_id), observation_from, int(bool(fresh)), text(answer_key),
            text(answers_version), text(model), text(release), asked_by.kind,
            text(asked_by.id), text(asked_by.name), text(client_ref), text(run_id), provenance)


def connect(path: Path | None) -> sqlite3.Connection:
    """The log's database with its table and guards; `None` keeps it in memory."""
    if path is None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        conn.execute("pragma journal_mode = wal")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


class IdentifyLog:
    """The live log. `record` never raises and never waits on the database: rows go on a
    queue (at most `max_pending`; beyond that they are counted as dropped and reported)
    and one writer thread, started by the first row, inserts them. `start=False` leaves
    the writer off (tests drive it with `drain`)."""

    def __init__(self, path: Path | None, note: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.time, max_pending: int = 20000,
                 retries: tuple[float, ...] = (1.0, 5.0, 30.0), start: bool = True,
                 conn: sqlite3.Connection | None = None):
        self.path, self.note, self.clock, self.retries = path, note, clock, retries
        self.conn = conn if conn is not None else connect(path)
        self._db = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._stats = {"written": 0, "dropped": 0, "failed": 0, "last_error": None}
        self._stats_lock = threading.Lock()
        self._idle = threading.Condition()
        self._busy = 0
        self._thread = None                   # started by the first record
        self._autostart = start
        self._start_lock = threading.Lock()

    # --- the request path: never raises, never waits ---------------------------------------

    def record(self, path: str, **fields) -> None:
        """Logs one answer, stamped now. Returns at once whatever happens: the row is made
        on the writer thread, where a field given as a function (a release read from the
        model card, say) is called."""
        try:
            entry = (path, self.clock(), fields)
            self._ensure_writer()
        except Exception as e:  # noqa: BLE001 - reported, never raised
            self._count("failed", f"{type(e).__name__}: {e}")
            return
        with self._idle:
            self._busy += 1
        try:
            self._queue.put_nowait(entry)
            return
        except queue.Full:
            n = self._count("dropped", "the queue is full")
            if n == 1 or n % 1000 == 0:
                self.note(f"[identify-log] queue full: {n} rows dropped so far")
        except Exception as e:  # noqa: BLE001
            self._count("failed", f"{type(e).__name__}: {e}")
        with self._idle:
            self._busy -= 1
            self._idle.notify_all()

    def _ensure_writer(self) -> None:
        if self._thread is None and self._autostart:
            with self._start_lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._writer, name="identify-log",
                                                    daemon=True)
                    self._thread.start()

    # --- the writer ------------------------------------------------------------------------

    def _writer(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < 500:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._write(batch, sleep=time.sleep)

    def drain(self) -> int:
        """Writes what is queued, on this thread (tests; a writer-less log). Rows written."""
        batch = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._write(batch, sleep=lambda s: None)
        return len(batch)

    def _write(self, batch: list[tuple], sleep) -> None:
        try:
            rows = []
            for path, at, fields in batch:
                try:
                    rows.append(make_row(path, at, **{k: v() if callable(v) else v
                                                      for k, v in fields.items()}))
                except Exception as e:  # noqa: BLE001 - this row only; counted
                    self._count("failed", f"{type(e).__name__}: {e}")
                    self.note(f"[identify-log] a {path} row could not be made: "
                              f"{type(e).__name__}: {e}")
            if not rows:
                return
            for attempt, wait in enumerate((*self.retries, None)):
                try:
                    self.insert(rows)
                    with self._stats_lock:
                        self._stats["written"] += len(rows)
                    return
                except Exception as e:  # noqa: BLE001 - retried, then counted
                    error = f"{type(e).__name__}: {e}"
                    if wait is None:
                        self._count("failed", error, len(rows))
                        self.note(f"[identify-log] {len(rows)} rows not written after "
                                  f"{attempt + 1} tries: {error}")
                        return
                    sleep(wait)
        finally:
            with self._idle:
                self._busy -= len(batch)
                self._idle.notify_all()

    def insert(self, rows: Iterable[tuple]) -> None:
        """Inserts rows (make_row tuples). The one way rows are written."""
        with self._db:
            with self.conn:
                self.conn.executemany(INSERT, list(rows))

    def flush(self, timeout: float = 10.0) -> bool:
        """Waits until every row recorded so far is written (or given up); False on
        timeout."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._busy > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._idle.wait(left)
        return True

    def _count(self, what: str, error: str | None, n: int = 1) -> int:
        with self._stats_lock:
            self._stats[what] += n
            if error:
                self._stats["last_error"] = error
            return self._stats[what]

    def state(self) -> dict:
        """For /api/health with the server key: rows written, waiting, dropped, failed."""
        with self._stats_lock:
            out = dict(self._stats)
        out["pending"] = self._queue.qsize()
        return out

    def first_answers(self, **kw) -> list[dict]:
        with self._db:
            return first_answers(self.conn, **kw)


# --- the read for the paper ---------------------------------------------------------------

def first_answers(conn: sqlite3.Connection, source: str | None = None,
                  since: float | None = None) -> list[dict]:
    """Per observation (source, id), the first time Vision answered it: `first_at` (the
    earliest row of any provenance), whether that time is exact or "at or before"
    (`first_exact`), its path, whether it was scored then (`first_fresh`) and its
    provenance; `first_live_at` and `first_live_path`, from the live log alone; and how
    many rows it has. `since`: only observations whose first answer is at or after this
    time (unix seconds). Ordered by first answer."""
    where, params = ["observation_id is not null"], []
    if source:
        where.append("source = ?")
        params.append(source)
    sql = f"""
    with rows as (
        select * from identify_log where {' and '.join(where)}
    ), ranked as (
        select *, row_number() over (partition by source, observation_id
                                     order by at, id) as n
        from rows
    ), live as (
        select source, observation_id, at, path,
               row_number() over (partition by source, observation_id order by at, id) as n
        from rows where provenance = 'live'
    ), counts as (
        select source, observation_id, count(*) as rows_ from rows group by 1, 2
    )
    select r.source, r.observation_id, r.at, r.at_utc, r.at_exact, r.path, r.fresh,
           r.provenance, l.at, l.path, c.rows_
    from ranked r
    join counts c on c.source = r.source and c.observation_id = r.observation_id
    left join live l on l.source = r.source and l.observation_id = r.observation_id
                    and l.n = 1
    where r.n = 1 {"and r.at >= ?" if since is not None else ""}
    order by r.at, r.source, r.observation_id
    """
    if since is not None:
        params.append(float(since))
    out = []
    for row in conn.execute(sql, params):
        out.append({"source": row[0], "observation_id": row[1], "first_at": row[3],
                    "first_exact": bool(row[4]), "first_path": row[5],
                    "first_fresh": bool(row[6]), "first_provenance": row[7],
                    "first_live_at": iso_utc(row[8]) if row[8] is not None else None,
                    "first_live_path": row[9], "rows": row[10]})
    return out


def summary(conn: sqlite3.Connection) -> dict:
    """Rows per provenance and path, with the first and last time (`mv identify-log`)."""
    out: dict = {}
    for prov, path, n, first, last in conn.execute(
            "select provenance, path, count(*), min(at), max(at) from identify_log "
            "group by 1, 2 order by 1, 2"):
        out.setdefault(prov, {})[path] = {"rows": n, "first": iso_utc(first),
                                          "last": iso_utc(last)}
    return out


FIRST_ABOUT = ("Per observation, the first time Vision answered it. first_exact false: a row "
               "recovered from an older store (first_provenance), meaning Vision first "
               "answered it at or before first_at. first_live_* is from the live log alone, "
               "which starts when it was deployed; a first row with first_fresh false is a "
               "kept answer served, so Vision had answered it earlier, before the log. "
               "Uploaded photos carry no observation id and are not here.")

FIRST_COLUMNS = ("source", "observation_id", "first_at", "first_exact", "first_path",
                 "first_fresh", "first_provenance", "first_live_at", "first_live_path", "rows")


def as_tsv(rows: list[dict]) -> str:
    def cell(v):
        return "" if v is None else str(int(v)) if isinstance(v, bool) else str(v)
    lines = ["\t".join(FIRST_COLUMNS)]
    lines += ["\t".join(cell(r[c]) for c in FIRST_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


# --- one-off recovery of what older stores hold (labelled, never exact) -------------------

def _already(conn: sqlite3.Connection, provenance: str) -> int:
    return conn.execute("select count(*) from identify_log where provenance = ?",
                        (provenance,)).fetchone()[0]


def import_answers(log: IdentifyLog, answers_path: Path) -> int:
    """Rows from observation-answers.sqlite: one per kept answer, at its answered_at, which
    is the LATEST time that key was answered (a rerun overwrites it), so Vision first
    answered it at or before then (`at_exact` 0, provenance FROM_ANSWERS). Refused when
    this store was imported before. Rows added."""
    if _already(log.conn, FROM_ANSWERS):
        raise ValueError("observation answers were imported already: the log is append-only, "
                         "so they are not imported twice")
    src = sqlite3.connect(Path(answers_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        cols = {r[1] for r in src.execute("pragma table_info(observation_answers)")}
        who = "asked_by" if "asked_by" in cols else "null"
        found = src.execute(f"select answer_key, version, answered_at, {who} "
                            "from observation_answers").fetchall()
    finally:
        src.close()
    rows = []
    for key, version, answered_at, asked in found:
        parts = str(key).split(":")
        if len(parts) < 4 or parts[0] not in ("site", "v1") or parts[1] not in ("inat", "mo"):
            continue
        kind, source, number, model = parts[0], parts[1], parts[2], parts[3]
        rows.append(make_row(
            SITE_OBSERVATION if kind == "site" else V1_OBSERVATION, float(answered_at),
            fresh=True, source=source, observation_id=number, observation_from="answer_key",
            answer_key=key, answers_version=version, model=model,
            asked_by=Asker(_answers_asker(asked)), provenance=FROM_ANSWERS, at_exact=False))
    log.insert(rows)
    return len(rows)


def _answers_asker(asked: str | None) -> str:
    """observation_answers.asked_by (site | mycomap.org | app | api) as an Asker kind."""
    return {"site": "site", "mycomap.org": "server", "app": "app", "api": "key"}.get(
        asked or "", "unknown")


def import_runs(log: IdentifyLog, index_path: Path, get_run: Callable[[str], dict | None]
                ) -> int:
    """Rows from the run links: every run the owner index (site-runs.sqlite) lists, read
    through `get_run` (RunStore.get), at the run's answered_at (`at_exact` 0: a run of a
    kept observation answer carries that answer's time), with who ran it. A run nobody
    signed in for is in no index and is not found. Refused when imported before."""
    if _already(log.conn, FROM_RUNS):
        raise ValueError("run links were imported already: the log is append-only, so they "
                         "are not imported twice")
    from datetime import datetime
    src = sqlite3.connect(Path(index_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        listed = src.execute("select run_id, min(created_at) from site_runs "
                             "group by run_id order by 2").fetchall()
    finally:
        src.close()
    rows = []
    for rid, _created in listed:
        run = get_run(rid)
        if not run or not run.get("answered_at"):
            continue
        try:
            at = datetime.fromisoformat(str(run["answered_at"]).replace("Z", "+00:00")
                                        ).timestamp()
        except ValueError:
            continue
        obs = run.get("observation") or {}
        owner = run.get("owner") or {}
        asker = (Asker("site", str(owner["id"]), owner.get("name")) if owner.get("id")
                 else Asker("anonymous"))
        for part in run.get("models") or [{}]:
            m = (part.get("result") or {}).get("model") or {}
            model = (f"{m.get('backbone')}/{m.get('method')}"
                     if m.get("backbone") and m.get("method") else None)
            rows.append(make_row(
                SITE_OBSERVATION if run.get("kind") == "observation" else SITE_UPLOAD, at,
                fresh=True, source=obs.get("source"), observation_id=obs.get("id"),
                observation_from="run" if obs.get("id") else None, model=model,
                release=run.get("release"), asked_by=asker, run_id=rid,
                provenance=FROM_RUNS, at_exact=False))
    log.insert(rows)
    return len(rows)
