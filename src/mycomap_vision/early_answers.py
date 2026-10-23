"""Answered before the DNA verdict: Vision's answers about an observation, frozen when they
are given, scored once the record turns green (Steve, 2026-10-23: "Score auto-validate
answers").

When mycomap.org's auto-validation runs on a project, every iNat and MO record in the job is
sent to Vision by observation (POST /api/v1/identify/{inat|mo}/{id}, .org's server key), and
Vision answers before the record has its DNA verdict. People ask the same way on the site
("from a link", POST /api/identify/observation). Each such fresh answer is kept here, its
ranked names DEPTH deep (identify.DEEP_RANKS) whatever the caller was shown, with when,
which model, release and answers version, the path, who asked (only the kind: mycomap.org,
site, app, api; never a person) and whether the observation or a twin of it was in the
reference and left out. No photo is kept.

The kept answers are a different store from the answer cache (answercache.py): that one
holds the latest answer per key and a rerun overwrites it. Here every fresh answer is a new
row and rows are never changed or deleted (the table refuses both); only the FIRST answer per
observation and model is scored, the earliest, so it is surely the one before the verdict.

Scoring (score_pending, nightly after the green records are read): a first answer is scored
once its record is green, against the DNA-validated name, by the advance predictions' rules
(prospective.report: the manifest's labels, taxonomy.labels_for, guests and non-fungi left
out) and name_equiv's answer key (strict, s.l., complex), at top 1/3/5/10. It is scored
only if it was given before the record's first green (green_moment). Left out, and counted:

    answered_after_verdict   given at or after the first green moment
    in_reference             the observation was itself a reference record when answered
                             (it was green then; answered with itself left out)
    twin_in_reference        a twin or cross-post of it was in the reference (photos left out)
    vision_assisted          green by .org's DNA + MycoMap Vision rule: Vision's own agreement
                             made the verdict, so scoring Vision on it is circular
                             (records.VISION_ASSISTED_FIELD)
    not_a_fungus_label       a guest genus or no fungus (as advance predictions)
    label_conflict           .org holds two names for it (waits, like the advance predictions)

The green feed (GET /api/vision/green-records) gives each slot's validation DAY
(validation_date_N, UTC, no time), and Vision knows when its nightly first listed the record
green (records.first_seen_at). Without a moment from .org (records.FIRST_GREEN_FIELD) the
answer must come before the start of the earliest green day, and before Vision first saw it
green: the most conservative rule those allow. With the moment, before it.

The paper's sealed test (v1-test: records first green after prospective.SEAL_DEFAULT) is not
scored while the seal holds (prospective.seal_time()): such answers are only counted, and a
report never carries a sealed score (research_public.SEAL on every page). A record counts as
sealed unless something shows it green before the seal.

State: state/early-answers.sqlite on the box (it outlives releases), beside the manifest
elsewhere. Aggregates only leave it (report; research_public.early_answers).
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from . import config

FILE = "early-answers.sqlite"
DEPTH = 10                         # identify.DEEP_RANKS
RANKS = ("species", "genus", "family")
KS = (1, 3, 5, 10)

V1_OBSERVATION = "v1_observation"           # identifylog.V1_OBSERVATION
SITE_OBSERVATION = "site_observation"       # identifylog.SITE_OBSERVATION
PATHS = (V1_OBSERVATION, SITE_OBSERVATION)
ASKERS = ("mycomap.org", "site", "app", "api")

LIVE = "live"
FROM_ANSWERS = "recovered:observation-answers"      # the answer cache's kept answer
FROM_ORG_RUNS = "recovered:org-record-vision-runs"  # .org's saved copy of a v1 answer
PROVENANCES = (LIVE, FROM_ANSWERS, FROM_ORG_RUNS)

SCORED = "scored"
AFTER = "answered_after_verdict"
IN_REFERENCE = "in_reference"
TWIN = "twin_in_reference"
VISION_ASSISTED = "vision_assisted"
NOT_FUNGUS = "not_a_fungus_label"
OUTCOMES = (SCORED, AFTER, IN_REFERENCE, TWIN, VISION_ASSISTED, NOT_FUNGUS)

SCHEMA = """
create table if not exists early_answers (
    id               integer primary key autoincrement,
    source           text not null,      -- inat | mo
    observation_id   text not null,
    model            text not null,      -- backbone/method
    answered_at      real not null,      -- unix seconds, UTC: when the model ran
    answered_utc     text not null,
    path             text not null,      -- early_answers.PATHS
    asked_by         text,               -- mycomap.org | site | app | api (never who)
    release          text,
    answers_version  text,
    code_version     text,
    place_used       integer,
    date_used        integer,
    excluded_reference integer,          -- 1: itself a reference record, left out of its answer
    left_out_photos  integer,            -- reference photos left out (its own, twins'); null: not known
    depth            integer not null,   -- how deep the ranked names go
    ranks_json       text not null,      -- {"species": [{"name", "confidence"}], "genus", "family"}
    provenance       text not null,      -- early_answers.PROVENANCES
    recorded_at      real not null
);
create index if not exists early_answers_first
    on early_answers (source, observation_id, model, answered_at, id);
create trigger if not exists early_answers_no_update before update on early_answers
begin select raise(abort, 'early_answers is append-only: no UPDATE'); end;
create trigger if not exists early_answers_no_delete before delete on early_answers
begin select raise(abort, 'early_answers is append-only: no DELETE'); end;

create table if not exists early_outcomes (
    answer_id        integer primary key, -- early_answers.id: the first answer of its observation and model
    record_key       text not null,
    decided_at       text not null,
    outcome          text not null,       -- early_answers.OUTCOMES
    truth_name       text,                -- the DNA-validated name when decided
    truth_species    text,
    truth_genus      text,
    validated_on     text,
    first_seen_at    text,
    first_green_at   text,
    green_before     text,                -- the moment the answer had to precede (green_moment)
    green_rule       text,                -- moment | day
    species_reference_records integer,    -- the true species' records when scored, itself left out
    sp_strict integer, sp_sl integer, sp_complex integer,   -- 0-based place of the first right
    ge_strict integer, ge_sl integer                        -- name, null: none in the list
);
create trigger if not exists early_outcomes_no_update before update on early_outcomes
begin select raise(abort, 'early_outcomes is a ledger: no UPDATE'); end;
create trigger if not exists early_outcomes_no_delete before delete on early_outcomes
begin select raise(abort, 'early_outcomes is a ledger: no DELETE'); end;
"""

COLUMNS = ("source", "observation_id", "model", "answered_at", "answered_utc", "path",
           "asked_by", "release", "answers_version", "code_version", "place_used", "date_used",
           "excluded_reference", "left_out_photos", "depth", "ranks_json", "provenance",
           "recorded_at")
INSERT = (f"insert into early_answers ({', '.join(COLUMNS)}) "
          f"values ({', '.join('?' for _ in COLUMNS)})")


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def default_path(manifest_path: Path | None = None) -> Path:
    """state/early-answers.sqlite on the box; beside the manifest elsewhere."""
    if config.RELEASE_ROOT:
        return config.RELEASE_ROOT / "state" / FILE
    return Path(manifest_path or config.MANIFEST_PATH).parent / FILE


def connect(path: Path | None) -> sqlite3.Connection:
    if path is None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        conn.execute("pragma journal_mode = wal")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --- what is frozen ----------------------------------------------------------------------

def ranked_lists(deep: list | None, ranks: dict | None, top_k: int) -> tuple[dict, int]:
    """(the ranked names to keep, how deep they go). `deep`: identify_vectors's ten-deep
    lists when it gave them; otherwise the answer's own `ranks`, `top_k` deep, or DEPTH
    when the answer named nothing (every photo left out: there is no deeper answer)."""
    if deep:
        got = deep[0]
        return {r: list(got.get(r) or [])[:DEPTH] for r in RANKS}, DEPTH
    lists = {r: [{"name": c["name"], "confidence": c.get("confidence")}
                 for c in ((ranks or {}).get(r) or [])[:DEPTH] if isinstance(c, dict)
                 and c.get("name")] for r in RANKS}
    if not any(lists.values()):
        return lists, DEPTH
    return lists, max(1, min(int(top_k), DEPTH))


def make_row(*, source: str, observation_id: str, model: str, answered_at: float, path: str,
             lists: dict, depth: int, asked_by: str | None = None, release=None,
             answers_version=None, code_version=None, place_used=None, date_used=None,
             excluded_reference=None, left_out_photos=None, provenance: str = LIVE,
             recorded_at: float | None = None) -> tuple:
    if path not in PATHS:
        raise ValueError(f"unknown path {path!r}")
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown provenance {provenance!r}")
    if source not in ("inat", "mo") or not str(observation_id).isdigit():
        raise ValueError(f"not an iNat or MO observation: {source!r} {observation_id!r}")
    if asked_by is not None and asked_by not in ASKERS:
        raise ValueError(f"unknown asker {asked_by!r}")

    def flag(v):
        return None if v is None else int(bool(v))
    body = json.dumps({r: [{"name": str(c["name"]),
                            "confidence": (None if c.get("confidence") is None
                                           else round(float(c["confidence"]), 4))}
                           for c in (lists.get(r) or [])[:DEPTH]] for r in RANKS},
                      separators=(",", ":"))
    text = (lambda v: None if v is None else str(v))  # noqa: E731
    return (source, str(observation_id), str(model), float(answered_at), iso_utc(answered_at),
            path, asked_by, text(release), text(answers_version), text(code_version),
            flag(place_used), flag(date_used), flag(excluded_reference),
            None if left_out_photos is None else int(left_out_photos), int(depth), body,
            provenance, float(time.time() if recorded_at is None else recorded_at))


class Store:
    """The kept answers. `record` never raises and never waits on the database: rows go on a
    queue and one writer thread inserts them (`start=False`: tests drive it with `drain`)."""

    def __init__(self, path: Path | None, note: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.time, start: bool = True,
                 max_pending: int = 20000):
        self.path, self.note, self.clock = path, note, clock
        self.conn = connect(path)
        self.lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._autostart = start
        self._thread = None
        self._start_lock = threading.Lock()
        self._busy = 0
        self._idle = threading.Condition()
        self.failed = 0

    def record(self, **fields) -> None:
        """Keeps one fresh answer (make_row's fields; answered now unless `answered_at`)."""
        try:
            fields.setdefault("answered_at", self.clock())
            if self._autostart and self._thread is None:
                with self._start_lock:
                    if self._thread is None:
                        self._thread = threading.Thread(target=self._writer, daemon=True,
                                                        name="early-answers")
                        self._thread.start()
            with self._idle:
                self._busy += 1
            try:
                self._queue.put_nowait(fields)
            except queue.Full:
                with self._idle:
                    self._busy -= 1
                self._failed("the queue is full")
        except Exception as e:  # noqa: BLE001 - never fails an answer
            self._failed(f"{type(e).__name__}: {e}")

    def _failed(self, why: str) -> None:
        self.failed += 1
        if self.failed == 1 or self.failed % 1000 == 0:
            self.note(f"[early-answers] {self.failed} answers not kept; last: {why}")

    def _writer(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < 200:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._write(batch)

    def drain(self) -> int:
        batch = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._write(batch)
        return len(batch)

    def _write(self, batch: list[dict]) -> None:
        try:
            rows = []
            for fields in batch:
                try:
                    rows.append(make_row(**fields))
                except Exception as e:  # noqa: BLE001 - this row only
                    self._failed(f"{type(e).__name__}: {e}")
            if rows:
                try:
                    self.insert(rows)
                except Exception as e:  # noqa: BLE001
                    self._failed(f"{type(e).__name__}: {e}")
        finally:
            with self._idle:
                self._busy -= len(batch)
                self._idle.notify_all()

    def flush(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._busy > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._idle.wait(left)
        return True

    def insert(self, rows: Iterable[tuple]) -> None:
        with self.lock:
            with self.conn:
                self.conn.executemany(INSERT, list(rows))


# --- the first answer --------------------------------------------------------------------

def first_answers(conn: sqlite3.Connection) -> list[dict]:
    """Per (source, observation, model) its earliest answer (ties: the first kept)."""
    cur = conn.execute("""
        select * from (
          select a.*, row_number() over (partition by source, observation_id, model
                                         order by answered_at, id) as n
          from early_answers a)
        where n = 1 order by answered_at, id""")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


# --- when the record first turned green ----------------------------------------------------

def _parse(moment: str | None) -> datetime | None:
    if not moment:
        return None
    try:
        t = datetime.fromisoformat(str(moment).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def green_moment(validated_on: str | None, first_seen_at: str | None,
                 first_green_at: str | None = None) -> tuple[datetime | None, str | None]:
    """(the moment an answer must precede to count, the rule): the earliest of .org's first
    green moment (`first_green_at`, rule "moment"), the start (00:00 UTC) of the earliest
    green day when no moment is given or the day is earlier than the moment's (rule "day"),
    and when Vision's nightly first listed the record green. (None, None) when nothing says:
    then nothing is scored."""
    found: list[datetime] = []
    moment = _parse(first_green_at)
    day = _parse(f"{validated_on[:10]}T00:00:00+00:00") if validated_on else None
    if moment is not None:
        found.append(moment)
    if day is not None and (moment is None or day.date() < moment.date()):
        found.append(day)
    seen = _parse(first_seen_at)
    if seen is not None:
        found.append(seen)
    if not found:
        return None, None
    return min(found), ("moment" if moment is not None else "day")


def sealed(validated_on: str | None, first_seen_at: str | None, first_green_at: str | None,
           seal: str | None) -> bool:
    """Is the record (possibly) in the sealed test, first green at or after `seal`? Only
    evidence of green before the seal clears it: .org's moment, a green day before the
    seal's day, or Vision's nightly having listed it green before. No evidence: sealed."""
    if seal is None:
        return False
    s = _parse(seal)
    moment, seen = _parse(first_green_at), _parse(first_seen_at)
    if moment is not None and moment < s:
        return False
    if seen is not None and seen < s:
        return False
    if validated_on and validated_on[:10] < s.date().isoformat():
        return False
    return True


# --- scoring -----------------------------------------------------------------------------

def _record_key(source: str, number: str) -> str:
    from .sources import record_key
    return record_key(source, number)


def _green_rows(manifest: sqlite3.Connection, keys: list[str]) -> dict[str, dict]:
    have = {r[1] for r in manifest.execute("pragma table_info(records)")}
    cols = ["observation_id", "scientific_name", "genus", "family", "validated_on",
            "first_seen_at", "label_conflict"]
    extra = [c for c in ("vision_assisted", "first_green_at") if c in have]
    out = {}
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        q = ",".join("?" for _ in part)
        cur = manifest.execute(f"select {', '.join(cols + extra)} from records "
                               f"where observation_id in ({q})", part)
        names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            out[row[0]] = dict(zip(names, row))
    return out


class Judge:
    """Names judged the advance predictions' way (prospective.report): the manifest's labels
    and taxonomy.labels_for for the truth, guests and non-fungi left out; then name_equiv's
    answer key and its s.l. and complex readings."""

    def __init__(self, manifest: sqlite3.Connection, extra_names: Iterable[str] = ()):
        from . import names, taxonomy
        from .name_equiv import ReferenceDepth
        self.labels = names.manifest_labels(manifest, extra=list(extra_names))
        self.tax = taxonomy.for_manifest(manifest)
        counts = manifest.execute("select scientific_name, count(*) from records "
                                  "where label_conflict = 0 group by 1").fetchall()
        self.depth = ReferenceDepth([(self.label(n), c) for n, c in counts if n])

    def label(self, name: str | None) -> str:
        return self.labels.get(name, name) if name else (name or "")

    def truth(self, name, genus, family):
        from . import guests, taxonomy
        lab = self.label(name) or ""
        labs = taxonomy.labels_for(lab, genus, family, lab.strip() != (name or "").strip(),
                                   self.tax)
        if guests.excluded(labs.genus) or taxonomy.nonfungal(self.tax, labs.genus):
            return None
        return labs

    def positions(self, lists: dict, species: str, genus: str) -> dict:
        """{column: 0-based place of the first right name, or None}; a right name at a
        stricter level counts at every looser one."""
        from . import name_equiv as ne
        out = {"sp_strict": None, "sp_sl": None, "sp_complex": None,
               "ge_strict": None, "ge_sl": None}
        if species:
            answers = ne.collapse([self.label(c["name"]) for c in lists.get("species") or []])
            for i, a in enumerate(answers):
                m = ne.species_match(a, species)
                strict = m["strict"] or a == species
                for col, hit in (("sp_strict", strict), ("sp_sl", strict or m["sl"]),
                                 ("sp_complex", strict or m["sl"] or m["complex"])):
                    if hit and out[col] is None:
                        out[col] = i
        if genus:
            seen: list[str] = []
            for c in lists.get("genus") or []:
                g = self.label(c["name"])
                if g not in seen:
                    seen.append(g)
            for i, g in enumerate(seen):
                m = ne.genus_match(g, genus)
                strict = m["strict"] or g == genus
                for col, hit in (("ge_strict", strict), ("ge_sl", strict or m["sl"])):
                    if hit and out[col] is None:
                        out[col] = i
        return out


def judge_one(a: dict, rec: dict, judge: Judge, seal: str | None) -> dict | None:
    """The outcome of first answer `a` for its green record `rec` (a records row), or None
    while it must wait (sealed, a label conflict)."""
    if rec.get("label_conflict"):
        return None
    vo, fs, fg = rec.get("validated_on"), rec.get("first_seen_at"), rec.get("first_green_at")
    if sealed(vo, fs, fg, seal):
        return None
    at, rule = green_moment(vo, fs, fg)
    base = {"truth_name": rec.get("scientific_name"), "validated_on": vo, "first_seen_at": fs,
            "first_green_at": fg, "green_before": at.isoformat() if at else None,
            "green_rule": rule}
    # Vision's own agreement made the verdict: circular whenever it was answered.
    if rec.get("vision_assisted"):
        return {**base, "outcome": VISION_ASSISTED}
    if at is None or a["answered_at"] >= at.timestamp():
        return {**base, "outcome": AFTER}
    if a.get("excluded_reference"):
        return {**base, "outcome": IN_REFERENCE}
    if (a.get("left_out_photos") or 0) > 0:
        return {**base, "outcome": TWIN}
    labs = judge.truth(rec.get("scientific_name"), rec.get("genus"), rec.get("family"))
    if labs is None:
        return {**base, "outcome": NOT_FUNGUS}
    lists = json.loads(a["ranks_json"])
    depth = judge.depth(labs.species) if labs.species else None
    return {**base, "outcome": SCORED, "truth_species": labs.species or None,
            "truth_genus": labs.genus or None,
            "species_reference_records": max(0, depth - 1) if depth is not None else None,
            **judge.positions(lists, labs.species, labs.genus)}


OUTCOME_COLUMNS = ("answer_id", "record_key", "decided_at", "outcome", "truth_name",
                   "truth_species", "truth_genus", "validated_on", "first_seen_at",
                   "first_green_at", "green_before", "green_rule", "species_reference_records",
                   "sp_strict", "sp_sl", "sp_complex", "ge_strict", "ge_sl")


def score_pending(store: sqlite3.Connection, manifest: sqlite3.Connection,
                  seal=None, now: str | None = None, lock=None) -> dict:
    """Decides every first answer whose record is green now and has no outcome yet (see the
    module's list), once: an outcome is never changed. `seal`: default
    prospective.seal_time(). Returns counts of what was decided and what still waits."""
    import contextlib
    from . import prospective
    seal = prospective.seal_time() if seal is None else (seal or None)
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    guard = lock or contextlib.nullcontext()
    with guard:
        firsts = first_answers(store)
        done = {r[0] for r in store.execute("select answer_id from early_outcomes")}
    todo = [a for a in firsts if a["id"] not in done]
    keys = {a["id"]: _record_key(a["source"], a["observation_id"]) for a in todo}
    green = _green_rows(manifest, sorted(set(keys.values())))
    stats = Counter()
    names_seen = [c["name"] for a in todo if keys[a["id"]] in green
                  for c in json.loads(a["ranks_json"]).get("species") or []]
    judge = Judge(manifest, names_seen) if any(keys[a["id"]] in green for a in todo) else None
    rows = []
    for a in todo:
        rec = green.get(keys[a["id"]])
        if rec is None:
            stats["awaiting_verdict"] += 1
            continue
        got = judge_one(a, rec, judge, seal)
        if got is None:
            stats["sealed" if not rec.get("label_conflict") else "label_conflict"] += 1
            continue
        stats[got["outcome"]] += 1
        full = {"answer_id": a["id"], "record_key": keys[a["id"]], "decided_at": now, **got}
        rows.append(tuple(full.get(c) for c in OUTCOME_COLUMNS))
    with guard:
        with store:
            store.executemany(f"insert or ignore into early_outcomes ({', '.join(OUTCOME_COLUMNS)}) "
                              f"values ({', '.join('?' for _ in OUTCOME_COLUMNS)})", rows)
    return dict(stats)


# --- the report --------------------------------------------------------------------------

DEPTH_BANDS = ((0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 99, "20-99"),
               (100, 10**9, "100+"))
LEVELS = (("species strict", "sp_strict"), ("species s.l.", "sp_sl"),
          ("species complex (beta)", "sp_complex"), ("genus strict", "ge_strict"),
          ("genus s.l.", "ge_sl"))
WITH_CONTEXT = "photos + place/date"
PHOTOS_ONLY = "photos only"


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _day(t: float | str | None) -> str | None:
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return iso_utc(t)[:10]
    return str(t)[:10]


_SEAL = object()


def report(store: sqlite3.Connection, manifest: sqlite3.Connection | None = None,
           sealed_after=_SEAL, lock=None) -> list[dict]:
    """Per model and context (photos only, or photos + place/date), aggregates only:
    how many first answers were decided and how (`left_out`), how many scored and over
    which days, the top 1/3/5/10 table (species strict, s.l., complex (beta); genus strict,
    s.l.), species top-1 and top-5 by the true species' reference records, and iNat CV
    ("not run"). Scores of records first green at or after `sealed_after` (default
    prospective.seal_time()) are never in it; with `manifest`, `sealed` counts the answers
    held back for them. No observation id, observer or per-record row."""
    import contextlib
    from . import prospective
    seal = prospective.seal_time() if sealed_after is _SEAL else sealed_after
    with lock or contextlib.nullcontext():
        firsts = first_answers(store)
        cur = store.execute("select * from early_outcomes")
        names = [d[0] for d in cur.description]
        outcomes = {r[0]: dict(zip(names, r)) for r in cur.fetchall()}
    green = {}
    if manifest is not None and seal is not None:
        keys = sorted({_record_key(a["source"], a["observation_id"]) for a in firsts
                       if a["id"] not in outcomes})
        green = _green_rows(manifest, keys)
    groups: dict[tuple, dict] = {}

    def group(a: dict) -> dict:
        backbone, _, method = a["model"].partition("/")
        ctx = WITH_CONTEXT if (a.get("place_used") or a.get("date_used")) else PHOTOS_ONLY
        return groups.setdefault((backbone, method, ctx), {
            "backbone": backbone, "method": method, "context": ctx, "answers": 0,
            "asked_by": Counter(), "left_out": Counter(), "awaiting_verdict": 0, "sealed": 0,
            "scored_rows": []})

    for a in firsts:
        g = group(a)
        g["answers"] += 1
        g["asked_by"][a.get("asked_by") or "unknown"] += 1
        o = outcomes.get(a["id"])
        if o is None:
            rec = green.get(_record_key(a["source"], a["observation_id"]))
            if rec is not None and not rec.get("label_conflict") and sealed(
                    rec.get("validated_on"), rec.get("first_seen_at"),
                    rec.get("first_green_at"), seal):
                at, _ = green_moment(rec.get("validated_on"), rec.get("first_seen_at"),
                                     rec.get("first_green_at"))
                if at is None or a["answered_at"] < at.timestamp():
                    g["sealed"] += 1
                    continue
            g["awaiting_verdict"] += 1
            continue
        if seal is not None and sealed(o["validated_on"], o["first_seen_at"],
                                       o["first_green_at"], seal):
            g["sealed"] += 1                   # decided after the seal was lifted: not here
            continue
        if o["outcome"] != SCORED:
            g["left_out"][o["outcome"]] += 1
            continue
        g["scored_rows"].append((a, o))
    out = []
    for key in sorted(groups):
        g = groups.pop(key)
        rows = g.pop("scored_rows")
        g["asked_by"] = dict(g["asked_by"])
        g["left_out"] = {k: g["left_out"].get(k, 0) for k in OUTCOMES if k != SCORED}
        g["scored"] = len(rows)
        g["answered"] = {"from": _day(min((a["answered_at"] for a, _ in rows), default=None)),
                         "through": _day(max((a["answered_at"] for a, _ in rows),
                                             default=None))}
        g["validated"] = {"from": _day(min((o["green_before"] for _, o in rows), default=None)),
                          "through": _day(max((o["green_before"] for _, o in rows),
                                              default=None))}
        g["green_rule"] = dict(Counter(o["green_rule"] for _, o in rows))
        g["top_k"] = _ladder(rows)
        g["species_by_true_species_reference_records"] = _by_depth(rows)
        g["inat_cv"] = {"status": "not run"}
        out.append(g)
    return out


def _ladder(rows: list[tuple[dict, dict]]) -> dict:
    """{level: {"n", "top1", "top3", "top5", "top10", "n_top10"}}. Top 10 only over answers
    kept ten deep (`n_top10`); answers recovered from older stores are five deep."""
    table = {}
    for label, col in LEVELS:
        want = "truth_species" if col.startswith("sp") else "truth_genus"
        use = [(a, o) for a, o in rows if o.get(want)]
        n = len(use)
        cells: dict = {"n": n}
        for k in KS:
            pool = [(a, o) for a, o in use if a["depth"] >= k]
            hits = sum(o[col] is not None and o[col] < k for _, o in pool)
            cells[f"top{k}"] = _rate(hits, len(pool))
            if k == max(KS):
                cells["n_top10"] = len(pool)
        table[label] = cells
    return table


def _by_depth(rows: list[tuple[dict, dict]]) -> dict:
    out = {}
    for lo, hi, label in DEPTH_BANDS:
        use = [o for _, o in rows if o.get("truth_species")
               and o.get("species_reference_records") is not None
               and lo <= o["species_reference_records"] <= hi]
        out[label] = {"n": len(use),
                      "top1": _rate(sum(o["sp_strict"] == 0 for o in use), len(use)),
                      "top5": _rate(sum(o["sp_strict"] is not None and o["sp_strict"] < 5
                                        for o in use), len(use))}
    return out


LEFT_OUT_LABELS = {
    AFTER: "answered on or after the verdict",
    IN_REFERENCE: "itself a reference record when answered",
    TWIN: "a twin was in the reference",
    VISION_ASSISTED: "validated with Vision's help",
    NOT_FUNGUS: "not a fungus label",
}


def format_report(groups: list[dict]) -> str:
    """The standard summary (top 1/3/5/10 x strict / s.l. / complex; genus; depth bands;
    iNat CV), every table with its n."""
    if not groups:
        return "No answers kept yet."
    lines = []
    for g in groups:
        lines.append(f"{g['backbone']}/{g['method']}  {g['context']}  "
                     f"(n = {g['scored']:,} scored of {g['answers']:,} first answers; "
                     f"answered {g['answered']['from']} to {g['answered']['through']}, "
                     f"green {g['validated']['from']} to {g['validated']['through']})")
        lines.append(f"    {'':<24}" + "".join(f"{'top ' + str(k):>8}" for k in KS)
                     + f"{'n':>7}{'n top10':>9}")
        for label, row in g["top_k"].items():
            cells = "".join(f"{100 * row[f'top{k}']:7.1f}%" if row[f"top{k}"] is not None
                            else f"{'-':>8}" for k in KS)
            lines.append(f"    {label:<24}{cells}{row['n']:>7,}{row['n_top10']:>9,}")
        lines.append("    species, by the true species' reference records (when scored)")
        for band, row in g["species_by_true_species_reference_records"].items():
            c = lambda r: f"{100 * r:7.1f}%" if r is not None else f"{'-':>8}"  # noqa: E731
            lines.append(f"    {band + ' records':<24}{c(row['top1'])}{c(row['top5'])}"
                         f"{row['n']:>7,}")
        left = ", ".join(f"{LEFT_OUT_LABELS[k]} {v:,}" for k, v in g["left_out"].items() if v)
        lines.append(f"    left out: {left or 'none'}; awaiting the verdict "
                     f"{g['awaiting_verdict']:,}" + (f"; sealed test held back {g['sealed']:,}"
                                                    if g.get("sealed") else ""))
        lines.append(f"    iNat CV: {g['inat_cv']['status']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- backfill from older stores ------------------------------------------------------------

def _already(store: sqlite3.Connection, provenance: str) -> int:
    return store.execute("select count(*) from early_answers where provenance = ?",
                         (provenance,)).fetchone()[0]


def rows_from_answer_cache(answers_path: Path) -> tuple[list[tuple], Counter]:
    """Rows from observation-answers.sqlite (answercache.py): one per kept by-observation
    answer, at its answered_at. That store keeps the LATEST answer per key (a rerun
    overwrites it), so a row is the answer Vision gave then: it counts only if that was
    before the first green, like any other. The v1 copy is as deep as its caller asked
    (mycomap.org: 5); the site's copy 5 deep."""
    src = sqlite3.connect(Path(answers_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        cols = {r[1] for r in src.execute("pragma table_info(observation_answers)")}
        who = "asked_by" if "asked_by" in cols else "null"
        found = src.execute(f"select answer_key, version, payload, answered_at, {who} "
                            "from observation_answers").fetchall()
    finally:
        src.close()
    rows, stats = [], Counter()
    for key, version, payload, answered_at, asked in found:
        parts = str(key).split(":")
        if len(parts) < 4 or parts[0] not in ("site", "v1") or parts[1] not in ("inat", "mo"):
            stats["not an observation answer"] += 1
            continue
        try:
            body = json.loads(payload)
        except ValueError:
            stats["unreadable"] += 1
            continue
        opts = dict(p.split("=", 1) for p in ":".join(parts[4:]).split(",") if "=" in p)
        if parts[0] == "v1":
            ranks, top_k = body.get("ranks"), int(opts.get("top_k") or 5)
            used = body.get("context_used") or {}
            own, left = body.get("excluded_reference"), body.get("left_out_photos")
            release = (body.get("model") or {}).get("version")
        else:
            ranks, top_k = (body.get("result") or {}).get("ranks"), 5
            used = body.get("used") or {}
            about = body.get("observation") or {}
            own, left = about.get("excluded_reference"), len(body.get("hidden") or [])
            release = body.get("release")
        lists, depth = ranked_lists(None, ranks, top_k)
        try:
            rows.append(make_row(
                source=parts[1], observation_id=parts[2], model=parts[3],
                answered_at=float(answered_at),
                path=V1_OBSERVATION if parts[0] == "v1" else SITE_OBSERVATION, lists=lists,
                depth=depth, asked_by=asked if asked in ASKERS else None, release=release,
                answers_version=version, place_used=bool(used.get("place_from")),
                date_used=bool(used.get("date_from")), excluded_reference=bool(own),
                left_out_photos=int(left or 0), provenance=FROM_ANSWERS))
            stats["rows"] += 1
        except ValueError:
            stats["not an observation answer"] += 1
    return rows, stats


ORG_RUNS_SQL = """
select row_to_json(t)::text from (
  select platform, observation_id, status, result, answers_version, created_at,
         finished_at
  from record_vision_runs where status = 'done' and result is not null
  order by id
) t
"""


def rows_from_org_runs(runs: list[dict], models_by_version: dict[str, str],
                       default_model: str | None = None) -> tuple[list[tuple], Counter]:
    """Rows from mycomap.org's record_vision_runs (its saved copies of Vision's v1 answers,
    five deep, read-only): at the answer's own answered.at (when Vision ran the model, which
    a kept answer carries from its first run), or, for runs saved before .org kept it, the
    run's finished_at, which is no earlier. Its model is found by the answers version in
    Vision's answer cache (`models_by_version`), else `default_model` when given (mycomap.org
    names no model, so it got the model served by default then: the operator says which);
    a run whose model stays unknown, or without answered.at, is skipped. Twins left out are
    not known (null)."""
    rows, stats = [], Counter()
    for run in runs:
        res = run.get("result") or {}
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except ValueError:
                stats["unreadable"] += 1
                continue
        at = _parse(res.get("answeredAt"))
        if at is None:
            # Runs saved before .org kept answered.at: when .org had the answer back, which
            # is at or after Vision ran the model, so an answer before the verdict by this
            # time surely was (finished_at is UTC without a zone, as .org's clock writes it).
            at = _parse(run.get("finished_at"))
            if at is None:
                stats["no answered time"] += 1
                continue
            stats["time from .org's finished_at (at or after the answer)"] += 1
        model = models_by_version.get(str(run.get("answers_version") or "")) or default_model
        if not model:
            stats["model unknown"] += 1
            continue
        ranks = {r: res.get("combined", {}).get(r) or [] for r in RANKS}
        lists, depth = ranked_lists(None, ranks, 5)
        try:
            rows.append(make_row(
                source=str(run.get("platform")), observation_id=str(run.get("observation_id")),
                model=model, answered_at=at.timestamp(), path=V1_OBSERVATION, lists=lists,
                depth=depth, asked_by="mycomap.org" if not res.get("kept")
                else ({"site": "site", "mycomap.org": "mycomap.org", "app": "app",
                       "api": "api"}.get(res.get("answeredBy") or "")),
                release=res.get("modelVersion"), answers_version=run.get("answers_version"),
                place_used=res.get("placeUsed"), date_used=res.get("dateUsed"),
                excluded_reference=res.get("excludedReference"), left_out_photos=None,
                provenance=FROM_ORG_RUNS))
            stats["rows"] += 1
        except ValueError:
            stats["not an observation answer"] += 1
    return rows, stats


def models_by_version(answers_path: Path) -> dict[str, str]:
    """{answers_version: backbone/method} from the answer cache's keys."""
    src = sqlite3.connect(Path(answers_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        found = src.execute("select answer_key, version from observation_answers").fetchall()
    finally:
        src.close()
    out = {}
    for key, version in found:
        parts = str(key).split(":")
        if len(parts) >= 4:
            out.setdefault(str(version), parts[3])
    return out


def without_repeats(rows: list[tuple], store: sqlite3.Connection) -> list[tuple]:
    """Rows not already in the store or earlier in `rows`: the same observation, model and
    answer time (to the second) is one answer, wherever it was recovered from."""
    seen = {(s, o, m, int(t)) for s, o, m, t in store.execute(
        "select source, observation_id, model, answered_at from early_answers")}
    out = []
    for r in rows:
        k = (r[0], r[1], r[2], int(r[3]))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def backfill(store: sqlite3.Connection, answers_path: Path | None,
             org_runs: list[dict] | None, apply: bool = False,
             default_model: str | None = None) -> dict:
    """What a backfill finds, and with `apply` keeps (refused for a source imported before:
    the store is append-only)."""
    out: dict = {"apply": apply}
    rows: list[tuple] = []
    sources_: list[str] = []
    if answers_path is not None:
        got, stats = rows_from_answer_cache(answers_path)
        out["answer_cache"] = dict(stats)
        rows += got
        sources_.append(FROM_ANSWERS)
    if org_runs is not None:
        versions = models_by_version(answers_path) if answers_path is not None else {}
        got, stats = rows_from_org_runs(org_runs, versions, default_model)
        out["org_runs"] = {"runs": len(org_runs), **dict(stats)}
        rows += got
        sources_.append(FROM_ORG_RUNS)
    fresh = without_repeats(rows, store)
    out["new_rows"] = len(fresh)
    out["observations"] = len({(r[0], r[1]) for r in fresh})
    if apply:
        for p in sources_:
            if _already(store, p):
                raise ValueError(f"{p} was imported already: the store is append-only")
        with store:
            store.executemany(INSERT, fresh)
    return out


def preview(rows_store: sqlite3.Connection, manifest: sqlite3.Connection,
            first_green: dict[str, str] | None = None) -> dict:
    """How the first answers in `rows_store` would be decided against `manifest` now, by the
    rule the box uses; with `first_green` ({record key: .org's first green moment}) also how
    many would score with exact moments. Counts only."""
    firsts = first_answers(rows_store)
    keys = {a["id"]: _record_key(a["source"], a["observation_id"]) for a in firsts}
    green = _green_rows(manifest, sorted(set(keys.values())))
    from . import prospective
    seal = prospective.seal_time()
    day, moment = Counter(), Counter()
    for a in firsts:
        rec = green.get(keys[a["id"]])
        if rec is None:
            day["awaiting_verdict"] += 1
            continue
        for tally, fg in ((day, rec.get("first_green_at")),
                          (moment, (first_green or {}).get(keys[a["id"]]))):
            if tally is moment and not first_green:
                continue
            if sealed(rec.get("validated_on"), rec.get("first_seen_at"), fg, seal):
                tally["sealed"] += 1
                continue
            at, _ = green_moment(rec.get("validated_on"), rec.get("first_seen_at"), fg)
            if rec.get("vision_assisted"):
                tally[VISION_ASSISTED] += 1
            elif at is None or a["answered_at"] >= at.timestamp():
                tally[AFTER] += 1
            elif a.get("excluded_reference"):
                tally[IN_REFERENCE] += 1
            elif (a.get("left_out_photos") or 0) > 0:
                tally[TWIN] += 1
            else:
                tally["would_score"] += 1
    out = {"first_answers": len(firsts), "green_now": sum(k in green for k in keys.values()),
           "by_day_rule": dict(day)}
    if first_green:
        out["with_exact_first_green"] = dict(moment)
    return out


FIRST_GREEN_SQL = """
select row_to_json(t)::text from (
  select observation, to_char(min(at), 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"') as first_green_at
  from validation_events where status in (1, 4) and observation in ({keys})
  group by observation
) t
"""


def org_first_green(keys: list[str], fetch: Callable[[str], str]) -> dict[str, str]:
    """{record key: .org's first green moment} from its validation events (status 1
    verified, 4 auto-verified; UTC), through the read-only SQL route. For a preview only:
    the box has no such route, and scoring there waits for the feed's FIRST_GREEN_FIELD."""
    from .records import parse_export
    out = {}
    as_org = {}
    for k in keys:
        source, _, number = k.partition(":") if ":" in k else ("inat", "", k)
        if source in ("inat", "mo") and number.isdigit():
            as_org[f"{source}:{number}"] = k
    names = sorted(as_org)
    for i in range(0, len(names), 500):
        part = names[i:i + 500]
        sql = FIRST_GREEN_SQL.replace("{keys}", ",".join(f"'{n}'" for n in part))
        for row in parse_export(fetch(sql)):
            out[as_org[row["observation"]]] = row["first_green_at"]
    return out
