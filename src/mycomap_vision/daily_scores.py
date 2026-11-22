"""Scores since training: a newer model, day by day, on the records first green after its
training cut (Steve, 2026-11-22: "for v2, once all records are trained against, I want to
have daily scores for whatever was validated that day, along with a cumulative total since
training, along with the number of new images that are added to the model since training").

A model declared here (MV_DAILY_SCORES_MODELS) was trained on every record green up to its
cut, so every record first green AFTER the cut is one its weights never saw: a fair test.
Each night, after the green records are read and tonight's photos are embedded
(api.run_nightly, after early_answers.score_pending), every such record not yet decided gets
the model's answer, frozen in an insert-only ledger and never asked again: the model's FIRST
answer about it. The answer is the served index's, from the record's own vectors with the
record itself, its cross-post twins and any copy of its photos left out of the reference
(identify.left_out), photos only (no place or date). A record whose photos have no vectors
for the model yet waits for a later night.

When a record turned green is early_answers.green_moment: the earliest of .org's first green
moment, the start of its first green day and when Vision's nightly first listed it. A record
counts as after the cut only when that moment is after it (nothing says it was green at the
cut); no moment, or one at or before the cut, and it is never scored. Its day (UTC) is the
moment's.

Left out, as early_answers leaves them out: a label conflict (waits), a record validated with
Vision's help (vision_assisted: circular), a guest or no fungus (not_a_fungus_label). Scored
against the DNA-validated name by early_answers.Judge (the manifest's labels,
taxonomy.labels_for): species and genus right first (strict). Record level, one answer per
record.

The paper's seals, kept as they are:

- v1 is never scored here. A declared model whose cut is before the paper's fixed seal
  (research_public.SEAL, v1's label snapshot), one named "v1", or one whose training data
  (the manifest's finetunes.trained_through) ends before the seal is refused, so no path
  scores v1, or any model of its era, on records first green after v1's seal. The v1 choice
  on the pages keeps today's sealed numbers (/api/how-good).
- The held records (MV_FEEDBACK_HOLD_KEYS, pinned by MV_FEEDBACK_HOLD_SHA256; the blind label
  check) are never scored and never counted in a report, read with feedback.read_hold_keys.
  It fails closed: a list that can't be trusted (named but missing, sha256 unset or wrong)
  refuses the whole step and every report (HoldListRefused); nothing names a held key.

Two settings, both off by default, so nothing changes on the box until someone turns them on:

    MV_DAILY_SCORES=1          compute: the nightly step runs
    MV_DAILY_SCORES_PUBLIC=1   show: /api/daily-scores serves the numbers

They are separate because whether these scores may be public before the v1 paper is submitted
(the records after v2's cut are also records first green after v1's seal) is the Paper
Coordinator's question, not this module's.

    MV_DAILY_SCORES_MODELS=v2=<backbone>/<method>@<cut, ISO UTC>[, v2-open=...@...]

declares each model by the id the pages' dropdown shows, so a later model (v2-open) joins the
dropdown with a config change only.

State: the ledger is a table in state/early-answers.sqlite (early_answers.default_path), which
outlives releases. Only aggregates leave it (report; research_public.daily_scores).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from . import config

COMPUTE_SETTING = "MV_DAILY_SCORES"
PUBLIC_SETTING = "MV_DAILY_SCORES_PUBLIC"
MODELS_SETTING = "MV_DAILY_SCORES_MODELS"
V1 = "v1"                         # the pages' sealed choice: /api/how-good, never this module
ROLLING_DAYS = 7
DEFAULT_LIMIT = 5000              # answers made in one night (the first night catches up)

SCORED = "scored"
VISION_ASSISTED = "vision_assisted"
NOT_FUNGUS = "not_a_fungus_label"
OUTCOMES = (SCORED, VISION_ASSISTED, NOT_FUNGUS)

SCHEMA = """
create table if not exists cut_answers (
    backbone         text not null,
    method           text not null,
    cut              text not null,       -- the model's training cut (ISO UTC)
    record_key       text not null,       -- private: never in a report
    decided_at       text not null,
    outcome          text not null,       -- daily_scores.OUTCOMES
    green_at         text not null,       -- early_answers.green_moment
    green_day        text not null,       -- its UTC day: the row of the daily table
    green_rule       text,                -- moment | day
    photos           integer,             -- the record's photos the model holds (none trained on)
    depth            integer,             -- how deep the frozen ranked names go
    ranks_json       text,                -- the frozen first answer (species, genus, family)
    truth_species    text,
    truth_genus      text,
    sp_right         integer,             -- 1/0: species right first (strict); null: no species label
    ge_right         integer,             -- 1/0: genus right first (strict); null: no genus label
    primary key (backbone, method, cut, record_key)
);
create trigger if not exists cut_answers_no_update before update on cut_answers
begin select raise(abort, 'cut_answers is a ledger: no UPDATE'); end;
create trigger if not exists cut_answers_no_delete before delete on cut_answers
begin select raise(abort, 'cut_answers is a ledger: no DELETE'); end;
"""

COLUMNS = ("backbone", "method", "cut", "record_key", "decided_at", "outcome", "green_at",
           "green_day", "green_rule", "photos", "depth", "ranks_json", "truth_species",
           "truth_genus", "sp_right", "ge_right")


class DeclarationRefused(ValueError):
    """MV_DAILY_SCORES_MODELS declares something this module must not score."""


class HoldListRefused(RuntimeError):
    """The held records' list can't be trusted: nothing is scored or reported."""


def _on(name: str) -> bool:
    return (config.setting(name) or "").strip().lower() in ("1", "true", "yes", "on")


def computing() -> bool:
    """MV_DAILY_SCORES: the nightly step runs (default off)."""
    return _on(COMPUTE_SETTING)


def public() -> bool:
    """MV_DAILY_SCORES_PUBLIC: /api/daily-scores serves the numbers (default off)."""
    return _on(PUBLIC_SETTING)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# --- which models --------------------------------------------------------------------------

def _moment(text: str) -> datetime | None:
    from .early_answers import _parse
    return _parse(text)


@dataclass(frozen=True)
class Declared:
    """A model scored here: the id the pages show, the served backbone and method, and the
    moment its training data was cut (every record green up to it was trained on)."""
    id: str
    backbone: str
    method: str
    cut: str

    @property
    def model(self) -> str:
        return f"{self.backbone}/{self.method}"

    @property
    def cut_at(self) -> datetime:
        return _moment(self.cut)


_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,31}$")
_ENTRY = re.compile(r"^(?P<id>[^=\s]+)=(?P<backbone>[^/@\s]+)/(?P<method>[^@\s]+)@(?P<cut>\S+)$")


def parse_models(text: str | None,
                 trained_through: dict[str, str] | None = None) -> list[Declared]:
    """The declared models (MV_DAILY_SCORES_MODELS), comma or space separated. Refused
    (DeclarationRefused), whole: a malformed entry, a repeated id, the id "v1", a cut that
    isn't a moment, and any model of v1's era: a cut before the paper's fixed seal
    (research_public.SEAL), or training data that ends before it (`trained_through`, the
    manifest's finetunes, by backbone). That is what keeps v1 from being scored on records
    first green after its seal."""
    from . import research_public
    seal = _moment(research_public.SEAL)
    out: list[Declared] = []
    for part in re.split(r"[,\s]+", (text or "").strip()):
        if not part:
            continue
        m = _ENTRY.match(part)
        if not m:
            raise DeclarationRefused(f"{MODELS_SETTING}: {part!r} is not "
                                     "<id>=<backbone>/<method>@<cut>")
        mid = m["id"].lower()
        if not _ID.match(mid):
            raise DeclarationRefused(f"{MODELS_SETTING}: {mid!r} is not a short id (v2, v2-open)")
        if mid == V1:
            raise DeclarationRefused(f"{MODELS_SETTING}: v1 is never scored here; its choice "
                                     "keeps the sealed numbers (/api/how-good)")
        if any(d.id == mid for d in out):
            raise DeclarationRefused(f"{MODELS_SETTING}: {mid} is declared twice")
        cut = _moment(m["cut"])
        if cut is None:
            raise DeclarationRefused(f"{MODELS_SETTING}: {mid}'s cut {m['cut']!r} is not a "
                                     "moment (ISO, UTC)")
        if cut < seal:
            raise DeclarationRefused(f"{MODELS_SETTING}: {mid}'s cut is before the paper's "
                                     f"seal ({research_public.SEAL}): a model of v1's era is "
                                     "never scored on records after it")
        through = (trained_through or {}).get(m["backbone"])
        if through and str(through)[:10] < research_public.SEAL[:10]:
            raise DeclarationRefused(f"{MODELS_SETTING}: {mid} ({m['backbone']}) was trained "
                                     f"through {str(through)[:10]}, before the paper's seal: "
                                     "a model of v1's era is never scored here")
        out.append(Declared(mid, m["backbone"], m["method"],
                            cut.astimezone(timezone.utc).isoformat(timespec="seconds")))
    return out


def declared(trained_through: dict[str, str] | None = None) -> list[Declared]:
    return parse_models(config.setting(MODELS_SETTING), trained_through)


# --- the held records ------------------------------------------------------------------------

def held_keys(path=None, sha256=None) -> frozenset:
    """The held record keys (feedback.read_hold_keys, MV_FEEDBACK_HOLD_KEYS pinned by
    MV_FEEDBACK_HOLD_SHA256 unless given). A list that can't be trusted refuses
    (HoldListRefused, saying why and naming no key); none named holds none, as everywhere."""
    from .feedback import read_hold_keys
    keys, hold_all, problem = read_hold_keys(
        path if path is not None else config.setting("MV_FEEDBACK_HOLD_KEYS"),
        sha256 if sha256 is not None else config.setting("MV_FEEDBACK_HOLD_SHA256"))
    if hold_all:
        raise HoldListRefused(f"held records: {problem}; nothing is scored or shown")
    return keys


# --- which records ---------------------------------------------------------------------------

def after_cut(rec: dict, cut: str) -> tuple[datetime, str] | None:
    """(when the record turned green, the rule) when that is after `cut`, else None: no
    moment, or one at or before the cut (it may have been trained on)."""
    from .early_answers import green_moment
    at, rule = green_moment(rec.get("validated_on"), rec.get("first_seen_at"),
                            rec.get("first_green_at"))
    if at is None or at <= _moment(cut):
        return None
    return at, rule


def candidates(manifest: sqlite3.Connection, model: Declared, done: set[str],
               held: frozenset) -> tuple[list[dict], Counter]:
    """The records first green after the model's cut with no decision yet, newest first,
    and counts of the rest (held ones only as a total)."""
    have = {r[1] for r in manifest.execute("pragma table_info(records)")}
    cols = ["observation_id", "source", "source_id", "scientific_name", "genus", "family",
            "validated_on", "first_seen_at", "label_conflict"]
    cols += [c for c in ("first_green_at", "vision_assisted") if c in have]
    # A superset, in SQL: a green day before the cut's day starts before the cut.
    cur = manifest.execute(f"select {', '.join(cols)} from records where validated_on is null "
                           "or substr(validated_on, 1, 10) >= ?", (model.cut[:10],))
    names = [d[0] for d in cur.description]
    out, stats = [], Counter()
    for row in cur.fetchall():
        rec = dict(zip(names, row))
        got = after_cut(rec, model.cut)
        if got is None:
            continue
        key = rec["observation_id"]
        if key in held:
            stats["held"] += 1
            continue
        if key in done:
            continue
        if rec.get("label_conflict"):
            stats["label_conflict"] += 1
            continue
        rec["green_at"], rec["green_rule"] = got
        out.append(rec)
    out.sort(key=lambda r: (r["green_at"], r["observation_id"]), reverse=True)
    return out, stats


# --- scoring --------------------------------------------------------------------------------

Answer = Callable[[Declared, dict], "dict | None"]
"""(model, record) -> {"lists": {"species": [{"name", "confidence"}], "genus", "family"},
"photos": n} the model's answer from the record's own vectors, itself and its twins left out;
None while it has no vectors for the model."""


def score_pending(store: sqlite3.Connection, manifest: sqlite3.Connection,
                  models: list[Declared], answer: Answer, held: frozenset,
                  now: str | None = None, limit: int = DEFAULT_LIMIT, lock=None) -> dict:
    """Decides, once, each record first green after a model's cut that has no decision yet
    (see the module's notes). Returns counts per model id."""
    import contextlib
    from .early_answers import DEPTH, RANKS, Judge, ranked_lists
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    guard = lock or contextlib.nullcontext()
    with guard:
        ensure_schema(store)
    out: dict = {}
    for model in models:
        with guard:
            done = {r[0] for r in store.execute(
                "select record_key from cut_answers where backbone = ? and method = ? "
                "and cut = ?", (model.backbone, model.method, model.cut))}
        todo, stats = candidates(manifest, model, done, held)
        decided: list[tuple[dict, str, dict | None]] = []
        asked = 0
        for rec in todo:
            if rec.get("vision_assisted"):
                decided.append((rec, VISION_ASSISTED, None))
                continue
            if asked >= limit:
                stats["waiting: tonight's limit"] += 1
                continue
            asked += 1
            got = answer(model, rec)
            if not got:
                stats["waiting: no vectors yet"] += 1
                continue
            decided.append((rec, SCORED, got))
        names = [c["name"] for _, _, g in decided if g
                 for c in (g.get("lists") or {}).get("species") or []]
        judge = Judge(manifest, names) if decided else None
        rows = []
        for rec, outcome, got in decided:
            base = {"backbone": model.backbone, "method": model.method, "cut": model.cut,
                    "record_key": rec["observation_id"], "decided_at": now,
                    "green_at": rec["green_at"].isoformat(timespec="seconds"),
                    "green_day": rec["green_at"].astimezone(timezone.utc).date().isoformat(),
                    "green_rule": rec["green_rule"]}
            if outcome == SCORED:
                labs = judge.truth(rec.get("scientific_name"), rec.get("genus"),
                                   rec.get("family"))
                if labs is None:
                    outcome = NOT_FUNGUS
                else:
                    lists, _ = ranked_lists(None, got["lists"], DEPTH)
                    lists = {r: lists.get(r) or [] for r in RANKS}
                    depth = max((len(v) for v in lists.values()), default=0)
                    pos = judge.positions(lists, labs.species, labs.genus)
                    base.update({
                        "photos": int(got.get("photos") or 0), "depth": depth,
                        "ranks_json": json.dumps(lists, separators=(",", ":")),
                        "truth_species": labs.species or None, "truth_genus": labs.genus or None,
                        "sp_right": int(pos["sp_strict"] == 0) if labs.species else None,
                        "ge_right": int(pos["ge_strict"] == 0) if labs.genus else None})
            base["outcome"] = outcome
            stats[outcome] += 1
            rows.append(tuple(base.get(c) for c in COLUMNS))
        with guard:
            with store:
                store.executemany(f"insert or ignore into cut_answers ({', '.join(COLUMNS)}) "
                                  f"values ({', '.join('?' for _ in COLUMNS)})", rows)
        out[model.id] = dict(stats)
    return out


# --- the report -----------------------------------------------------------------------------

def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _rank(rows: list[dict], col: str) -> dict:
    """{"right", "n", "top1", "ci95"}: right first over the rows with that label; ci95 the
    Wilson score interval (heldout_report.wilson)."""
    from .heldout_report import wilson
    use = [r[col] for r in rows if r[col] is not None]
    k, n = sum(use), len(use)
    ci = wilson(k, n)
    return {"right": k, "n": n, "top1": _rate(k, n), "ci95": list(ci) if ci else None}


def _days(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def report(store: sqlite3.Connection, model: Declared, held: frozenset, lock=None) -> dict:
    """One model's scores since its cut, aggregates only: per UTC green day the records
    scored, their photos and species and genus right first (each with its n), the 7-day
    rolling value ending that day, and the cumulative total with its 95% interval. Held
    records are left out whatever the ledger holds. No record key, name, photo or place."""
    import contextlib
    with lock or contextlib.nullcontext():
        ensure_schema(store)
        cur = store.execute("select * from cut_answers where backbone = ? and method = ? "
                            "and cut = ?", (model.backbone, model.method, model.cut))
        names = [d[0] for d in cur.description]
        found = [dict(zip(names, r)) for r in cur.fetchall()]
    found = [r for r in found if r["record_key"] not in held]
    rows = [r for r in found if r["outcome"] == SCORED]
    left = Counter(r["outcome"] for r in found if r["outcome"] != SCORED)
    as_of = max((r["decided_at"] for r in found), default=None)
    base = {"id": model.id, "backbone": model.backbone, "method": model.method,
            "trained_through": model.cut, "context": "photos only", "as_of": as_of,
            "left_out": {k: left.get(k, 0) for k in OUTCOMES if k != SCORED}}
    if not rows:
        return {**base, "records": 0, "new_photos": 0, "days": [], "cumulative": None,
                "rolling_7_days": None}
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["green_day"], []).append(r)
    span = _days(model.cut[:10], max(by_day))
    as_of_day = (as_of or "")[:10]
    days = []
    for i, d in enumerate(span):
        today = by_day.get(d, [])
        window = [r for w in span[max(0, i - ROLLING_DAYS + 1):i + 1] for r in by_day.get(w, [])]
        days.append({
            "day": d, "records": len(today), "photos": sum(r["photos"] or 0 for r in today),
            "species": _rank(today, "sp_right"), "genus": _rank(today, "ge_right"),
            "rolling_7_days": {"records": len(window), "species": _rank(window, "sp_right"),
                               "genus": _rank(window, "ge_right")},
            # A day the nightly has not finished with: more of its records may still come.
            "complete": bool(as_of_day) and d < as_of_day})
    photos = sum(r["photos"] or 0 for r in rows)
    return {**base, "records": len(rows), "new_photos": photos, "days": days,
            "cumulative": {"from": span[0], "through": span[-1], "records": len(rows),
                           "photos": photos, "species": _rank(rows, "sp_right"),
                           "genus": _rank(rows, "ge_right")},
            "rolling_7_days": {"through": span[-1], **days[-1]["rolling_7_days"]}}


def public_payload(store: sqlite3.Connection | None, models: list[Declared] | None = None,
                   held: frozenset | None = None, show: bool | None = None, lock=None) -> dict:
    """/api/daily-scores: which models the pages may offer (v1 always, its sealed numbers
    from /api/how-good), the default (the first declared model with a score, when shown;
    else v1), and each shown model's report. Nothing but v1 while MV_DAILY_SCORES_PUBLIC is
    off, the declaration is refused or the hold list can't be trusted."""
    show = public() if show is None else show
    out: dict = {"public": bool(show), "available": [V1], "default": V1, "models": {}}
    if not show or store is None:
        return out
    try:
        models = declared() if models is None else models
        held = held_keys() if held is None else held
    except (DeclarationRefused, HoldListRefused):
        return out
    for m in models:
        rep = report(store, m, held, lock=lock)
        if rep["records"]:
            out["models"][m.id] = rep
            out["available"].append(m.id)
    if len(out["available"]) > 1:
        out["default"] = out["available"][1]
    return out
