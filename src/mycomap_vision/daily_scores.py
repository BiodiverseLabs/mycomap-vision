"""Scores since training: a newer model, day by day, on the records first green after its
training cut (Steve, 2026-11-22: "for v2, once all records are trained against, I want to
have daily scores for whatever was validated that day, along with a cumulative total since
training, along with the number of new images that are added to the model since training").

A model declared here (MV_DAILY_SCORES_MODELS) was trained on every record green up to its
cut, so every record first green AFTER the cut is one its weights never saw: a fair test.
Each night, after the green records are read and tonight's photos are embedded
(api.run_nightly, after early_answers.score_pending), every such record not yet decided gets
the model's answer, frozen in an insert-only ledger and never asked again: the model's FIRST
answer about it. A record whose photos have no vectors for the model yet waits.

Measured the way v1's How it works numbers are (Steve, 2026-11-22: like with like): each
record is identified from the records verified before it, with its photos and its PUBLIC
place and date, by the served place method (nearest+mean-records-balanced+place). Tonight's
records are the test, the rest of the served index the reference: every one of tonight's
records, its cross-post twins and any copy of its photos are left out of the photo reference
(identify.left_out), and the place prior is fitted again without them (the served method's
own prior counts them), so no record is scored against itself, its place or its date. The
place is what the source itself shows (dataset_release.public_place: iNat's point as the
anonymous API gives it, obscured as-is, none when private; MO's shown GPS, else its location
box's centre), never .org's point; the date is .org's, else iNat's (evaluate.load_records).
A method with no place and date weighting is refused (PlaceRefused), never scored on photos
alone.

When a record turned green is early_answers.green_moment: the earliest of .org's first green
moment, the start of its first green day and when Vision's nightly first listed it. A record
counts as after the cut only when that moment is after it (nothing says it was green at the
cut); no moment, or one at or before the cut, and it is never scored. Its day (UTC) is the
moment's.

Left out, as early_answers leaves them out: a label conflict (waits), a record validated with
Vision's help (vision_assisted: circular), a guest or no fungus (not_a_fungus_label). The
truth is the DNA-validated name by early_answers.Judge (the manifest's labels,
taxonomy.labels_for, so family is iNat's for the genus from the taxonomy cache, else .org's,
as in evaluate.load_records). Judged as evaluate.evaluate judges v1's runs, at write time and
frozen: species by the answer key (name_equiv.first_credited over the collapsed list), genus
and family as written; the place of the first right name in the ten-deep answer, so top
1/3/5/10 at each rank. Each record also keeps its true species' reference depth AT the cut
(the North American, unconflicted records of that species, by the answer key, first green at
or before it: v2's reference then), for the bands of "Every sequenced record makes it
better". A band with fewer than MIN_BAND_RECORDS records gives no rate.

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

    MV_DAILY_SCORES_MODELS=v2=<backbone>/nearest+mean-records-balanced+place@<cut, ISO UTC>
                           [, v2-open=...@...]

declares each model by the id the pages' dropdown shows, so a later model (v2-open) joins the
dropdown with a config change only. Its method is the served place method, v1's: a method
without place and date weighting is refused.

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

RANKS = ("species", "genus", "family")
KS = (1, 3, 5, 10)                 # evaluate.STANDARD_K
MIN_BAND_RECORDS = 30              # a depth band with fewer gives no rate: "too few yet"
CONTEXT = "photos + public place and date"

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
    place_used       integer,             -- 1: answered with a public place
    date_used        integer,             -- 1: answered with a date
    answer_key       text,                -- name_equiv's species key it was judged by
    depth            integer,             -- how deep the frozen ranked names go
    ranks_json       text,                -- the frozen first answer (species, genus, family)
    truth_species    text,
    truth_genus      text,
    truth_family     text,
    sp_pos           integer,             -- 0-based place of the first right species; null: none
    ge_pos           integer,             -- ... genus
    fa_pos           integer,             -- ... family
    species_depth_at_cut integer,         -- the true species' reference records at the cut
    primary key (backbone, method, cut, record_key)
);
create trigger if not exists cut_answers_no_update before update on cut_answers
begin select raise(abort, 'cut_answers is a ledger: no UPDATE'); end;
create trigger if not exists cut_answers_no_delete before delete on cut_answers
begin select raise(abort, 'cut_answers is a ledger: no DELETE'); end;
"""

COLUMNS = ("backbone", "method", "cut", "record_key", "decided_at", "outcome", "green_at",
           "green_day", "green_rule", "photos", "place_used", "date_used", "answer_key",
           "depth", "ranks_json", "truth_species", "truth_genus", "truth_family", "sp_pos",
           "ge_pos", "fa_pos", "species_depth_at_cut")
POS = {"species": "sp_pos", "genus": "ge_pos", "family": "fa_pos"}
TRUTH = {"species": "truth_species", "genus": "truth_genus", "family": "truth_family"}


class DeclarationRefused(ValueError):
    """MV_DAILY_SCORES_MODELS declares something this module must not score."""


class HoldListRefused(RuntimeError):
    """The held records' list can't be trusted: nothing is scored or reported."""


class PlaceRefused(RuntimeError):
    """The model can't be given a record's place and date: it is not scored on photos alone."""


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

Answer = Callable[[Declared, list], dict]
"""(model, tonight's records) -> {record key: {"lists": {"species": [{"name", "confidence"}],
"genus", "family"}, "photos": n, "place_used": bool, "date_used": bool}, or None while it has
no vectors for the model}. Each record arrives with "public" (public_context). All of them
are answered as tonight's test: from the records verified before them, every one of them
left out of the reference and of the place prior (api.daily_answers)."""


def public_context(manifest: sqlite3.Connection, keys: list[str]) -> dict[str, dict]:
    """{record key: {"latitude", "longitude", "observed_on", "uuid", "kind"}}: the record's
    PUBLIC place (dataset_release.public_place: never .org's point; None when the source
    shows none or was never read) and its date (.org's, else iNat's, as evaluate.load_records)."""
    from .dataset_release import public_place
    from .dates import real_date
    tables = {r[0] for r in manifest.execute("select name from sqlite_master where type = 'table'")}
    points = "mo_public_points" in tables
    pt = ("p.latitude, p.longitude, p.gps_hidden, p.box_north, p.box_south, p.box_east, "
          "p.box_west, p.fetched_at, p.status" if points else ", ".join(["null"] * 9))
    join = "left join mo_public_points p on p.observation_id = r.observation_id" if points else ""
    out: dict[str, dict] = {}
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        q = ",".join("?" for _ in part)
        for row in manifest.execute(f"""
            select r.observation_id, r.source, r.observed_on, o.observed_on, o.status,
                   o.inat_latitude, o.inat_longitude, o.obscured, o.geoprivacy,
                   o.taxon_geoprivacy, o.positional_accuracy, o.fetched_at, o.uuid, {pt}
            from records r
            left join inat_observations o on o.observation_id = r.observation_id
            {join}
            where r.observation_id in ({q})""", part):
            (key, source, org_day, inat_day, i_status, i_lat, i_lon, i_obs, i_geo, i_tgeo,
             i_acc, i_at, uuid, m_lat, m_lon, m_hidden, m_n, m_s, m_e, m_w, m_at, m_status) = row
            inat = None if i_status is None else {
                "latitude": i_lat, "longitude": i_lon, "obscured": i_obs, "geoprivacy": i_geo,
                "taxon_geoprivacy": i_tgeo, "positional_accuracy": i_acc, "fetched_at": i_at}
            mo = None if m_at is None else ({
                "latitude": m_lat, "longitude": m_lon, "gps_hidden": m_hidden, "box_north": m_n,
                "box_south": m_s, "box_east": m_e, "box_west": m_w, "fetched_at": m_at}
                if m_status == "ok" else {"fetched_at": m_at})
            pp = public_place(key, source, inat, mo) or {}
            lat, lon = pp.get("public_lat_micro"), pp.get("public_lon_micro")
            known = lat is not None and lon is not None
            out[key] = {"latitude": lat / 1e6 if known else None,
                        "longitude": lon / 1e6 if known else None,
                        "observed_on": real_date(org_day) or real_date(inat_day),
                        "uuid": (uuid or None) if source == "inat" else None,
                        "kind": pp.get("public_kind", "none")}
    return out


def depth_at_cut(manifest: sqlite3.Connection, cut: str, judge, answer_key: str):
    """name_equiv.ReferenceDepth of the North American, unconflicted, fungal records first
    green at or before `cut` (v2's reference then), by their species labels."""
    from .early_answers import green_moment
    from .name_equiv import ReferenceDepth
    have = {r[1] for r in manifest.execute("pragma table_info(records)")}
    fg = "first_green_at" if "first_green_at" in have else "null"
    at_cut = _moment(cut)
    labels = []
    for name, genus, family, vo, fs, g in manifest.execute(
            f"select scientific_name, genus, family, validated_on, first_seen_at, {fg} "
            "from records where label_conflict = 0 and north_america = 1 "
            "and (validated_on is null or substr(validated_on, 1, 10) <= ?)", (cut[:10],)):
        moment, _ = green_moment(vo, fs, g)
        if moment is None or moment > at_cut:
            continue
        labs = judge.truth(name, genus, family)
        if labs is not None and labs.species:
            labels.append(labs.species)
    return ReferenceDepth(labels, answer_key)


def judge_lists(lists: dict, labs, judge, answer_key: str) -> dict:
    """{"sp_pos", "ge_pos", "fa_pos"}: the 0-based place of the first right name at each rank
    (evaluate.evaluate's rules: species by the answer key over the collapsed list, genus and
    family as written), None when none in the list or no label at that rank."""
    from . import name_equiv
    out = {"sp_pos": None, "ge_pos": None, "fa_pos": None}
    if labs.species:
        out["sp_pos"] = name_equiv.first_credited(
            [judge.label(c["name"]) for c in lists.get("species") or []], labs.species,
            answer_key)
    for rank, col, truth in (("genus", "ge_pos", labs.genus), ("family", "fa_pos", labs.family)):
        if not truth:
            continue
        seen: list[str] = []
        for c in lists.get(rank) or []:
            name = judge.label(c["name"])
            if name not in seen:
                seen.append(name)
        out[col] = seen.index(truth) if truth in seen else None
    return out


def score_pending(store: sqlite3.Connection, manifest: sqlite3.Connection,
                  models: list[Declared], answer: Answer, held: frozenset,
                  now: str | None = None, limit: int = DEFAULT_LIMIT, lock=None,
                  answer_key: str | None = None) -> dict:
    """Decides, once, each record first green after a model's cut that has no decision yet
    (see the module's notes). Returns counts per model id."""
    import contextlib
    from . import name_equiv
    from .early_answers import DEPTH, Judge, ranked_lists
    answer_key = answer_key or name_equiv.ANSWER_KEY
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
        batch = []
        for rec in todo:
            if rec.get("vision_assisted"):
                decided.append((rec, VISION_ASSISTED, None))
            elif len(batch) >= limit:
                stats["waiting: tonight's limit"] += 1
            else:
                batch.append(rec)
        if batch:
            places = public_context(manifest, [r["observation_id"] for r in batch])
            for rec in batch:
                rec["public"] = places.get(rec["observation_id"]) or {}
            got = answer(model, batch)
            for rec in batch:
                g = got.get(rec["observation_id"])
                if not g:
                    stats["waiting: no vectors yet"] += 1
                else:
                    decided.append((rec, SCORED, g))
        names = [c["name"] for _, _, g in decided if g
                 for c in (g.get("lists") or {}).get("species") or []]
        judge = Judge(manifest, names) if decided else None
        depth = (depth_at_cut(manifest, model.cut, judge, answer_key)
                 if any(o == SCORED for _, o, _ in decided) else None)
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
                    base.update({
                        "photos": int(got.get("photos") or 0),
                        "place_used": int(bool(got.get("place_used"))),
                        "date_used": int(bool(got.get("date_used"))),
                        "answer_key": answer_key,
                        "depth": max((len(v) for v in lists.values()), default=0),
                        "ranks_json": json.dumps(lists, separators=(",", ":")),
                        "truth_species": labs.species or None, "truth_genus": labs.genus or None,
                        "truth_family": labs.family or None,
                        "species_depth_at_cut": depth(labs.species) if labs.species else None,
                        **judge_lists(lists, labs, judge, answer_key)})
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


def _labelled(rows: list[dict], rank: str) -> list[dict]:
    return [r for r in rows if r[TRUTH[rank]]]


def _rank(rows: list[dict], rank: str) -> dict:
    """{"right", "n", "top1", "ci95"}: right first over the rows with a label at `rank`;
    ci95 the Wilson score interval (heldout_report.wilson)."""
    from .heldout_report import wilson
    use = _labelled(rows, rank)
    k, n = sum(r[POS[rank]] == 0 for r in use), len(use)
    ci = wilson(k, n)
    return {"right": k, "n": n, "top1": _rate(k, n), "ci95": list(ci) if ci else None}


def _top_k(rows: list[dict], rank: str) -> dict:
    """{"n", "top1", "top3", "top5", "top10"}: the right name within the first k (as
    evaluate's standard table)."""
    use = _labelled(rows, rank)
    out = {"n": len(use)}
    for k in KS:
        out[f"top{k}"] = _rate(sum(r[POS[rank]] is not None and r[POS[rank]] < k for r in use),
                               len(use))
    return out


def _by_depth(rows: list[dict]) -> list[dict]:
    """Species by the true species' reference records at the cut, in evaluate's bands; a band
    with fewer than MIN_BAND_RECORDS records gives no rate (`too_few`)."""
    from .evaluate import STANDARD_DEPTH, standard_band
    use = [r for r in _labelled(rows, "species") if r["species_depth_at_cut"] is not None]
    out = []
    for _, _, band in STANDARD_DEPTH:
        mine = [r for r in use if standard_band(r["species_depth_at_cut"]) == band]
        cell = _top_k(mine, "species")
        too_few = cell["n"] < MIN_BAND_RECORDS
        if too_few:
            cell = {"n": cell["n"], **{f"top{k}": None for k in KS}}
        out.append({"band": band, **cell, "too_few": too_few})
    return out


def _days(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def _three(rows: list[dict]) -> dict:
    return {rank: _rank(rows, rank) for rank in RANKS}


def report(store: sqlite3.Connection, model: Declared, held: frozenset, lock=None) -> dict:
    """One model's scores since its cut, aggregates only: per UTC green day the records
    scored, their photos and species, genus and family right first (each with its n), the
    7-day rolling value ending that day; the cumulative total with its 95% intervals; top
    1/3/5/10 at each rank; species by reference depth at the cut. Held records are left out
    whatever the ledger holds. No record key, name, photo or place."""
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
            "trained_through": model.cut, "context": CONTEXT,
            "answer_key": next((r["answer_key"] for r in rows if r["answer_key"]), None),
            "as_of": as_of, "min_band_records": MIN_BAND_RECORDS,
            "left_out": {k: left.get(k, 0) for k in OUTCOMES if k != SCORED}}
    if not rows:
        return {**base, "records": 0, "new_photos": 0, "with_place": 0, "with_date": 0,
                "days": [], "cumulative": None, "rolling_7_days": None, "top_k": None,
                "species_by_depth": None}
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
            **_three(today),
            "rolling_7_days": {"records": len(window), **_three(window)},
            # A day the nightly has not finished with: more of its records may still come.
            "complete": bool(as_of_day) and d < as_of_day})
    photos = sum(r["photos"] or 0 for r in rows)
    return {**base, "records": len(rows), "new_photos": photos,
            "with_place": sum(bool(r["place_used"]) for r in rows),
            "with_date": sum(bool(r["date_used"]) for r in rows), "days": days,
            "cumulative": {"from": span[0], "through": span[-1], "records": len(rows),
                           "photos": photos, **_three(rows)},
            "rolling_7_days": {"through": span[-1], **days[-1]["rolling_7_days"]},
            "top_k": {"k": list(KS), **{rank: _top_k(rows, rank) for rank in RANKS}},
            "species_by_depth": _by_depth(rows)}


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
