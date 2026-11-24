"""Scores since training: a newer model, day by day, on the records first green after its
training cut (Steve, 2026-11-22: "for v2, once all records are trained against, I want to
have daily scores for whatever was validated that day, along with a cumulative total since
training, along with the number of new images that are added to the model since training"),
and, paired with it on the same records, the served v1 (Steve, 2026-11-24: "A", v2 against v1
side by side on records validated after 22 Nov; "any new autovalidation that has a vision call
should be scored on both v1 and v2, with v1 kept internal for scoring purposes"; "any record
that gets green checked that has vision should be scored and any green check that doesnt have
vision should have it run").

The Paper Coordinator's ruling (research repo f79c7d2, paper/PLAN.md, 24 Nov 2026) sets the
conditions this module keeps:

- Internal scoring of v1 and the beta, paired, on records first validated after PAIRED_FROM
  (2026-11-22T08:24:03Z, the beta's cut: after v1-test's close, so no v1-test record and none
  on the E17 hold list). A declared model cut before PAIRED_FROM is refused.
- (a) Each ledger row records the model id, the served release and the reference state it
  answered from, the scoring time, .org's first_green_at, the validation route (manual or
  auto, the auto rules version, Vision's role) and the scoring code's commit. Insert-only.
- (b) The PRIMARY comparison leaves out Vision-assisted validations and auto-validations
  Vision gated (agreement_validated, penalty_lift, second_source_vision) and, fail-closed,
  every record whose route is not known yet. Those records are still answered by both models
  and scored: each is its own labelled group (GROUPS), counted, never mixed into the primary.
- (c) Each record's own photos and its twins (the cross-posted copy, by the served release's
  twin pass, twins.ReferenceTwins) are hidden from BOTH references, the same records for both
  (api.daily_answers). A model with no twin table is refused (TwinsRefused).
- (d) The ledger stops at S2: a record first validated after S2_STOP (2027-01-15T00:00:00Z,
  the v2-test seal) is refused, fail-closed (s2_allows), until the v2-test sealed run is done
  and the Paper Coordinator gives the word. S2_STOP is a constant changed by commit, never a
  setting; the ledger's own trigger refuses such a row too.
- (e) Nothing from this ledger enters the v1 paper. The beta's row is descriptive: a served
  all-data model.
- Public display: v1's numbers on live records and the v2-against-v1 comparison stay off the
  public payload until the v1 paper is submitted or posted (target 8 Dec), behind their own
  setting (MV_DAILY_SCORES_PUBLIC_V1, off); research_public.daily_scores refuses them otherwise.

EVERY record first green after the cut is answered by every model of the pair, whether or not
mycomap.org called Vision when it validated it: the box runs each model on it itself. Whether
Vision was called, and its part, decide only the record's GROUP (route_of), never whether it is
answered or scored; no path here reads a Vision field to skip a record. Records with no Vision
call at all are their own sub-count inside the primary group (internal view).

A model declared here (MV_DAILY_SCORES_MODELS) was trained on every record green up to its
cut, so every record first green AFTER the cut is one its weights never saw: a fair test. v1
(MV_DAILY_SCORES_V1: its served backbone and +place method) was trained long before, so those
records are new to it too. Each night, after the green records are read and tonight's photos
are embedded (api.run_nightly, after early_answers.score_pending), every such record not yet
decided gets each model's answer, frozen in the insert-only ledger and never asked again: each
model's FIRST answer about it. A record waits until every model of the pair has answered it
from the same photos; then all its rows are written together.

Measured the way v1's How it works numbers are (Steve, 2026-11-22: like with like): each
record is identified from the records verified before it, with its photos and the place and
date recorded with it, by the served place method (nearest+mean-records-balanced+place).
Tonight's records are the test, the rest of each served index the reference: every one of
tonight's records, its cross-post twins and any copy of its photos are left out of the photo
reference (identify.left_out), and the place prior is fitted again without them (the served
method's own prior counts them), so no record is scored against itself, its place or its date.
The place and date are the ones v1's runs score with (Steve, 2026-11-22): evaluate.load_records'
record, through evaluate.context_of: .org's coordinates (records.latitude/longitude, used for
scores and never shown), .org's date, else iNat's, and the iNat uuid. A record with no
coordinates is answered with its date alone, as v1's are. None of it leaves the server
(research_public.daily_scores refuses any coordinate field). A method with no place and date
weighting is refused (PlaceRefused), never scored on photos alone.

When a record turned green is early_answers.green_moment: the earliest of .org's first green
moment, the start of its first green day and when Vision's nightly first listed it. A record
counts as after the cut only when that moment is after it (nothing says it was green at the
cut); no moment, or one at or before the cut, and it is never scored. Its day (UTC) is the
moment's.

Waits: a label conflict. Never answered: a held record. Answered and frozen but given no rank:
a guest or no fungus (not_a_fungus_label). The truth is the DNA-validated name by
early_answers.Judge (the manifest's labels, taxonomy.labels_for, so family is iNat's for the
genus from the taxonomy cache, else .org's, as in evaluate.load_records). Judged as
evaluate.evaluate judges v1's runs, at write time and frozen: species by the answer key
(name_equiv.first_credited over the collapsed list), genus and family as written; the place of
the first right name in the ten-deep answer, so top 1/3/5/10 at each rank. Each record also
keeps its true species' reference depth AT the cut (the North American, unconflicted records of
that species, by the answer key, first green at or before it: v2's reference then), for the
bands of "Every sequenced record makes it better". A band with fewer than MIN_BAND_RECORDS
records gives no rate.

The validation route comes from mycomap.org's green feed (records.ROUTE_METHOD_FIELD and the
fields beside it; proposed in docs/proposals/org-green-records-validation-route.md, not in the
feed yet). Until the feed carries it, every record that is not Vision-assisted is
ROUTE_UNKNOWN: answered, scored, counted, and out of the primary. A route learned on a later
night is appended to cut_routes (insert-only), and the reports group the record by it from
then on; the answers themselves never change.

The held records (MV_FEEDBACK_HOLD_KEYS, pinned by MV_FEEDBACK_HOLD_SHA256; the blind label
check) are never scored and never counted in a report, read with feedback.read_hold_keys. It
fails closed: a list that can't be trusted (named but missing, sha256 unset or wrong) refuses
the whole step and every report (HoldListRefused); nothing names a held key.

Settings, all off by default, so nothing changes on the box until someone turns them on:

    MV_DAILY_SCORES=1            compute: the nightly step runs
    MV_DAILY_SCORES_PUBLIC=1     show: /api/daily-scores serves the declared models' primary
                                 figures (v2 only; Steve's call)
    MV_DAILY_SCORES_PUBLIC_V1=1  also show v1's live figures and the paired comparison
                                 publicly (off until the v1 paper is out and Steve says so)

    MV_DAILY_SCORES_MODELS=v2=<backbone>/nearest+mean-records-balanced+place@<cut, ISO UTC>
                           [, v2-open=...@...]
    MV_DAILY_SCORES_V1=<v1 backbone>/nearest+mean-records-balanced+place

declare each newer model by the id the pages' dropdown shows, and the v1 each of them is paired
with (internal: /api/daily-scores/internal, mycomap.org's server key or an admin). Without
MV_DAILY_SCORES_V1 a newer model is scored alone, as before. A model named "v1" in
MV_DAILY_SCORES_MODELS, or one cut before the paper's seal or PAIRED_FROM, is refused: the
pages' v1 choice keeps its sealed numbers (/api/how-good).

State: the ledger is a table in state/early-answers.sqlite (early_answers.default_path), which
outlives releases. Only aggregates leave it (report, paired, internal_payload;
research_public.daily_scores).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from . import config

COMPUTE_SETTING = "MV_DAILY_SCORES"
PUBLIC_SETTING = "MV_DAILY_SCORES_PUBLIC"
PUBLIC_V1_SETTING = "MV_DAILY_SCORES_PUBLIC_V1"
MODELS_SETTING = "MV_DAILY_SCORES_MODELS"
COMPANION_SETTING = "MV_DAILY_SCORES_V1"
V1 = "v1"                         # the pages' sealed choice: /api/how-good, never public here
ROLLING_DAYS = 7
DEFAULT_LIMIT = 5000              # answers made in one night (the first night catches up)

# The Paper Coordinator's ruling (research f79c7d2): the records this ledger may hold. Both
# are changed by a commit the coordinator asks for, never by a setting.
PAIRED_FROM = "2026-11-22T08:24:03+00:00"   # first validated after: the v2 beta's cut
S2_STOP = "2027-01-15T00:00:00+00:00"       # first validated after: refused (v2-test seal)

RANKS = ("species", "genus", "family")
KS = (1, 3, 5, 10)                 # evaluate.STANDARD_K
MIN_BAND_RECORDS = 30              # a depth band with fewer gives no rate: "too few yet"
CONTEXT = "photos + place and date"
BOOTSTRAP_RESAMPLES = 2000         # the paired interval: records resampled with replacement
BOOTSTRAP_SEED = 20261124

SCORED = "scored"
NOT_FUNGUS = "not_a_fungus_label"
OUTCOMES = (SCORED, NOT_FUNGUS)

# The groups a record is compared in (ruling (b)). Every group is answered and scored; only
# PRIMARY makes the primary comparison and the public figure.
PRIMARY = "primary"
VISION_GATED = "vision_gated"
VISION_ASSISTED = "vision_assisted"
ROUTE_UNKNOWN = "route_unknown"
GROUPS = (PRIMARY, VISION_GATED, VISION_ASSISTED, ROUTE_UNKNOWN)
GROUP_LABELS = {
    PRIMARY: "Primary: validated with Vision taking no part in the decision",
    VISION_GATED: "Auto-validated where Vision gated the decision (agreement, penalty lift "
                  "or second source); the served v1 chose these",
    VISION_ASSISTED: "Validated with Vision's help (DNA + MycoMap Vision, or Vision's answer "
                     "on screen when a person validated)",
    ROUTE_UNKNOWN: "Validation route not known yet (mycomap.org's feed does not say): kept "
                   "out of the primary until it does",
}
# .org's vision_decision_events kinds (lib/validation-engine visionDecisionEvents.ts) that
# put a record in a group other than the primary.
GATED_ROLES = frozenset({"agreement_validated", "penalty_lift", "second_source_vision"})
SHOWN_ROLES = frozenset({"vision_shown_at_manual_validation"})
ROUTE_METHODS = ("manual", "auto", "mixed")

SCHEMA = """
create table if not exists cut_answers (
    cohort           text not null,       -- the declared model whose cut picks the records
    model_id         text not null,       -- the model that answered: the cohort's, or v1
    backbone         text not null,
    method           text not null,
    cut              text not null,       -- the cohort model's training cut (ISO UTC)
    record_key       text not null,       -- private: never in a report
    decided_at       text not null,       -- the scoring time
    outcome          text not null,       -- daily_scores.OUTCOMES
    green_at         text not null,       -- early_answers.green_moment, UTC
    green_day        text not null,       -- its UTC day: the row of the daily table
    green_rule       text,                -- moment | day
    first_green_at   text,                -- .org's first green moment as its feed gave it
    release          text,                -- the served release the model answered under
    reference_state  text,                -- the served index's answers version
    code_commit      text not null,       -- the scoring code's commit (config.code_version)
    route_group      text not null,       -- daily_scores.GROUPS, when it was scored
    route_method     text,                -- manual | auto | mixed; null: not known
    route_rules_version text,             -- the auto-validation rules version(s)
    vision_roles     text,                -- JSON list of Vision's parts; null: not known
    vision_called    integer,             -- 1/0: .org called Vision for it; null: not known
    vision_assisted  integer not null,    -- .org's validated_with_vision
    held_by_vision   integer not null,    -- .org's held_by_vision
    photos_sha       text,                -- the photos answered from (the same for the pair)
    hidden_sha       text,                -- the records hidden (the same for the pair)
    photos           integer,             -- the record's photos answered from (none trained on)
    place_used       integer,             -- 1: answered with a place (.org's, as v1)
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
    primary key (cohort, model_id, record_key)
);
create trigger if not exists cut_answers_no_update before update on cut_answers
begin select raise(abort, 'cut_answers is a ledger: no UPDATE'); end;
create trigger if not exists cut_answers_no_delete before delete on cut_answers
begin select raise(abort, 'cut_answers is a ledger: no DELETE'); end;
create table if not exists cut_routes (
    cohort           text not null,
    record_key       text not null,       -- private: never in a report
    learned_at       text not null,
    route_group      text not null,
    route_method     text,
    route_rules_version text,
    vision_roles     text,
    vision_called    integer,
    vision_assisted  integer not null,
    held_by_vision   integer not null,
    code_commit      text not null,
    primary key (cohort, record_key)
);
create trigger if not exists cut_routes_no_update before update on cut_routes
begin select raise(abort, 'cut_routes is a ledger: no UPDATE'); end;
create trigger if not exists cut_routes_no_delete before delete on cut_routes
begin select raise(abort, 'cut_routes is a ledger: no DELETE'); end;
"""

# Remade each time from S2_STOP, so a commit that moves the stop moves the ledger's own refusal.
S2_TRIGGER = """
drop trigger if exists cut_answers_s2_stop;
create trigger cut_answers_s2_stop before insert on cut_answers
when new.green_at > '{stop}'
begin select raise(abort, 'cut_answers stops at S2: a record first green after {stop}'); end;
"""

COLUMNS = ("cohort", "model_id", "backbone", "method", "cut", "record_key", "decided_at",
           "outcome", "green_at", "green_day", "green_rule", "first_green_at", "release",
           "reference_state", "code_commit", "route_group", "route_method",
           "route_rules_version", "vision_roles", "vision_called", "vision_assisted",
           "held_by_vision", "photos_sha", "hidden_sha", "photos", "place_used", "date_used",
           "answer_key", "depth", "ranks_json", "truth_species", "truth_genus", "truth_family",
           "sp_pos", "ge_pos", "fa_pos", "species_depth_at_cut")
ROUTE_COLUMNS = ("route_group", "route_method", "route_rules_version", "vision_roles",
                 "vision_called", "vision_assisted", "held_by_vision")
POS = {"species": "sp_pos", "genus": "ge_pos", "family": "fa_pos"}
TRUTH = {"species": "truth_species", "genus": "truth_genus", "family": "truth_family"}


class DeclarationRefused(ValueError):
    """MV_DAILY_SCORES_MODELS or MV_DAILY_SCORES_V1 declares something not to score."""


class HoldListRefused(RuntimeError):
    """The held records' list can't be trusted: nothing is scored or reported."""


class PlaceRefused(RuntimeError):
    """The model can't be given a record's place and date: it is not scored on photos alone."""


class TwinsRefused(RuntimeError):
    """The model's reference has no twin table: a record's twins can't be hidden (ruling c)."""


def _on(name: str) -> bool:
    return (config.setting(name) or "").strip().lower() in ("1", "true", "yes", "on")


def computing() -> bool:
    """MV_DAILY_SCORES: the nightly step runs (default off)."""
    return _on(COMPUTE_SETTING)


def public() -> bool:
    """MV_DAILY_SCORES_PUBLIC: /api/daily-scores serves the newer models' figures (default
    off)."""
    return _on(PUBLIC_SETTING)


def public_v1() -> bool:
    """MV_DAILY_SCORES_PUBLIC_V1: /api/daily-scores also serves v1's live figures and the
    paired comparison (default off; only with MV_DAILY_SCORES_PUBLIC too)."""
    return _on(PUBLIC_V1_SETTING)


class LedgerSchemaRefused(RuntimeError):
    """A cut_answers table from before the paired ledger: never written to or read as one."""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """The ledger's tables and triggers. A cut_answers table of the earlier, v2-only layout
    (no cohort or code_commit column; feat/v2-daily-scores before 24 Nov, never deployed) is
    refused rather than mixed in: score into a fresh store."""
    have = {r[1] for r in conn.execute("pragma table_info(cut_answers)")}
    if have and not {"cohort", "model_id", "code_commit", "route_group"} <= have:
        raise LedgerSchemaRefused("cut_answers has the earlier v2-only layout: start a fresh "
                                  "store; it is never altered or deleted here")
    conn.executescript(SCHEMA + S2_TRIGGER.format(stop=S2_STOP))
    conn.commit()


# --- which models --------------------------------------------------------------------------

def _moment(text: str) -> datetime | None:
    from .early_answers import _parse
    return _parse(text)


@dataclass(frozen=True)
class Declared:
    """A model scored here: the id the pages show, the served backbone and method, and the
    moment its training data was cut (every record green up to it was trained on). v1 paired
    with a cohort carries the cohort's cut: the records are the cohort's."""
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


@dataclass(frozen=True)
class Companion:
    """The served v1 (MV_DAILY_SCORES_V1), answering every cohort's records beside it."""
    backbone: str
    method: str

    @property
    def model(self) -> str:
        return f"{self.backbone}/{self.method}"

    def paired(self, cohort: Declared) -> Declared:
        return Declared(V1, self.backbone, self.method, cohort.cut)


_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,31}$")
_ENTRY = re.compile(r"^(?P<id>[^=\s]+)=(?P<backbone>[^/@\s]+)/(?P<method>[^@\s]+)@(?P<cut>\S+)$")
_MODEL = re.compile(r"^(?P<backbone>[^/@\s]+)/(?P<method>[^@\s]+)$")


def parse_models(text: str | None,
                 trained_through: dict[str, str] | None = None) -> list[Declared]:
    """The declared models (MV_DAILY_SCORES_MODELS), comma or space separated. Refused
    (DeclarationRefused), whole: a malformed entry, a repeated id, the id "v1" (v1 is only
    ever the internal companion, MV_DAILY_SCORES_V1), a cut that isn't a moment, any model of
    v1's era (a cut before the paper's fixed seal, research_public.SEAL, or training data that
    ends before it: `trained_through`, the manifest's finetunes, by backbone), and a cut
    before PAIRED_FROM (the ruling covers only records first validated after it)."""
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
            raise DeclarationRefused(f"{MODELS_SETTING}: v1 is never scored here as a model of "
                                     f"its own; it is paired internally ({COMPANION_SETTING}) "
                                     "and its choice keeps the sealed numbers (/api/how-good)")
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
        if cut < _moment(PAIRED_FROM):
            raise DeclarationRefused(f"{MODELS_SETTING}: {mid}'s cut is before {PAIRED_FROM}: "
                                     "the Paper Coordinator's ruling covers only records first "
                                     "validated after it")
        out.append(Declared(mid, m["backbone"], m["method"],
                            cut.astimezone(timezone.utc).isoformat(timespec="seconds")))
    return out


def declared(trained_through: dict[str, str] | None = None) -> list[Declared]:
    return parse_models(config.setting(MODELS_SETTING), trained_through)


def parse_companion(text: str | None, models: list[Declared] = ()) -> Companion | None:
    """MV_DAILY_SCORES_V1: v1's served backbone and method, or None when unset. Refused: a
    malformed value, and one that is a declared model's own backbone (it would be paired with
    itself)."""
    text = (text or "").strip()
    if not text:
        return None
    m = _MODEL.match(text)
    if not m:
        raise DeclarationRefused(f"{COMPANION_SETTING}: {text!r} is not <backbone>/<method>")
    for d in models:
        if d.backbone == m["backbone"]:
            raise DeclarationRefused(f"{COMPANION_SETTING}: {m['backbone']} is {d.id}'s own "
                                     "backbone, not v1's")
    return Companion(m["backbone"], m["method"])


def companion(models: list[Declared] = ()) -> Companion | None:
    return parse_companion(config.setting(COMPANION_SETTING), models)


def pair_of(model: Declared, comp: Companion | None) -> list[Declared]:
    """The models that answer a cohort's records: the cohort model, then v1 when declared."""
    return [model] + ([comp.paired(model)] if comp else [])


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


def s2_allows(rec: dict, stop: str = S2_STOP) -> bool:
    """Whether the record is known to have been first validated at or before `stop` (S2):
    .org's first green moment, Vision's first listing, or the end of its first green day is
    at or before it. Fail-closed: nothing that shows it, or every sign after it, refuses."""
    at = _moment(stop)
    bounds = [m for m in (_moment(rec.get("first_green_at")), _moment(rec.get("first_seen_at")))
              if m is not None]
    vo = rec.get("validated_on")
    if vo:
        try:
            day = date.fromisoformat(str(vo)[:10])
            bounds.append(datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
                          + timedelta(days=1))
        except ValueError:
            pass
    return bool(bounds) and min(bounds) <= at


def parse_route(raw) -> dict | None:
    """records.validation_route (JSON, records.route_of) as {"method", "rules_version",
    "vision_roles", "vision_called"}; None when the feed gave none or it can't be read."""
    if not raw:
        return None
    try:
        r = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(r, dict):
        return None
    method = r.get("method") if r.get("method") in ROUTE_METHODS else None
    roles = r.get("vision_roles")
    roles = (sorted({str(x) for x in roles}) if isinstance(roles, list)
             and all(isinstance(x, str) for x in roles) else None)
    called = r.get("vision_called")
    rules = r.get("rules_version")
    return {"method": method, "rules_version": str(rules) if rules not in (None, "") else None,
            "vision_roles": roles, "vision_called": called if isinstance(called, bool) else None}


def route_of(rec: dict) -> dict:
    """The record's validation route and its group (ruling (b)), from the manifest's fields:
    {"route_group", "route_method", "route_rules_version", "vision_roles" (JSON or None),
    "vision_called", "vision_assisted", "held_by_vision"}. Decides the group only: every
    record is answered and scored whatever it says.

      Vision-assisted   .org's validated_with_vision, or Vision's answer on screen when a
                        person validated (vision_shown_at_manual_validation)
      Vision-gated      Vision was the agreement, the penalty lift or the second source
      route unknown     the feed gives no route, or no Vision roles: fail-closed
      primary           a known route with no such part for Vision"""
    assisted = int(bool(rec.get("vision_assisted")))
    held = int(bool(rec.get("held_by_vision")))
    route = parse_route(rec.get("validation_route")) or {}
    method, roles = route.get("method"), route.get("vision_roles")
    if assisted:
        group = VISION_ASSISTED
    elif roles is not None and GATED_ROLES & set(roles):
        group = VISION_GATED
    elif roles is not None and SHOWN_ROLES & set(roles):
        group = VISION_ASSISTED
    elif method is None or roles is None:
        group = ROUTE_UNKNOWN
    else:
        group = PRIMARY
    called = route.get("vision_called")
    return {"route_group": group, "route_method": method,
            "route_rules_version": route.get("rules_version"),
            "vision_roles": json.dumps(roles) if roles is not None else None,
            "vision_called": None if called is None else int(called),
            "vision_assisted": assisted, "held_by_vision": held}


_OPTIONAL = ("first_green_at", "vision_assisted", "held_by_vision", "validation_route")


def candidates(manifest: sqlite3.Connection, model: Declared, done: set[str],
               held: frozenset) -> tuple[list[dict], Counter]:
    """The records first green after the model's cut with no decision yet, newest first,
    and counts of the rest (held ones only as a total). Nothing here looks at whether Vision
    was called: such fields only travel with the record, for its group."""
    have = {r[1] for r in manifest.execute("pragma table_info(records)")}
    cols = ["observation_id", "source", "source_id", "scientific_name", "genus", "family",
            "validated_on", "first_seen_at", "label_conflict"]
    cols += [c for c in _OPTIONAL if c in have]
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
        if not s2_allows(rec):
            stats["refused: first green after S2 (v2-test seal)"] += 1
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
"genus", "family"}, "photos": n, "place_used": bool, "date_used": bool, and when the server
knows them "release", "reference_state", "photos_sha", "hidden_sha"}, or None while it can't
answer it}. Called for every model of a pair with the same records; the server answers them
all from the same photos with the same records hidden (api.daily_answers). All of them are
answered as tonight's test: from the records verified before them, with the place and date
v1's runs use (evaluate.context_of), every one of them left out of the reference and of the
place prior."""


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


def _ledger(store: sqlite3.Connection, cohort: str, model: Declared) -> list[dict]:
    cur = store.execute("select * from cut_answers where cohort = ? and model_id = ? and "
                        "backbone = ? and method = ? and cut = ?",
                        (cohort, model.id, model.backbone, model.method, model.cut))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def learn_routes(store: sqlite3.Connection, manifest: sqlite3.Connection, model: Declared,
                 now: str, code: str) -> int:
    """For the cohort's records scored while their route was unknown, the route the
    manifest knows now, appended once to cut_routes (insert-only). Returns how many."""
    unknown = {r[0] for r in store.execute(
        "select record_key from cut_answers where cohort = ? and model_id = ? and backbone = ? "
        "and method = ? and cut = ? and route_group = ?",
        (model.id, model.id, model.backbone, model.method, model.cut, ROUTE_UNKNOWN))}
    unknown -= {r[0] for r in store.execute("select record_key from cut_routes where cohort = ?",
                                            (model.id,))}
    if not unknown:
        return 0
    have = {r[1] for r in manifest.execute("pragma table_info(records)")}
    cols = ["observation_id"] + [c for c in _OPTIONAL if c in have]
    keys, rows = sorted(unknown), []
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        cur = manifest.execute(f"select {', '.join(cols)} from records where observation_id in "
                               f"({', '.join('?' for _ in part)})", part)
        for row in cur.fetchall():
            route = route_of(dict(zip(cols, row)))
            if route["route_group"] != ROUTE_UNKNOWN:
                rows.append((model.id, row[0], now, *(route[c] for c in ROUTE_COLUMNS), code))
    with store:
        store.executemany("insert or ignore into cut_routes (cohort, record_key, learned_at, "
                          f"{', '.join(ROUTE_COLUMNS)}, code_commit) values "
                          f"({', '.join('?' for _ in range(len(ROUTE_COLUMNS) + 4))})", rows)
    return len(rows)


def _same(values) -> bool:
    return len({v for v in values}) <= 1


def score_pending(store: sqlite3.Connection, manifest: sqlite3.Connection,
                  models: list[Declared], answer: Answer, held: frozenset,
                  now: str | None = None, limit: int = DEFAULT_LIMIT, lock=None,
                  answer_key: str | None = None, companion: Companion | None = None,
                  code: str | None = None) -> dict:
    """Decides, once, each record first green after a model's cut that has no decision yet
    (see the module's notes): every model of the pair (pair_of) answers it, and its rows are
    written together once all have answered from the same photos. Returns counts per model
    id (records, not rows), with the records by group."""
    from . import name_equiv
    from .early_answers import DEPTH, Judge, ranked_lists
    answer_key = answer_key or name_equiv.ANSWER_KEY
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    code = code or config.code_version()
    guard = lock or contextlib.nullcontext()
    with guard:
        ensure_schema(store)
    out: dict = {}
    for model in models:
        pair = pair_of(model, companion)
        with guard:
            done = {r["record_key"] for r in _ledger(store, model.id, model)}
        todo, stats = candidates(manifest, model, done, held)
        batch = todo[:limit]
        if len(todo) > len(batch):
            stats["waiting: tonight's limit"] += len(todo) - len(batch)
        # Every record of the batch goes to every model: none is skipped for how .org
        # validated it, or for whether .org ever called Vision on it.
        got = {m.id: (answer(m, batch) or {}) if batch else {} for m in pair}
        decided: list[tuple[dict, list[dict]]] = []
        for rec in batch:
            gs = [got[m.id].get(rec["observation_id"]) for m in pair]
            if not all(gs):
                stats["waiting: no vectors yet"] += 1
            elif not (_same(g.get("photos_sha") for g in gs)
                      and _same(g.get("hidden_sha") for g in gs)):
                stats["waiting: the models' photos or hidden records differ"] += 1
            else:
                decided.append((rec, gs))
        names = [c["name"] for _, gs in decided for g in gs
                 for c in (g.get("lists") or {}).get("species") or []]
        judge = Judge(manifest, names) if decided else None
        depth = depth_at_cut(manifest, model.cut, judge, answer_key) if decided else None
        rows, groups = [], Counter()
        for rec, gs in decided:
            labs = judge.truth(rec.get("scientific_name"), rec.get("genus"), rec.get("family"))
            outcome = SCORED if labs is not None else NOT_FUNGUS
            route = route_of(rec)
            green = rec["green_at"].astimezone(timezone.utc)
            for m, g in zip(pair, gs):
                base = {"cohort": model.id, "model_id": m.id, "backbone": m.backbone,
                        "method": m.method, "cut": model.cut,
                        "record_key": rec["observation_id"], "decided_at": now,
                        "outcome": outcome, "green_at": green.isoformat(timespec="seconds"),
                        "green_day": green.date().isoformat(), "green_rule": rec["green_rule"],
                        "first_green_at": rec.get("first_green_at"),
                        "release": g.get("release"), "reference_state": g.get("reference_state"),
                        "code_commit": code, "photos_sha": g.get("photos_sha"),
                        "hidden_sha": g.get("hidden_sha"), **route}
                if outcome == SCORED:
                    lists, _ = ranked_lists(None, g["lists"], DEPTH)
                    lists = {r: lists.get(r) or [] for r in RANKS}
                    base.update({
                        "photos": int(g.get("photos") or 0),
                        "place_used": int(bool(g.get("place_used"))),
                        "date_used": int(bool(g.get("date_used"))),
                        "answer_key": answer_key,
                        "depth": max((len(v) for v in lists.values()), default=0),
                        "ranks_json": json.dumps(lists, separators=(",", ":")),
                        "truth_species": labs.species or None,
                        "truth_genus": labs.genus or None,
                        "truth_family": labs.family or None,
                        "species_depth_at_cut": depth(labs.species) if labs.species else None,
                        **judge_lists(lists, labs, judge, answer_key)})
                rows.append(tuple(base.get(c) for c in COLUMNS))
            stats[outcome] += 1
            groups[route["route_group"]] += 1
        with guard:
            with store:
                store.executemany(f"insert or ignore into cut_answers ({', '.join(COLUMNS)}) "
                                  f"values ({', '.join('?' for _ in COLUMNS)})", rows)
            learned = learn_routes(store, manifest, model, now, code)
        if learned:
            stats["routes learned"] += learned
        res = dict(stats)
        if groups:
            res["groups"] = dict(groups)
        out[model.id] = res
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


def routes(store: sqlite3.Connection, cohort: Declared) -> dict[str, dict]:
    """Each of the cohort's records' route as the reports use it: the one it was scored
    with, or, for one scored with its route unknown, the first route learned since
    (cut_routes). {record key: {"route_group", "vision_called", "held_by_vision", ...}}."""
    out = {r["record_key"]: {c: r[c] for c in ROUTE_COLUMNS}
           for r in _ledger(store, cohort.id, cohort)}
    cur = store.execute(f"select record_key, {', '.join(ROUTE_COLUMNS)} from cut_routes "
                        "where cohort = ?", (cohort.id,))
    for row in cur.fetchall():
        if row[0] in out and out[row[0]]["route_group"] == ROUTE_UNKNOWN:
            out[row[0]] = dict(zip(ROUTE_COLUMNS, row[1:]))
    return out


def _rows(store, model: Declared, cohort: Declared, held: frozenset, lock=None
          ) -> tuple[list[dict], dict[str, dict]]:
    with lock or contextlib.nullcontext():
        ensure_schema(store)
        found = _ledger(store, cohort.id, model)
        route = routes(store, cohort)
    found = [r for r in found if r["record_key"] not in held]
    for r in found:
        r.update(route.get(r["record_key"]) or {})
    return found, route


def report(store: sqlite3.Connection, model: Declared, held: frozenset, lock=None,
           cohort: Declared | None = None, groups=None) -> dict:
    """One model's scores on a cohort's records (its own, by default), aggregates only: per
    UTC green day the records scored, their photos and species, genus and family right first
    (each with its n), the 7-day rolling value ending that day; the cumulative total with its
    95% intervals; top 1/3/5/10 at each rank; species by reference depth at the cut. With
    `groups`, only the records in those groups (route_of; routes), the others counted in
    `left_out` by group. Held records are left out whatever the ledger holds. No record key,
    name, photo or place."""
    cohort = cohort or model
    found, _ = _rows(store, model, cohort, held, lock)
    keep = set(groups) if groups is not None else set(GROUPS)
    outside = Counter(r["route_group"] for r in found if r["route_group"] not in keep)
    mine = [r for r in found if r["route_group"] in keep]
    rows = [r for r in mine if r["outcome"] == SCORED]
    left = Counter(r["outcome"] for r in mine if r["outcome"] != SCORED)
    as_of = max((r["decided_at"] for r in found), default=None)
    left_out = {NOT_FUNGUS: left.get(NOT_FUNGUS, 0)}
    if groups is not None:
        left_out.update({g: outside.get(g, 0) for g in GROUPS if g not in keep})
    base = {"id": model.id, "backbone": model.backbone, "method": model.method,
            "trained_through": cohort.cut, "context": CONTEXT,
            "answer_key": next((r["answer_key"] for r in rows if r["answer_key"]), None),
            "as_of": as_of, "min_band_records": MIN_BAND_RECORDS, "left_out": left_out}
    if groups is not None:
        base["groups"] = [g for g in GROUPS if g in keep]
    if not rows:
        return {**base, "records": 0, "new_photos": 0, "with_place": 0, "with_date": 0,
                "days": [], "cumulative": None, "rolling_7_days": None, "top_k": None,
                "species_by_depth": None}
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["green_day"], []).append(r)
    span = _days(cohort.cut[:10], max(by_day))
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


# --- the paired comparison --------------------------------------------------------------------

def mcnemar_p(b: int, c: int) -> float | None:
    """The exact two-sided McNemar p-value for b and c discordant pairs (a binomial test of
    b against b + c at one half); None with no discordant pair."""
    n = b + c
    if not n:
        return None
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def paired_interval(a: list[bool], b: list[bool], resamples: int = BOOTSTRAP_RESAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    """The 95% paired bootstrap interval of mean(a) - mean(b): records resampled with
    replacement, both models' answers about a record kept together. None with no records."""
    import numpy as np
    n = len(a)
    if not n:
        return None
    d = np.asarray(a, dtype=np.int8) - np.asarray(b, dtype=np.int8)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    step = max(1, min(resamples, 2_000_000 // n))
    for s in range(0, resamples, step):
        k = min(step, resamples - s)
        means[s:s + k] = d[rng.integers(0, n, size=(k, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def paired(store: sqlite3.Connection, cohort: Declared, comp: Companion, held: frozenset,
           lock=None, groups=(PRIMARY,)) -> dict:
    """The cohort model against v1 on the same records (both scored, same photos, same records
    hidden), in `groups`: per rank, the records with a label there, how many each got right
    first, both, neither, the difference (cohort minus v1) with its 95% paired bootstrap
    interval, and McNemar's exact p over the discordant records. Aggregates only."""
    v1 = comp.paired(cohort)
    a_rows, _ = _rows(store, cohort, cohort, held, lock)
    b_rows, _ = _rows(store, v1, cohort, held, lock)
    keep = set(groups)
    a = {r["record_key"]: r for r in a_rows if r["outcome"] == SCORED and r["route_group"] in keep}
    b = {r["record_key"]: r for r in b_rows if r["outcome"] == SCORED}
    both = sorted(set(a) & set(b))
    out: dict = {"cohort": cohort.id, "against": V1, "groups": [g for g in GROUPS if g in keep],
                 "records": len(both), "unpaired": len(set(a) - set(b))}
    for rank in RANKS:
        keys = [k for k in both if a[k][TRUTH[rank]]]
        ra = [a[k][POS[rank]] == 0 for k in keys]
        rb = [b[k][POS[rank]] == 0 for k in keys]
        n = len(keys)
        only_a = sum(x and not y for x, y in zip(ra, rb))
        only_b = sum(y and not x for x, y in zip(ra, rb))
        out[rank] = {"n": n, "both_right": sum(x and y for x, y in zip(ra, rb)),
                     "only_cohort_right": only_a, "only_v1_right": only_b,
                     "both_wrong": sum(not x and not y for x, y in zip(ra, rb)),
                     "top1": {cohort.id: _rate(sum(ra), n), V1: _rate(sum(rb), n)},
                     "difference": _rate(sum(ra) - sum(rb), n),
                     "ci95": paired_interval(ra, rb), "mcnemar_p": mcnemar_p(only_a, only_b)}
    return out


def _summary(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["outcome"] == SCORED]
    return {"records": len(scored), **_three(scored),
            "top_k": {"k": list(KS), **{rank: _top_k(scored, rank) for rank in RANKS}}}


def internal_payload(store: sqlite3.Connection | None, models: list[Declared] | None = None,
                     comp: Companion | None = None, held: frozenset | None = None,
                     lock=None, trained_through: dict | None = None) -> dict:
    """The internal view (/api/daily-scores/internal): for each declared model, every group's
    records (answered by both models), with the primary's sub-counts (no Vision call at all,
    Vision call not known, held by Vision once), each model's figures per group, the paired
    comparison per group, and the primary's full reports for both models. Aggregates only; it
    may name v1. Refused whole (its reason only) when a declaration or the hold list is."""
    out: dict = {"internal": True, "paired_from": PAIRED_FROM, "s2_stop": S2_STOP,
                 "public": public(), "public_v1": public_v1(), "cohorts": {}}
    if store is None:
        return out
    try:
        models = declared(trained_through) if models is None else models
        comp = companion(models) if comp is None else comp
        held = held_keys() if held is None else held
    except (DeclarationRefused, HoldListRefused) as e:
        return {**out, "refused": str(e)}
    out["v1"] = comp.model if comp else None
    try:
        with lock or contextlib.nullcontext():
            ensure_schema(store)
    except LedgerSchemaRefused as e:
        return {**out, "refused": str(e)}
    for m in models:
        pair = pair_of(m, comp)
        by_model = {p.id: _rows(store, p, m, held, lock)[0] for p in pair}
        cohort_rows = by_model[m.id]
        groups: dict = {}
        for g in GROUPS:
            mine = [r for r in cohort_rows if r["route_group"] == g]
            entry: dict = {"label": GROUP_LABELS[g], "records": len(mine),
                           "scored": sum(r["outcome"] == SCORED for r in mine)}
            if g == PRIMARY:
                entry["no_vision_call"] = sum(r["vision_called"] == 0 for r in mine)
                entry["vision_call_not_known"] = sum(r["vision_called"] is None for r in mine)
                entry["held_by_vision"] = sum(bool(r["held_by_vision"]) for r in mine)
            keys = {r["record_key"] for r in mine}
            entry["models"] = {p.id: _summary([r for r in by_model[p.id]
                                               if r["record_key"] in keys]) for p in pair}
            if comp:
                entry["paired"] = paired(store, m, comp, held, lock, groups=(g,))
            groups[g] = entry
        out["cohorts"][m.id] = {
            "trained_through": m.cut, "models": {p.id: p.model for p in pair},
            "records": len(cohort_rows), "groups": groups,
            "primary": {p.id: report(store, p, held, lock=lock, cohort=m, groups=(PRIMARY,))
                        for p in pair}}
    return out


def public_payload(store: sqlite3.Connection | None, models: list[Declared] | None = None,
                   held: frozenset | None = None, show: bool | None = None, lock=None,
                   comp: Companion | None = None, show_v1: bool | None = None) -> dict:
    """/api/daily-scores: which models the pages may offer (v1 always, its sealed numbers
    from /api/how-good), the default (the first declared model with a score, when shown;
    else v1), and each shown model's report on its PRIMARY records. v1's live figures and the
    paired comparison are never in it unless MV_DAILY_SCORES_PUBLIC_V1 is on as well
    ("v1_live", "paired"). Nothing but v1 while MV_DAILY_SCORES_PUBLIC is off, the
    declaration is refused or the hold list can't be trusted."""
    show = public() if show is None else show
    show_v1 = public_v1() if show_v1 is None else show_v1
    out: dict = {"public": bool(show), "available": [V1], "default": V1, "models": {}}
    if not show or store is None:
        return out
    try:
        models = declared() if models is None else models
        held = held_keys() if held is None else held
        if show_v1 and comp is None:
            comp = companion(models)
    except (DeclarationRefused, HoldListRefused):
        return out
    shown: dict = {"models": {}, "available": [V1]}
    try:
        for m in models:
            rep = report(store, m, held, lock=lock, groups=(PRIMARY,))
            if rep["records"]:
                shown["models"][m.id] = rep
                shown["available"].append(m.id)
        if show_v1 and comp:
            shown["v1_live"] = {m.id: report(store, comp.paired(m), held, lock=lock, cohort=m,
                                             groups=(PRIMARY,))
                                for m in models if m.id in shown["models"]}
            shown["paired"] = {m.id: paired(store, m, comp, held, lock) for m in models
                               if m.id in shown["models"]}
    except LedgerSchemaRefused:
        return out
    out.update(shown)
    if len(out["available"]) > 1:
        out["default"] = out["available"][1]
    return out


def same_find_records(tonight, loaded_lists) -> set[str]:
    """The records that may be another site's copy of one of `tonight`'s (evaluate.Record):
    the same label and the same observation day, from the other site. The served release's
    twin pass (twins.ReferenceTwins) covers the records it was built with; this covers the
    cross-posts newer than it, for ruling (c). It may hide a few records of another find of
    that name and day too: they are hidden from every model alike."""
    finds: dict = {}
    for loaded in loaded_lists:
        for r in loaded:
            if r.unit and r.observed_on:
                finds.setdefault((r.unit, r.observed_on), set()).add((r.observation_id, r.source))
    out: set[str] = set()
    for r in tonight:
        for other, source in finds.get((r.unit, r.observed_on), ()):
            if source != r.source:
                out.add(other)
    return out


def photo_sha(photo_ids) -> str:
    """The photos a record was answered from, as the ledger keeps them (not the ids)."""
    return hashlib.sha256(",".join(str(int(p)) for p in photo_ids).encode()).hexdigest()[:16]


def records_sha(keys) -> str:
    """The records hidden from a night's references, as the ledger keeps them."""
    return hashlib.sha256("\n".join(sorted(str(k) for k in keys)).encode()).hexdigest()[:16]
