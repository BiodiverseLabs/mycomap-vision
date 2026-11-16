"""Every nightly photo goes to the photo review (Steve, 2026-11-07): "whenever there is a nightly
update, all of our putative exclusions are added. They are not excluded until human validation.
But it should be a dynamic process for each record that is pushed in. Tagger runs for all. Human
review still required to save them".

So, with MV_NIGHTLY_PHOTO_REVIEW=1, each photo the nightly update embeds (nightly.run_once) is
tagged by the served photo-type tagger (phototypes.PhotoTagger), and each one it tags as anything
but `specimen`, in the bands MV_NIGHTLY_PHOTO_REVIEW_BANDS names (default high, medium and low,
like the past seeds), is put on the Human Feedback photo review ("Photos that show no specimen",
feedback.TYPES["non_specimen_photo"]) as a putative exclusion.

Nothing here excludes a photo. A photo leaves Vision only when a person decides
(feedback.Store.excluded_photos reads decisions only), as for every other flag.

How the tag is made. The tagger reads the photo's FROZEN base vector (bioclip-2-336 for v2.1-336),
the one the not-a-fungus gate reads; the nightly embeds with the served fine-tune. The fine-tune
changed only its last blocks, so models.FinetunedBackbone.encode_with_base runs the shared blocks
once and the tail twice (trained, and with the base's own weights): the base vector costs the
tail again, 4 of 24 blocks for the served model, about 1/6 more compute a photo, in the same pass
and under the same model lock (nightly.Locked) while the photo file is still on the box. Photos
are deleted once embedded, so this is the only time a night can read them for free.

Each tag is kept in the layer's manifest (photo_type_tags), so a tag a night made is queued even
when the queueing failed or the seal refused it: each night retries the tags not yet queued,
and once the paper's seal is switched off (MV_PROSPECTIVE_SEAL=off) the next night queues them
without reading a photo again.

Every row goes through feedback.Store.seed with the box's seal gate (feedback.SealGate): a photo
of a record first green after the seal is refused and counted, never queued. No duplicates: a
photo already on the queue keeps its item (an open one gains the nightly flag, FLAG_MERGE_TYPES);
a decided one never reopens.

A failure here never fails or rolls back the nightly update: each step is caught, logged and
reported (`photo_review` in the night's report, /api/health and `mv nightly`).

`mv feedback seed-nightly-tags --since YYYY-MM-DD [--dry-run]` (catch_up) is the backfill: the
photos of every record first green on or after a date that are not yet queued or decided, tagged
(fetched again from their source, within the download budgets, and read by the frozen base model)
and queued through the same gate.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from . import feedback as fb
from . import phototypes

SETTING = "MV_NIGHTLY_PHOTO_REVIEW"
BANDS_SETTING = "MV_NIGHTLY_PHOTO_REVIEW_BANDS"
DEFAULT_BANDS = (phototypes.HIGH, phototypes.MEDIUM, phototypes.LOW)
ITEM_TYPE = "non_specimen_photo"
SEED_PREFIX = "nightly-photo-review"
RULE_PREFIX = "nightly"
# When the box has no timing of its own yet: seconds a photo for one full tower pass on the
# Lightsail box's 4 vCPUs (the nightly's embed_runs, 2-7 Nov 2026: 2.3-2.5 s, ~5 s throttled).
FALLBACK_SECONDS_PER_PHOTO = 2.4

SCHEMA = """
-- The photo-type tag of each photo a night (or the catch-up) tagged, and what queueing did.
create table if not exists photo_type_tags (
  photo_id     integer not null,
  fingerprint  text not null,        -- phototypes.PhotoTagger.fingerprint()
  version      text,                 -- the pinned tagger version (v2.1-336)
  base         text,                 -- whose frozen vectors it read
  type         text not null,        -- the top class ("specimen": an ordinary photo)
  p            real not null,
  band         text,                 -- high | medium | low; null for a specimen photo
  tagged_at    text not null,
  tagged_by    text not null,        -- "nightly <run id>" | "catch-up"
  queue_state  text,                 -- null: not tried yet; else QUEUE_STATES
  queued_at    text,
  seed         text,
  primary key (photo_id, fingerprint)
);
create index if not exists photo_type_tags_state on photo_type_tags (queue_state);
"""

# What queueing made of a tag. FINAL ones are never tried again; the others are retried each
# night (a refused one once the seal is off; an out-of-band one if the bands change).
QUEUE_STATES = ("queued", "already_queued", "already_decided", "specimen", "out_of_band",
                "refused_sealed", "refused_unknown", "no_record")
FINAL = ("queued", "already_queued", "already_decided", "specimen")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- settings ----------------------------------------------------------------------------------

def enabled() -> bool:
    """MV_NIGHTLY_PHOTO_REVIEW=1 (off unless switched on)."""
    return (config.setting(SETTING) or "").lower() in ("1", "true", "yes", "on")


def bands_from_env() -> tuple[str, ...]:
    """The bands queued (MV_NIGHTLY_PHOTO_REVIEW_BANDS, comma-separated; default
    high,medium,low). ValueError on a band the tagger doesn't have, or none."""
    raw = config.setting(BANDS_SETTING)
    if not raw:
        return DEFAULT_BANDS
    bands = tuple(dict.fromkeys(b.strip().lower() for b in raw.split(",") if b.strip()))
    unknown = [b for b in bands if b not in phototypes.BANDS]
    if unknown or not bands:
        raise ValueError(f"{BANDS_SETTING} names bands {', '.join(phototypes.BANDS)}, "
                         f"not {raw!r}")
    return bands


# --- the base vectors ----------------------------------------------------------------------------

class SameBase:
    """A served backbone that IS the tagger's base (a frozen one): its vectors are the base
    vectors, so no second pass is needed."""

    def __init__(self, inner):
        self._inner = inner
        self.name, self.dim = inner.name, inner.dim

    def encode(self, images):
        return self._inner.encode(images)

    def encode_with_base(self, images):
        v = self._inner.encode(images)
        return v, v

    def __getattr__(self, name):
        return getattr(self._inner, name)


def with_base_model(model, base: str):
    """(`model` able to give `base` vectors from its own pass, or None; why not). A fine-tune of
    `base` forks its tail (models.FinetunedBackbone.encode_with_base); the base itself gives
    its own vectors (SameBase); any other backbone can't."""
    if getattr(model, "base_name", None) == base and callable(
            getattr(model, "encode_with_base", None)):
        return model, None
    if getattr(model, "name", None) == base:
        return SameBase(model), None
    return None, (f"the served backbone {getattr(model, 'name', '?')} gives no {base} "
                  "vectors (the photo-type tagger's base)")


def save_tags(conn: sqlite3.Connection, tagger: phototypes.PhotoTagger, photo_ids,
              base_vectors, tagged_by: str) -> int:
    """Tag the photos from their base vectors and keep each tag (one per photo and tagger
    fingerprint: a photo tagged before keeps its first tag). Returns the tags made."""
    ensure_schema(conn)
    ids = [int(p) for p in photo_ids]
    if not ids:
        return 0
    tags = tagger.tag(np.asarray(base_vectors, np.float32))
    print_ = tagger.fingerprint()
    now = now_iso()
    with conn:
        conn.executemany(
            "insert or ignore into photo_type_tags (photo_id, fingerprint, version, base, type, "
            "p, band, tagged_at, tagged_by) values (?,?,?,?,?,?,?,?,?)",
            [(pid, print_, tagger.version, tagger.base, t["type"], float(t["p"]), t["band"], now,
              tagged_by) for pid, t in zip(ids, tags)])
    return len(ids)


@dataclass
class NightSink:
    """What embed.embed_photos hands each batch's base vectors to, during one night. Never
    raises: a failure is counted and logged, and the night goes on."""
    conn: sqlite3.Connection
    tagger: phototypes.PhotoTagger
    tagged_by: str
    log: Callable[[str], None] = print
    tagged: int = 0
    errors: list = field(default_factory=list)

    def __call__(self, photo_ids, base_vectors) -> None:
        try:
            self.tagged += save_tags(self.conn, self.tagger, photo_ids, base_vectors,
                                     self.tagged_by)
        except Exception as e:  # noqa: BLE001 - tagging never fails the night
            err = f"{type(e).__name__}: {e}"[:300]
            self.errors.append(err)
            self.log(f"[nightly] photo tags failed for a batch, carrying on: {err}")


# --- queue rows --------------------------------------------------------------------------------

def rule_of(tag: dict) -> str:
    """The item's rule: what the review's tabs read (photo_review.parse_flags: the tagger's
    kind, p and band), naming the served tagger: 'nightly v2.1-336 micrograph p=0.910 (high)'."""
    return f"{RULE_PREFIX} {tag['version']} {tag['type']} p={float(tag['p']):.3f} ({tag['band']})"


def _photo_records(conn: sqlite3.Connection, photo_ids: list[int]) -> dict[int, dict]:
    """Per photo: its key and the record it is reviewed under, the one of its records first
    green last (no date first), so the seal judges the strictest; with that record's green
    projects."""
    from .photo_review import photo_key_of
    out: dict[int, dict] = {}
    for i in range(0, len(photo_ids), 500):
        chunk = photo_ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for pid, source, spid, key, green, day in conn.execute(
                f"select p.photo_id, p.source, p.source_photo_id, r.observation_id, "
                f"r.green_projects, r.validated_on from photos p "
                f"join observation_photos op on op.photo_id = p.photo_id "
                f"join records r on r.observation_id = op.observation_id and r.source = p.source "
                f"where p.photo_id in ({marks}) "
                f"order by p.photo_id, (r.validated_on is null) desc, r.validated_on desc, "
                f"r.observation_id", chunk):
            if pid in out:
                continue
            out[pid] = {"photo_key": photo_key_of(int(pid), source, spid), "record": key,
                        "projects": sorted(set(json.loads(green or "[]")))}
    return out


def queue_row(tag: dict, where: dict, run_label: str) -> dict:
    """One non_specimen_photo item, shaped like the past photo seeds: the photo key as subject,
    the record's link, the tagger's rule and evidence, the record's green projects."""
    return {"item_type": ITEM_TYPE, "subject": where["photo_key"], "granularity": "photo",
            "rule": rule_of(tag), "model_flagged": False, "record_keys": [],
            "links": [fb.link(where["record"])],
            "evidence": {"evidence": (
                f"Photo-type tagger {tag['version']} ({tag['fingerprint']}) on the frozen "
                f"{tag['base']} vector, {run_label}: tagged {tag['type']} p={float(tag['p']):.3f} "
                f"({tag['band']} band). A putative exclusion: the photo stays in Vision until a "
                "person decides."),
                "tagger": tag["version"], "fingerprint": tag["fingerprint"],
                "type": tag["type"], "p": float(tag["p"]), "band": tag["band"],
                "in_v1": "", "v1_val_outcome": ""},
            "projects": where["projects"], "records": 1}


def _set_state(conn, rows: list[tuple[str, str | None, int, str]]) -> None:
    now = now_iso()
    with conn:
        conn.executemany("update photo_type_tags set queue_state = ?, seed = ?, queued_at = ? "
                         "where photo_id = ? and fingerprint = ?",
                         [(state, seed, now, pid, fp) for state, seed, pid, fp in rows])


def queue_tags(conn: sqlite3.Connection, store: fb.Store, bands, seed_name: str,
               run_label: str, photo_ids=None, dry_run: bool = False, log=print) -> dict:
    """Queue the kept tags not queued yet (all of them, or only `photo_ids`): each non-specimen
    tag in `bands` goes through store.seed (with its seal gate) as one photo item. Counts:
    candidates (tags tried), specimen, out_of_band, queued (new items), already_queued (an open
    item: it gains the flag when it lacks it, flags_added), already_decided, refused_sealed,
    refused_unknown, no_record (the photo is in no record now). `dry_run`: the same counts,
    judged by the same gate and queue, with nothing seeded or marked."""
    ensure_schema(conn)
    bands = tuple(bands)
    sql = ("select photo_id, fingerprint, version, base, type, p, band from photo_type_tags "
           "where (queue_state is null or queue_state not in (%s))" % ",".join("?" * len(FINAL)))
    params: list = list(FINAL)
    tags = [dict(zip(("photo_id", "fingerprint", "version", "base", "type", "p", "band"), r))
            for r in conn.execute(sql, params)]
    if photo_ids is not None:
        wanted = {int(p) for p in photo_ids}
        tags = [t for t in tags if t["photo_id"] in wanted]
    counts = {"candidates": len(tags), "specimen": 0, "out_of_band": 0, "queued": 0,
              "already_queued": 0, "flags_added": 0, "already_decided": 0,
              "refused_sealed": 0, "refused_unknown": 0, "no_record": 0}
    if not tags:
        return {**counts, "seed": None, "bands": list(bands)}
    where = _photo_records(conn, sorted({t["photo_id"] for t in tags}))
    states: list[tuple[str, str | None, int, str]] = []
    rows, row_tags = [], []
    for t in tags:
        mark = (t["photo_id"], t["fingerprint"])
        if t["type"] == phototypes.SPECIMEN:
            counts["specimen"] += 1
            states.append(("specimen", None, *mark))
        elif t["band"] not in bands:
            counts["out_of_band"] += 1
            states.append(("out_of_band", None, *mark))
        elif t["photo_id"] not in where:
            counts["no_record"] += 1
            states.append(("no_record", None, *mark))
        else:
            rows.append(queue_row(t, where[t["photo_id"]], run_label))
            row_tags.append(t)
    if rows:
        # Each row's outcome, judged as Store.seed judges it (the same gate, the same queue),
        # so a refused tag is tried again on a later night and a queued one never is.
        gate = store.gate
        existing = dict(store.conn.execute(
            "select subject, status from items where item_type = ?", (ITEM_TYPE,)).fetchall())
        for r, t in zip(rows, row_tags):
            verdict = gate.judge_row(r)[0] if gate is not None else "ok"
            state = (f"refused_{verdict}" if verdict != "ok" else
                     "already_decided" if existing.get(r["subject"], "open") != "open" else
                     "already_queued" if r["subject"] in existing else "queued")
            states.append((state, seed_name if verdict == "ok" else None,
                           t["photo_id"], t["fingerprint"]))
        if dry_run:                     # the rows' outcomes, as judged: nothing seeded
            for state, *_ in states[len(states) - len(rows):]:
                counts[state] += 1
            return {**counts, "seed": None, "bands": list(bands), "dry_run": True}
        out = store.seed(rows, seed_name)
        counts["queued"] = out["added"]
        counts["flags_added"] = out.get("flags_added", 0)
        counts["already_decided"] = out.get("already_decided", 0)
        counts["refused_sealed"] = out.get("refused_sealed", 0)
        counts["refused_unknown"] = out.get("refused_unknown", 0)
        counts["already_queued"] = (len(rows) - counts["queued"] - counts["already_decided"]
                                    - counts["refused_sealed"] - counts["refused_unknown"])
    if dry_run:
        return {**counts, "seed": None, "bands": list(bands), "dry_run": True}
    _set_state(conn, states)
    log(f"[nightly] photo review: {counts['queued']:,} queued, {counts['already_queued']:,} "
        f"already queued, {counts['already_decided']:,} already decided, "
        f"{counts['refused_sealed']:,} refused by the seal ({counts['refused_unknown']:,} "
        f"unknown), {counts['specimen']:,} specimen, {counts['out_of_band']:,} out of band")
    return {**counts, "seed": seed_name if rows else None, "bands": list(bands)}


# --- one night ---------------------------------------------------------------------------------

@dataclass
class NightlyReview:
    """The photo review's part of one nightly run (nightly.run_once's `photo_review`).
    `tagger` None: switched on but no tagger is served, so nothing is tagged (reported)."""
    tagger: phototypes.PhotoTagger | None
    store: fb.Store | None
    bands: tuple = DEFAULT_BANDS
    problem: str | None = None
    sink: NightSink | None = field(default=None, init=False)
    tap_error: str | None = field(default=None, init=False)

    @classmethod
    def from_env(cls, tagger, store) -> "NightlyReview | None":
        """The review when MV_NIGHTLY_PHOTO_REVIEW is on, else None. A setting that can't be
        read, or no tagger, is reported by the night instead of stopping it."""
        if not enabled():
            return None
        try:
            bands = bands_from_env()
        except ValueError as e:
            return cls(None, store, DEFAULT_BANDS, problem=str(e))
        if tagger is None:
            return cls(None, store, bands, problem="no photo-type tagger is served "
                                                   "(MV_PHOTO_TAGGER)")
        if store is None:
            return cls(tagger, None, bands, problem="no Human Feedback store")
        return cls(tagger, store, bands)

    def tap(self, conn: sqlite3.Connection, model, run_id: int, log=print):
        """(the model embed_photos should use, the sink for its base vectors or None). The
        first served backbone that can give the tagger's base vectors tags the night's photos;
        one tag a photo. Never raises."""
        if self.problem or self.tagger is None or self.sink is not None:
            return model, None
        try:
            fork, why = with_base_model(model, self.tagger.base)
            if fork is None:
                self.tap_error = why
                log(f"[nightly] photo review: {why}; nothing tagged")
                return model, None
            ensure_schema(conn)
            self.sink = NightSink(conn, self.tagger, f"nightly {run_id}", log=log)
            self.tap_error = None
            return fork, self.sink
        except Exception as e:  # noqa: BLE001
            self.tap_error = f"{type(e).__name__}: {e}"[:300]
            log(f"[nightly] photo review: {self.tap_error}; nothing tagged")
            return model, None

    def queue(self, conn: sqlite3.Connection, run_id: int, log=print) -> dict:
        """Queue tonight's tags, and earlier ones not queued yet. The night's report block."""
        out: dict = {"tagger": self.tagger.version if self.tagger else None,
                     "tagged": self.sink.tagged if self.sink else 0}
        if self.sink and self.sink.errors:
            out["tag_errors"] = len(self.sink.errors)
            out["error"] = self.sink.errors[0]
        if self.problem:
            return {**out, "error": self.problem}
        if self.tap_error:
            out["error"] = self.tap_error
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        counts = queue_tags(conn, self.store, self.bands, f"{SEED_PREFIX}-{run_id}",
                            f"in the nightly update of {stamp} (run {run_id})", log=log)
        return {**out, **counts}


# --- the catch-up: photos of records green since a date -----------------------------------------

def backlog(conn: sqlite3.Connection, store: fb.Store, since: str,
            fingerprint: str | None) -> dict:
    """The photos of every record first green on or after `since` (a day), each once:
    {"photos": [{photo_id, photo_key, record, source_url, host, source, license_class}],
     "records": n, "already_queued", "already_decided", "tagged" (a kept tag under
     `fingerprint`, not queued yet), "to_tag"}."""
    from . import holdouts, sources
    from .photo_review import photo_key_of
    ensure_schema(conn)
    holdouts.ensure_schema(conn)
    rows = conn.execute(
        f"select distinct p.photo_id, p.source, p.source_photo_id, p.source_url, p.host, "
        f"p.license_class, r.observation_id from records r "
        f"join observation_photos op on op.observation_id = r.observation_id "
        f"join photos p on p.photo_id = op.photo_id and p.source = r.source "
        f"where r.validated_on >= ? and r.source in {sources.PHOTO_SOURCES_SQL} "
        f"and {holdouts.not_held_out('r.observation_id')} "
        f"order by p.photo_id", (since[:10],)).fetchall()
    items = dict(store.conn.execute("select subject, status from items where item_type = ?",
                                    (ITEM_TYPE,)).fetchall())
    kept = ({r[0]: r[1] for r in conn.execute(
        "select photo_id, queue_state from photo_type_tags where fingerprint = ?",
        (fingerprint,))} if fingerprint else {})
    seen: set[int] = set()
    records: set[str] = set()
    out = {"already_queued": 0, "already_decided": 0, "tagged": 0, "tagged_done": 0,
           "to_tag": []}
    for pid, source, spid, url, host, lic, record in rows:
        records.add(record)
        if pid in seen:
            continue
        seen.add(pid)
        key = photo_key_of(int(pid), source, spid)
        status = items.get(key)
        if status is not None:
            out["already_queued" if status == "open" else "already_decided"] += 1
            continue
        if pid in kept:                  # tagged before: queued from its kept tag
            out["tagged" if kept[pid] not in FINAL else "tagged_done"] += 1
            continue
        out["to_tag"].append({"photo_id": int(pid), "photo_key": key, "record": record,
                              "source_url": url, "host": host, "source": source,
                              "license_class": lic})
    out["photos"] = len(seen)
    out["records"] = len(records)
    return out


def seconds_per_photo(conn: sqlite3.Connection) -> float:
    """One full tower pass a photo on this machine, from its last embedding runs (embed_runs:
    the nightly's, on the box's CPU), else FALLBACK_SECONDS_PER_PHOTO."""
    try:
        rows = conn.execute("select photos, seconds from embed_runs where photos >= 50 "
                            "order by rowid desc limit 10").fetchall()
    except sqlite3.OperationalError:
        rows = []
    rates = sorted(s / n for n, s in rows if n and s)
    return round(rates[len(rates) // 2], 2) if rates else FALLBACK_SECONDS_PER_PHOTO


def catch_up(conn: sqlite3.Connection, store: fb.Store, tagger: phototypes.PhotoTagger,
             since: str, bands=DEFAULT_BANDS, dry_run: bool = True, limit: int | None = None,
             load_model: Callable[[], object] | None = None,
             fetch: Callable[[list[dict], Path], dict[int, Path]] | None = None,
             work_dir: Path | None = None, busy: Callable[[], bool] | None = None,
             log=print) -> dict:
    """`mv feedback seed-nightly-tags`: tag and queue the photos of every record first green on
    or after `since` that are not queued or decided yet, through store.gate (the seal). The
    dry run changes nothing: it says how many photos it would tag, how many the seal would
    refuse, and the CPU time the tagging would take.

    The seal is judged first, per photo's record, so a photo it would refuse is never fetched
    or read: with the seal on (before the paper's sealed run) a real run queues nothing and
    costs nothing. A kept tag (a night's, or an earlier run's) is queued without the photo.

    `load_model()`: a backbone giving the tagger's base vectors (the frozen base itself, one
    pass a photo); `fetch(rows, work_dir)`: {photo_id: file} for the photos it could get (from
    their source, within the download budgets); every file is deleted after tagging."""
    if store.gate is None:
        store.gate = fb.SealGate.from_manifest(conn)
    gate = store.gate
    found = backlog(conn, store, since, tagger.fingerprint())
    todo = found.pop("to_tag")
    ok, sealed, unknown = [], [], []
    for r in todo:
        verdict = gate.judge(r["record"])
        (ok if verdict == "ok" else sealed if verdict == "sealed" else unknown).append(r)
    every = len(ok)
    if limit is not None:
        ok = ok[:max(0, int(limit))]
    per = seconds_per_photo(conn)
    by_host: dict[str, int] = {}
    for r in ok:
        by_host[r["host"] or "?"] = by_host.get(r["host"] or "?", 0) + 1
    # The kept tags (a night's, or an earlier run's) it would queue now: exact, by the same
    # gate and queue. Photos still to tag: only once tagged can anyone say.
    kept = queue_tags(conn, store, bands, "dry-run", "dry run", dry_run=True, log=lambda s: None)
    plan = {"since": since[:10], "seal": gate.seal, "tagger": tagger.version,
            "bands": list(bands), "records": found["records"], "photos": found["photos"],
            "already_queued": found["already_queued"],
            "already_decided": found["already_decided"],
            "tagged_not_queued": found["tagged"], "tagged_specimen": found["tagged_done"],
            "to_tag": every, "this_run": len(ok),
            "refused_sealed": len(sealed), "refused_unknown": len(unknown),
            "refused_records": len({r["record"] for r in sealed + unknown}),
            "kept_tags_would_queue": kept["queued"],
            "kept_tags_refused": kept["refused_sealed"] + kept["refused_unknown"],
            "downloads_by_host": by_host,
            "seconds_per_photo": per, "cpu_seconds": round(len(ok) * per),
            "cpu_hours": round(len(ok) * per / 3600, 2),
            "cpu_hours_all": round(every * per / 3600, 2)}
    if dry_run:
        plan["dry_run"] = True
        return plan
    if busy is not None and busy():
        raise RuntimeError("a nightly run holds the layer: run the catch-up after it")
    tagged = 0
    started = time.monotonic()
    if ok:
        work_dir = Path(work_dir or (config.DATA_DIR / "nightly-review-photos"))
        work_dir.mkdir(parents=True, exist_ok=True)
        files = (fetch or _fetch_photos(conn))(ok, work_dir)
        try:
            model = load_model() if load_model is not None else None
            fork, why = with_base_model(model, tagger.base) if model is not None else (None, None)
            if fork is None:
                raise RuntimeError(why or "no model to read the photos with")
            tagged = _tag_files(conn, tagger, fork, files, log)
        finally:
            for f in files.values():
                Path(f).unlink(missing_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counts = queue_tags(conn, store, bands, f"{SEED_PREFIX}-catch-up-{stamp}",
                        f"in the catch-up of {stamp} for records green since {since[:10]}",
                        photo_ids=None, log=log)
    return {**plan, "dry_run": False, "tagged": tagged,
            "tag_seconds": round(time.monotonic() - started, 1), **counts}


def _tag_files(conn, tagger, model, files: dict[int, Path], log, batch: int = 8) -> int:
    from .embed import decode
    prepare = getattr(model, "prepare", None) or (lambda im: im)
    ids = sorted(files)
    done = 0
    for i in range(0, len(ids), batch):
        good, imgs = [], []
        for pid in ids[i:i + batch]:
            try:
                imgs.append(prepare(decode(Path(files[pid]).read_bytes())))
                good.append(pid)
            except Exception as e:  # noqa: BLE001 - an unreadable photo is skipped
                log(f"  photo {pid} unreadable: {type(e).__name__}")
        if not good:
            continue
        pair = model.encode_with_base(imgs)
        if pair is None:
            raise RuntimeError("the model gave no base vectors")
        done += save_tags(conn, tagger, good, pair[1], "catch-up")
        if done and done % 400 < batch:
            log(f"  {done:,}/{len(ids):,} photos tagged")
    return done


def _fetch_photos(conn: sqlite3.Connection):
    """Fetch photos from their source at the nightly's size, within the shared download budgets
    (photos.run_downloads: the same hosts, pacing and caps as every download). Each copy is
    recorded under its own store, then its file and record are dropped after tagging."""
    def fetch(rows: list[dict], work_dir: Path) -> dict[int, Path]:
        from . import photos
        from .storage import LocalStore
        size = config.setting("MV_NIGHTLY_PHOTO_SIZE", "large")
        store = LocalStore(work_dir)
        photos.run_downloads(conn, rows, store, size,
                             lambda r, now: photos.save_result(conn, r, size, now,
                                                               store.location))
        got = {int(pid): work_dir / rel for pid, rel in conn.execute(
            "select photo_id, path from photo_copies where store = ?", (store.location,))}
        with conn:
            conn.execute("delete from photo_copies where store = ?", (store.location,))
        return {pid: f for pid, f in got.items() if f.is_file()}
    return fetch
