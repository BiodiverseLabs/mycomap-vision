"""Research > Excluded photos (Steve, 2026-10-26): every photo people left out of Vision in
Human Feedback, by why it was left out, and "where Vision looked" (looked.py) on any of them,
from the served model (v1), for research on attention. v1 only (Steve, 2026-10-26): the BioCLIP
base model and the side-by-side view are gone, so its tower and vectors are never loaded here;
a request for another model is refused (404), and base maps kept before are never served.

**Which photos.** A photo item of the "Photos that show no specimen" queue whose latest decision
is `exclude_photo`; its category is that decision's value (photo_review.REASONS; a retired code
as its successor, so both old drawing reasons are "Drawings"). Decisions are
written only by a signed-in person (feedback_api), so a tagger's flag alone never puts a photo
here. People photos (Steve, 2026-10-26) are listed only to Human Feedback reviewers.

**What is shown** is what Human Feedback shows a read-only viewer (feedback_public.py): a photo
anyone may see now (Creative Commons, or its owner said yes), by its public URL; every other
photo a placeholder with no photographer, no bytes and no map. Never who decided, or their note.

**A map** explains the photo's served score for its record's DNA-verified name, with the
record itself (and its cross-post twins) left out of the reference, as if the photo were
someone's new find: the record's other photos are DNA-verified specimens of the same name and
would otherwise light up whatever the photo shares with them. Beside it, where the photo alone
puts that name among every species (top RANK_DEPTH, else "not in its top 10"). Made on request
(signed in), one at a time on the run maps' worker, kept here and open to anyone after.

**The paper's seal** (Paper Coordinator, 2026-10-26; PLAN §8): this page is research-facing, so
no map is made or shown for a photo whose record is not judged "ok" by feedback.SealGate (a v1
record, or first green on a day before the seal); the photo still shows. One switch lifts it,
MV_EXCLUDED_LOOKED_UNSEALED=1, set only once the paper's v1-test scoring is committed. Requests
keep no requester and write nothing to the identify log.

**Crops** (Recovered Labels, recovered_labels.py): each kept specimen crop of a slip photo has a
map of its own, made on the photo with the crop's box applied in memory (as embed.py embeds it;
no cropped file is ever kept), keyed `<photo key>#crop<n>`. A crop's map is made only once the
served model has the crop's vector (the nightly embeds it), and explains the record's name with
the record, crops included, left out, like the photo's.

**Preload** (Steve, 2026-10-27, the approved plan .drafts/excluded-maps-preload-plan.md): the
Preloader makes every slip photo's and crop's map ahead of anyone asking, one batch photo at a
time on the same worker, behind every identification and every person's ask, so Identify waits
at most one map. It runs in a time window (MV_EXCLUDED_PRELOAD_WINDOW, UTC, default
PRELOAD_WINDOW; "any": whenever the box is idle), never while the nightly update runs, and off
with MV_EXCLUDED_PRELOAD=0; it is on by default only on the box (a served release), and
MV_EXCLUDED_PRELOAD=1 turns it on elsewhere. A pass makes the embedded crops' maps first, then
the slip photos', and a slip photo whose record's name still has a kept crop waiting for its
vector goes last, so as many maps as can be are made against the reference set with the crops
in it
(MV_EXCLUDED_PRELOAD_CROP_WAIT=0: slips first, in no such order). Slips never wait for the
nightly (2026-10-30): with 20-26k crops waiting and the nightly embedding 3,000 a night, the old
wait (every slip held while any crop waited) made nothing for most of each day for a week. A
target that can't be made (held by the seal, not shown, given up, no map possible) is passed
over once per pass. When nothing is made for STALL_MINUTES, one line says why
("[excluded preload] nothing made for 30 min: ..."), once per state until a map is made. The
seal and the public-photo rule apply as for an ask.

**Stale maps.** Every map carries a stamp of what it was made from (STAMP_KEYS). A map is
served, and counts as preloaded, only while its stamp matches what is served now (is_current):
a new served model, release or method, a record's new name, or a crop's new box makes it
stale, and a stale map is made again on the next ask or preload pass. The calibration run and
the nightly layer's last run are stamped for the record but don't make a map stale: the rank is
read from raw scores, and a night adds well under 1% of the reference photos.

**No map possible** (2026-10-28: the first window's 153 "failed" maps were all this): a photo
whose record's name is not in the reference set, or has no other DNA-verified record to compare
with (looked.NeverMap), can never be mapped from what is served now. That outcome is kept beside
the maps with its reason and the same stamp, shown on the page instead of "map coming", and not
tried again while its stamp holds and for NO_MAP_DAYS (the nightly may add a second record of
the name). Any other failure is logged ("[excluded preload] map failed: <key>: <why>") and tried
on later passes, at most GIVE_UP_AFTER times per photo while the server runs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import feedback as fb
from . import photo_review as pr

FILE = "excluded-looked.sqlite"
VERSION = 1
MODEL = "v1"
REVIEWERS_ONLY = frozenset({"people"})
CATEGORY_LABELS = {
    "micrograph": "Micrographs", "slip": "Field slips and labels",
    "habitat": "Habitat and scenes", "people": "People", "not_fungus": "Not a fungus",
    "dried": "Dried specimens", "spore_print": "Spore prints", "uv": "UV reactions",
    "chemical": "Chemical spot tests", "phylo_tree": "Phylogenetic trees / BLAST",
    "drawing": "Drawings",
    "species_description": "Species descriptions", "blurry": "Blurry or badly exposed",
    "other": "Other"}
assert set(CATEGORY_LABELS) == set(pr.REASON_CODES)
RANK_DEPTH = 10
PAGE_MAX = 96
UNSEAL_SETTING = "MV_EXCLUDED_LOOKED_UNSEALED"
PRELOAD_WINDOW = "07:45-12:45"          # UTC: after the 07:00 nightly, before US mornings
PRELOAD_CATEGORIES = ("slip",)
# What a map is made from; a map is current while these match what is served now.
STAMP_KEYS = ("map_version", "backbone", "method", "release", "name", "box")
NO_MAP_DAYS = 7                  # a kept "no map possible" is tried again after this long
GIVE_UP_AFTER = 3                # failed tries per photo before the preload leaves it alone
STALL_MINUTES = 30               # nothing made this long: the preload logs one line saying why
CROP_MARK = "#crop"

# Each item's latest decision, grouped once (0.6 s on the live store's 79k decisions; the
# per-item max of feedback.LATEST_DECISION took 2.5 s).
EXCLUDED_SQL = (
    "with latest as (select item_id, max(id) as id from decisions group by item_id) "
    "select i.subject, i.record_keys, i.links, d.value, d.id from latest "
    "join decisions d on d.id = latest.id join items i on i.id = d.item_id "
    "where i.item_type = ? and d.decision = 'exclude_photo' order by d.id desc")


@dataclass(frozen=True)
class Excluded:
    photo_key: str
    record_key: str
    category: str


def read_excluded(conn: sqlite3.Connection) -> list[Excluded]:
    """Every photo a person's latest decision leaves out, newest decision first. A photo whose
    record can't be told, or whose reason is not a known category, is left off."""
    out = []
    for subject, record_keys, links, value, _ in conn.execute(EXCLUDED_SQL, (pr.ITEM_TYPE,)):
        rk = pr.record_of_item({"record_keys": json.loads(record_keys or "[]"),
                                "links": json.loads(links or "[]")})
        value = pr.shown_reason(value)
        if rk and value in CATEGORY_LABELS:
            out.append(Excluded(subject, rk, value))
    return out


def visible(rows: list[Excluded], reviewer: bool) -> list[Excluded]:
    """The rows this viewer may list: People only to Human Feedback reviewers."""
    return rows if reviewer else [r for r in rows if r.category not in REVIEWERS_ONLY]


def categories(rows: list[Excluded]) -> list[dict]:
    """{code, label, count} per category with photos, most photos first."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.category] = counts.get(r.category, 0) + 1
    return [{"code": c, "label": CATEGORY_LABELS[c], "count": n}
            for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def select(rows: list[Excluded], category: str, mapped: set[str] | None = None) -> list[Excluded]:
    """The rows of a category ("all": every one), only those in `mapped` when given."""
    if category != "all" and category not in CATEGORY_LABELS:
        raise ValueError(f"no such category: {category}")
    return [r for r in rows if (category == "all" or r.category == category)
            and (mapped is None or r.photo_key in mapped)]


def held_back(gate: "fb.SealGate | None", record_key: str, unsealed: bool) -> bool:
    """Whether the paper's seal holds this record's maps back (module docstring)."""
    return not unsealed and gate is not None and gate.judge(record_key) != "ok"


def rank_of(ident, vector, name: str, hidden) -> int | None:
    """Where the photo alone puts `name` among every species (1-based), its own record left
    out (`hidden`, Identifier.left_out); None when below RANK_DEPTH."""
    import numpy as np
    leave = hidden if hidden is not None and len(hidden) and ident.can_leave_out() else None
    got = ident.identify_vectors_ungated(np.asarray(vector, dtype=np.float32)[None],
                                         top_k=RANK_DEPTH, n_specimens=0, n_photo_specimens=0,
                                         photo_lookup=lambda ids: {}, hidden=leave)
    names = [c["name"] for c in got["ranks"]["species"]]
    return names.index(name) + 1 if name in names else None


def crop_key(photo_key: str, n: int) -> str:
    """The key a crop's map is kept under: `inat:123#crop2`."""
    return f"{photo_key}{CROP_MARK}{int(n)}"


def is_current(doc: dict | None, now: dict) -> bool:
    """Whether a kept map was made from what is served now (`now`: STAMP_KEYS' values). A map
    kept before stamps were written is never current."""
    stamp = (doc or {}).get("stamp")
    return isinstance(stamp, dict) and all(stamp.get(k) == now.get(k) for k in STAMP_KEYS)


def no_map_current(doc: dict | None, now: dict, when: datetime) -> bool:
    """Whether a kept "no map possible" still holds: made from what is served now (is_current)
    and less than NO_MAP_DAYS ago."""
    if not is_current(doc, now):
        return False
    try:
        made = datetime.strptime(doc["made_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return False
    return (when - made).total_seconds() < NO_MAP_DAYS * 86400


def preload_on(setting: str | None, on_box: bool) -> bool:
    """MV_EXCLUDED_PRELOAD: "1" on, "0" off; unset, on only on the box (a served release)."""
    return setting == "1" or (setting != "0" and on_box)


def parse_window(value: str | None) -> tuple[int, int] | None:
    """MV_EXCLUDED_PRELOAD_WINDOW: "HH:MM-HH:MM" in UTC (it may run past midnight) as minutes
    of the day, or None for "any" (whenever the box is idle). Unset: PRELOAD_WINDOW."""
    value = (value or PRELOAD_WINDOW).strip().lower()
    if value in ("any", "always", "idle"):
        return None
    try:
        a, b = value.split("-")
        start, end = (int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in (a, b))
    except (ValueError, IndexError):
        raise ValueError(f"the preload window is HH:MM-HH:MM in UTC, or any: {value!r}") from None
    if not (0 <= start < 1440 and 0 <= end < 1440) or start == end:
        raise ValueError(f"the preload window is HH:MM-HH:MM in UTC, or any: {value!r}")
    return start, end


def in_window(window: tuple[int, int] | None, when: datetime) -> bool:
    if window is None:
        return True
    minute = when.astimezone(timezone.utc).hour * 60 + when.astimezone(timezone.utc).minute
    start, end = window
    return start <= minute < end if start < end else (minute >= start or minute < end)


def record_url(record_key: str) -> str:
    if record_key.startswith("mo:"):
        return f"https://mushroomobserver.org/observations/{record_key[3:]}"
    return f"https://www.inaturalist.org/observations/{record_key}"


class Maps:
    """The kept maps: one row per photo and model, insert or replace (a newer VERSION), never
    who asked. Only MODEL's are made or read now; rows of the old base model stay untouched."""

    SCHEMA = ("create table if not exists maps (photo_key text not null, model text not null, "
              "version integer not null, doc text not null, made_at text not null, "
              "primary key (photo_key, model))")

    # A photo (or crop) no map can be made for from what is served now (looked.NeverMap), with
    # why and the stamp it was tried under (module docstring, "No map possible").
    NO_MAP_SCHEMA = ("create table if not exists no_maps (photo_key text not null, model text "
                     "not null, reason text not null, stamp text not null, made_at text not null, "
                     "primary key (photo_key, model))")

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.execute(self.SCHEMA)
            self.conn.execute(self.NO_MAP_SCHEMA)
        self._mapped: dict[str, set[str]] = {}         # model -> keys, until a map is kept

    def get(self, photo_key: str, model: str) -> dict | None:
        with self.lock:
            r = self.conn.execute("select doc from maps where photo_key = ? and model = ? "
                                  "and version = ?", (photo_key, model, VERSION)).fetchone()
        return json.loads(r[0]) if r else None

    def keep(self, photo_key: str, model: str, doc: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute("insert or replace into maps values (?,?,?,?,?)",
                              (photo_key, model, VERSION, json.dumps(doc, separators=(",", ":")),
                               doc["made_at"]))
            self.conn.execute("delete from no_maps where photo_key = ? and model = ?",
                              (photo_key, model))
            self._mapped.clear()

    def keep_no_map(self, photo_key: str, model: str, reason: str, stamp: dict,
                    made_at: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("insert or replace into no_maps values (?,?,?,?,?)",
                              (photo_key, model, reason, json.dumps(stamp, separators=(",", ":")),
                               made_at))

    def no_map(self, photo_key: str, model: str = MODEL) -> dict | None:
        """{"reason", "stamp", "made_at"} when no map could be made for this key, else None."""
        with self.lock:
            r = self.conn.execute("select reason, stamp, made_at from no_maps where photo_key = ? "
                                  "and model = ?", (photo_key, model)).fetchone()
        return {"reason": r[0], "stamp": json.loads(r[1]), "made_at": r[2]} if r else None

    def no_maps(self, model: str = MODEL) -> dict[str, dict]:
        """Every kept "no map possible", by key (a preload pass reads them once)."""
        with self.lock:
            rows = self.conn.execute("select photo_key, reason, stamp, made_at from no_maps "
                                     "where model = ?", (model,)).fetchall()
        return {k: {"reason": r, "stamp": json.loads(st), "made_at": m} for k, r, st, m in rows}

    def mapped(self, model: str = MODEL) -> set[str]:
        """The photo keys with a kept map from `model` (read once, again after a keep)."""
        with self.lock:
            if model not in self._mapped:
                self._mapped[model] = {k for (k,) in self.conn.execute(
                    "select photo_key from maps where model = ? and version = ?",
                    (model, VERSION))}
            return self._mapped[model]

    def stamps(self, model: str = MODEL) -> dict[str, dict | None]:
        """Every kept map's stamp, by key (a preload pass reads them once)."""
        with self.lock:
            rows = self.conn.execute("select photo_key, doc from maps where model = ? and "
                                     "version = ?", (model, VERSION)).fetchall()
        return {k: json.loads(d).get("stamp") for k, d in rows}


class Listing:
    """read_excluded, read again only when the store's decisions change (checked at most every
    `every` seconds), on a read-only connection of its own: the store is in WAL mode, so the
    read never holds up a reviewer's save."""

    def __init__(self, store: fb.Store, every: float = 30.0, clock=time.monotonic):
        self.store, self.every, self.clock = store, every, clock
        self._at, self._version, self._rows = -1e9, None, []
        self._by_key: dict[str, Excluded] = {}
        self._conn: sqlite3.Connection | None = None
        self.lock = threading.Lock()

    def rows(self) -> list[Excluded]:
        with self.lock:
            if self.clock() - self._at >= self.every:
                self._at = self.clock()
                if self._conn is None:
                    self._conn = sqlite3.connect(
                        f"file:{Path(self.store.path).as_posix()}?mode=ro", uri=True,
                        check_same_thread=False)
                version = self._conn.execute("select max(id), count(*) from decisions").fetchone()
                if version != self._version:
                    self._rows = read_excluded(self._conn)
                    self._by_key = {r.photo_key: r for r in self._rows}
                    self._version = version
            return self._rows

    def get(self, photo_key: str) -> Excluded | None:
        """One photo's row (read as rows() reads them), or None."""
        self.rows()
        return self._by_key.get(photo_key)


# --- the preload -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """One map the preload makes: a slip photo's (crop 0) or one of its crops'."""
    hit: Excluded
    crop: int = 0
    box: tuple | None = None

    @property
    def key(self) -> str:
        return crop_key(self.hit.photo_key, self.crop) if self.crop else self.hit.photo_key


@dataclass
class Pass:
    """One pass over every target: what was already current, what is left in the order it is
    made (`todo`), what was skipped, and how many slips go last for their name's crops."""
    todo: deque
    total: dict[str, int]
    done: dict[str, int]
    skipped: dict[str, int] = field(default_factory=dict)
    crops_waiting: int = 0
    at: float = 0.0
    no_map: dict[str, int] = field(default_factory=lambda: {"slips": 0, "crops": 0})
    slips_last: int = 0


class Preloader:
    """Makes the targets' maps one batch photo at a time (module docstring). The server gives
    it its reads and its worker:

    - `targets()` -> (slip photos, embedded crops as Targets, kept crops still waiting for
      their vector: a count, or {record name: count} so a slip of such a name goes last);
    - `stamp_now(target)` -> the stamp a map made now would carry (None: the model is not
      ready, nothing is made); `stamps()` every kept map's stamp by key, `stamp_of(key)` one;
      `no_maps()` every kept "no map possible" by key (Maps.no_maps);
    - `skip(target)` -> why it can't be mapped now ("held", "not shown"), or None;
    - `job(target, on_done)` -> the looked.Job that makes and keeps it, calling
      on_done(ok, why, never): `never` when no map is possible (looked.NeverMap);
    - `nightly_running()`, and the worker (`submit`, `batch_pending`).

    step() is one tick of its thread: it submits at most one photo, and only while the worker
    has no batch photo queued or in hand. A pass's targets are queued once, in the order they
    are made, so a target passed over costs one look per pass and none per tick."""

    def __init__(self, *, enabled: bool, window: tuple[int, int] | None, crop_wait: bool,
                 worker, targets: Callable, stamp_now: Callable, stamps: Callable,
                 stamp_of: Callable, skip: Callable, job: Callable, nightly_running: Callable[[], bool],
                 no_maps: Callable[[], dict] = dict, rescan_seconds: float = 3600.0,
                 stall_minutes: float = STALL_MINUTES, clock=time.monotonic,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc), note=print):
        self.enabled, self.window, self.crop_wait = enabled, window, crop_wait
        self.worker, self.targets, self.stamp_now, self.stamps = worker, targets, stamp_now, stamps
        self.stamp_of, self.no_maps = stamp_of, no_maps
        self.skip_reason, self.job, self.nightly_running = skip, job, nightly_running
        self.rescan_seconds, self.clock, self.now, self.note = rescan_seconds, clock, now, note
        self.state = "off" if not enabled else "starting"
        self.pass_: Pass | None = None
        self.last_pass_seconds: float | None = None
        self._retry_at = -1e9                     # after an error, the next try (clock)
        self.made = self.failed = self.never = 0
        self.fails: dict[str, int] = {}           # failed tries by key, while the server runs
        self.last_map_at: str | None = None
        self.lock = threading.Lock()
        # The stall line: nothing made since `_progress_at`, in `state` since `_state_since`.
        self.stall_seconds = stall_minutes * 60.0
        self._progress_at = self._state_since = self.clock()
        self._state_was: str | None = None
        self._said: set[str] = set()

    def scan(self) -> Pass:
        """A new pass: every target whose kept map is not current, slip photos first. Logged
        at its start and end."""
        started = self.clock()
        self.note("[excluded preload] pass starting")
        slips, crops, waiting = self.targets()
        names = set(waiting) if isinstance(waiting, Mapping) else None
        waiting = sum(waiting.values()) if isinstance(waiting, Mapping) else int(waiting or 0)
        kept, never, when = self.stamps(), self.no_maps(), self.now()
        total = {"slips": len(slips), "crops": len(crops)}
        done = {"slips": 0, "crops": 0}
        no_map = {"slips": 0, "crops": 0}
        left: dict[str, list[Target]] = {"slips": [], "crops": [], "last": []}
        for kind, group in (("slips", slips), ("crops", crops)):
            for t in group:
                now = self.stamp_now(t)
                if now is not None and is_current({"stamp": kept.get(t.key)}, now):
                    done[kind] += 1
                elif now is not None and no_map_current(never.get(t.key), now, when):
                    no_map[kind] += 1
                elif (kind == "slips" and self.crop_wait and waiting
                      and (names is None or (now or {}).get("name") in names)):
                    left["last"].append(t)       # its name's crops still wait for the nightly
                else:
                    left[kind].append(t)
        todo = deque(left["crops"] + left["slips"] + left["last"] if self.crop_wait
                     else left["slips"] + left["crops"])
        self.last_pass_seconds = round(self.clock() - started, 1)
        self.note(f"[excluded preload] pass ready in {self.last_pass_seconds} s: "
                  f"{total['slips']} slips ({done['slips']} current), {total['crops']} crops "
                  f"({done['crops']} current), {sum(no_map.values())} no map possible, "
                  f"{len(todo)} to make, {waiting} crops waiting for their vector"
                  + (f" ({len(left['last'])} slips of their names go last)" if left["last"]
                     else ""))
        return Pass(todo, total, done, {}, waiting, self.clock(), no_map, len(left["last"]))

    def step(self) -> str:
        """One tick: the state it is in afterwards. An error is logged, shown in the health
        ("error: ..."), and tried again a minute later."""
        with self.lock:
            if self.clock() < self._retry_at:
                return self.state
            try:
                self.state = self._step()
            except Exception as e:  # noqa: BLE001 - say so, never stop the server
                self.note(f"[excluded preload] {type(e).__name__}: {e}")
                self.state = f"error: {type(e).__name__}: {e}"[:200]
                self._retry_at = self.clock() + 60.0
            self._watch()
            return self.state

    def _watch(self) -> None:
        """The stall line: when nothing has been made for STALL_MINUTES and the state has held
        that long, say why, once per state until a map is made."""
        t = self.clock()
        if self.state != self._state_was:
            self._state_was, self._state_since = self.state, t
        if (self.state == "off" or self.state in self._said
                or t - max(self._progress_at, self._state_since) < self.stall_seconds):
            return
        self._said.add(self.state)
        self.note(f"[excluded preload] nothing made for {int((t - self._progress_at) // 60)} "
                  f"min: {self.why()}")

    def why(self) -> str:
        """Why nothing is being made now, with what the pass has left."""
        reason = {
            "outside its window": f"outside its window ({self.window_text()})",
            "waiting for the nightly update": "the nightly update is running",
            "making a map": "the map worker has had a preload photo in hand all that time",
            "waiting for the model": "the served model is not ready",
            "done": "nothing left to make",
        }.get(self.state, self.state)
        p = self.pass_
        if p is None:
            return reason + "; no pass yet"
        return (f"{reason}; {len(p.todo)} to make ({p.slips_last} slips last), "
                f"{p.done['slips'] + p.done['crops']} current, "
                f"{p.no_map['slips'] + p.no_map['crops']} no map possible, "
                f"skipped {dict(sorted(p.skipped.items()))}, "
                f"{p.crops_waiting} crops waiting for their vector")

    def window_text(self) -> str:
        return "any" if self.window is None else "%02d:%02d-%02d:%02d UTC" % (
            self.window[0] // 60, self.window[0] % 60, self.window[1] // 60, self.window[1] % 60)

    def _step(self) -> str:
        if not self.enabled:
            return "off"
        if not in_window(self.window, self.now()):
            return "outside its window"
        if self.nightly_running():
            return "waiting for the nightly update"
        if self.worker.batch_pending():
            return "making a map"
        # A fresh pass every rescan_seconds, done or not: new exclusions, crops the night
        # embedded (which lets slips go), maps a new release or rename made stale.
        if self.pass_ is None or self.clock() - self.pass_.at >= self.rescan_seconds:
            self.pass_ = self.scan()
        p = self.pass_
        while p.todo:
            t = p.todo.popleft()
            kind = "crops" if t.crop else "slips"
            why = self.skip_reason(t)
            if why is None and self.fails.get(t.key, 0) >= GIVE_UP_AFTER:
                why = "given up"
            if why is not None:
                p.skipped[why] = p.skipped.get(why, 0) + 1
                continue
            stamp = self.stamp_now(t)
            if stamp is None:
                p.todo.appendleft(t)
                return "waiting for the model"
            if is_current({"stamp": self.stamp_of(t.key)}, stamp):    # asked for meanwhile
                p.done[kind] += 1
                continue
            try:
                self.worker.submit(self.job(
                    t, lambda ok, why=None, never=False, kind=kind, key=t.key:
                    self._made(kind, key, ok, why, never)))
            except Exception as e:  # noqa: BLE001 - this photo alone; the pass goes on
                self._made(kind, t.key, False, f"{type(e).__name__}: {e}")
            return "making a map"
        return "done"

    def _made(self, kind: str, key: str, ok: bool, why: str | None = None,
              never: bool = False) -> None:
        """A job's outcome: made; no map possible (kept by the job, counted, not retried); or
        failed (logged with why, tried again on a later pass up to GIVE_UP_AFTER times)."""
        if ok or never:                           # an outcome: the stall line may say so again
            self._progress_at = self.clock()
            self._said.clear()
        if ok:
            self.made += 1
            if self.pass_:
                self.pass_.done[kind] += 1
            self.last_map_at = self.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        elif never:
            self.never += 1
            if self.pass_:
                self.pass_.no_map[kind] += 1
            self.note(f"[excluded preload] no map possible: {key}: {why}")
        else:
            self.failed += 1
            n = self.fails[key] = self.fails.get(key, 0) + 1
            if self.pass_:
                self.pass_.skipped["failed"] = self.pass_.skipped.get("failed", 0) + 1
            self.note(f"[excluded preload] map failed: {key}: {why}"
                      + (f" (try {n}; given up until the server restarts)"
                         if n >= GIVE_UP_AFTER else f" (try {n})"))

    def health(self) -> dict:
        """For /api/health: on or off, its window, what it is doing, and per kind how many
        maps are current, left and skipped, with the last map's time."""
        p = self.pass_
        out = {"enabled": self.enabled, "window": self.window_text(),
               # "doing", not "state": deploy.sh reads the health's two "state" keys.
               "doing": self.state, "made": self.made, "failed": self.failed,
               "no_map_found": self.never,
               "given_up": sum(1 for n in self.fails.values() if n >= GIVE_UP_AFTER),
               "last_map_at": self.last_map_at, "last_pass_seconds": self.last_pass_seconds,
               "last_map_seconds": (round(self.worker.last_seconds, 2)
                                    if getattr(self.worker, "last_seconds", None) else None)}
        if p is not None:
            left = {"slips": sum(1 for t in p.todo if not t.crop),
                    "crops": sum(1 for t in p.todo if t.crop)}
            out.update({k: {"total": p.total[k], "done": p.done[k], "remaining": left[k]}
                        for k in ("slips", "crops")})
            out.update(skipped=dict(sorted(p.skipped.items())), crops_waiting=p.crops_waiting,
                       slips_last=p.slips_last, no_map_possible=dict(p.no_map))
        return out
