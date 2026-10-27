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
MV_EXCLUDED_PRELOAD=1 turns it on elsewhere. Slip photos wait until no kept crop is still waiting for its vector
(MV_EXCLUDED_PRELOAD_CROP_WAIT=0 skips that wait), so their maps are made against the reference
set with the crops in it. The seal and the public-photo rule apply as for an ask.

**Stale maps.** Every map carries a stamp of what it was made from (STAMP_KEYS). A map is
served, and counts as preloaded, only while its stamp matches what is served now (is_current):
a new served model, release or method, a record's new name, or a crop's new box makes it
stale, and a stale map is made again on the next ask or preload pass. The calibration run and
the nightly layer's last run are stamped for the record but don't make a map stale: the rank is
read from raw scores, and a night adds well under 1% of the reference photos.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
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

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.execute(self.SCHEMA)
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
            self._mapped.clear()

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
    """One pass over every target: what was already current, what is left, what was skipped."""
    todo: list[Target]
    total: dict[str, int]
    done: dict[str, int]
    skipped: dict[str, int] = field(default_factory=dict)
    crops_waiting: int = 0
    at: float = 0.0


class Preloader:
    """Makes the targets' maps one batch photo at a time (module docstring). The server gives
    it its reads and its worker:

    - `targets()` -> (slip photos, embedded crops as Targets, kept crops still waiting for
      their vector);
    - `stamp_now(target)` -> the stamp a map made now would carry (None: the model is not
      ready, nothing is made); `stamps()` every kept map's stamp by key, `stamp_of(key)` one;
    - `skip(target)` -> why it can't be mapped now ("held", "not shown"), or None;
    - `job(target, on_done)` -> the looked.Job that makes and keeps it, calling on_done(ok);
    - `nightly_running()`, and the worker (`submit`, `batch_pending`).

    step() is one tick of its thread: it submits at most one photo, and only while the worker
    has no batch photo queued or in hand."""

    def __init__(self, *, enabled: bool, window: tuple[int, int] | None, crop_wait: bool,
                 worker, targets: Callable, stamp_now: Callable, stamps: Callable,
                 stamp_of: Callable, skip: Callable, job: Callable, nightly_running: Callable[[], bool],
                 rescan_seconds: float = 3600.0, clock=time.monotonic,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc), note=print):
        self.enabled, self.window, self.crop_wait = enabled, window, crop_wait
        self.worker, self.targets, self.stamp_now, self.stamps = worker, targets, stamp_now, stamps
        self.stamp_of = stamp_of
        self.skip_reason, self.job, self.nightly_running = skip, job, nightly_running
        self.rescan_seconds, self.clock, self.now, self.note = rescan_seconds, clock, now, note
        self.state = "off" if not enabled else "starting"
        self.pass_: Pass | None = None
        self.made = self.failed = 0
        self.last_map_at: str | None = None
        self.lock = threading.Lock()

    def scan(self) -> Pass:
        """A new pass: every target whose kept map is not current, slip photos first."""
        slips, crops, waiting = self.targets()
        kept = self.stamps()
        total = {"slips": len(slips), "crops": len(crops)}
        done = {"slips": 0, "crops": 0}
        todo: list[Target] = []
        for kind, group in (("slips", slips), ("crops", crops)):
            for t in group:
                now = self.stamp_now(t)
                if now is not None and is_current({"stamp": kept.get(t.key)}, now):
                    done[kind] += 1
                else:
                    todo.append(t)
        return Pass(todo, total, done, {}, waiting, self.clock())

    def _due(self, t: Target) -> bool:
        return not (t.crop == 0 and self.crop_wait and self.pass_ and self.pass_.crops_waiting)

    def step(self) -> str:
        """One tick: the state it is in afterwards."""
        with self.lock:
            self.state = self._step()
            return self.state

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
        while True:
            t = next((t for t in p.todo if self._due(t)), None)
            if t is None:
                if p.todo:                       # only slip photos left, crops still embedding
                    return "waiting for crops to be embedded"
                return "done"
            p.todo.remove(t)
            kind = "crops" if t.crop else "slips"
            why = self.skip_reason(t)
            if why is not None:
                p.skipped[why] = p.skipped.get(why, 0) + 1
                continue
            stamp = self.stamp_now(t)
            if stamp is None:
                p.todo.insert(0, t)
                return "waiting for the model"
            if is_current({"stamp": self.stamp_of(t.key)}, stamp):    # asked for meanwhile
                p.done[kind] += 1
                continue
            try:
                self.worker.submit(self.job(t, lambda ok, kind=kind: self._made(kind, ok)))
            except Exception as e:  # noqa: BLE001 - this photo alone; the pass goes on
                self.note(f"[preload] {t.key}: {type(e).__name__}: {e}")
                self._made(kind, False)
            return "making a map"

    def _made(self, kind: str, ok: bool) -> None:
        if ok:
            self.made += 1
            if self.pass_:
                self.pass_.done[kind] += 1
            self.last_map_at = self.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            self.failed += 1
            if self.pass_:
                self.pass_.skipped["failed"] = self.pass_.skipped.get("failed", 0) + 1

    def health(self) -> dict:
        """For /api/health: on or off, its window, what it is doing, and per kind how many
        maps are current, left and skipped, with the last map's time."""
        p = self.pass_
        out = {"enabled": self.enabled,
               "window": "any" if self.window is None else "%02d:%02d-%02d:%02d UTC" % (
                   self.window[0] // 60, self.window[0] % 60, self.window[1] // 60,
                   self.window[1] % 60),
               # "doing", not "state": deploy.sh reads the health's two "state" keys.
               "doing": self.state, "made": self.made, "failed": self.failed,
               "last_map_at": self.last_map_at,
               "last_map_seconds": (round(self.worker.last_seconds, 2)
                                    if getattr(self.worker, "last_seconds", None) else None)}
        if p is not None:
            left = {"slips": sum(1 for t in p.todo if not t.crop),
                    "crops": sum(1 for t in p.todo if t.crop)}
            out.update({k: {"total": p.total[k], "done": p.done[k], "remaining": left[k]}
                        for k in ("slips", "crops")})
            out.update(skipped=dict(sorted(p.skipped.items())), crops_waiting=p.crops_waiting)
        return out
