"""The Human Feedback queues' item lists kept across saves (feedback_api.items_cached).

The photo review's tabs (Summary, record view), its counts, Photo views and Recovered labels all
read a whole queue (Store.items_of, the record names filled from the served manifest). They used
to be read again after every decision: Steve's round of "open a tab, Save page, open the next
tab" paid for all 150,024 photo items each time, 12-14 s on the box (2026-11-20).

Here each queue is read whole once, then kept: after a decision, a batch or a reopen only the
items that changed are read again (Store.changed_since; reopens found by counting open items)
and put in their place, their names filled again. The list is then exactly what a fresh read
would give: the same items, in the same order (by id), with the same fields. What cannot be
traced item by item reads the queue whole again: a seed (new items, or a flag added to an open
item), items added or removed any other way, a decision removed, and any commit from another
connection (a `mv feedback` command or sqlite3 by hand on the box: SQLite's data_version), whose
writes need not be Store's decisions. The names from the manifest are filled again for the whole
list every NAMES_TTL.

A whole read reads every kept photo queue in one pass over the table (Store.latest_rows), the
store's lock held one id range at a time, and keeps each item already in the old list whose row
is unchanged (the same row hash; never one patched since): only new and changed rows are turned
into items again. The first read after a restart builds every item.

So no person waits for a whole read, Warmer reads the queues in the background on a low-priority
thread of its own: at startup once the preload is done, after a seed, a reopen or a system close
(Store.notify), and whenever a cheap look (Store.quick_mark, every POLL_SECONDS) says a list must
be read whole (another process's write). Triggers that come while it runs make one more run.
A request that comes during a whole read waits for it (the lists' lock) instead of starting its own.

Lists are kept for every project (an admin's view); a reviewer's projects, and a read-only
viewer's, are cut from it with Store.visible, as Store.items_of(type, projects) cuts them.
A list handed out is never changed afterwards: a patch makes a new one.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import feedback as fb

# The names filled from the served manifest are read again this often even when nothing in the
# store changed for an item: the manifest can change under a kept list (a release, a record's
# new DNA name), which a full read every Save used to pick up.
NAMES_TTL = 600.0
# The queues people read whole (items_cached): every photo queue (the photo review, Photo views,
# Recovered labels; the Summary's counts of each).
WARM_TYPES = tuple(t.code for t in fb.TYPES.values() if t.granularity == "photo")
# How often the warmer looks for a write it was not told of (another process's).
POLL_SECONDS = 20.0
# Seconds the warmer leaves between id ranges of a whole read, for the requests.
WARM_PAUSE = 0.02


@dataclass
class _List:
    mark: dict
    raw: list[dict]                     # as Store.items_of gives them
    items: list[dict]                   # with `fill` applied
    pos: dict[int, int]                 # item id -> its index
    open: int                           # how many are open
    filled_at: float
    sig: dict[int, int] = field(default_factory=dict)   # id -> its row's hash at the whole read
    patched: frozenset = frozenset()    # ids patched since: their sig no longer says their item


@dataclass
class ItemLists:
    store: fb.Store
    # The names for a queue's items (photo_review.fill_names with the served manifest), or None.
    fill: Callable[[str, list[dict]], list[dict]] | None = None
    names_ttl: float = NAMES_TTL
    clock: Callable[[], float] = time.monotonic
    # The queues read together in a whole read (any other one asked for is added).
    types: tuple = WARM_TYPES
    read_step: int = 20_000             # ids per range of a whole read (Store.latest_rows)
    lists: dict[str, _List] = field(default_factory=dict)
    # One lock for every list: a whole read covers them all, and a request that comes during
    # it waits for it rather than reading too.
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # For the tests and the log: how each list was last brought up to date, and counts.
    last: dict[str, str] = field(default_factory=dict)
    full_reads: int = 0
    rows_built: int = 0
    rows_kept: int = 0

    def of(self, item_type: str, projects=None) -> list[dict]:
        """The queue's items as Store.items_of(item_type, projects) would read them now, with
        the names filled."""
        items = self.current(item_type).items
        if projects is None:
            return items
        return [i for i in items if fb.Store.visible(i, projects)]

    def current(self, item_type: str) -> _List:
        with self._lock:
            if item_type not in self.types:
                self.types = self.types + (item_type,)
            got = self._up_to_date(item_type)
            if got is None:
                self._read_whole(self.types)
                got = self.lists[item_type]
            return got

    def warm(self, pause: float = 0.0) -> dict[str, str]:
        """Every kept queue brought up to date (Warmer): what a request would do, before one
        asks. {queue: how}."""
        with self._lock:
            need = [t for t in self.types if self._up_to_date(t) is None]
            if need:
                self._read_whole(tuple(need), pause=pause)
            return {t: self.last.get(t, "") for t in self.types}

    def needs_whole(self) -> bool:
        """A list missing, or a write since its read that only a whole read follows (cheap)."""
        lists = dict(self.lists)
        if any(t not in lists for t in self.types):
            return True
        q = self.store.quick_mark()
        return any(_quick(lst.mark) != q for lst in lists.values())

    # --- the ways a list is brought up to date --------------------------------------------
    def _up_to_date(self, item_type: str) -> _List | None:
        have = self.lists.get(item_type)
        got = self._patched(item_type, have) if have is not None else None
        if got is not None and self.clock() - got.filled_at > self.names_ttl:
            got = self._refilled(item_type, got)
        if got is not None:
            self.lists[item_type] = got
        return got

    def _filled(self, item_type: str, raw: list[dict]) -> list[dict]:
        return self.fill(item_type, raw) if self.fill is not None else raw

    def _read_whole(self, types: tuple, pause: float = 0.0) -> None:
        mark = self.store.change_mark()       # before the read: anything later is found later
        shape, rows = self.store.latest_rows(types, step=self.read_step, pause=pause)
        self.full_reads += 1
        for t in types:
            old = self.lists.get(t)
            memo: dict = {}
            raw, sig, built = [], {}, 0
            for r in rows[t]:
                i, h = r[0], hash(r)
                sig[i] = h
                if old is not None and old.sig.get(i) == h and i not in old.patched:
                    raw.append(old.raw[old.pos[i]])       # its row is unchanged: so is its item
                else:
                    raw.append(shape.item(r, memo))
                    built += 1
            self.rows_built += built
            self.rows_kept += len(raw) - built
            self.last[t] = "full"
            self.lists[t] = _List(mark, raw, self._filled(t, raw),
                                  {it["id"]: k for k, it in enumerate(raw)},
                                  sum(1 for it in raw if it["status"] == "open"), self.clock(),
                                  sig)

    def _refilled(self, item_type: str, have: _List) -> _List:
        self.last[item_type] = "names"
        return _List(have.mark, have.raw, self._filled(item_type, have.raw), have.pos,
                     have.open, self.clock(), have.sig, have.patched)

    def _patched(self, item_type: str, have: _List) -> _List | None:
        """`have` brought up to date by reading again only what changed since its mark, or
        None when that cannot be done safely (the caller reads the list whole)."""
        store = self.store
        with store._lock:                      # one snapshot of the marks and counts
            began = _begin(store)
            try:
                mark = store.change_mark()
                if mark == have.mark and store.open_count(item_type) == have.open:
                    self.last[item_type] = "same"
                    return have
                changed = store.changed_since(have.mark, mark)
                if changed is None:
                    return None
                fresh = store.items_by_id(changed, item_type) if changed else []
                open_now = store.open_count(item_type)
            finally:
                _end(store, began)
        raw, items = list(have.raw), list(have.items)
        n_open = have.open
        for it in fresh:
            k = have.pos.get(it["id"])
            if k is None:                      # not in the list: it was not read whole
                return None
            n_open += (it["status"] == "open") - (raw[k]["status"] == "open")
            raw[k] = it
        filled = self._filled(item_type, fresh)
        for it in filled:
            items[have.pos[it["id"]]] = it
        touched = {it["id"] for it in fresh}
        if n_open != open_now:
            # A reopen (it writes no decision), or a decision made between the reads above:
            # every item whose status differs from the list's is read again.
            now = store.statuses(item_type)
            if len(now) != len(raw):
                return None
            again = [i for i, s in now.items() if i not in have.pos or raw[have.pos[i]]["status"] != s]
            if any(i not in have.pos for i in again):
                return None
            more = store.items_by_id(again, item_type)
            for it, f in zip(more, self._filled(item_type, more)):
                k = have.pos[it["id"]]
                raw[k], items[k] = it, f
            touched.update(it["id"] for it in more)
            n_open = sum(1 for it in raw if it["status"] == "open")
        self.last[item_type] = "patch"
        return _List(mark, raw, items, have.pos, n_open, have.filled_at, have.sig,
                     have.patched | touched)


def _quick(mark: dict) -> tuple:
    """Store.quick_mark's fields out of a change_mark."""
    return (*mark["items"], *mark["seeds"], mark["data_version"])


def _begin(store: fb.Store) -> bool:
    """A read transaction on the store's connection (the caller holds its lock), so the marks
    and counts read together are one snapshot even with another process writing."""
    if store.conn.in_transaction:
        return False
    store.conn.execute("begin")
    return True


def _end(store: fb.Store, began: bool) -> None:
    if began:
        store.conn.execute("commit")


class Warmer:
    """Keeps ItemLists up to date in the background so no person waits for a whole read. One
    daemon thread, at the lowest CPU priority it can set for itself (Linux), one run at a time.
    `trigger` (Store.notify's events) asks for a run; triggers that arrive during a run make
    one more run after it. Between triggers it looks every `poll` seconds whether a list must
    be read whole (ItemLists.needs_whole). `after` blocks until the server is ready for it (the
    preload done); the first run is the startup warm-up."""

    def __init__(self, lists: ItemLists, *, after: Callable[[], object] | None = None,
                 poll: float = POLL_SECONDS, pause: float = WARM_PAUSE, note=print,
                 nice: int = 19):
        self.lists, self.after, self.poll, self.pause = lists, after, poll, pause
        self.note, self.nice = note, nice
        self._wake = threading.Event()
        self._cond = threading.Condition()
        self._pending: list[str] = ["startup"]
        self._running = False
        self.runs: list[dict] = []          # the last few runs, for the tests and health
        self.thread: threading.Thread | None = None

    def start(self) -> "Warmer":
        self.thread = threading.Thread(target=self._loop, name="feedback lists warm-up",
                                       daemon=True)
        self.thread.start()
        return self

    def trigger(self, reason: str) -> None:
        with self._cond:
            self._pending.append(reason)
        self._wake.set()

    def idle(self, timeout: float | None = None) -> bool:
        """Wait until nothing is running or waiting to run (the tests). False on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._running and not self._pending,
                                       timeout)

    def health(self) -> dict:
        with self._cond:
            last = self.runs[-1] if self.runs else None
        return {"running": self._running, "runs": len(self.runs), "last": last}

    def _loop(self) -> None:
        _lower_priority(self.nice)
        if self.after is not None:
            try:
                self.after()
            except Exception as e:  # noqa: BLE001 - warm up anyway
                self.note(f"[feedback lists] waiting for the preload: {type(e).__name__}: {e}")
        while True:
            with self._cond:
                reasons, self._pending = self._pending, []
                self._running = bool(reasons)
            if not reasons:
                try:
                    if self.lists.needs_whole():
                        reasons = ["a write from another process"]
                        with self._cond:
                            self._running = True
                except Exception as e:  # noqa: BLE001 - look again next time
                    self.note(f"[feedback lists] look: {type(e).__name__}: {e}")
            if reasons:
                self._run(reasons)
                continue                # a trigger during the run: run again before waiting
            self._wake.wait(self.poll)
            self._wake.clear()

    def _run(self, reasons: list[str]) -> None:
        start, full = time.monotonic(), self.lists.full_reads
        try:
            hows = self.lists.warm(pause=self.pause)
            error = None
        except Exception as e:  # noqa: BLE001 - the next request reads it itself
            hows, error = {}, f"{type(e).__name__}: {e}"
        took = round(time.monotonic() - start, 2)
        run = {"reasons": sorted(set(reasons)), "hows": hows, "seconds": took,
               "whole": self.lists.full_reads > full, "error": error}
        if run["whole"] or error:
            self.note(f"[feedback lists] warm-up ({', '.join(run['reasons'])}): "
                      f"{error or ', '.join(f'{t} {h}' for t, h in hows.items())} in {took} s "
                      f"({self.lists.rows_built} rows built, {self.lists.rows_kept} kept so far)")
        with self._cond:
            self.runs = (self.runs + [run])[-20:]
            self._running = bool(self._pending)
            self._cond.notify_all()


def _lower_priority(nice: int) -> None:
    """This thread alone at `nice` (Linux: a thread is its own task); nothing elsewhere."""
    setp = getattr(os, "setpriority", None)
    if setp is None or not hasattr(threading, "get_native_id"):
        return
    try:
        setp(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except OSError:
        pass
