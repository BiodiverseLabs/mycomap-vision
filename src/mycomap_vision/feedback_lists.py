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

Lists are kept for every project (an admin's view); a reviewer's projects, and a read-only
viewer's, are cut from it with Store.visible, as Store.items_of(type, projects) cuts them.
A list handed out is never changed afterwards: a patch makes a new one.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import feedback as fb

# The names filled from the served manifest are read again this often even when nothing in the
# store changed for an item: the manifest can change under a kept list (a release, a record's
# new DNA name), which a full read every Save used to pick up.
NAMES_TTL = 600.0


@dataclass
class _List:
    mark: dict
    raw: list[dict]                     # as Store.items_of gives them
    items: list[dict]                   # with `fill` applied
    pos: dict[int, int]                 # item id -> its index
    open: int                           # how many are open
    filled_at: float
    full_reads: int = 0
    patches: int = 0


@dataclass
class ItemLists:
    store: fb.Store
    # The names for a queue's items (photo_review.fill_names with the served manifest), or None.
    fill: Callable[[str, list[dict]], list[dict]] | None = None
    names_ttl: float = NAMES_TTL
    clock: Callable[[], float] = time.monotonic
    lists: dict[str, _List] = field(default_factory=dict)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)
    _guard: threading.Lock = field(default_factory=threading.Lock)
    # For the tests and the log: how each list was last brought up to date.
    last: dict[str, str] = field(default_factory=dict)

    def of(self, item_type: str, projects=None) -> list[dict]:
        """The queue's items as Store.items_of(item_type, projects) would read them now, with
        the names filled."""
        items = self.current(item_type).items
        if projects is None:
            return items
        return [i for i in items if fb.Store.visible(i, projects)]

    def current(self, item_type: str) -> _List:
        with self._guard:
            lock = self._locks.setdefault(item_type, threading.Lock())
        with lock:
            have = self.lists.get(item_type)
            got = self._patched(item_type, have) if have is not None else None
            if got is None:
                got = self._read_whole(item_type, have)
            elif self.clock() - got.filled_at > self.names_ttl:
                got = self._refilled(item_type, got)
            self.lists[item_type] = got
            return got

    # --- the two ways a list is brought up to date ----------------------------------------
    def _filled(self, item_type: str, raw: list[dict]) -> list[dict]:
        return self.fill(item_type, raw) if self.fill is not None else raw

    def _read_whole(self, item_type: str, old: _List | None) -> _List:
        mark = self.store.change_mark()       # before the read: anything later is found later
        raw = self.store.items_of(item_type)
        self.last[item_type] = "full"
        return _List(mark, raw, self._filled(item_type, raw),
                     {it["id"]: k for k, it in enumerate(raw)},
                     sum(1 for it in raw if it["status"] == "open"), self.clock(),
                     full_reads=(old.full_reads if old else 0) + 1,
                     patches=old.patches if old else 0)

    def _refilled(self, item_type: str, have: _List) -> _List:
        self.last[item_type] = "names"
        return _List(have.mark, have.raw, self._filled(item_type, have.raw), have.pos,
                     have.open, self.clock(), have.full_reads, have.patches)

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
            n_open = sum(1 for it in raw if it["status"] == "open")
        self.last[item_type] = "patch"
        return _List(mark, raw, items, have.pos, n_open, have.filled_at, have.full_reads,
                     have.patches + 1)


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
