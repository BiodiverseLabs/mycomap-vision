"""Often confused with: what DNA said when people called a fungus by a candidate's name.

mycomap.org builds "commonly misidentified as" every night from every iNaturalist
identification on its DNA-checked iNat records (public page mycomap.org/misid). A name
*prompts* when .org is 80% sure DNA says another species on at least a quarter of the
finds it was called on (a Jeffreys half-count, no minimum number of finds). .org decides
that rule; Vision only reads the answer:

    GET <MV_ORG_BASE_URL>/api/misidentifications/all     public, no key, ~1 MB

once a night (nightly.run_once, its own step), and keeps the last good answer gzipped
beside the served manifest (CACHE). A failed read, a 503 (.org has not built it yet) or
an empty or malformed answer keeps the previous cache; the night's report says so and
counts the failures in a row.

Each species candidate whose name prompts gets `confused_with` (docs/api.md §4):

    {"names": [{"name": "Gymnopilus subspectabilis", "finds": 8}, ...],   # top 3, "w" only
     "called": 86, "wrong": 29,
     "note": "Often confused with G. subspectabilis · 9% of DNA-checked finds"}

and nothing otherwise (no cache, the name is unknown to .org or doesn't prompt).

Matching a candidate to a row, first hit wins:
    1. .org's own key for the name (name_key: lower case, single spaces, quotes made
       straight, " var. / subsp. / ssp. / f. " dropped), against each row's own name;
    2. the same key against the rows' aliases (other spellings, iNat and MycoMap synonyms);
    3. Vision's spelling key (names.variant_key: the rule Vision's labels use, so
       'IN7' / 'IN07' and quote glyphs fold) against rows' names, then their aliases.
At each step the row called most often wins, as on .org's /api/misidentifications/name.
A row that matches but doesn't prompt stops the search: the name has its own record.

Memory: only prompting rows are kept as rows; every other name is one key in the
lookup, so a not-prompting name can't fall through to a synonym's numbers.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

from . import config, names

ALL_PATH = "/api/misidentifications/all"
CACHE = "misidentifications.json.gz"
STATE = "misidentifications.state.json"
TIMEOUT = 120                           # as the nightly's other reads of mycomap.org
RETRY_PAUSES = (30.0, 120.0)            # a failed read is asked again after these pauses
TOP = 3                                 # names listed in confused_with
FEW = 10                                # under this many finds the note gives counts
COLUMNS = ("name", "called", "wrong", "broaderSpecies", "people", "confidence", "prompts",
           "dnaSaidInstead")            # .org's first eight columns, in order (aliases ninth)


class Unavailable(RuntimeError):
    """mycomap.org's answer was not taken; the previous cache stays."""


# ---------------------------------------------------------------------------
# Names

_ORG_QUOTES = re.compile("[‘’“”\"]")
_ORG_SPACES = re.compile("[" + names.JS_SPACE + "]+")
_ORG_RANKS = re.compile(r" (var|subsp|ssp|f|fo|forma)\. ")


def org_key(name: str | None) -> str:
    """mycomap.org's name key (services/misidentifications.ts nameKey), rule for rule."""
    t = _ORG_SPACES.sub(" ", _ORG_QUOTES.sub("'", name or "")).strip(names.JS_SPACE).lower()
    return _ORG_RANKS.sub(" ", t)


def spelling_key(name: str | None) -> str:
    """Vision's key: spellings of one label share it (names.variant_key). Not through its
    cache: that keeps every name it has seen for good, and .org's 50,000 names and aliases
    a night would hold ~13 MB there and grow with each build."""
    return names._parse.__wrapped__(name or "").key


def same_genus(a: str, b: str) -> bool:
    return bool(a and b) and a.split(" ", 1)[0].casefold() == b.split(" ", 1)[0].casefold()


def short_name(name: str, candidate: str) -> str:
    """'G. subspectabilis' when `name` is in the candidate's genus, else the whole name."""
    genus, _, rest = name.partition(" ")
    return f"{genus[:1]}. {rest}" if rest and same_genus(name, candidate) else name


def note_text(candidate: str, called: int, top_name: str, top_finds: int) -> str:
    """The line under the candidate: counts while there are fewer than ten finds, a
    percentage of all the finds called this name from ten on ("under 1%" rather than 0%)."""
    if called < FEW:
        amount = f"{top_finds} of {called} DNA-checked find{'' if called == 1 else 's'}"
    else:
        pct = int(100 * top_finds / called + 0.5)           # JavaScript's Math.round, as .org
        amount = (f"{pct}% of DNA-checked finds" if pct >= 1
                  else "under 1% of DNA-checked finds")
    return f"Often confused with {short_name(top_name, candidate)} · {amount}"


# ---------------------------------------------------------------------------
# The table

@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompting name as .org counts it."""
    name: str
    called: int
    wrong: int
    instead: tuple[tuple[str, int], ...]     # what DNA said ("w" answers), most finds first


@dataclass(frozen=True)
class Table:
    rows: tuple[Prompt, ...]                 # prompting names only
    own: dict[str, int]                      # org key of a row's name -> row, -1: not prompting
    alias: dict[str, int]                    # org key of an alias -> row, -1: not prompting
    spelling: dict[str, int]                 # Vision's key of a name or alias -> row or -1
    build: dict                              # .org's build: id, builtAt, counts
    fetched_at: str | None
    total: int                               # names in .org's answer

    @classmethod
    def from_answer(cls, payload: dict, fetched_at: str | None = None) -> "Table":
        rows_in = check_answer(payload)["names"]
        # Most-called first, so the first row to claim a key keeps it (.org's tie rule).
        order = sorted(range(len(rows_in)), key=lambda i: -rows_in[i][1])
        rows: list[Prompt] = []
        own: dict[str, int] = {}
        alias: dict[str, int] = {}
        spelling: dict[str, int] = {}
        later: list[tuple[str, int]] = []       # aliases, for the spelling step after names
        intern = _Interner()
        for i in order:
            r = rows_in[i]
            at = -1
            if r[6]:
                instead = sorted(((intern(a[0]), int(a[1])) for a in r[7] if a[3] == "w"),
                                 key=lambda a: (-a[1], a[0]))
                at = len(rows)
                rows.append(Prompt(intern(r[0]), int(r[1]), int(r[2]), tuple(instead)))
            key = intern(org_key(r[0]))
            own.setdefault(key, at)
            _add_spelling(spelling, r[0], key, at, intern)
            for a in (r[8] if len(r) > 8 else ()):
                key = intern(org_key(a))
                alias.setdefault(key, at)
                later.append((a, key, at))
        for a, key, at in later:
            _add_spelling(spelling, a, key, at, intern)
        return cls(tuple(rows), own, alias, spelling, dict(payload.get("build") or {}),
                   fetched_at, len(rows_in))

    def lookup(self, name: str) -> Prompt | None:
        """The prompting row for a candidate's name, or None (see the module's steps)."""
        key, spelled = org_key(name), spelling_key(name)
        # A spelling key that equals a row's own key is held only in own / alias.
        for found in (self.own.get(key), self.alias.get(key), self.own.get(spelled),
                      self.spelling.get(spelled), self.alias.get(spelled)):
            if found is not None:
                return self.rows[found] if found >= 0 else None
        return None

    def confused_with(self, name: str) -> dict | None:
        """The candidate's `confused_with` field, or None when it doesn't prompt."""
        row = self.lookup(name)
        if row is None:
            return None
        # An answer that is the candidate itself in another spelling is not a confusion.
        own = {org_key(name), spelling_key(name), org_key(row.name), spelling_key(row.name)}
        instead = [(n, f) for n, f in row.instead
                   if org_key(n) not in own and spelling_key(n) not in own][:TOP]
        if not instead:
            return None
        return {"names": [{"name": n, "finds": f} for n, f in instead],
                "called": row.called, "wrong": row.wrong,
                "note": note_text(name, row.called, *instead[0])}

    def summary(self) -> dict:
        return {"build_id": self.build.get("id"), "built_at": self.build.get("builtAt"),
                "fetched_at": self.fetched_at, "names": self.total, "prompting": len(self.rows)}


def _add_spelling(spelling: dict, name: str, key: str, at: int, intern) -> None:
    """Only spellings whose key differs from .org's: where they agree, the name or alias
    step already found the row (most names: 'gymnopilus luteus' either way)."""
    sk = spelling_key(name)
    if sk != key:
        spelling.setdefault(intern(sk), at)


class _Interner:
    """One str object per distinct text (aliases repeat names across rows)."""

    def __init__(self):
        self.seen: dict[str, str] = {}

    def __call__(self, s: str) -> str:
        return self.seen.setdefault(s, s)


def check_answer(payload) -> dict:
    """.org's /all answer, or Unavailable saying what is wrong with it. Any bad row refuses
    the whole answer: a half-read table would quietly drop prompts."""
    if not isinstance(payload, dict):
        raise Unavailable("the answer is not a JSON object")
    columns = payload.get("columns")
    if (not isinstance(columns, list) or len(columns) < len(COLUMNS)
            or any(not isinstance(c, str) or not c.startswith(want)
                   for c, want in zip(columns, COLUMNS))):
        raise Unavailable("the answer's columns are not the ones Vision reads")
    rows = payload.get("names")
    if not isinstance(rows, list) or not rows:
        raise Unavailable("mycomap.org listed no names")
    for n, r in enumerate(rows):
        if not _good_row(r):
            raise Unavailable(f"row {n + 1} of {len(rows):,} is malformed")
    return payload


def _count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _good_row(r) -> bool:
    if not isinstance(r, list) or len(r) < len(COLUMNS):
        return False
    name, called, wrong, _broader, _people, _conf, prompts, instead = r[:8]
    if not (isinstance(name, str) and name.strip() and _count(called) and _count(wrong)
            and wrong <= called and isinstance(prompts, bool) and isinstance(instead, list)):
        return False
    for a in instead:
        if not (isinstance(a, list) and len(a) >= 4 and isinstance(a[0], str) and a[0]
                and _count(a[1]) and a[3] in ("w", "b")):
            return False
    aliases = r[8] if len(r) > 8 else []
    return isinstance(aliases, list) and all(isinstance(a, str) for a in aliases)


def annotate(result: dict, table: Table | None) -> int:
    """Adds `confused_with` to every species candidate that prompts, wherever species
    names appear (ranks, the likely set, each photo's ranks). Returns how many."""
    if table is None or not isinstance(result, dict):
        return 0
    lists = [(result.get("ranks") or {}).get("species") or [],
             ((result.get("likely") or {}).get("species") or {}).get("names") or []]
    lists += [((p or {}).get("ranks") or {}).get("species") or []
              for p in result.get("per_photo") or []]
    seen: dict[str, dict | None] = {}
    added = 0
    for candidates in lists:
        for c in candidates:
            name = c.get("name") if isinstance(c, dict) else None
            if not name:
                continue
            if name not in seen:
                seen[name] = table.confused_with(name)
            if seen[name] is not None:
                c["confused_with"] = dict(seen[name], names=[dict(x) for x in seen[name]["names"]])
                added += 1
    return added


# ---------------------------------------------------------------------------
# Reading mycomap.org

def fetch_all(base_url: str, session: requests.Session | None = None,
              timeout: float = TIMEOUT, pauses: Iterable[float] = RETRY_PAUSES,
              sleep: Callable[[float], None] = time.sleep, log=print) -> dict:
    """.org's /all answer, checked (check_answer). No key: the route is public. Network
    failures and 5xx are asked again after each pause; .org's own 503 not_ready (no build
    yet) and other refusals are not. Raises Unavailable."""
    session = session or requests.Session()
    url = base_url.rstrip("/") + ALL_PATH
    headers = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    pauses = list(pauses)
    for attempt in range(len(pauses) + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            why = f"could not reach mycomap.org: {type(e).__name__}"
        else:
            if resp.status_code == 503 and _code(resp) == "not_ready":
                raise Unavailable("mycomap.org has not built the table yet (503 not_ready)")
            if resp.status_code < 500 and resp.status_code != 200:
                raise Unavailable(f"mycomap.org answered {resp.status_code}")
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    raise Unavailable("the answer was not JSON") from None
                return check_answer(payload)
            why = f"mycomap.org answered {resp.status_code}"
        if attempt == len(pauses):
            raise Unavailable(f"{why}, {attempt + 1} tries")
        log(f"[confused-with] {why}; asking again in {pauses[attempt]:.0f} s")
        sleep(pauses[attempt])
    raise AssertionError("unreachable")


def _code(resp) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    return body.get("code") if isinstance(body, dict) else None


# ---------------------------------------------------------------------------
# The cache

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cache_path(manifest_path: Path) -> Path:
    """Beside the served manifest: the nightly layer on the box, data/ on a laptop."""
    return Path(manifest_path).parent / CACHE


def read_state(cache: Path) -> dict:
    try:
        return json.loads((cache.parent / STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(cache: Path, state: dict) -> None:
    tmp = cache.parent / (STATE + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, cache.parent / STATE)


def load(cache: Path) -> Table | None:
    """The cached table, or None when there is none."""
    try:
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            held = json.load(f)
    except FileNotFoundError:
        return None
    return Table.from_answer(held["answer"], held.get("fetched_at"))


def refresh(cache: Path, fetch: Callable[[], dict], labels: Iterable[str] | None = None,
            log=print) -> dict:
    """One read of .org into `cache`. Whatever goes wrong before the new file is in place
    (a failed read, a 503, an empty or malformed answer) leaves the previous cache as it
    was; the report says so, with the failures in a row. `labels`: Vision's species names,
    to count how many of them prompt."""
    state = read_state(cache)
    state["last_attempt_at"] = now_iso()
    try:
        payload = fetch()
        table = Table.from_answer(payload, now_iso())
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.parent / (CACHE + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump({"fetched_at": table.fetched_at, "answer": payload}, f,
                      separators=(",", ":"))
        os.replace(tmp, cache)
    except Exception as e:  # noqa: BLE001 - reported; the previous cache serves on
        error = str(e) if isinstance(e, Unavailable) else f"{type(e).__name__}: {e}"
        state["failures_in_a_row"] = int(state.get("failures_in_a_row") or 0) + 1
        state["last_error"] = error[:500]
        _write_state(cache, state)
        kept = state.get("cached")
        log(f"[confused-with] not updated ({error}); "
            + (f"keeping the table built {kept.get('built_at')}" if kept else "no table yet"))
        return {"ok": False, "error": error[:500], "kept_previous": kept,
                "failures_in_a_row": state["failures_in_a_row"]}
    out = {"ok": True, **table.summary(), "bytes": cache.stat().st_size}
    if labels is not None:
        species = {n for n in labels if n and " " in n.strip()}
        out["vision_species"] = len(species)
        out["vision_species_prompting"] = sum(1 for n in species if table.confused_with(n))
    state.update(cached=table.summary(), last_ok_at=table.fetched_at, failures_in_a_row=0,
                 last_error=None)
    _write_state(cache, state)
    log(f"[confused-with] {out['names']:,} names from mycomap.org, {out['prompting']:,} prompt"
        + (f"; {out['vision_species_prompting']:,} of Vision's {out['vision_species']:,} "
           "species names" if labels is not None else ""))
    return out


def refresh_from_settings(cache: Path, labels: Iterable[str] | None = None, log=print) -> dict:
    """The nightly step: .org at MV_ORG_BASE_URL (skipped when unset)."""
    base_url = config.setting("MV_ORG_BASE_URL")
    if not base_url:
        return {"skipped": "MV_ORG_BASE_URL not set"}
    return refresh(cache, lambda: fetch_all(base_url, log=log), labels, log=log)


def health(cache: Path) -> dict:
    """For /api/health (public: no error text)."""
    state = read_state(cache)
    cached = state.get("cached") or {}
    return {"cached": bool(cached) and cache.is_file(), "built_at": cached.get("built_at"),
            "fetched_at": cached.get("fetched_at"), "prompting": cached.get("prompting"),
            "failures_in_a_row": int(state.get("failures_in_a_row") or 0)}


class Served:
    """The table the server answers with. Read at startup, and again when the cache file
    changes (a night, or `mv confused-with --fetch`): looked at no more than once every
    `check_every` seconds, reloaded by one thread while the others go on with the old one.
    A cache that can't be read leaves the old table serving."""

    def __init__(self, cache: Path, check_every: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, note=print):
        self.cache, self.check_every, self.clock, self.note = Path(cache), check_every, clock, note
        self._table: Table | None = None
        self._sig: tuple | None = None
        self._checked = -1e18
        self._lock = threading.Lock()

    def table(self) -> Table | None:
        if self.clock() - self._checked >= self.check_every:
            self.reload(block=False)
        return self._table

    def reload(self, block: bool = True) -> Table | None:
        if not self._lock.acquire(blocking=block):
            return self._table
        try:
            self._checked = self.clock()
            try:
                st = self.cache.stat()
                sig = (st.st_mtime_ns, st.st_size)
            except FileNotFoundError:
                sig = None
            if sig == self._sig:
                return self._table
            try:
                self._table = load(self.cache) if sig else None
                self._sig = sig
                if self._table is not None:
                    s = self._table.summary()
                    self.note(f"[confused-with] serving .org's table built {s['built_at']}: "
                              f"{s['prompting']:,} of {s['names']:,} names prompt")
            except Exception as e:  # noqa: BLE001 - the old table serves on
                self.note(f"[confused-with] cache unreadable, old table kept: "
                          f"{type(e).__name__}: {e}")
                self._sig = sig
            return self._table
        finally:
            self._lock.release()
