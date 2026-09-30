"""Family, order, class and phylum from iNaturalist's taxonomy, one answer per genus.

Steve's decisions (2026-09-30):

- The higher ranks come from iNaturalist, per genus: every record of a genus gets
  the same family, order, class and phylum, whatever .org's columns say. A genus in
  any doubt is listed for a person (`mv taxonomy`), never guessed, and keeps .org's
  values: no fungal genus of that name on iNat, more than one, an inactive one (its
  current replacement is named), a provisional one, one iNat puts in no family, or a
  name that isn't a Latin genus (a code, a note, a typo). A genus whose iNat family
  differs from the family most of its records carry on .org is applied, and listed.
- A one-word name ("Russula", "Agaricales", "Fungi") is never a species
  (`labels_for`): it counts at genus when the word is a genus, and at the ranks above.

The lookup (`mv fetch-taxonomy`) asks iNat's public API, read-only, at most one
request a second, with a User-Agent, backing off on 429 and 5xx. Names are searched
within Fungi (taxon 47170): a plant or animal genus of the same name is ignored.
Every answer and every taxon fetched is kept in <data>/taxonomy/inat_genera.sqlite,
beside the manifest and never inside it; a re-run skips names already answered
(`--refresh` asks them all again, `--older-than DAYS` those answered before then).
Labels read that file when records are loaded (evaluate.load_records), and a
release ships it (release.py), so the site labels records the same way.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests

from . import config, names
from .ratelimit import MinInterval

API = "https://api.inaturalist.org/v1"
FUNGI = 47170
HIGHER = ("phylum", "class", "order", "family")
GENUS_LEVEL = 20                   # iNat's rank_level of a genus; higher ranks are larger
CACHE = Path("taxonomy") / "inat_genera.sqlite"
BATCH = 30                         # taxa per /v1/taxa/<id,id,...> request

# A genus as it is written: capital, lower-case letters (a hyphen in a few). Anything
# else ('Sistotrema5', "'Mycena'", 'Mycena PNW05', 'galerina', '-') is not asked.
_LATIN = re.compile(r"^[A-Z][a-z]+(?:-[a-z]+)?\Z")
# Words written in the name column that are not a taxon.
NOT_A_TAXON = frozenset({"unknown", "unidentified", "none", "null", "sample", "sequence",
                         "delete", "uncultured", "environmental", "mixed", "fungus", "mushroom",
                         "lichen", "mold", "mould", "slime"})

# Why a name is listed for a person (the first six are never applied).
NOT_LATIN = "not a Latin genus name"
NOT_FOUND = "no fungal genus of that name on iNat"
SEVERAL = "more than one fungal genus of that name on iNat"
INACTIVE = "inactive on iNat"
PROVISIONAL = "provisional on iNat"
NO_FAMILY = "iNat puts it in no family"
DIFFERS = "iNat's family differs from most records on .org (iNat's applied)"
# (a subgenus or section, e.g. Dermocybe, Telamonia, is not looked up: listed here too)
HIGHER_NOT_FOUND = "no fungal taxon of that name above genus on iNat"

SCHEMA = """
create table if not exists answers (
  name            text not null,     -- as written in the records
  kind            text not null,     -- 'genus', or 'higher' (a one-word name above genus)
  status          text not null,     -- 'ok' (applied) or 'doubt' (listed, not applied)
  reason          text,
  taxon_id        integer,
  candidates      text,              -- JSON: the fungal taxon ids that matched the name
  replacement_ids text,              -- JSON: iNat's current taxa for an inactive one
  homonyms        integer not null default 0,   -- matches outside Fungi, ignored
  asked_at        text not null,
  primary key (name, kind)
);
create table if not exists taxa (
  id              integer primary key,
  name            text,
  rank            text,
  rank_level      real,
  is_active       integer,
  provisional     integer,
  ancestor_ids    text,              -- JSON, from the root down to the taxon itself
  current_ids     text,              -- JSON: current_synonymous_taxon_ids
  fetched_at      text not null
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_latin_word(name: str) -> bool:
    return bool(_LATIN.match(name or "")) and name.lower() not in NOT_A_TAXON


def one_word(label: str) -> bool:
    """A name of one word ('Russula', 'Agaricales'): never a species (Steve, 2026-09-30)."""
    return len((label or "").split()) == 1


# --- iNat, politely ------------------------------------------------------------------

class InatTaxa:
    """GET requests to iNat's API at most one a second, retried with backoff."""

    def __init__(self, session: requests.Session | None = None, interval: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep, attempts: int = 6):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.pace = MinInterval(interval, sleep=sleep)
        self.attempts = attempts
        self.calls = 0

    def get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(self.attempts):
            self.pace.wait()
            self.calls += 1
            try:
                r = self.session.get(API + path, params=params, timeout=60)
            except requests.RequestException:
                self.pace.pause(10 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                wait = 30 * (attempt + 1)
                after = str((getattr(r, "headers", None) or {}).get("Retry-After", ""))
                if after.isdigit():
                    wait = max(wait, int(after))
                self.pace.pause(wait)
                continue
            raise RuntimeError(f"iNat answered {r.status_code}: {r.text[:200]}")
        raise RuntimeError("iNat kept failing; stopped. Run it again to resume "
                           "(every answer so far is kept).")


# --- the cache ---------------------------------------------------------------------

def cache_path(data_dir: Path | None = None) -> Path:
    return (data_dir or config.DATA_DIR) / CACHE


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def _save_taxon(db: sqlite3.Connection, t: dict, at: str) -> None:
    db.execute("insert or replace into taxa values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (int(t["id"]), t.get("name"), t.get("rank"), t.get("rank_level"),
                None if t.get("is_active") is None else int(bool(t.get("is_active"))),
                int(bool(t.get("provisional"))), json.dumps(t.get("ancestor_ids") or []),
                json.dumps(t.get("current_synonymous_taxon_ids") or []), at))


def _save_answer(db: sqlite3.Connection, a: "Answer", at: str) -> None:
    with db:
        db.execute("insert or replace into answers values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (a.name, a.kind, a.status, a.reason, a.taxon_id, json.dumps(a.candidates),
                    json.dumps(a.replacement_ids), a.homonyms, at))


# --- asking about one name ------------------------------------------------------------

@dataclass
class Answer:
    name: str
    kind: str
    status: str                     # 'ok' or 'doubt'
    reason: str | None = None
    taxon_id: int | None = None
    candidates: list[int] = field(default_factory=list)
    replacement_ids: list[int] = field(default_factory=list)
    homonyms: int = 0


def in_fungi(t: dict) -> bool:
    return int(t.get("id") or 0) == FUNGI or FUNGI in (t.get("ancestor_ids") or [])


def _matches(results: list[dict], name: str, rank: str | None) -> tuple[list[dict], int]:
    """Taxa named exactly `name` (any case): (those in Fungi, how many outside it)."""
    exact = [t for t in results if str(t.get("name", "")).lower() == name.lower()
             and (rank is None or t.get("rank") == rank)]
    fungal = [t for t in exact if in_fungi(t)]
    return fungal, len(exact) - len(fungal)


def ask_genus(client: InatTaxa, db: sqlite3.Connection, name: str) -> Answer:
    """Is `name` a fungal genus on iNat, and which one?"""
    if not is_latin_word(name):
        return Answer(name, "genus", "doubt", NOT_LATIN)
    at = now_iso()
    body = client.get("/taxa", {"q": name, "rank": "genus", "per_page": 30})
    fungal, homonyms = _matches(body.get("results") or [], name, "genus")
    for t in fungal:
        _save_taxon(db, t, at)
    if len(fungal) > 1:
        return Answer(name, "genus", "doubt", SEVERAL, candidates=[int(t["id"]) for t in fungal],
                      homonyms=homonyms)
    if len(fungal) == 1:
        t = fungal[0]
        if t.get("provisional"):
            return Answer(name, "genus", "doubt", PROVISIONAL, int(t["id"]), [int(t["id"])],
                          homonyms=homonyms)
        return Answer(name, "genus", "ok", None, int(t["id"]), [int(t["id"])], homonyms=homonyms)
    # No active one: an inactive fungal genus of that name, with iNat's replacement?
    body = client.get("/taxa", {"q": name, "rank": "genus", "is_active": "false",
                                "per_page": 30})
    gone, more = _matches(body.get("results") or [], name, "genus")
    homonyms = max(homonyms, more)
    for t in gone:
        _save_taxon(db, t, at)
    if gone:
        current = sorted({int(i) for t in gone for i in t.get("current_synonymous_taxon_ids") or []})
        return Answer(name, "genus", "doubt", INACTIVE,
                      int(gone[0]["id"]) if len(gone) == 1 else None,
                      [int(t["id"]) for t in gone], current, homonyms)
    return Answer(name, "genus", "doubt", NOT_FOUND, homonyms=homonyms)


def ask_higher(client: InatTaxa, db: sqlite3.Connection, name: str) -> Answer:
    """A one-word name that is no genus: the fungal taxon above genus it names, if any."""
    at = now_iso()
    body = client.get("/taxa", {"q": name, "per_page": 30})
    fungal, homonyms = _matches(body.get("results") or [], name, None)
    fungal = [t for t in fungal if float(t.get("rank_level") or 0) > GENUS_LEVEL]
    for t in fungal:
        _save_taxon(db, t, at)
    if len(fungal) == 1:
        t = fungal[0]
        return Answer(name, "higher", "ok", None, int(t["id"]), [int(t["id"])], homonyms=homonyms)
    if fungal:
        return Answer(name, "higher", "doubt", SEVERAL.replace("genus", "taxon"),
                      candidates=[int(t["id"]) for t in fungal], homonyms=homonyms)
    return Answer(name, "higher", "doubt", HIGHER_NOT_FOUND, homonyms=homonyms)


def resolve_ancestors(client: InatTaxa, db: sqlite3.Connection, log=print) -> int:
    """Fetch every taxon the answers need that isn't cached yet: the ancestors of each
    answered taxon (to read their ranks) and the replacements of inactive ones."""
    have = {r[0] for r in db.execute("select id from taxa")}
    want: set[int] = set()
    for tid, reps in db.execute("select taxon_id, replacement_ids from answers"):
        want.update(json.loads(reps or "[]"))
        if tid is not None:
            want.add(int(tid))
    for tid in list(want):
        row = db.execute("select ancestor_ids from taxa where id = ?", (tid,)).fetchone()
        if row:
            want.update(int(i) for i in json.loads(row[0] or "[]"))
    missing = sorted(want - have)
    # A replacement's own ancestors are needed too: a second pass picks them up.
    fetched = 0
    while missing:
        for i in range(0, len(missing), BATCH):
            ids = missing[i:i + BATCH]
            body = client.get("/taxa/" + ",".join(str(x) for x in ids))
            at = now_iso()
            with db:
                for t in body.get("results") or []:
                    _save_taxon(db, t, at)
                    fetched += 1
                # A taxon iNat no longer returns: remember it, so it isn't asked forever.
                got = {int(t["id"]) for t in body.get("results") or []}
                for x in set(ids) - got:
                    db.execute("insert or ignore into taxa (id, fetched_at) values (?, ?)",
                               (x, at))
        have = {r[0] for r in db.execute("select id from taxa")}
        extra: set[int] = set()
        for x in missing:
            row = db.execute("select ancestor_ids from taxa where id = ?", (x,)).fetchone()
            if row and row[0]:
                extra.update(int(i) for i in json.loads(row[0]))
        missing = sorted(extra - have)
    if fetched:
        log(f"  {fetched} taxa fetched to read ranks")
    return fetched


# --- which names the records need ------------------------------------------------------

def clean(s) -> str:
    return (s or "").strip()


@dataclass
class RecordNames:
    """For every record: its label and the columns the labels are built from."""
    rows: list[tuple[str, str, str, bool]]      # (label, genus column, family column, respelled)

    @classmethod
    def from_manifest(cls, conn: sqlite3.Connection) -> "RecordNames":
        labels = names.manifest_labels(conn)
        rows = []
        for name, genus, family in conn.execute(
                "select scientific_name, genus, family from records "
                "where coalesce(scientific_name, '') <> ''"):
            label = clean(labels.get(name, name))
            if label:
                rows.append((label, clean(genus), clean(family), label != clean(name)))
        return cls(rows)

    def genera(self) -> Counter:
        """Records per genus name to look up: the genus a record's label uses (the genus
        column, the first word of a respelled or genus-less name), and every one-word name."""
        out = Counter()
        for label, genus, _family, respelled in self.rows:
            words = label.split()
            if len(words) == 1:
                out[label] += 1
            else:
                out[words[0] if respelled else (genus or words[0])] += 1
        return out

    def one_word(self) -> Counter:
        return Counter(label for label, *_ in self.rows if one_word(label))


def fetch(conn: sqlite3.Connection, cache: Path | None = None, client: InatTaxa | None = None,
          refresh: bool = False, older_than_days: float | None = None, limit: int | None = None,
          log=print) -> dict:
    """Ask iNat about every genus (and one-word name) the records use that the cache
    hasn't answered; then fetch the ranks of their ancestors. Resumable: each answer is
    saved as it comes."""
    client = client or InatTaxa()
    db = open_cache(cache or cache_path())
    try:
        rn = RecordNames.from_manifest(conn)
        genera = rn.genera()
        words = rn.one_word()
        cutoff = None
        if older_than_days is not None:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=older_than_days)).isoformat(timespec="seconds")

        def asked(name: str, kind: str) -> bool:
            if refresh:
                return False
            row = db.execute("select asked_at from answers where name = ? and kind = ?",
                             (name, kind)).fetchone()
            return bool(row) and (cutoff is None or row[0] >= cutoff)

        todo = [g for g, _ in sorted(genera.items(), key=lambda kv: (-kv[1], kv[0]))
                if not asked(g, "genus")]
        if limit is not None:
            todo = todo[:limit]
        log(f"{len(genera):,} genus names, {len(todo):,} to ask iNat about "
            f"(~{len(todo) * 1.3 / 60:.0f} min at 1 request/s)")
        stats = Counter()
        started = time.monotonic()
        for i, g in enumerate(todo, 1):
            a = ask_genus(client, db, g)
            _save_answer(db, a, now_iso())
            stats[a.reason or "ok"] += 1
            if i % 50 == 0:
                log(f"  {i:,}/{len(todo):,} genera, {client.calls:,} requests, "
                    f"{time.monotonic() - started:.0f} s")
        # One-word names iNat has as no genus: perhaps a family, an order, "Fungi".
        higher = [w for w in sorted(words)
                  if (row := db.execute("select reason from answers where name = ? and "
                                        "kind = 'genus'", (w,)).fetchone())
                  and row[0] == NOT_FOUND and not asked(w, "higher")]
        for w in higher:
            a = ask_higher(client, db, w)
            _save_answer(db, a, now_iso())
            stats["higher: " + (a.reason or "ok")] += 1
        resolve_ancestors(client, db, log)
        return {"genera": len(genera), "asked": len(todo), "higher_asked": len(higher),
                "requests": client.calls, "answers": dict(stats)}
    finally:
        db.close()


# --- applying the answers ----------------------------------------------------------------

@dataclass
class Taxonomy:
    """What the cache says, ready for labelling records."""
    genera: dict[str, dict[str, str]]          # applied: genus -> {phylum, class, order, family}
    higher: dict[str, dict[str, str]]          # one-word higher name -> its ranks (itself too)
    genus_words: set[str]                      # names iNat has as a fungal genus (any status)
    not_genus: set[str]                        # names that are no genus (above genus, or no taxon)
    answers: dict[tuple[str, str], tuple] = field(default_factory=dict, repr=False)
    taxa: dict[int, tuple] = field(default_factory=dict, repr=False)

    def is_genus(self, word: str) -> bool | None:
        """True / False when iNat's answer settles it; None when it doesn't (not asked,
        or no fungal taxon of that name at all: perhaps a genus iNat doesn't have yet)."""
        if word in self.genus_words:
            return True
        if word in self.not_genus:
            return False
        return None

    def name_of(self, taxon_id: int | None) -> str:
        """A taxon's name, with its rank when it isn't a genus ('Nolanea (subgenus)')."""
        t = self.taxa.get(taxon_id) if taxon_id is not None else None
        if not t or not t[0]:
            return ""
        return t[0] if t[1] in (None, "genus") else f"{t[0]} ({t[1]})"


def ranks_of(taxa: dict[int, tuple], taxon_id: int) -> dict[str, str] | None:
    """{phylum, class, order, family} along a taxon's ancestry ('' where iNat has none);
    None if a step of it isn't cached yet."""
    t = taxa.get(taxon_id)
    if not t or t[3] is None:
        return None
    out = {rank: "" for rank in HIGHER}
    for aid in t[3] or [taxon_id]:
        a = taxa.get(int(aid))
        if a is None or a[1] is None:
            return None
        if a[1] in out:
            out[a[1]] = a[0]
    return out


def load(path: Path) -> Taxonomy | None:
    if not path.is_file():
        return None
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        taxa = {int(r[0]): (r[1], r[2], r[3], json.loads(r[4]) if r[4] else None)
                for r in db.execute("select id, name, rank, rank_level, ancestor_ids from taxa")}
        answers = {(r[0], r[1]): r[2:] for r in db.execute(
            "select name, kind, status, reason, taxon_id, replacement_ids, homonyms "
            "from answers")}
    finally:
        db.close()
    genera, higher, genus_words, not_genus = {}, {}, set(), set()
    for (name, kind), (status, reason, tid, _reps, _homonyms) in answers.items():
        if kind == "genus":
            if reason in (NOT_LATIN,):
                not_genus.add(name)
            elif reason != NOT_FOUND:
                genus_words.add(name)          # ok, inactive, provisional, several: a genus
            if status == "ok" and tid is not None:
                r = ranks_of(taxa, int(tid))
                if r and r["family"]:
                    genera[name] = r
        elif kind == "higher" and status == "ok" and tid is not None:
            not_genus.add(name)
            r = ranks_of(taxa, int(tid))
            if r:
                higher[name] = r
    return Taxonomy(genera, higher, genus_words, not_genus, answers, taxa)


_LOADED: dict[str, tuple[tuple, Taxonomy | None]] = {}


def for_manifest(conn: sqlite3.Connection) -> Taxonomy | None:
    """The cache beside this manifest (<its folder>/taxonomy/inat_genera.sqlite), or
    None; read again only when the file changes."""
    main = next((r[2] for r in conn.execute("pragma database_list") if r[1] == "main"), "")
    if not main:
        return None
    path = Path(main).parent / CACHE
    try:
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    hit = _LOADED.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1]
    tax = load(path)
    _LOADED[str(path)] = (stamp, tax)
    return tax


# --- a record's labels ----------------------------------------------------------------------

@dataclass(frozen=True)
class Labels:
    species: str        # '' for a one-word name
    genus: str
    family: str
    unit: str           # what the index groups the record under: the species, or the one word


def labels_for(label: str, genus_col: str, family_col: str, respelled: bool,
               tax: Taxonomy | None) -> Labels:
    """A record's species, genus and family.

    Species: the label, unless it is one word, then none. Genus: a merged spelling
    names its own genus, otherwise the genus column (the first word if blank); a
    one-word name is its own genus when iNat has it as a genus, or, when iNat can't
    say, when the genus column agrees. Family: iNat's for that genus when the cache
    has an answer free of doubt, else a one-word higher name's own, else .org's."""
    label, genus_col, family_col = clean(label), clean(genus_col), clean(family_col)
    words = label.split()
    if not words:
        return Labels("", "", "", "")
    if len(words) > 1:
        species, unit = label, label
        genus = words[0] if respelled else (genus_col or words[0])
    else:
        species, unit = "", label
        is_genus = tax.is_genus(label) if tax else None
        if is_genus is None:
            is_genus = genus_col == label and is_latin_word(label)
        genus = label if is_genus else ""
    family = family_col
    if tax:
        if genus and genus in tax.genera:
            family = tax.genera[genus]["family"]
        elif not genus and label in tax.higher:
            family = tax.higher[label]["family"]
    return Labels(species, genus, family, unit)


# --- the report for a person -----------------------------------------------------------------

def _org_families(fams: Counter) -> str:
    return "; ".join(f"{f or '(blank)'} {n}" for f, n in fams.most_common())


def report(conn: sqlite3.Connection, cache: Path | None = None, out_dir: Path | None = None,
           top: int = 15) -> dict:
    """Numbers, and a CSV of every name in doubt (and every disagreement with .org)."""
    path = cache or cache_path()
    tax = load(path)
    if tax is None:
        raise FileNotFoundError(f"no taxonomy cache at {path}: run mv fetch-taxonomy first")
    rn = RecordNames.from_manifest(conn)
    genera = rn.genera()
    fams: dict[str, Counter] = defaultdict(Counter)
    filled = changed = 0
    for label, genus, family, respelled in rn.rows:
        lab = labels_for(label, genus, family, respelled, tax)
        g = lab.genus or (label if one_word(label) else "")
        if g:
            fams[g][family] += 1
        if lab.family != family:
            if family:
                changed += 1
            else:
                filled += 1
    rows, doubt = [], Counter()
    for g, n in sorted(genera.items(), key=lambda kv: (-kv[1], kv[0])):
        a = tax.answers.get((g, "genus"))
        if a is None:
            continue
        status, reason, tid, reps, homonyms = a
        inat_family = tax.genera.get(g, {}).get("family", "")
        if status == "ok" and g not in tax.genera:
            reason = NO_FAMILY
        org = fams.get(g, Counter())
        named = [(f, c) for f, c in org.most_common() if f]
        if status == "ok" and inat_family and named and named[0][0] != inat_family:
            reason = DIFFERS
        if reason is None:
            continue
        if reason == NOT_FOUND and (g, "higher") in tax.answers:
            h = tax.answers[(g, "higher")]
            if h[0] == "ok":
                continue                       # a one-word family or order, not a genus
            reason = h[1] or reason
        doubt[reason] += 1
        replacement = ", ".join(tax.name_of(int(i)) or str(i) for i in json.loads(reps or "[]"))
        rows.append({"genus": g, "records": n, "inat_family": inat_family,
                     "org_families": _org_families(org), "why": reason,
                     "inat_taxon": tid or "", "inat_replacement": replacement,
                     "homonyms_outside_fungi": homonyms or "",
                     "records_off_inat_family": sum(c for f, c in named if f != inat_family)
                     if reason == DIFFERS else ""})
    out_dir = out_dir or config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "taxonomy-doubts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["genus", "records", "inat_family", "org_families",
                                          "why", "inat_taxon", "inat_replacement",
                                          "homonyms_outside_fungi",
                                          "records_off_inat_family"])
        w.writeheader()
        w.writerows(rows)
    asked = [k for k in tax.answers if k[1] == "genus" and k[0] in genera]
    disagree = sorted((r for r in rows if r["why"] == DIFFERS),
                      key=lambda r: -r["records_off_inat_family"])
    one = rn.one_word()
    return {
        "genus_names": len(genera),
        "asked": len(asked),
        "not_asked_yet": len(genera) - len(asked),
        "applied": sum(1 for g in genera if g in tax.genera),
        "in_doubt": dict(doubt.most_common()),
        "records": len(rn.rows),
        "records_family_filled": filled,
        "records_family_changed": changed,
        "top_disagreements": [
            {k: r[k] for k in ("genus", "records", "inat_family", "org_families")}
            for r in disagree[:top]],
        "one_word_names": len(one),
        "one_word_records": sum(one.values()),
        "one_word_genera": sum(1 for w in one if tax.is_genus(w)),
        "one_word_higher": sum(1 for w in one if w in tax.higher),
        "report": str(csv_path),
    }
