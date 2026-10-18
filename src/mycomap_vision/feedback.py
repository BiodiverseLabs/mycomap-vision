"""Human Feedback: a review queue of the places where Vision and our reference data disagree,
for mycomap.org's admins, project leaders and contributors to clean up (Steve, 2026-10-18).

Most of the work is cleanup: one name written three ways, a group label, a retired name, a
record whose own DNA says something else, a photo of a field slip. Each queue item says which
RULE picked it (never "the model got this wrong" on its own: the outside review of 2026-10-18,
model disagreement alone is no reason to change a label), what a reviewer needs to see, and the
decisions it allows. A decision is stored here with who made it; where it takes effect is part
of the decision:

    org       a name update on mycomap.org (its Name Update / Validate Records flows; the page
              links there in a new tab) and, through .org, on the legacy .com site
    vision    applied by Vision itself (a photo left out of the reference, a twin counted once)
    note      recorded only (a look-alike pair, a hard but valid example)

Fairness (the paper coordinator, 2026-10-18): a fix found here applies to every record of the
name or rule, not only where Vision missed, so name-level items carry every record of the
name; and model-flagged queues are mixed with a seeded blind sample of finds Vision got right,
shown the same way, so the error rate reviewers find is measured, not assumed.

The queue lives in a SQLite file kept across releases (<release root>/state/, like the usage
counts). `mv feedback seed <tsv>` loads the review lists (the v1 miss review's
review_queue_seed.tsv and crosswalk_candidates.tsv); a seed never duplicates an item.
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Option:
    code: str
    label: str
    applies: str               # org | vision | note
    needs_value: bool = False  # e.g. the name to rename to


@dataclass(frozen=True)
class ItemType:
    code: str
    title: str
    about: str                 # one paragraph shown above the queue
    shows: tuple[str, ...]     # what a reviewer sees on each item
    options: tuple[Option, ...]
    model_flagged: bool = False
    granularity: str = "record"


KEEP = Option("keep", "Keep as it is", "note")
RENAME = Option("rename", "Rename the record(s) to…", "org", needs_value=True)
SEE = ("photos", "label now on mycomap.org", "Vision's top 5", "sequences with BLAST",
       "the rule that picked it")

TYPES: dict[str, ItemType] = {t.code: t for t in (
    ItemType("name_writing_split", "One name, several writings",
             "The same taxon is written more than one way (var. / subsp. / ssp., IN7 / IN07, "
             "hyphens). Vision counts them as one name already; the records should still use "
             "one writing.",
             ("every writing with its record count", "the proposed writing"),
             (Option("use_writing", "Use this writing for all", "org", needs_value=True),
              Option("different", "These are different taxa", "note")),
             granularity="name group"),
    ItemType("group_label", "Group or complex labels",
             "A label like 'Laccaria laccata group' names no single species, so no answer can "
             "be right. Rename it at source to a species or a provisional code.",
             ("records under the label", "photos", "sequences with BLAST", "Vision's top 5"),
             (RENAME, Option("keep_group", "Keep the group (no species decision possible)",
                             "note")), granularity="name"),
    ItemType("name_retired", "Names retired on MycoMap's synonym list",
             "Records still use a name MycoMap's own synonym list retires.",
             ("the retired and accepted names", "records under the retired name"),
             (Option("accept", "Use the accepted name for all", "org"), KEEP, RENAME),
             granularity="name"),
    ItemType("name_inactive_on_inat", "Names iNaturalist no longer uses",
             "iNaturalist has made the taxon inactive and points to a current name.",
             ("iNat's current name", "records under the name"),
             (Option("accept", "Use iNat's current name for all", "org"), KEEP, RENAME),
             granularity="name"),
    ItemType("crosswalk", "Provisional code and formal name: one lineage?",
             "A provisional code and a formal (or group) name that may be the same lineage, or "
             "one inside the other: their DNA, a shared epithet, a name authority or Vision "
             "link them. Decided at source, never in scoring alone.",
             ("both names with record counts", "the shared DNA and example sequences",
              "Vision's confusions between them", "name authorities"),
             (Option("same", "Same lineage: rename one to the other", "org", needs_value=True),
              Option("nested", "Nested: the code is a lineage within the formal name", "org"),
              Option("different", "Different lineages", "note")),
             granularity="name pair"),
    ItemType("label_vs_own_dna", "The record's own DNA names another taxon",
             "A sequence of this record carries another name. Often the sequence's name is the "
             "stale one (a code refined later); sometimes the label is wrong or the sequence is "
             "a contaminant. Open the BLAST results before deciding.",
             SEE, (Option("label_right", "Label is right (the sequence's name is stale)", "org"),
                   RENAME,
                   Option("bad_sequence", "The sequence is a contaminant or the wrong specimen",
                          "org"),
                   Option("unsure", "Needs another look (BLAST, microscopy)", "note"))),
    ItemType("photos_unlike_label", "Photos unlike the label",
             "Vision ranks the label far down although it knows the species well, and no photo "
             "even reaches the label's genus: the photos may show another organism or no "
             "usable specimen.",
             SEE, (RENAME,
                   Option("exclude_photos", "Wrong or unusable photos: leave them out of Vision",
                          "vision"),
                   Option("hard_example", "Label and photos are right (a hard example)", "note")),
             model_flagged=True),
    ItemType("vision_names_same_taxon", "Vision may name the same taxon",
             "Vision's first answer looks like the same taxon under another name (a variant, a "
             "synonym, a code on the same epithet).",
             SEE, (Option("same", "Same taxon: rename the label to Vision's name", "org"),
                   Option("crosswalk", "A code/formal-name question: send to the crosswalk",
                          "note"),
                   Option("different", "Different taxa: Vision is wrong", "note")),
             model_flagged=True),
    ItemType("reference_label_doubt", "Reference records in doubt",
             "A reference record that pulled other finds to its name, and whose own DNA or "
             "later name disagrees with its label.",
             SEE, (Option("label_right", "Label is right", "note"), RENAME,
                   Option("exclude_record", "Leave it out of Vision's reference until fixed",
                          "vision")),
             model_flagged=True),
    ItemType("non_specimen_photo", "Photos that show no specimen",
             "Micrographs, microscope set-ups, field slips, habitat scenes, or photos the "
             "not-a-fungus gate turns away. Useful on the record, but not for matching photos.",
             ("the photo", "why it was flagged"),
             (Option("exclude_photo", "Leave this photo out of Vision", "vision"),
              Option("keep_photo", "Keep it (it shows the specimen)", "note")),
             granularity="photo"),
    ItemType("twin_location_conflict", "One find, two places",
             "The same find posted to both Mushroom Observer and iNaturalist, whose two copies "
             "are more than 5 km apart, so maps and the Atlas count it as two sites. Check "
             "both copies' maps and fix the copy that is wrong at its source.",
             ("both records' links", "how far apart (km)", "how each copy is placed",
              "which copy looks wrong"),
             # Codes kept stable: a later .org hand-off may act on them (the coordinator).
             (Option("fix_mo_location", "The MO copy's location is wrong: fix it on MO / .org",
                     "org"),
              Option("fix_inat_location", "The iNat copy's location is wrong: fix it on iNat",
                     "org"),
              Option("both_right", "Both are right (two places)", "note")),
             granularity="record pair"),
    ItemType("crosspost_twin", "One find posted to both iNaturalist and Mushroom Observer",
             "Two records with the same name and the same photo. Vision should count them as "
             "one specimen.",
             ("both records' photos", "both links"),
             (Option("same_find", "Same find: count once", "vision"),
              Option("different_finds", "Different finds", "note")),
             granularity="record pair"),
)}

SCHEMA = """
create table if not exists items (
  id integer primary key, item_type text not null, subject text not null,
  granularity text not null, rule text not null, model_flagged integer not null default 0,
  blind integer not null default 0, record_keys text not null default '[]',
  links text not null default '[]', current text, proposed text,
  evidence text not null default '{}', projects text not null default '[]',
  records integer not null default 1, status text not null default 'open',
  seed text not null, created_at real not null,
  unique (item_type, subject)
);
create index if not exists items_type_status on items (item_type, status);
create table if not exists decisions (
  id integer primary key, item_id integer not null references items(id), decision text not null,
  value text, note text, photo_keys text not null default '[]', applies text not null,
  by_user text not null, by_name text, at real not null,
  applied integer not null default 0, applied_at real, applied_by text
);
create index if not exists decisions_item on decisions (item_id);
create table if not exists refreshes (
  id integer primary key, item_id integer not null references items(id), at real not null,
  by_user text not null, name_now text, top5 text not null default '[]',
  still_disagrees integer
);
create table if not exists seeds (
  name text primary key, file_sha256 text, rows integer, added integer, at real not null
);
"""


# Queues reviewed blind: a reviewer judges the label from the photos and the record, Vision's
# answer, the rule and the outcome evidence hidden until they decide, so the finds Vision got
# right that are mixed in (blind_sample) look exactly like the others.
BLIND_TYPES = ("photos_unlike_label", "reference_label_doubt")
MASKED_EVIDENCE = ("in_v1", "v1_val_outcome", "evidence")


def reviewer_view(item: dict) -> dict:
    """An item as reviewers receive it: the blind-sample marks never leave the server, and an
    open item of a BLIND_TYPES queue carries no Vision answer, rule or outcome evidence."""
    out = {k: v for k, v in item.items() if k not in ("blind", "seed", "hidden_until_decided")}
    masked = item["item_type"] in BLIND_TYPES and item["status"] == "open"
    out["masked"] = masked
    if masked:
        out["rule"] = "label check"
        out["proposed"] = ""
        out["evidence"] = {k: v for k, v in (item.get("evidence") or {}).items()
                           if k not in MASKED_EVIDENCE}
        out["refreshes"] = []
    return out


def blind_order(subject: str) -> str:
    """The order a BLIND_TYPES queue is shown in: a fixed shuffle by subject. The blind sample
    is seeded after the model-flagged items, so listing by id would put every blind item last
    and give it away."""
    import hashlib
    return hashlib.sha256(f"blind-order:{subject}".encode()).hexdigest()


class FeedbackError(ValueError):
    pass


@dataclass
class Store:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("pragma journal_mode=wal")
        self.conn.executescript(SCHEMA)

    # --- seeding ------------------------------------------------------------------------------
    def seed(self, rows: list[dict], name: str, file_sha256: str | None = None) -> dict:
        """Add queue items (dicts with item_type, subject, rule and optionally granularity,
        model_flagged, blind, record_keys, links, current, proposed, evidence, projects,
        records). An item already queued (same type and subject) is left as it is, decisions
        and all. Refuses an unknown item type."""
        now = time.time()
        added = 0
        with self._lock, self.conn:
            for r in rows:
                t = TYPES.get(r["item_type"])
                if t is None:
                    raise FeedbackError(f"unknown item type {r['item_type']!r}")
                cur = self.conn.execute(
                    "insert or ignore into items (item_type, subject, granularity, rule, "
                    "model_flagged, blind, record_keys, links, current, proposed, evidence, "
                    "projects, records, seed, created_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (t.code, r["subject"], r.get("granularity") or t.granularity, r["rule"],
                     int(bool(r.get("model_flagged", t.model_flagged))), int(bool(r.get("blind"))),
                     json.dumps(r.get("record_keys") or []), json.dumps(r.get("links") or []),
                     r.get("current"), r.get("proposed"), json.dumps(r.get("evidence") or {}),
                     json.dumps(r.get("projects") or []), int(r.get("records") or 1), name, now))
                added += cur.rowcount
            self.conn.execute("insert or replace into seeds values (?,?,?,?,?)",
                              (name, file_sha256, len(rows), added, now))
        return {"seed": name, "rows": len(rows), "added": added}

    # --- reading -------------------------------------------------------------------------------
    def summary(self, projects: set[str] | None = None) -> list[dict]:
        """Per item type: its title, description and open / decided counts (within
        `projects` when given: a leader or contributor sees their projects' items)."""
        rows = self._visible("select item_type, status, projects from items", (), projects)
        counts: dict[str, dict] = {}
        for r in rows:
            c = counts.setdefault(r["item_type"], {"open": 0, "decided": 0})
            c["open" if r["status"] == "open" else "decided"] += 1
        return [{"type": t.code, "title": t.title, "about": t.about,
                 "granularity": t.granularity, "model_flagged": t.model_flagged,
                 **counts.get(t.code, {"open": 0, "decided": 0})} for t in TYPES.values()]

    def items(self, item_type: str, status: str = "open", offset: int = 0, limit: int = 25,
              projects: set[str] | None = None, query: str = "") -> dict:
        if item_type not in TYPES:
            raise FeedbackError(f"unknown item type {item_type!r}")
        sql = ("select * from items where item_type = ? and status = ?"
               + (" and (subject like ? or current like ? or proposed like ?)" if query else "")
               + " order by records desc, id")
        params: tuple = (item_type, status) + ((f"%{query}%",) * 3 if query else ())
        rows = self._visible(sql, params, projects)
        if item_type in BLIND_TYPES:
            rows = sorted(rows, key=lambda r: blind_order(r["subject"]))
        return {"total": len(rows),
                "items": [self._item(r) for r in rows[offset:offset + limit]]}

    def items_of(self, item_type: str, projects: set[str] | None = None) -> list[dict]:
        """Every item of a type (within `projects` when given), each with its latest decision
        as "last_decision" (None while undecided): the photo review groups them by record."""
        if item_type not in TYPES:
            raise FeedbackError(f"unknown item type {item_type!r}")
        rows = self._visible("select * from items where item_type = ? order by id",
                             (item_type,), projects)
        last = {r["item_id"]: dict(r) for r in self.conn.execute(
            "select d.* from decisions d join items i on i.id = d.item_id where i.item_type = ? "
            "and d.id in (select max(id) from decisions group by item_id)", (item_type,))}
        return [self._item(r) | {"last_decision": last.get(r["id"])} for r in rows]

    def get(self, item_id: int, projects: set[str] | None = None) -> dict:
        r = self.conn.execute("select * from items where id = ?", (item_id,)).fetchone()
        if r is None or (projects is not None and not self._in(r, projects)):
            raise KeyError(item_id)
        item = self._item(r)
        item["decisions"] = [dict(d) | {"photo_keys": json.loads(d["photo_keys"])}
                             for d in self.conn.execute(
                                 "select * from decisions where item_id = ? order by at",
                                 (item_id,))]
        item["refreshes"] = [dict(x) | {"top5": json.loads(x["top5"])} for x in self.conn.execute(
            "select * from refreshes where item_id = ? order by at", (item_id,))]
        return item

    # --- writing -------------------------------------------------------------------------------
    def decide(self, item_id: int, decision: str, person: dict, value: str | None = None,
               note: str | None = None, photo_keys: list[str] | None = None,
               projects: set[str] | None = None) -> dict:
        item = self.get(item_id, projects)
        t = TYPES[item["item_type"]]
        opt = next((o for o in t.options if o.code == decision), None)
        if opt is None:
            raise FeedbackError(f"{decision!r} is not a decision for {t.code} "
                                f"({', '.join(o.code for o in t.options)})")
        if opt.needs_value and not (value or "").strip():
            raise FeedbackError(f"{decision!r} needs a value (the name to use)")
        with self._lock, self.conn:
            self.conn.execute(
                "insert into decisions (item_id, decision, value, note, photo_keys, applies, "
                "by_user, by_name, at) values (?,?,?,?,?,?,?,?,?)",
                (item_id, decision, (value or "").strip() or None, note,
                 json.dumps(photo_keys or []), opt.applies, str(person["sub"]),
                 person.get("name"), time.time()))
            self.conn.execute("update items set status = 'decided' where id = ?", (item_id,))
        return self.get(item_id, projects)

    def reopen(self, item_id: int, projects: set[str] | None = None) -> dict:
        self.get(item_id, projects)
        with self._lock, self.conn:
            self.conn.execute("update items set status = 'open' where id = ?", (item_id,))
        return self.get(item_id, projects)

    def record_refresh(self, item_id: int, person: dict, name_now: str | None,
                       top5: list, still_disagrees: bool | None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "insert into refreshes (item_id, at, by_user, name_now, top5, still_disagrees) "
                "values (?,?,?,?,?,?)", (item_id, time.time(), str(person["sub"]), name_now,
                                         json.dumps(top5), None if still_disagrees is None
                                         else int(still_disagrees)))

    def mark_applied(self, decision_id: int, person: dict) -> None:
        with self._lock, self.conn:
            self.conn.execute("update decisions set applied = 1, applied_at = ?, applied_by = ? "
                              "where id = ?", (time.time(), str(person["sub"]), decision_id))

    # --- outputs ---------------------------------------------------------------------------------
    def changes(self, applies: str | None = None, pending_only: bool = True) -> list[dict]:
        """Decisions to carry out (on .org, or by Vision), newest last, with the item."""
        sql = ("select d.*, i.item_type, i.subject, i.current, i.proposed, i.record_keys, "
               "i.records, i.rule, i.blind from decisions d join items i on i.id = d.item_id "
               "where d.id in (select max(id) from decisions group by item_id)")
        params: list = []
        if applies:
            sql += " and d.applies = ?"
            params.append(applies)
        if pending_only:
            sql += " and d.applied = 0"
        return [dict(r) | {"record_keys": json.loads(r["record_keys"]),
                           "photo_keys": json.loads(r["photo_keys"])}
                for r in self.conn.execute(sql + " order by d.at", params)]

    def excluded_photos(self) -> set[str]:
        """Photo keys a current decision leaves out of Vision (applied by Vision itself)."""
        out: set[str] = set()
        for c in self.changes(applies="vision", pending_only=False):
            if c["decision"] == "exclude_photo":
                out.add(c["subject"])
            elif c["decision"] == "exclude_photos":
                out.update(c["photo_keys"])
        return out

    # --- helpers ---------------------------------------------------------------------------------
    def _visible(self, sql, params, projects):
        rows = self.conn.execute(sql, params).fetchall()
        return rows if projects is None else [r for r in rows if self._in(r, projects)]

    @staticmethod
    def _in(row, projects: set[str]) -> bool:
        return bool(set(json.loads(row["projects"] or "[]")) & projects)

    @staticmethod
    def _item(r) -> dict:
        d = dict(r)
        for k in ("record_keys", "links", "projects"):
            d[k] = json.loads(d[k] or "[]")
        d["evidence"] = json.loads(d["evidence"] or "{}")
        d["model_flagged"] = bool(d["model_flagged"])
        d["blind"] = bool(d["blind"])
        if d["blind"]:
            d["hidden_until_decided"] = d["status"] == "open"
        return d


def type_catalogue() -> list[dict]:
    """The item types, what reviewers see and the decisions with where each applies."""
    return [{"type": t.code, "title": t.title, "about": t.about, "shows": list(t.shows),
             "granularity": t.granularity, "model_flagged": t.model_flagged,
             "options": [{"code": o.code, "label": o.label, "applies": o.applies,
                          "needs_value": o.needs_value} for o in t.options]}
            for t in TYPES.values()]


# --- seed files ----------------------------------------------------------------------------------

def link(record_key: str) -> str:
    return (f"https://mushroomobserver.org/obs/{record_key[3:]}" if record_key.startswith("mo:")
            else f"https://www.inaturalist.org/observations/{record_key}")


def rows_from_queue_seed(path: Path) -> list[dict]:
    """review_queue_seed.tsv (the v1 miss review, s15) as queue rows."""
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            keys = [k for k in r["subject"].replace(" + ", "\n").split("\n")
                    if r["granularity"] in ("record", "record pair")]
            out.append({"item_type": r["item_type"], "subject": r["subject"],
                        "granularity": r["granularity"], "rule": r["selection_rule"],
                        "model_flagged": r["model_flagged"] == "1", "record_keys": keys,
                        "links": [x.strip() for x in r["link"].split(";") if x.strip()],
                        "current": r["current"], "proposed": r["proposed"],
                        "records": int(r["records"] or 1),
                        "evidence": {"evidence": r["evidence"], "in_v1": r["in_v1"],
                                     "v1_val_outcome": r["v1_val_outcome"]}})
    return out


def rows_from_crosswalk(path: Path) -> list[dict]:
    """crosswalk_candidates.tsv (s16) as queue rows."""
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.append({"item_type": "crosswalk", "subject": f"{r['name_a']} <-> {r['name_b']}",
                        "rule": "crosswalk: " + r["evidence"],
                        "model_flagged": r["evidence"] == "vision",
                        "current": r["name_a"], "proposed": r["name_b"],
                        "records": int(r["records_a"] or 0) + int(r["records_b"] or 0),
                        "evidence": {k: r[k] for k in (
                            "kind_a", "records_a", "kind_b", "records_b", "evidence",
                            "v1_val_misses_resolved", "misses_detail", "dna_records",
                            "dna_detail", "example_sequences", "name_authority",
                            "shared_stem", "inat_status")}})
    return out


def rows_from_twin_locations(path: Path) -> list[dict]:
    """The Atlas lane's twins whose copies are over 5 km apart (columns distance_km,
    mo_org_id, mo_number, mo_placement, inat_org_id, inat_number, inat_accuracy_m, taxon,
    likely_wrong_copy). No coordinates are read or kept: only the distance, how each copy
    is placed, and which one looks wrong."""
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="	"):
            mo, inat = f"mo:{r['mo_number']}", str(r["inat_number"])
            km = float(r["distance_km"])
            out.append({
                "item_type": "twin_location_conflict", "subject": f"{mo} + {inat}",
                "rule": "MO <-> iNat twins (same label, same photo) placed over 5 km apart",
                "record_keys": [mo, inat], "links": [link(mo), link(inat)],
                "current": r["taxon"], "proposed": r["likely_wrong_copy"], "records": 2,
                "evidence": {"distance_km": f"{km:.0f}" if km >= 10 else f"{km:.1f}",
                             "mo_placement": r["mo_placement"],
                             "inat_accuracy_m": r["inat_accuracy_m"] or "not recorded",
                             "likely_wrong_copy": r["likely_wrong_copy"]}})
    return out


def blind_sample(hits: list[dict], item_type: str, n: int, seed: int) -> list[dict]:
    """A seeded sample of finds Vision got right, queued like the model-flagged items of
    `item_type` (blind = 1), so reviewers' error rate on them can be compared."""
    rng = random.Random(seed)
    pick = rng.sample(hits, min(n, len(hits)))
    return [{"item_type": item_type, "subject": h["record_key"], "rule": "blind sample of "
             f"finds Vision answered right (seed {seed})", "model_flagged": True, "blind": True,
             "record_keys": [h["record_key"]], "links": [link(h["record_key"])],
             "current": h["label"], "proposed": "", "evidence": {}} for h in pick]


def read_hits(path: Path) -> list[dict]:
    """Finds Vision answered right, for the blind sample (TSV: record_key, label)."""
    with open(path, encoding="utf-8", newline="") as f:
        return [{"record_key": r["record_key"], "label": r["label"]}
                for r in csv.DictReader(f, delimiter="	")]


def with_projects(rows: list[dict], conn: sqlite3.Connection) -> list[dict]:
    """Each row's projects, from the manifest: its records' green projects, and for a
    name-level item every record under any of its names (a leader sees a name-level item
    when any record of the name is in their project)."""
    by_record, by_name = {}, {}
    for key, name, green in conn.execute(
            "select observation_id, scientific_name, green_projects from records"):
        ps = set(json.loads(green or "[]"))
        by_record[key] = ps
        if name:
            by_name.setdefault(name, set()).update(ps)
    for r in rows:
        ps = set(r.get("projects") or [])
        for k in r.get("record_keys") or []:
            ps |= by_record.get(k, set())
        if not r.get("record_keys"):
            names = {r.get("current"), r.get("proposed"), *str(r["subject"]).replace(
                " <-> ", "; ").split("; ")}
            for n in names:
                ps |= by_name.get((n or "").strip(), set())
        r["projects"] = sorted(ps)
    return rows


def seed_files(store: Store, conn: sqlite3.Connection, queue: Path | None = None,
               crosswalk: Path | None = None, hits: Path | None = None,
               blind_ratio: float = 0.25, seed: int = 20261018,
               twin_locations: Path | None = None) -> list[dict]:
    """Seed the queue from the miss review's files. With `hits`, every model-flagged type
    also gets a blind sample of finds Vision answered right: blind_ratio of its items."""
    import hashlib
    out = []
    for path, reader in ((queue, rows_from_queue_seed), (crosswalk, rows_from_crosswalk),
                         (twin_locations, rows_from_twin_locations)):
        if path is None:
            continue
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        rows = with_projects(reader(Path(path)), conn)
        out.append(store.seed(rows, Path(path).name, sha))
        if hits is not None:
            pool = read_hits(Path(hits))
            for t in TYPES.values():
                n = sum(1 for r in rows if r["item_type"] == t.code and r.get("model_flagged"))
                if t.code in BLIND_TYPES and n:
                    blind = with_projects(blind_sample(pool, t.code, round(n * blind_ratio),
                                                       seed), conn)
                    out.append(store.seed(blind, f"blind-{t.code}-{seed}"))
    return out


# --- photos Vision leaves out (decisions it applies itself) ------------------------------------
# Synced into the manifest Vision serves from, at startup and before each nightly update, so
# the reference set is rebuilt at most once a night, not on every decision. evaluate.load_records
# skips these photos (the served index and training); a frozen dataset release is not affected.
EXCLUSIONS_SCHEMA = """
create table if not exists feedback_photo_exclusions (
  photo_id integer primary key, photo_key text not null
);
"""
MO_PHOTO_ID_OFFSET = 1_000_000_000_000      # manifest photo ids of MO photos


def ensure_exclusions_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(EXCLUSIONS_SCHEMA)


def photo_id_of(key: str) -> int | None:
    """The manifest photo id of a photo key ('inat:123', 'mo:45'), or None."""
    source, _, number = (key or "").partition(":")
    if not number.isdigit():
        return None
    if source == "inat":
        return int(number)
    if source == "mo":
        return MO_PHOTO_ID_OFFSET + int(number)
    return None


def sync_exclusions(store: Store, conn: sqlite3.Connection) -> dict:
    """Make the manifest's exclusion table the decisions' current set."""
    import hashlib
    keys = sorted(store.excluded_photos())
    rows = [(photo_id_of(k), k) for k in keys if photo_id_of(k) is not None]
    ensure_exclusions_schema(conn)
    with conn:
        conn.execute("delete from feedback_photo_exclusions")
        conn.executemany("insert or ignore into feedback_photo_exclusions values (?, ?)", rows)
    return {"photos": len(rows), "fingerprint": exclusions_fingerprint(conn),
            "unreadable_keys": len(keys) - len(rows)}


def exclusions_fingerprint(conn: sqlite3.Connection) -> str:
    """Changes whenever the set of excluded photos changes (the index is then rebuilt)."""
    import hashlib
    try:                      # read on every index lookup: no schema work here
        ids = [r[0] for r in conn.execute("select photo_id from feedback_photo_exclusions "
                                          "order by photo_id")]
    except sqlite3.OperationalError:          # never synced on this manifest: none
        ids = []
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:12]
