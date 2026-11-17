"""E17: the blind review page (Steve, 2026-11-16; item schema v1.1).

A reviewer is shown, one item at a time, a name, its DNA evidence (BLAST hits against the
local database and GenBank) and photos, and says whether the name is supported. Blindness
is the point: nothing this module serves may reveal a record id, an observation link or
the source site, a person, an exact date or coordinates, or Vision's answer. So it reads
only the packet the builder made (codes, names, hits, place to state or province, month)
and its own store, and imports nothing that identifies, logs or keeps answers (a test
checks the imports).

Packet (what `mv e17 load` accepts; nothing else is loaded):
    manifest.json                {"schema": "e17-packet/1", "count", "items", "photos"}
    items/<code>.json            one per item (ITEM_FIELDS below)
    photos/<code>/<n>.jpg        JPEG, no Exif/XMP (APP1), no IPTC (APP13), no comment
The pinned sha is sha256(manifest.json); every listed file's sha256 is checked, and an
unlisted or missing file, an extra field anywhere, or a string that looks like an id, a
link, a site or a full date (R1-R4) refuses the whole load. The loader prints the item
code, the field path and the rule, never the value.

Where it lives: MV_E17_DIR, else <release root>/state/e17-review on the box (kept across
releases), else beside the manifest. There: review.sqlite, manifest.json, items/, photos/.
Never the Human Feedback store, which is keyed by record.

Responses are insert-only (triggers refuse UPDATE and DELETE); the current answer for an
item is its latest row. `mv e17 export` writes latest.csv and history.csv; the page has no
download.

Serving (registered in api.py), behind a path token and the mycomap.org session:
    GET  /review/<token>                         the page (its own bundle, not the site shell)
    GET  /review/<token>/assets/<file>           the page's JS and CSS
    GET  /api/e17/<token>/items                  codes, positions, answered or not, count
    GET  /api/e17/<token>/item/<code>            the item's own fields + the latest response
    GET  /api/e17/<token>/photo/<code>/<n>       a photo
    POST /api/e17/<token>/response/<code>        a new response row; the sub is the session's
MV_E17_TOKEN_SHA256 (hex sha256 of the token) and MV_E17_REVIEWERS (mycomap.org subs,
comma-separated) must both be set and a packet loaded, or every e17 path is a 404. A wrong
token, no session, or a sub not on the list is the same plain 404.
"""

import argparse
import csv
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import config

DIR_NAME = "e17-review"
DB_FILE = "review.sqlite"
PACKET_SCHEMA = "e17-packet/1"
ITEM_SCHEMA = "e17-item/1"
MAX_PHOTO_BYTES = 8 * 1024 * 1024
MAX_PHOTOS = 40
MAX_HITS = 50
VERDICTS = ("supported", "wrong", "cannot_tell")
SHOWN_BY = ("sequences", "photographs", "both")
MAX_BETTER_NAME = 200
MAX_NOTES = 4000

# Every response the e17 paths send, whatever answers (the page, a 404, the sign-in gate).
HEADERS = {
    "Referrer-Policy": "no-referrer",
    "X-Robots-Tag": "noindex, nofollow",
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": ("default-src 'self'; img-src 'self'; connect-src 'self'; "
                                "style-src 'self'; script-src 'self'; base-uri 'none'; "
                                "form-action 'self'; frame-ancestors 'none'"),
}
PAGE_PREFIX = "/review/"
API_PREFIX = "/api/e17/"
NOT_FOUND_BODY = "Not found"

CODE_RE = re.compile(r"^[A-Z0-9]{5}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|"
          "December")
WHEN_RE = re.compile(rf"^(({MONTHS}) [0-9]{{4}}|not recorded)$")
ACCESSION_RE = re.compile(r"^[A-Z]{1,6}_?[0-9]{5,9}(\.[0-9]+)?$")
PHOTO_PATH_RE = re.compile(r"^photos/([A-Z0-9]{5})/([1-9][0-9]{0,2})\.jpg$")
ASSET_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}\.(js|css)$")

# The loader's second line behind the builder's own scan: applied to every string value of
# an item except "accession" and "code"; numbers are never scanned.
RULES = (
    ("R1", re.compile(r"\d{6,}")),
    ("R2", re.compile(r"\b(inat|mo|mc|org|obs)\s*[:#/]\s*\d", re.IGNORECASE)),
    ("R3", re.compile(r"inaturalist|mushroomobserver|mycomap|observation|https?://|www\.|@",
                      re.IGNORECASE)),
    ("R4", re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")),
)
UNSCANNED_KEYS = ("accession", "code")


def scan_string(value: str) -> str | None:
    """The first rule (R1-R4) a string breaks, or None."""
    for rule, pattern in RULES:
        if pattern.search(value):
            return rule
    return None


# --- settings --------------------------------------------------------------------------

def default_dir(manifest_path: Path | None = None) -> Path:
    """MV_E17_DIR; else <release root>/state/e17-review on the box; else beside the
    manifest (a laptop, a test's temporary manifest), as identifylog and feedback do."""
    override = config.setting("MV_E17_DIR")
    if override:
        return Path(override)
    if config.RELEASE_ROOT:
        return config.RELEASE_ROOT / "state" / DIR_NAME
    return Path(manifest_path or config.MANIFEST_PATH).parent / DIR_NAME


@dataclass(frozen=True)
class Settings:
    token_sha256: str | None
    reviewers: frozenset
    root: Path

    @property
    def enabled(self) -> bool:
        return bool(self.token_sha256 and HEX64_RE.match(self.token_sha256) and self.reviewers)

    @classmethod
    def from_settings(cls, manifest_path: Path | None = None) -> "Settings":
        reviewers = config.setting("MV_E17_REVIEWERS") or ""
        return cls(token_sha256=(config.setting("MV_E17_TOKEN_SHA256") or "").lower() or None,
                   reviewers=frozenset(s.strip() for s in reviewers.split(",") if s.strip()),
                   root=default_dir(manifest_path))

    def token_ok(self, token: str) -> bool:
        """A url-safe token of 32+ characters whose sha256 is the configured one, compared
        in constant time."""
        if not self.enabled or not isinstance(token, str) or not TOKEN_RE.match(token):
            return False
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        return hmac.compare_digest(digest, self.token_sha256 or "")


# --- the store -------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS items (
    code TEXT PRIMARY KEY,
    position INTEGER NOT NULL,
    packet_sha TEXT NOT NULL,
    loaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    sub TEXT NOT NULL,
    verdict TEXT,
    better_name TEXT,
    shown_by TEXT,
    recognised INTEGER NOT NULL DEFAULT 0,
    looked_up INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    client_saved_at TEXT,
    saved_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS responses_code ON responses (code, id);
CREATE TRIGGER IF NOT EXISTS responses_no_update BEFORE UPDATE ON responses
BEGIN SELECT RAISE(ABORT, 'e17 responses are insert-only'); END;
CREATE TRIGGER IF NOT EXISTS responses_no_delete BEFORE DELETE ON responses
BEGIN SELECT RAISE(ABORT, 'e17 responses are insert-only'); END;
"""

RESPONSE_FIELDS = ("verdict", "better_name", "shown_by", "recognised", "looked_up", "notes",
                   "client_saved_at", "saved_at")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Store:
    """review.sqlite with the packet's files beside it."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db = self.root / DB_FILE

    def connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        return conn

    def loaded(self) -> bool:
        return (self.root / "manifest.json").is_file() and self.db.is_file()

    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def packet_sha(self) -> str | None:
        p = self.root / "manifest.json"
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

    def positions(self) -> dict[str, int]:
        conn = self.connect()
        try:
            return {r["code"]: r["position"] for r in conn.execute("select code, position from items")}
        finally:
            conn.close()

    def item(self, code: str) -> dict | None:
        if not CODE_RE.match(code) or code not in self.positions():
            return None
        path = self.root / "items" / f"{code}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def photo(self, code: str, n: int) -> Path | None:
        if not CODE_RE.match(code) or code not in self.positions():
            return None
        path = self.root / "photos" / code / f"{n}.jpg"
        return path if path.is_file() else None

    def latest(self, code: str) -> dict | None:
        conn = self.connect()
        try:
            row = conn.execute("select * from responses where code = ? order by id desc limit 1",
                               (code,)).fetchone()
        finally:
            conn.close()
        return self._public(row) if row else None

    def answered(self) -> set[str]:
        """Codes whose latest response gives a verdict."""
        conn = self.connect()
        try:
            rows = conn.execute(
                "select r.code, r.verdict from responses r join (select code, max(id) id "
                "from responses group by code) m on m.id = r.id").fetchall()
        finally:
            conn.close()
        return {r["code"] for r in rows if r["verdict"]}

    def add(self, code: str, sub: str, answer: dict, saved_at: str | None = None) -> dict:
        conn = self.connect()
        try:
            with conn:
                cur = conn.execute(
                    "insert into responses (code, sub, verdict, better_name, shown_by, "
                    "recognised, looked_up, notes, client_saved_at, saved_at) "
                    "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (code, sub, answer.get("verdict"), answer.get("better_name"),
                     answer.get("shown_by"), int(bool(answer.get("recognised"))),
                     int(bool(answer.get("looked_up"))), answer.get("notes"),
                     answer.get("client_saved_at"), saved_at or utc_now()))
                row = conn.execute("select * from responses where id = ?",
                                   (cur.lastrowid,)).fetchone()
        finally:
            conn.close()
        return self._public(row)

    def history(self) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute("select * from responses order by id").fetchall()
        finally:
            conn.close()

    @staticmethod
    def _public(row) -> dict:
        """A response as the page gets it: never who saved it."""
        out = {k: row[k] for k in RESPONSE_FIELDS}
        out["recognised"] = bool(out["recognised"])
        out["looked_up"] = bool(out["looked_up"])
        return out


def clean_answer(body) -> dict | None:
    """A response body with exactly the page's fields and allowed values, or None."""
    allowed = {"verdict", "better_name", "shown_by", "recognised", "looked_up", "notes",
               "client_saved_at"}
    if not isinstance(body, dict) or set(body) - allowed:
        return None
    out = {}
    v = body.get("verdict")
    if v is not None and v not in VERDICTS:
        return None
    out["verdict"] = v
    s = body.get("shown_by")
    if s is not None and s not in SHOWN_BY:
        return None
    out["shown_by"] = s
    for key in ("recognised", "looked_up"):
        b = body.get(key, False)
        if not isinstance(b, bool):
            return None
        out[key] = b
    for key, cap in (("better_name", MAX_BETTER_NAME), ("notes", MAX_NOTES),
                     ("client_saved_at", 40)):
        t = body.get(key)
        if t is not None and (not isinstance(t, str) or len(t) > cap):
            return None
        out[key] = t if t else None
    return out


# --- the packet check ------------------------------------------------------------------

@dataclass(frozen=True)
class Refusal:
    code: str          # the item code, or "manifest" / "packet"
    path: str          # the field path (or file) within it
    rule: str

    def line(self) -> str:
        return f"{self.code}\t{self.path or '-'}\t{self.rule}"


class LoadRefused(Exception):
    def __init__(self, refusals: list[Refusal]):
        super().__init__(f"{len(refusals)} refusal(s)")
        self.refusals = refusals


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


class _Checker:
    def __init__(self, code: str):
        self.code = code
        self.out: list[Refusal] = []

    def no(self, path: str, rule: str) -> None:
        self.out.append(Refusal(self.code, path, rule))

    def obj(self, value, path: str, keys: tuple[str, ...]) -> bool:
        """An object with exactly these keys."""
        if not isinstance(value, dict):
            self.no(path, "schema:type")
            return False
        ok = True
        for k in value:
            if k not in keys:
                self.no(f"{path}.{_safe_key(k)}" if path else _safe_key(k), "schema:additional")
                ok = False
        for k in keys:
            if k not in value:
                self.no(f"{path}.{k}" if path else k, "schema:missing")
                ok = False
        return ok

    def string(self, value, path: str, lo: int = 1, hi: int = 200,
               pattern: re.Pattern | None = None) -> None:
        if not isinstance(value, str):
            self.no(path, "schema:type")
            return
        if not lo <= len(value) <= hi:
            self.no(path, "schema:length")
        if pattern is not None and not pattern.match(value):
            self.no(path, "schema:pattern")

    def integer(self, value, path: str, lo: int | None = None, hi: int | None = None,
                nullable: bool = False) -> None:
        if value is None and nullable:
            return
        if not _is_int(value):
            self.no(path, "schema:type")
        elif (lo is not None and value < lo) or (hi is not None and value > hi):
            self.no(path, "schema:range")

    def number(self, value, path: str, lo: float, hi: float) -> None:
        if not _is_num(value):
            self.no(path, "schema:type")
        elif not lo <= value <= hi:
            self.no(path, "schema:range")

    def array(self, value, path: str, hi: int) -> bool:
        if not isinstance(value, list):
            self.no(path, "schema:type")
            return False
        if len(value) > hi:
            self.no(path, "schema:length")
        return True


def _safe_key(k) -> str:
    """An unexpected key, named only when it is a plain short identifier (never a value)."""
    return k if isinstance(k, str) and re.fullmatch(r"[a-z_]{1,32}", k) else "<key>"


def check_manifest(m) -> list[Refusal]:
    c = _Checker("manifest")
    if not c.obj(m, "", ("schema", "count", "items", "photos")):
        return c.out
    if m["schema"] != PACKET_SCHEMA:
        c.no("schema", "schema:value")
    c.integer(m["count"], "count", lo=1)
    if c.array(m["items"], "items", hi=10_000):
        codes, positions = set(), set()
        for i, it in enumerate(m["items"]):
            p = f"items[{i}]"
            if not c.obj(it, p, ("code", "position", "sha256")):
                continue
            c.string(it["code"], f"{p}.code", 5, 5, CODE_RE)
            c.string(it["sha256"], f"{p}.sha256", 64, 64, HEX64_RE)
            c.integer(it["position"], f"{p}.position", lo=1,
                      hi=m["count"] if _is_int(m["count"]) else None)
            if it["code"] in codes:
                c.no(f"{p}.code", "schema:unique")
            if it["position"] in positions:
                c.no(f"{p}.position", "schema:unique")
            if isinstance(it["code"], str):
                codes.add(it["code"])
            if _is_int(it["position"]):
                positions.add(it["position"])
        if _is_int(m["count"]) and m["count"] != len(m["items"]):
            c.no("count", "schema:count")
    if c.array(m["photos"], "photos", hi=10_000 * MAX_PHOTOS):
        paths = set()
        for i, ph in enumerate(m["photos"]):
            p = f"photos[{i}]"
            if not c.obj(ph, p, ("path", "sha256")):
                continue
            c.string(ph["path"], f"{p}.path", 1, 64, PHOTO_PATH_RE)
            c.string(ph["sha256"], f"{p}.sha256", 64, 64, HEX64_RE)
            if ph["path"] in paths:
                c.no(f"{p}.path", "schema:unique")
            if isinstance(ph["path"], str):
                paths.add(ph["path"])
    return c.out


def check_item(item, code: str) -> list[Refusal]:
    """Schema v1.1 and R1-R4 for one item; `code` is the file's own name."""
    c = _Checker(code)
    keys = ("schema", "code", "name_under_review", "place", "when", "photos", "sequences")
    if not c.obj(item, "", keys):
        return c.out
    if item["schema"] != ITEM_SCHEMA:
        c.no("schema", "schema:value")
    c.string(item["code"], "code", 5, 5, CODE_RE)
    if item["code"] != code:
        c.no("code", "schema:file-name")
    c.string(item["name_under_review"], "name_under_review", 1, 200)
    c.string(item["place"], "place", 1, 80)
    c.string(item["when"], "when", 1, 40, WHEN_RE)
    if c.array(item["photos"], "photos", MAX_PHOTOS):
        for i, ph in enumerate(item["photos"]):
            p = f"photos[{i}]"
            if c.obj(ph, p, ("n", "width", "height")):
                c.integer(ph["n"], f"{p}.n", lo=1)
                if ph["n"] != i + 1:
                    c.no(f"{p}.n", "schema:contiguous")
                c.integer(ph["width"], f"{p}.width", lo=1, hi=100_000)
                c.integer(ph["height"], f"{p}.height", lo=1, hi=100_000)
    if c.array(item["sequences"], "sequences", 1000):
        for i, seq in enumerate(item["sequences"]):
            p = f"sequences[{i}]"
            if not c.obj(seq, p, ("n", "withdrawn", "length", "local", "genbank")):
                continue
            c.integer(seq["n"], f"{p}.n", lo=1)
            if seq["n"] != i + 1:
                c.no(f"{p}.n", "schema:contiguous")
            if not isinstance(seq["withdrawn"], bool):
                c.no(f"{p}.withdrawn", "schema:type")
            c.integer(seq["length"], f"{p}.length", lo=1, hi=10_000_000, nullable=True)
            for table, extra in (("local", ()), ("genbank", ("accession",))):
                tp = f"{p}.{table}"
                if not c.array(seq[table], tp, MAX_HITS):
                    continue
                seen = set()
                for j, hit in enumerate(seq[table]):
                    hp = f"{tp}[{j}]"
                    if not c.obj(hit, hp, ("hit_no", "name", "identity", "coverage", *extra)):
                        continue
                    c.integer(hit["hit_no"], f"{hp}.hit_no", lo=1)
                    if hit["hit_no"] in seen:
                        c.no(f"{hp}.hit_no", "schema:unique")
                    if _is_int(hit["hit_no"]):
                        seen.add(hit["hit_no"])
                    c.string(hit["name"], f"{hp}.name", 1, 200)
                    c.number(hit["identity"], f"{hp}.identity", 0, 100)
                    c.number(hit["coverage"], f"{hp}.coverage", 0, 100)
                    if extra:
                        c.string(hit["accession"], f"{hp}.accession", 1, 20, ACCESSION_RE)
    _scan(item, "", c)
    return c.out


def _scan(value, path: str, c: _Checker) -> None:
    """R1-R4 over every string value except accession and code; numbers never."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in UNSCANNED_KEYS:
                continue
            _scan(v, f"{path}.{_safe_key(k)}" if path else _safe_key(k), c)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _scan(v, f"{path}[{i}]", c)
    elif isinstance(value, str):
        rule = scan_string(value)
        if rule:
            c.no(path, rule)


def jpeg_check(data: bytes) -> tuple[list[str], tuple[int, int] | None]:
    """Problems with a photo's bytes (rule names) and its (width, height) from the frame
    header: JPEG only, no APP1 (Exif, XMP: where GPS lives), no APP13 (IPTC), no comment,
    at most 8 MB."""
    problems: list[str] = []
    if len(data) > MAX_PHOTO_BYTES:
        problems.append("photo:bytes")
    if data[:2] != b"\xff\xd8":
        return problems + ["photo:jpeg"], None
    size = None
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            return problems + ["photo:jpeg"], size
        while i < n and data[i] == 0xFF:      # fill bytes
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:                    # end of image before a scan
            break
        if i + 2 > n:
            return problems + ["photo:jpeg"], size
        length = int.from_bytes(data[i:i + 2], "big")
        if length < 2 or i + length > n:
            return problems + ["photo:jpeg"], size
        if marker == 0xE1:
            problems.append("photo:app1-exif-xmp")
        elif marker == 0xED:
            problems.append("photo:app13-iptc")
        elif marker == 0xFE:
            problems.append("photo:comment")
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC) and length >= 7:
            h = int.from_bytes(data[i + 3:i + 5], "big")
            w = int.from_bytes(data[i + 5:i + 7], "big")
            size = (w, h)
        if marker == 0xDA:                    # start of scan: no more headers to read
            break
        i += length
    if size is None:
        problems.append("photo:jpeg")
    return sorted(set(problems)), size


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Packet:
    root: Path
    sha: str
    manifest: dict
    items: dict[str, dict]


def check_packet(packet: Path, pinned_sha: str) -> Packet:
    """The whole packet checked, or LoadRefused with every refusal found."""
    packet = Path(packet)
    mpath = packet / "manifest.json"
    pinned = (pinned_sha or "").strip().lower()
    if not mpath.is_file() or mpath.is_symlink():
        raise LoadRefused([Refusal("manifest", "manifest.json", "file:missing")])
    sha = sha256_file(mpath)
    if not HEX64_RE.match(pinned) or not hmac.compare_digest(sha, pinned):
        raise LoadRefused([Refusal("manifest", "manifest.json", "sha:pinned")])
    try:
        manifest = _load_json(mpath)
    except ValueError:
        raise LoadRefused([Refusal("manifest", "manifest.json", "json")])
    out = check_manifest(manifest)
    if out:
        raise LoadRefused(out)

    listed: dict[str, tuple[str, str]] = {"manifest.json": ("manifest", "")}
    for it in manifest["items"]:
        listed[f"items/{it['code']}.json"] = (it["code"], it["sha256"])
    for ph in manifest["photos"]:
        listed[ph["path"]] = (PHOTO_PATH_RE.match(ph["path"]).group(1), ph["sha256"])

    present: set[str] = set()
    unlisted = 0
    for dirpath, dirnames, filenames in os.walk(packet, followlinks=False):
        for name in list(dirnames):
            if (Path(dirpath) / name).is_symlink():
                out.append(Refusal("packet", "<link>", "file:not-regular"))
                dirnames.remove(name)
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(packet).as_posix()
            if rel not in listed:
                unlisted += 1
                top = rel.split("/", 1)[0] if "/" in rel else ""
                out.append(Refusal("packet", f"unlisted file {unlisted}"
                                   + (f" under {_safe_key(top)}/" if top else ""),
                                   "file:unlisted"))
            elif full.is_symlink() or not full.is_file():
                out.append(Refusal(listed[rel][0], rel, "file:not-regular"))
            else:
                present.add(rel)
    for rel, (code, _) in listed.items():
        if rel not in present and not any(r.path == rel for r in out):
            out.append(Refusal(code, rel, "file:missing"))
    for rel, (code, want) in listed.items():
        if rel != "manifest.json" and rel in present and sha256_file(packet / rel) != want:
            out.append(Refusal(code, rel, "sha:file"))
    if out:
        raise LoadRefused(out)

    items: dict[str, dict] = {}
    for it in manifest["items"]:
        code = it["code"]
        try:
            item = _load_json(packet / "items" / f"{code}.json")
        except ValueError:
            out.append(Refusal(code, "", "json"))
            continue
        problems = check_item(item, code)
        out.extend(problems)
        if not problems:
            items[code] = item
    if out:
        raise LoadRefused(out)

    # The photos the items list are exactly the photos the manifest lists.
    want = {f"photos/{code}/{ph['n']}.jpg": (code, i, ph)
            for code, item in items.items() for i, ph in enumerate(item["photos"])}
    for ph in manifest["photos"]:
        if ph["path"] not in want:
            out.append(Refusal(PHOTO_PATH_RE.match(ph["path"]).group(1), ph["path"],
                               "photo:not-in-item"))
    listed_photos = {ph["path"] for ph in manifest["photos"]}
    for path, (code, i, ph) in want.items():
        if path not in listed_photos:
            out.append(Refusal(code, f"photos[{i}]", "photo:not-in-manifest"))
            continue
        problems, size = jpeg_check((packet / path).read_bytes())
        for rule in problems:
            out.append(Refusal(code, f"photos[{i}]", rule))
        if size is not None and size != (ph["width"], ph["height"]):
            out.append(Refusal(code, f"photos[{i}]", "photo:dimensions"))
    if out:
        raise LoadRefused(out)
    return Packet(packet, sha, manifest, items)


def _load_json(path: Path):
    """Strict JSON: UTF-8, no NaN or Infinity, no duplicate keys."""
    def no_constants(name):
        raise ValueError(name)

    def no_duplicates(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate key")
            out[k] = v
        return out
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("utf-8") from e
    return json.loads(text, parse_constant=no_constants, object_pairs_hook=no_duplicates)


# --- loading ---------------------------------------------------------------------------

LIVE = ("manifest.json", "items", "photos")


def load(packet_dir: Path, pinned_sha: str, root: Path, replace: bool = False,
         now: Callable[[], str] = utc_now) -> tuple[str, Packet]:
    """Check the packet, then copy it into `root` and list its items. Returns ("loaded" |
    "unchanged", packet). Any refusal raises LoadRefused before anything live changes.
    A packet already loaded is kept unless `replace` (responses stay, keyed by code)."""
    packet = check_packet(Path(packet_dir), pinned_sha)
    store = Store(root)
    current = store.packet_sha()
    if current == packet.sha:
        return "unchanged", packet
    if current is not None and not replace:
        raise LoadRefused([Refusal("packet", "", "loaded:another-packet (use --replace)")])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tag = secrets.token_hex(6)
    stage, old = root / f".load-{tag}", root / f".old-{tag}"
    try:
        (stage / "items").mkdir(parents=True)
        (stage / "photos").mkdir()
        shutil.copyfile(packet.root / "manifest.json", stage / "manifest.json")
        for code in packet.items:
            shutil.copyfile(packet.root / "items" / f"{code}.json", stage / "items" / f"{code}.json")
            (stage / "photos" / code).mkdir()
        for ph in packet.manifest["photos"]:
            shutil.copyfile(packet.root / ph["path"], stage / ph["path"])
        # The copies are checked again before they go live.
        for it in packet.manifest["items"]:
            if sha256_file(stage / "items" / f"{it['code']}.json") != it["sha256"]:
                raise LoadRefused([Refusal(it["code"], "", "sha:copy")])
        for ph in packet.manifest["photos"]:
            if sha256_file(stage / ph["path"]) != ph["sha256"]:
                raise LoadRefused([Refusal(PHOTO_PATH_RE.match(ph["path"]).group(1),
                                           ph["path"], "sha:copy")])
        old.mkdir()
        moved: list[str] = []
        conn = store.connect()
        try:
            for name in LIVE:
                if (root / name).exists():
                    os.replace(root / name, old / name)
                moved.append(name)
                os.replace(stage / name, root / name)
            loaded_at = now()
            with conn:
                conn.execute("delete from items")
                conn.executemany(
                    "insert into items (code, position, packet_sha, loaded_at) values (?, ?, ?, ?)",
                    [(it["code"], it["position"], packet.sha, loaded_at)
                     for it in packet.manifest["items"]])
        except BaseException:
            for name in reversed(moved):        # put the previous packet back
                _remove(root / name)
                if (old / name).exists():
                    os.replace(old / name, root / name)
            raise
        finally:
            conn.close()
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    return "loaded", packet


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


# --- export ----------------------------------------------------------------------------

EXPORT_COLUMNS = ("code", "position", "sub", "verdict", "better_name", "shown_by",
                  "recognised", "looked_up", "notes", "client_saved_at", "saved_at")


def export(root: Path, out_dir: Path) -> tuple[int, int]:
    """latest.csv (one row per loaded code: its current answer, blank when none) and
    history.csv (every response row, oldest first). Returns their row counts."""
    store = Store(root)
    positions = store.positions()
    rows = store.history()
    latest: dict[str, sqlite3.Row] = {}
    for r in rows:
        latest[r["code"]] = r
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def as_row(code: str, r) -> list:
        if r is None:
            return [code, positions.get(code, "")] + [""] * (len(EXPORT_COLUMNS) - 2)
        return [code, positions.get(code, ""), r["sub"], r["verdict"] or "",
                r["better_name"] or "", r["shown_by"] or "", int(r["recognised"]),
                int(r["looked_up"]), r["notes"] or "", r["client_saved_at"] or "", r["saved_at"]]

    codes = sorted(set(positions) | set(latest), key=lambda c: (positions.get(c, 10**9), c))
    with open(out_dir / "latest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EXPORT_COLUMNS)
        for code in codes:
            w.writerow(as_row(code, latest.get(code)))
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("id",) + EXPORT_COLUMNS)
        for r in rows:
            w.writerow([r["id"]] + as_row(r["code"], r))
    return len(codes), len(rows)


# --- serving ---------------------------------------------------------------------------

def is_e17_path(path: str) -> bool:
    return path.startswith(PAGE_PREFIX) or path == PAGE_PREFIX.rstrip("/") \
        or path.startswith(API_PREFIX) or path == API_PREFIX.rstrip("/")


class TokenRedactor(logging.Filter):
    """Keeps the review link's token out of the server's access log."""
    _re = re.compile(r"^(/review/|/api/e17/)[^/?]+")

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str) \
                and self._re.match(args[2]):
            record.args = (*args[:2], self._re.sub(r"\1(token)", args[2]), *args[3:])
        return True


def redact_access_log() -> None:
    log = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, TokenRedactor) for f in log.filters):
        log.addFilter(TokenRedactor())


def page_html(token: str, entry: dict) -> str:
    """The page's own document: no site shell, no fonts, nothing from another host."""
    base = f"/review/{token}/assets/"
    css = "".join(f'<link rel="stylesheet" href="{base}{Path(c).name}">'
                  for c in entry.get("css", []))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta name=\"robots\" content=\"noindex,nofollow\">"
            "<meta name=\"referrer\" content=\"no-referrer\">"
            f"<title>Review</title>{css}"
            f"<script type=\"module\" src=\"{base}{Path(entry['file']).name}\"></script>"
            "</head><body><div id=\"root\"></div></body></html>")


def bundle_entry(bundle_dir: Path) -> dict | None:
    """The page's built entry from Vite's manifest (web/dist/e17review/.vite/manifest.json)."""
    path = bundle_dir / ".vite" / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    for entry in manifest.values():
        if isinstance(entry, dict) and entry.get("isEntry") and entry.get("file"):
            return entry
    return None


BUNDLE_DIR = "e17review"


def register(app, *, settings: Settings, person: Callable, web_dist: Path | None) -> None:
    """(No `from __future__ import annotations` in this module: FastAPI reads the routes'
    `Request` annotations, imported here, as real objects.)

    The e17 routes and their headers. Every path under /review/ and /api/e17/ answers
    here (a catch-all 404 included), so none falls through to the site's shell."""
    from fastapi import Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
    from starlette.concurrency import run_in_threadpool

    store = Store(settings.root)
    bundle = (Path(web_dist) / BUNDLE_DIR) if web_dist else None

    def not_found():
        return PlainTextResponse(NOT_FOUND_BODY, status_code=404, headers=HEADERS)

    def reviewer(request: Request, token: str) -> str | None:
        """The reviewer's sub when the token, the session and the list all agree."""
        if not settings.enabled or not settings.token_ok(token):
            return None
        p = person(request)
        sub = str(p.get("sub") or "") if isinstance(p, dict) else ""
        if not sub or sub not in settings.reviewers:
            return None
        if not store.loaded():
            return None
        return sub

    @app.middleware("http")
    async def e17_headers(request: Request, call_next):
        response = await call_next(request)
        if is_e17_path(request.url.path):
            for k, v in HEADERS.items():
                response.headers[k] = v
        return response

    @app.get("/review/{token}", include_in_schema=False)
    def e17_page(request: Request, token: str):
        if reviewer(request, token) is None or bundle is None:
            return not_found()
        entry = bundle_entry(bundle)
        if entry is None:
            return not_found()
        return HTMLResponse(page_html(token, entry), headers=HEADERS)

    @app.get("/review/{token}/assets/{name}", include_in_schema=False)
    def e17_asset(request: Request, token: str, name: str):
        if reviewer(request, token) is None or bundle is None or not ASSET_RE.match(name):
            return not_found()
        path = bundle / "assets" / name
        if not path.is_file():
            return not_found()
        media = "text/javascript" if name.endswith(".js") else "text/css"
        return FileResponse(path, media_type=media, headers=HEADERS)

    @app.get("/api/e17/{token}/items", include_in_schema=False)
    def e17_items(request: Request, token: str):
        if reviewer(request, token) is None:
            return not_found()
        positions = store.positions()
        answered = store.answered()
        items = [{"code": c, "position": p, "answered": c in answered}
                 for c, p in sorted(positions.items(), key=lambda kv: kv[1])]
        return JSONResponse({"count": store.manifest()["count"], "items": items},
                            headers=HEADERS)

    @app.get("/api/e17/{token}/item/{code}", include_in_schema=False)
    def e17_item(request: Request, token: str, code: str):
        if reviewer(request, token) is None:
            return not_found()
        item = store.item(code)
        if item is None:
            return not_found()
        return JSONResponse({**item, "response": store.latest(code)}, headers=HEADERS)

    @app.get("/api/e17/{token}/photo/{code}/{n}", include_in_schema=False)
    def e17_photo(request: Request, token: str, code: str, n: str):
        if reviewer(request, token) is None or not re.fullmatch(r"[1-9][0-9]{0,2}", n):
            return not_found()
        path = store.photo(code, int(n))
        if path is None:
            return not_found()
        return FileResponse(path, media_type="image/jpeg", headers=HEADERS)

    @app.post("/api/e17/{token}/response/{code}", include_in_schema=False)
    async def e17_respond(request: Request, token: str, code: str):
        sub = reviewer(request, token)
        if sub is None or not CODE_RE.match(code) or code not in store.positions():
            return not_found()
        if not (request.headers.get("content-type") or "").startswith("application/json"):
            return JSONResponse({"detail": "send JSON"}, status_code=415, headers=HEADERS)
        try:
            body = await request.json()
        except ValueError:
            body = None
        answer = clean_answer(body)
        if answer is None:
            return JSONResponse({"detail": "malformed response"}, status_code=400,
                                headers=HEADERS)
        row = await run_in_threadpool(store.add, code, sub, answer)
        return JSONResponse({"ok": True, "response": row}, headers=HEADERS)

    @app.api_route("/review/{rest:path}", include_in_schema=False,
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def e17_page_other(rest: str):
        return not_found()

    @app.api_route("/api/e17/{rest:path}", include_in_schema=False,
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def e17_api_other(rest: str):
        return not_found()

    @app.api_route("/review", include_in_schema=False,
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def e17_review_root():
        return not_found()


# --- the command line ------------------------------------------------------------------

def add_commands(sub) -> None:
    p = sub.add_parser("e17", help="E17 blind review page: load a packet, export responses "
                                   "(e17review.py)")
    esub = p.add_subparsers(dest="action", required=True)
    q = esub.add_parser("load", help="check a packet and copy it in (all or nothing)")
    q.add_argument("packet", help="the packet folder (manifest.json, items/, photos/)")
    q.add_argument("--sha", required=True, help="the pinned sha256 of manifest.json")
    q.add_argument("--replace", action="store_true",
                   help="replace a different packet already loaded (responses are kept)")
    q = esub.add_parser("export", help="write latest.csv and history.csv")
    q.add_argument("out", help="the folder to write them to")
    esub.add_parser("status", help="where the store is, items loaded and answered")
    esub.add_parser("new-token", help="a fresh path token and the sha256 to set as "
                                      "MV_E17_TOKEN_SHA256")


def run_cli(args, out=print, err=None) -> int:
    err = err or (lambda s: print(s, file=sys.stderr))
    root = default_dir()
    if args.action == "load":
        try:
            state, packet = load(Path(args.packet), args.sha, root, replace=args.replace)
        except LoadRefused as e:
            err(f"refused: nothing was loaded ({len(e.refusals)} problem(s)); "
                "code, field path, rule:")
            for r in e.refusals:
                err("  " + r.line())
            return 1
        out(f"{state}: {len(packet.items)} items, {len(packet.manifest['photos'])} photos, "
            f"packet {packet.sha[:12]} in {root}")
        return 0
    if args.action == "export":
        n_latest, n_history = export(root, Path(args.out))
        out(f"wrote latest.csv ({n_latest} rows) and history.csv ({n_history} rows) "
            f"to {args.out}")
        return 0
    if args.action == "status":
        store = Store(root)
        if not store.loaded():
            out(f"no packet loaded in {root}")
            return 0
        out(f"{root}: packet {store.packet_sha()[:12]}, {len(store.positions())} items, "
            f"{len(store.answered())} answered, {len(store.history())} response rows")
        return 0
    if args.action == "new-token":
        token = secrets.token_urlsafe(32)
        out(f"token (the link is /review/<token>; keep it out of chat and email): {token}")
        out(f"MV_E17_TOKEN_SHA256={hashlib.sha256(token.encode('ascii')).hexdigest()}")
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mv")
    add_commands(parser.add_subparsers(dest="command", required=True))
    return run_cli(parser.parse_args(argv))
