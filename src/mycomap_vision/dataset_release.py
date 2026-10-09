"""Dataset releases: a frozen, verifiable snapshot of the data, for research.

The full specification is docs/dataset-release.md; read it first. In short:

- A release (`v1`, `v2`, ...; a draft is `dry-<stamp>`) is a folder under
  data/research-releases/<id>/ holding release.sqlite (every candidate record and photo,
  in or out with one reason, labels with provenance, splits, derivation recipes and the
  hashes of derived images), private.sqlite (coordinates, owners' numeric ids, where the
  originals are), MANIFEST.json (written last), SHA256SUMS and a dataset card.
- It is never edited: files are read-only, the reader opens SQLite immutable, a build
  refuses an existing id, and `verify` recomputes every hash.
- `release_hash` depends only on content, so the same inputs give the same hash and any
  change to a record, label, photo, inclusion, split or recipe changes it.
- Photos are not copied: a release records each original's sha256 and deterministic
  recipes; `derive` makes a derived image from our original and checks its stored hash.
- `vN-cc` is the public variant: CC photos only, no coordinates, built from `vN`.

A dataset release is not a serving release (release.py, `mv release`). Building reads a
private copy of the manifest and never writes to it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from . import config

FORMAT = 1
ROOT = config.DATA_DIR / "research-releases"
FROZEN_ID = re.compile(r"^v[1-9][0-9]{0,3}(?:-(?:cc|test[0-9]{0,2}))?\Z")
DRAFT_ID = re.compile(r"^dry-[0-9]{8}-[0-9]{6}(?:-(?:cc|test[0-9]{0,2}))?\Z")
LABEL_RULE = ("2026-10-09: the iNat Species Name Override beats the Provisional Species Name; "
              "the MycoBank number is looked up from that name; name and number agree")
REQUIRED_MIGRATION = "record-sources-v1"     # sources.MIGRATION (feat/record-sources-mo)
PHOTO_SOURCES = ("inat", "mo")
CC_CLASSES = ("open", "nc")                  # licenses.license_class: every CC licence
SPLITS = ("train", "val", "test")
TRAINABLE = ("train", "val")
DEFAULT_VAL_WEEKS = 8

# The order reasons are tried in: a record or photo gets the first that applies.
RECORD_REASONS = ("source_mycoportal", "source_com_sequence", "source_genbank",
                  "source_unknown", "held_out", "missing_at_source", "not_north_america",
                  "label_conflict", "no_label", "guest", "review", "no_photos", "ok")
PHOTO_REASONS = ("record_excluded", "no_original", "permission_withdrawn", "review", "ok")

MANIFEST = "MANIFEST.json"
SUMS = "SHA256SUMS"
CARD = "DATASET_CARD.md"
RELEASE_DB = "release.sqlite"
PRIVATE_DB = "private.sqlite"


class ReleaseError(RuntimeError):
    """A release is missing, unfinished, altered, or asked for something it refuses."""


class SealedLeak(ReleaseError):
    """Records that may not be trained on were asked for as training records."""


# --- recipes --------------------------------------------------------------------------------

def library_versions() -> dict:
    import PIL
    from PIL import features
    return {"pillow": PIL.__version__, "libjpeg": features.version("jpg") or "unknown"}


def _recipe(name: str, long_side: int | None, quality: int | None) -> dict:
    steps = ["identity"] if long_side is None else [
        "exif_transpose", "convert:RGB", f"resize_long_side:{long_side}:lanczos:no_upscale",
        f"jpeg:quality={quality}:subsampling=4:2:0:baseline:no_metadata"]
    return {"name": name, "version": 1, "steps": steps, "long_side": long_side,
            "quality": quality}


# Immutable once a release has used them: a change is a new name.
RECIPES = {
    "original": _recipe("original", None, None),
    # The usual public size (iNat "medium", FungiTastic 500p).
    "long500-q90": _recipe("long500-q90", 500, 90),
}


def recipe_with_library(name: str) -> dict:
    if name not in RECIPES:
        raise ReleaseError(f"unknown recipe {name!r}; known: {', '.join(RECIPES)}")
    r = dict(RECIPES[name])
    if r["long_side"] is not None:
        r["library"] = library_versions()
    return r


@dataclass(frozen=True)
class Derived:
    body: bytes
    sha256: str
    pixels_sha256: str | None
    width: int | None
    height: int | None


def derive_bytes(original: bytes, recipe: dict) -> Derived:
    """The derived image a recipe makes from an original: the same bytes every time with
    the same library versions (no randomness, no metadata, fixed encoder settings)."""
    if recipe["long_side"] is None:
        return Derived(original, sha256_bytes(original), None, None, None)
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(original))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    side = recipe["long_side"]
    if max(w, h) > side:
        size = (side, max(1, round(h * side / w))) if w >= h else \
            (max(1, round(w * side / h)), side)
        img = img.resize(size, Image.Resampling.LANCZOS, reducing_gap=None)
    img.info = {}
    w, h = img.size
    pixels = sha256_bytes(f"{w}x{h}\n".encode() + img.tobytes())
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=recipe["quality"], subsampling=2, optimize=False,
             progressive=False)
    body = buf.getvalue()
    return Derived(body, sha256_bytes(body), pixels, w, h)


# --- hashing --------------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_line(row: dict) -> bytes:
    for k, v in row.items():
        if isinstance(v, float):
            raise ReleaseError(f"hashed column {k!r} holds a float ({v}); store text or integers")
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def stream_hash(name: str, rows) -> str:
    h = hashlib.sha256(f"{name}:{FORMAT}\n".encode())
    for row in rows:
        h.update(canonical_line(row))
    return h.hexdigest()


def table_rows(conn: sqlite3.Connection, table: str, order_by: str, columns: list[str] | None = None,
               where: str = ""):
    cols = columns or [r[1] for r in conn.execute(f'pragma table_info("{table}")')]
    sel = ", ".join(f'"{c}"' for c in cols)
    for row in conn.execute(f'select {sel} from "{table}" {where} order by {order_by}'):
        yield dict(zip(cols, row))


# What each component hash covers, in the order release_hash lists them.
COMPONENTS = ("records", "labels", "photos", "splits", "recipes", "derived", "private")


def content_hashes(db: sqlite3.Connection, private: sqlite3.Connection) -> dict[str, str]:
    out = {
        "records": stream_hash("records", table_rows(db, "records", "record_key")),
        "labels": stream_hash("labels", table_rows(
            db, "records", "record_key", ["record_key", "label", "label_rank", "mycobank_number"],
            "where included = 1")),
        "photos": stream_hash("photos", table_rows(db, "photos", "photo_key")),
        "splits": stream_hash("splits", table_rows(db, "splits", "split")),
        "recipes": stream_hash("recipes", table_rows(db, "recipes", "name")),
        "derived": stream_hash("derived", table_rows(db, "derived", "photo_key, recipe")),
    }
    h = hashlib.sha256(f"private:{FORMAT}\n".encode())
    for table in sorted(r[0] for r in private.execute(
            "select name from sqlite_master where type = 'table'")):
        h.update(f"table:{table}\n".encode())
        first = [r[1] for r in private.execute(f'pragma table_info("{table}")')][0]
        for row in table_rows(private, table, f'"{first}"'):
            h.update(canonical_line(row))
    out["private"] = h.hexdigest()
    out["reference"] = reference_hash(db)
    out["release"] = release_hash(out)
    return out


def release_hash(hashes: dict[str, str]) -> str:
    return sha256_bytes("".join(f"{n}={hashes[n]}\n" for n in COMPONENTS).encode())


def reference_hash(db: sqlite3.Connection) -> str:
    """The reference index's records, labels and photos: train + val, each record's key,
    label and the sorted original sha256s of its included photos."""
    shas: dict[str, list[str]] = defaultdict(list)
    for key, sha in db.execute("select record_key, original_sha256 from photos where included = 1"):
        shas[key].append(sha)
    rows = ({"record_key": k, "label": lab, "photos": sorted(shas[k])}
            for k, lab in db.execute(
                "select record_key, label from records where included = 1 and split in "
                "('train', 'val') order by record_key"))
    return stream_hash("reference", rows)


# --- schema ---------------------------------------------------------------------------------

RELEASE_SCHEMA = """
create table info (key text primary key, value text not null);
create table records (
  record_key text primary key, source text not null, source_id text, org_source text,
  org_name text, mycobank_number text, label text, label_rank text,
  species text, genus text, family text, name_kind text,
  label_rule text not null, label_exported_at text,
  validated_on text, green_projects text, observed_on text, country text, state text,
  north_america integer not null, observer_login text, inat_uuid text,
  included integer not null, reason text not null, split text,
  benchmark text, benchmark_split text
);
create table photos (
  photo_key text primary key, manifest_photo_id integer not null, record_key text not null,
  position integer, source text not null, source_photo_id integer, source_url text,
  original_path text, original_sha256 text, original_bytes integer,
  license_code text, license_class text not null, license_checked_at text,
  owner_login text, owner_name text, attribution text,
  included integer not null, reason text not null
);
create index photos_record on photos(record_key);
create table splits (split text primary key, rule text not null, records integer not null,
                     photos integer not null);
create table recipes (name text primary key, recipe_json text not null, sha256 text not null);
create table derived (photo_key text not null, recipe text not null, derived_sha256 text not null,
                      pixels_sha256 text, width integer, height integer,
                      primary key (photo_key, recipe));
"""

PRIVATE_SCHEMA = """
create table record_places (record_key text primary key, lat_micro integer, lon_micro integer);
create table photo_owners (photo_key text primary key, owner_user_id integer,
                           source_owner_id integer, permission text);
create table stores (store_id integer primary key, location text not null);
create table photo_stores (photo_key text primary key, store_id integer not null);
"""


# --- reading the manifest -------------------------------------------------------------------

def _columns(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f'pragma table_info("{table}")')}


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("select name from sqlite_master where type = 'table'")}


def check_migrated(conn: sqlite3.Connection) -> None:
    """A release needs the explicit record sources (feat/record-sources-mo): before that
    migration a numeric id was guessed to be iNat's and MO/MyCoPortal records carried
    unrelated iNat photos."""
    ok = "manifest_migrations" in _tables(conn) and conn.execute(
        "select 1 from manifest_migrations where name = ?", (REQUIRED_MIGRATION,)).fetchone()
    if not ok or not {"source_id", "org_source"} <= _columns(conn, "records"):
        raise ReleaseError(f"the manifest has not had the {REQUIRED_MIGRATION} migration "
                           "(explicit record sources); a release cannot be cut from it")


def micro(v) -> int | None:
    return None if v is None else int(round(float(v) * 1_000_000))


def name_kind(name: str) -> str:
    from .names import parse_name
    words = (name or "").split()
    if len(words) <= 1:
        return "one-word"
    return "provisional" if parse_name(name).code else "formal"


def read_review_lists(paths: list[Path]) -> tuple[dict, dict, list[dict]]:
    """TSV files with kind (record|photo), key, reason, note -> {key: 'review:<list>'}
    for records and photos, and each list's name, path and sha256."""
    recs: dict[str, str] = {}
    photos: dict[str, str] = {}
    lists = []
    for p in paths:
        p = Path(p)
        name = re.sub(r"[^a-z0-9-]+", "-", p.stem.lower()).strip("-")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for r in rows:
            kind, key = (r.get("kind") or "").strip(), (r.get("key") or "").strip()
            if kind not in ("record", "photo") or not key:
                raise ReleaseError(f"{p.name}: every row needs kind record|photo and a key")
            (recs if kind == "record" else photos).setdefault(key, f"review:{name}")
        lists.append({"name": name, "file": p.name, "sha256": sha256_file(p), "rows": len(rows)})
    return recs, photos, lists


@dataclass
class Built:
    records: list[dict]
    photos: list[dict]
    places: list[dict]
    owners: list[dict]
    stores: dict[str, int]
    photo_stores: list[dict]
    info: dict


def _photo_key(source: str, manifest_id: int, source_photo_id) -> str:
    if source == "inat":
        return f"inat:{manifest_id}"
    return f"{source}:{source_photo_id if source_photo_id is not None else manifest_id}"


def collect(conn: sqlite3.Connection, *, review_records: dict | None = None,
            review_photos: dict | None = None, north_america_only: bool = True,
            only_keys: set[str] | None = None) -> Built:
    """Every candidate record and photo from a manifest (a private copy), with labels,
    inclusion reasons and photos' originals. Splits are assigned afterwards."""
    from . import guests, holdouts, names, taxonomy
    from .evaluate import clean
    from .permissions import EXCLUDED_FROM_USE_SQL
    from .permissions import SCHEMA as PERMISSIONS_SCHEMA
    conn.executescript(PERMISSIONS_SCHEMA)
    holdouts.ensure_schema(conn)
    review_records, review_photos = review_records or {}, review_photos or {}
    tables = _tables(conn)
    has_mo = "mo_observations" in tables
    rcols = _columns(conn, "records")
    pcols = _columns(conn, "photos")
    mb = "r.mycobank_number" if "mycobank_number" in rcols else "null"
    mo_sel = ("mo.status, mo.owner_login" if has_mo else "null, null")
    mo_join = ("left join mo_observations mo on mo.observation_id = r.observation_id"
               if has_mo else "")
    held = holdouts.held_out_ids(conn)
    bench: dict[str, tuple[str, str]] = {}
    if "heldout_records" in tables:
        for b, oid, split in conn.execute(
                "select benchmark, observation_id, split from heldout_records order by benchmark"):
            bench.setdefault(oid, (b, split))
    labels = names.manifest_labels(conn)
    tax = taxonomy.for_manifest(conn)

    records, places, keep = [], [], {}
    for row in conn.execute(f"""
        select r.observation_id, r.source, r.source_id, r.org_source, r.scientific_name, {mb},
               r.genus, r.family, r.exported_at, r.validated_on, r.green_projects,
               r.observed_on, r.country, r.state, r.north_america, r.label_conflict,
               r.latitude, r.longitude, o.status, o.user_login, o.uuid, {mo_sel}
        from records r
        left join inat_observations o on o.observation_id = r.observation_id
        {mo_join}
        order by r.observation_id"""):
        (key, source, source_id, org_source, name, mbn, genus, family, exported, vdate, projects,
         observed, country, state, na, conflict, lat, lon, inat_status, inat_login, uuid,
         mo_status, mo_login) = row
        if only_keys is not None and key not in only_keys:
            continue
        name = clean(name)
        label = clean(labels.get(name, name)) if name else ""
        lab = taxonomy.labels_for(label, genus, family, label != name, tax) if label else None
        rank = ("species" if lab and lab.species else "genus" if lab and lab.genus
                else "family" if lab and lab.family else None)
        status = inat_status if source == "inat" else mo_status if source == "mo" else None
        if source not in PHOTO_SOURCES:
            reason = f"source_{source}" if f"source_{source}" in RECORD_REASONS \
                else "source_unknown"
        elif key in held:
            reason = "held_out"
        elif status != "ok":
            reason = "missing_at_source"
        elif north_america_only and not na:
            reason = "not_north_america"
        elif conflict:
            reason = "label_conflict"
        elif not rank:
            reason = "no_label"
        elif guests.excluded(lab.genus):
            reason = "guest"
        elif key in review_records:
            reason = review_records[key]
        else:
            reason = "ok"
        b = bench.get(key, (None, None))
        records.append({
            "record_key": key, "source": source, "source_id": source_id, "org_source": org_source,
            "org_name": name or None, "mycobank_number": None if mbn is None else str(mbn),
            "label": (lab.unit if lab else None) or None, "label_rank": rank,
            "species": (lab.species if lab else None) or None,
            "genus": (lab.genus if lab else None) or None,
            "family": (lab.family if lab else None) or None,
            "name_kind": name_kind(label) if label else None,
            "label_rule": LABEL_RULE, "label_exported_at": exported,
            "validated_on": vdate, "green_projects": projects, "observed_on": observed,
            "country": country, "state": state, "north_america": int(bool(na)),
            "observer_login": inat_login if source == "inat" else mo_login,
            "inat_uuid": uuid if source == "inat" else None,
            "included": 0, "reason": reason, "split": None,
            "benchmark": b[0], "benchmark_split": b[1]})
        places.append({"record_key": key, "lat_micro": micro(lat), "lon_micro": micro(lon)})
        keep[key] = records[-1]

    # Photos: every photo linked to a candidate record, with its kept original (the
    # 'large' copy, S3 first) and the permission rule of evaluate.load_records.
    withdrawn = {r[0] for r in conn.execute(
        f"select op.photo_id from observation_photos op where not ({EXCLUDED_FROM_USE_SQL})")}
    copies: dict[int, tuple] = {}
    for pid, store, path, nbytes, sha in conn.execute(
            "select photo_id, store, path, bytes, sha256 from photo_copies "
            "where size = 'large' and sha256 is not null "
            "order by photo_id, (store like 's3://%') desc, store"):
        copies.setdefault(pid, (store, path, nbytes, sha))
    psrc = "p.source" if "source" in pcols else "'inat'"
    pspid = "p.source_photo_id" if "source_photo_id" in pcols else "null"
    psoid = "p.source_owner_id" if "source_owner_id" in pcols else "null"
    answers = dict(conn.execute("select inat_user_id, status from photo_permissions"))
    photos, owners, photo_stores, stores = [], [], [], {}
    seen = set()
    for (oid, pid, pos, psource, spid, owner_id, soid, login, owner_name, lic, lclass, attrib,
         url, checked) in conn.execute(f"""
        select op.observation_id, op.photo_id, op.position, {psrc}, {pspid}, p.owner_user_id,
               {psoid}, p.owner_login, p.owner_name, p.license_code, p.license_class,
               p.attribution, p.source_url, p.license_checked_at
        from observation_photos op join photos p on p.photo_id = op.photo_id
        order by op.observation_id, op.position, op.photo_id"""):
        rec = keep.get(oid)
        if rec is None:
            continue
        pkey = _photo_key(psource, pid, spid)
        if pkey in seen:                 # one photo on two records: the first keeps it
            continue
        seen.add(pkey)
        copy = copies.get(pid)
        if rec["reason"] != "ok":
            reason = "record_excluded"
        elif copy is None:
            reason = "no_original"
        elif pid in withdrawn:
            reason = "permission_withdrawn"
        elif pkey in review_photos:
            reason = review_photos[pkey]
        else:
            reason = "ok"
        photos.append({
            "photo_key": pkey, "manifest_photo_id": pid, "record_key": oid, "position": pos,
            "source": psource, "source_photo_id": spid if psource != "inat" else pid,
            "source_url": url, "original_path": copy[1] if copy else None,
            "original_sha256": copy[3] if copy else None,
            "original_bytes": copy[2] if copy else None,
            "license_code": lic or "", "license_class": lclass, "license_checked_at": checked,
            "owner_login": login, "owner_name": owner_name, "attribution": attrib,
            "included": int(reason == "ok"), "reason": reason})
        owners.append({"photo_key": pkey, "owner_user_id": owner_id, "source_owner_id": soid,
                       "permission": (answers.get(owner_id, "no_answer") if lclass == "arr"
                                      else None)})
        if copy:
            sid = stores.setdefault(copy[0], len(stores) + 1)
            photo_stores.append({"photo_key": pkey, "store_id": sid})

    unknown = (set(review_records) - set(keep)) | (set(review_photos) - seen)
    if unknown:
        raise ReleaseError(f"{len(unknown)} review-list keys are not in the manifest "
                           f"(e.g. {sorted(unknown)[:5]}); a stale list must not exclude nothing")
    has_photo = {p["record_key"] for p in photos if p["included"]}
    for rec in records:
        if rec["reason"] == "ok" and rec["record_key"] not in has_photo:
            rec["reason"] = "no_photos"
        rec["included"] = int(rec["reason"] == "ok")
    exported = [r["label_exported_at"] for r in records if r["label_exported_at"]]
    info = {"label_snapshot_at": max(exported) if exported else None,
            "north_america_only": north_america_only}
    return Built(records, photos, places, owners, stores, photo_stores, info)


def assign_splits(records: list[dict], *, val_cutoff: str | None = None,
                  val_weeks: int = DEFAULT_VAL_WEEKS, test_only: bool = False) -> dict:
    """Time-based splits of the included records (never random; CLAUDE.md). A test
    release puts every included record in 'test'. Returns the split rules."""
    inc = [r for r in records if r["included"]]
    if test_only:
        for r in inc:
            r["split"] = "test"
        return {"test": "every included record of this test release"}
    if val_cutoff is None:
        dated = [r["validated_on"][:10] for r in inc if r["validated_on"]]
        if not dated:
            raise ReleaseError("no included record has a validation date to split by")
        val_cutoff = (date.fromisoformat(max(dated)) - timedelta(weeks=val_weeks)).isoformat()
    for r in inc:
        v = (r["validated_on"] or "")[:10]
        r["split"] = "val" if v and v > val_cutoff else "train"
    return {"train": f"first green validation on or before {val_cutoff}, or no date",
            "val": f"first green validation after {val_cutoff}",
            "val_cutoff": val_cutoff}


# --- writing --------------------------------------------------------------------------------

def _insert(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    conn.executemany(f'insert into "{table}" ({", ".join(cols)}) values '
                     f'({", ".join("?" * len(cols))})', [tuple(r[c] for c in cols) for r in rows])


def _new_db(path: Path, schema: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("pragma page_size = 4096")
    conn.executescript(schema)
    return conn


def _seal_db(conn: sqlite3.Connection) -> None:
    conn.commit()
    conn.execute("vacuum")
    conn.close()


def is_draft(rid: str) -> bool:
    return bool(DRAFT_ID.match(rid))


def check_id(rid: str) -> str:
    if not (FROZEN_ID.match(rid) or DRAFT_ID.match(rid)):
        raise ReleaseError(f"release ids are v1, v2, v1-cc, v1-test (frozen) or "
                           f"dry-YYYYMMDD-HHMMSS (draft): {rid!r}")
    return rid


def draft_id(now: datetime | None = None, suffix: str = "") -> str:
    return "dry-" + (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S") + suffix


def snapshot_manifest(src: Path, dest: Path) -> dict:
    """A private copy of the manifest, read without writing to it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    uri = "file:" + quote(Path(src).resolve().as_posix()) + "?mode=ro"
    s = sqlite3.connect(uri, uri=True)
    out = sqlite3.connect(dest)
    try:
        s.backup(out)
    finally:
        out.close()
        s.close()
    return {"path_name": Path(src).name, "copied_at": now_iso(), "bytes": dest.stat().st_size}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_read_only(folder: Path) -> None:
    for p in folder.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)


def _writable(folder: Path) -> None:
    for p in folder.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def write_release(folder: Path, rid: str, built: Built, split_rules: dict, *,
                  recipes: list[str], derived: list[dict], inputs: dict,
                  review_files: list[Path] = (), extra_info: dict | None = None,
                  log=print) -> dict:
    """Write <root>/<rid>/ (through <rid>.partial), MANIFEST.json last, files read-only."""
    final = folder / rid
    if final.exists():
        raise ReleaseError(f"release {rid} exists; a release is never rebuilt or edited "
                           "(a correction is the next release)")
    partial = folder / f"{rid}.partial"
    if partial.exists():
        _writable(partial)
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    db = _new_db(partial / RELEASE_DB, RELEASE_SCHEMA)
    priv = _new_db(partial / PRIVATE_DB, PRIVATE_SCHEMA)
    _insert(db, "records", built.records)
    _insert(db, "photos", built.photos)
    split_counts = Counter(r["split"] for r in built.records if r["included"])
    photo_split = Counter()
    split_of = {r["record_key"]: r["split"] for r in built.records if r["included"]}
    for p in built.photos:
        if p["included"]:
            photo_split[split_of.get(p["record_key"])] += 1
    _insert(db, "splits", [{"split": s, "rule": split_rules.get(s, ""),
                            "records": split_counts[s], "photos": photo_split[s]}
                           for s in SPLITS if split_counts[s]])
    rec_rows = []
    for name in recipes:
        r = recipe_with_library(name)
        text = json.dumps(r, sort_keys=True, separators=(",", ":"))
        rec_rows.append({"name": name, "recipe_json": text, "sha256": sha256_bytes(text.encode())})
    _insert(db, "recipes", rec_rows)
    _insert(db, "derived", derived)
    _insert(priv, "record_places", built.places)
    _insert(priv, "photo_owners", built.owners)
    _insert(priv, "stores", [{"store_id": i, "location": loc} for loc, i in built.stores.items()])
    _insert(priv, "photo_stores", built.photo_stores)
    db.commit()
    priv.commit()
    hashes = content_hashes(db, priv)
    info = {"id": rid, "draft": is_draft(rid), "format": FORMAT, "created": now_iso(),
            "code_commit": config.code_version(), "label_rule": LABEL_RULE,
            "splits": split_rules, "inputs": inputs, **built.info, **(extra_info or {})}
    _insert(db, "info", [{"key": k, "value": json.dumps(v, sort_keys=True)}
                         for k, v in sorted(info.items())])
    _seal_db(db)
    _seal_db(priv)
    counts = summary_counts(built)
    if review_files:
        (partial / "inputs").mkdir()
        for p in review_files:
            shutil.copyfile(p, partial / "inputs" / Path(p).name)
    (partial / CARD).write_text(dataset_card(info, counts, hashes), encoding="utf-8")
    files = sorted(p for p in partial.rglob("*") if p.is_file())
    listed = [{"path": p.relative_to(partial).as_posix(), "bytes": p.stat().st_size,
               "sha256": sha256_file(p)} for p in files]
    (partial / SUMS).write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in listed),
                                encoding="utf-8")
    manifest = {**info, "hashes": hashes, "counts": counts, "files": listed,
                "sums_sha256": sha256_file(partial / SUMS)}
    (partial / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                    encoding="utf-8")
    _make_read_only(partial)
    partial.rename(final)
    log(f"release {rid}: release_hash {hashes['release']}")
    return manifest


def summary_counts(built: Built) -> dict:
    by = lambda rows, *keys: dict(sorted(Counter(  # noqa: E731
        "/".join(str(r[k]) for k in keys) for r in rows).items()))
    inc_photos = [p for p in built.photos if p["included"]]
    return {"records": len(built.records), "records_included": sum(r["included"] for r in built.records),
            "records_by_source_reason": by(built.records, "source", "reason"),
            "records_by_split": by([r for r in built.records if r["included"]], "split"),
            "records_by_source_split": by([r for r in built.records if r["included"]],
                                          "source", "split"),
            "photos": len(built.photos), "photos_included": len(inc_photos),
            "photos_by_source_reason": by(built.photos, "source", "reason"),
            "photos_included_by_licence": by(inc_photos, "source", "license_class"),
            "labels_included": len({r["label"] for r in built.records if r["included"]})}


def dataset_card(info: dict, counts: dict, hashes: dict) -> str:
    lines = [f"# MycoMap Vision dataset release {info['id']}", ""]
    if info.get("draft"):
        lines += ["**DRAFT (dry run), not a frozen release.**", ""]
    lines += [
        f"- Created {info['created']} from code {info['code_commit']}.",
        f"- release_hash `{hashes['release']}`; reference_hash `{hashes['reference']}`.",
        f"- Labels: {LABEL_RULE}. Label snapshot: {info.get('label_snapshot_at')}.",
        "- Records: DNA-validated (green in a mycomap.org project) iNaturalist and Mushroom "
        "Observer records. MyCoPortal, .com Sequences and GenBank records are listed as "
        "excluded (no field photos of their own).",
        "- Photos: originals are our kept copies (iNat 'large', MO 960 px). Derived images are "
        "made on demand by the recipes listed, and checked against their stored hashes.",
        "- Splits are by date of first DNA validation, never random.",
        "- Licences: all-rights-reserved photos are used for training only while permission "
        "is sought, and are never published; the public variant holds CC photos only.",
        "- Coordinates are never published (obscured records stay obscured).", "",
        "## Counts", "", "```", json.dumps(counts, indent=2), "```", ""]
    return "\n".join(lines)


def build(*, name: str, manifest_path: Path | None = None, root: Path | None = None,
          review_files: list[Path] = (), recipes: list[str] = ("original", "long500-q90"),
          derive_photos: str = "all", val_cutoff: str | None = None,
          val_weeks: int = DEFAULT_VAL_WEEKS, north_america_only: bool = True,
          freeze_approved: str | None = None, test_for: str | None = None,
          test_ids: Path | None = None, store_reader=None, log=print) -> dict:
    """Cut a release (frozen ids need freeze_approved: who gave the trigger, when) or a
    draft. `derive_photos` 'all' computes every derived hash (reads every original);
    'none' or a number (a sample, drafts only) do less."""
    root = Path(root or ROOT)
    check_id(name)
    if not is_draft(name) and not freeze_approved:
        raise ReleaseError(f"{name} is a frozen release id: cut it only with "
                           "--freeze-approved '<who, when>' after Steve's freeze trigger")
    if not is_draft(name) and derive_photos != "all":
        raise ReleaseError("a frozen release stores every derived hash (--derive all)")
    if (root / name).exists():
        raise ReleaseError(f"release {name} exists; a release is never rebuilt or edited")
    manifest_path = Path(manifest_path or config.MANIFEST_PATH)
    work = root / f".work-{name}"
    if work.exists():
        shutil.rmtree(work)
    try:
        snap = snapshot_manifest(manifest_path, work / "manifest.sqlite")
        conn = sqlite3.connect(work / "manifest.sqlite")
        try:
            check_migrated(conn)
            rrev, prev, lists = read_review_lists(list(review_files))
            only = base = None
            if test_for:
                base = load_release(test_for, root=root, allow_draft=is_draft(test_for))
                only = read_ids(test_ids)
                if not only:
                    raise ReleaseError("a test release needs --ids with its records")
            built = collect(conn, review_records=rrev, review_photos=prev,
                            north_america_only=north_america_only, only_keys=only)
        finally:
            conn.close()
        extra = {"freeze_approved": freeze_approved}
        if base is not None:
            missing = only - {r["record_key"] for r in built.records}
            if missing:
                raise ReleaseError(f"{len(missing)} test ids are not in the manifest "
                                   f"(e.g. {sorted(missing)[:5]})")
            check_disjoint(base, built)
            extra.update({"test_for": base.id, "test_for_release_hash": base.release_hash})
        rules = assign_splits(built.records, val_cutoff=val_cutoff, val_weeks=val_weeks,
                              test_only=base is not None)
        derived = derive_all(built, recipes, derive_photos, store_reader, log=log)
        extra["derived_complete"] = derive_photos == "all"
        inputs = {"manifest": snap, "review_lists": lists,
                  "migration": REQUIRED_MIGRATION}
        return write_release(root, name, built, rules, recipes=list(recipes), derived=derived,
                             inputs=inputs, review_files=list(review_files), extra_info=extra,
                             log=log)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_list(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def read_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    col = "record_key" if rows and "record_key" in rows[0] else "observation_id"
    return {(r.get(col) or "").strip() for r in rows if (r.get(col) or "").strip()}


def check_disjoint(base: "Release", built: Built) -> None:
    """A test release shares no record and no photo (by original sha256) with its base."""
    keys = {r["record_key"] for r in built.records}
    shared = keys & set(base.record_keys())
    if shared:
        raise SealedLeak(f"{len(shared)} test records are in {base.id} already "
                         f"(e.g. {sorted(shared)[:5]})")
    shas = {p["original_sha256"] for p in built.photos if p["original_sha256"]}
    same = shas & base.photo_shas()
    if same:
        raise SealedLeak(f"{len(same)} test photos are byte-identical to photos in {base.id}")


def derive_all(built: Built, recipes, derive_photos: str, store_reader, log=print) -> list[dict]:
    """Derived hashes for included photos × recipes. 'original' needs no reading."""
    inc = [p for p in built.photos if p["included"]]
    out = [{"photo_key": p["photo_key"], "recipe": "original",
            "derived_sha256": p["original_sha256"], "pixels_sha256": None, "width": None,
            "height": None} for p in inc] if "original" in recipes else []
    others = [r for r in recipes if r != "original"]
    if not others or derive_photos == "none":
        return out
    todo = inc if derive_photos == "all" else inc[:int(derive_photos)]
    if store_reader is None:
        raise ReleaseError("deriving images needs the photo store (store_reader)")
    store_of = {ps["photo_key"]: ps["store_id"] for ps in built.photo_stores}
    location = {i: loc for loc, i in built.stores.items()}
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        body = store_reader(location[store_of[p["photo_key"]]], p["original_path"])
        if sha256_bytes(body) != p["original_sha256"]:
            raise ReleaseError(f"{p['photo_key']}: the original's sha256 differs from the manifest")
        rows = []
        for name in others:
            d = derive_bytes(body, recipe_with_library(name))
            rows.append({"photo_key": p["photo_key"], "recipe": name,
                         "derived_sha256": d.sha256, "pixels_sha256": d.pixels_sha256,
                         "width": d.width, "height": d.height})
        return rows

    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, rows in enumerate(ex.map(one, todo), 1):
            out.extend(rows)
            if i % 5000 == 0:
                log(f"  derived {i:,} / {len(todo):,} photos")
    return out


def default_store_reader():
    """(store location, relpath) -> bytes, one store object per location."""
    from .storage import open_store
    cache = {}

    def read(location: str, relpath: str) -> bytes:
        if location not in cache:
            cache[location] = open_store(location, Path(location) if "://" not in location
                                         else config.DATA_DIR)
        return cache[location].get(relpath)
    return read


# --- reading a release ----------------------------------------------------------------------

def _ro(path: Path) -> sqlite3.Connection:
    uri = "file:" + quote(Path(path).resolve().as_posix()) + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class Release:
    id: str
    folder: Path
    manifest: dict
    db: sqlite3.Connection = field(repr=False)
    _private: sqlite3.Connection | None = field(default=None, repr=False)

    @property
    def release_hash(self) -> str:
        return self.manifest["hashes"]["release"]

    @property
    def reference_hash(self) -> str:
        return self.manifest["hashes"]["reference"]

    @property
    def labels_hash(self) -> str:
        return self.manifest["hashes"]["labels"]

    @property
    def code_commit(self) -> str:
        return self.manifest["code_commit"]

    @property
    def val_cutoff(self) -> str | None:
        return self.manifest.get("splits", {}).get("val_cutoff")

    @property
    def private(self) -> sqlite3.Connection:
        if self._private is None:
            self._private = _ro(self.folder / PRIVATE_DB)
        return self._private

    def cite(self) -> dict:
        """What an experiment writes into its outputs (and the registry entry)."""
        return {"dataset_release": self.id, "release_hash": self.release_hash,
                "reference_hash": self.reference_hash, "labels_hash": self.labels_hash,
                "release_code_commit": self.code_commit, "draft": self.manifest["draft"]}

    def record_keys(self) -> list[str]:
        return [r[0] for r in self.db.execute("select record_key from records")]

    def photo_shas(self) -> set[str]:
        return {r[0] for r in self.db.execute(
            "select original_sha256 from photos where included = 1 and original_sha256 is not null")}

    def records(self, split: str | tuple[str, ...]) -> list[sqlite3.Row]:
        splits = (split,) if isinstance(split, str) else tuple(split)
        bad = set(splits) - set(SPLITS)
        if bad:
            raise ReleaseError(f"unknown split {sorted(bad)}")
        q = ",".join("?" * len(splits))
        return self.db.execute(f"select * from records where included = 1 and split in ({q}) "
                               "order by record_key", splits).fetchall()

    def training_records(self, include_val: bool = False) -> list[sqlite3.Row]:
        """Records a model may learn from: train (and val when asked). Never test."""
        if self.manifest.get("test_for"):
            raise SealedLeak(f"{self.id} is a test release: none of it may be trained on")
        rows = self.records(TRAINABLE if include_val else ("train",))
        for r in rows:
            if not r["included"] or r["split"] not in TRAINABLE:
                raise SealedLeak(f"{r['record_key']} ({r['split']}) is not trainable")
        return rows

    def reference_records(self) -> list[sqlite3.Row]:
        """The reference index: train + val, never test."""
        return self.training_records(include_val=True)

    def photos(self, record_key: str) -> list[sqlite3.Row]:
        return self.db.execute("select * from photos where record_key = ? and included = 1 "
                               "order by position, photo_key", (record_key,)).fetchall()

    def photo_ids(self, splits=TRAINABLE) -> list[int]:
        q = ",".join("?" * len(splits))
        return [r[0] for r in self.db.execute(
            f"select p.manifest_photo_id from photos p join records r using (record_key) "
            f"where p.included = 1 and r.included = 1 and r.split in ({q})", tuple(splits))]

    def as_records(self, splits=TRAINABLE, photo_row: dict[int, int] | None = None) -> list:
        """evaluate.Record objects for these splits: labels, dates and photos from the
        release, coordinates from private.sqlite. photo_rows hold photo_row[photo id] for
        photos with a vector (all included photos when photo_row is None)."""
        from .evaluate import Record, real_date
        splits = (splits,) if isinstance(splits, str) else tuple(splits)
        places = {r[0]: (r[1], r[2]) for r in self.private.execute(
            "select record_key, lat_micro, lon_micro from record_places")}
        photos: dict[str, list[int]] = defaultdict(list)
        for key, pid in self.db.execute(
                "select record_key, manifest_photo_id from photos where included = 1 "
                "order by record_key, position, photo_key"):
            photos[key].append(pid)
        out = []
        for r in self.records(splits):
            pids = photos.get(r["record_key"], [])
            rows = [photo_row[p] for p in pids if p in photo_row] if photo_row is not None else pids
            if not rows:
                continue
            lat, lon = places.get(r["record_key"], (None, None))
            out.append(Record(
                r["record_key"], r["species"] or "", r["genus"] or "", r["family"] or "",
                real_date(r["validated_on"]), r["observer_login"], rows,
                latitude=None if lat is None else lat / 1e6,
                longitude=None if lon is None else lon / 1e6,
                observed_on=real_date(r["observed_on"]),
                projects=tuple(json.loads(r["green_projects"] or "[]")),
                stored_name=r["org_name"] if r["org_name"] != r["label"] else "",
                taxon="" if r["species"] else (r["label"] or ""),
                uuid=r["inat_uuid"] or ""))
        return out

    def check_photos_match(self, conn: sqlite3.Connection) -> None:
        """Vectors are looked up by photo id in the live manifest: refuse when a photo's
        kept original there is no longer the one the release recorded (re-downloaded,
        replaced), since its vector may then be of another image."""
        want = {pid: sha for pid, sha in self.db.execute(
            "select manifest_photo_id, original_sha256 from photos where included = 1")}
        changed = [pid for pid, sha in conn.execute(
            "select photo_id, sha256 from photo_copies where size = 'large' and sha256 is not null")
            if pid in want and sha != want[pid]]
        if changed:
            raise ReleaseError(f"{len(changed)} photos of {self.id} have another original in "
                               f"this manifest now (e.g. {sorted(changed)[:5]}); re-embed them "
                               "from the release's originals first")

    def derive(self, photo_key: str, recipe: str, store_reader=None) -> Derived:
        """The derived image from our original, checked against the stored hashes."""
        p = self.db.execute("select * from photos where photo_key = ?", (photo_key,)).fetchone()
        if p is None:
            raise ReleaseError(f"{photo_key} is not in {self.id}")
        r = self.db.execute("select recipe_json from recipes where name = ?", (recipe,)).fetchone()
        if r is None:
            raise ReleaseError(f"{self.id} has no recipe {recipe!r}")
        stored_recipe = json.loads(r[0])
        loc = self.private.execute(
            "select s.location from photo_stores ps join stores s using (store_id) "
            "where ps.photo_key = ?", (photo_key,)).fetchone()
        if loc is None:
            raise ReleaseError(f"{photo_key}: no kept original")
        body = (store_reader or default_store_reader())(loc[0], p["original_path"])
        if sha256_bytes(body) != p["original_sha256"]:
            raise ReleaseError(f"{photo_key}: our original no longer matches its sha256")
        d = derive_bytes(body, stored_recipe)
        want = self.db.execute("select derived_sha256, pixels_sha256 from derived "
                               "where photo_key = ? and recipe = ?", (photo_key, recipe)).fetchone()
        if want is not None and d.sha256 != want[0]:
            pixels = "the pixels match" if d.pixels_sha256 == want[1] else "the pixels differ too"
            raise ReleaseError(
                f"{photo_key} {recipe}: derived bytes differ from the release ({pixels}); "
                f"recipe made with {stored_recipe.get('library')}, here {library_versions()}")
        return d

    def close(self) -> None:
        self.db.close()
        if self._private is not None:
            self._private.close()


def load_release(rid: str, root: Path | None = None, *, allow_draft: bool = False,
                 check_files: bool = True) -> Release:
    """Open a finished release read-only. Checks MANIFEST.json and (by default) every
    file's sha256. Drafts are refused unless allow_draft."""
    root = Path(root or ROOT)
    check_id(rid)
    folder = root / rid
    mpath = folder / MANIFEST
    if not mpath.is_file():
        raise ReleaseError(f"release {rid} is missing or unfinished (no {MANIFEST}) in {root}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("id") != rid:
        raise ReleaseError(f"{MANIFEST} of {rid} names {manifest.get('id')!r}")
    if manifest.get("draft") and not allow_draft:
        raise ReleaseError(f"{rid} is a draft (dry run), not a frozen release; "
                           "research results need a frozen one (--allow-draft to try)")
    if check_files:
        for f in manifest["files"]:
            p = folder / f["path"]
            if not p.is_file() or p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
                raise ReleaseError(f"{rid}: {f['path']} does not match {MANIFEST}")
    return Release(rid, folder, manifest, _ro(folder / RELEASE_DB))


def verify(rid: str, root: Path | None = None, *, photos: str = "0", store_reader=None,
           allow_draft: bool = True) -> dict:
    """Recompute every file hash and content hash; with photos (a number or 'all'),
    re-read originals from the store and check their sha256."""
    rel = load_release(rid, root, allow_draft=allow_draft)
    try:
        hashes = content_hashes(rel.db, rel.private)
        bad = {k: (rel.manifest["hashes"].get(k), v) for k, v in hashes.items()
               if rel.manifest["hashes"].get(k) != v}
        if bad:
            raise ReleaseError(f"{rid}: content hashes differ from {MANIFEST}: {sorted(bad)}")
        checked = 0
        if photos != "0":
            reader = store_reader or default_store_reader()
            q = "select photo_key, original_path, original_sha256 from photos where included = 1"
            if photos != "all":
                q += f" order by photo_key limit {int(photos)}"
            stores = dict(rel.private.execute(
                "select ps.photo_key, s.location from photo_stores ps join stores s using (store_id)"))
            for key, path, sha in rel.db.execute(q):
                if sha256_bytes(reader(stores[key], path)) != sha:
                    raise ReleaseError(f"{rid}: original of {key} no longer matches its sha256")
                checked += 1
        return {"id": rid, "release_hash": hashes["release"], "files": len(rel.manifest["files"]),
                "originals_checked": checked, "ok": True}
    finally:
        rel.close()


# --- the public variant ---------------------------------------------------------------------

PUBLIC_DROP_PHOTO_COLUMNS = ("manifest_photo_id",)


def public_variant(rid: str, root: Path | None = None, log=print) -> dict:
    """<rid>-cc: CC photos only (every CC licence), no ARR photo or its hashes, no
    coordinates or owners' numeric ids (private.sqlite is left empty)."""
    root = Path(root or ROOT)
    if rid.endswith("-cc"):
        raise ReleaseError("already a public variant")
    base = load_release(rid, root, allow_draft=is_draft(rid))
    try:
        records = [dict(r) for r in base.db.execute("select * from records order by record_key")]
        photos = [dict(r) for r in base.db.execute(
            "select * from photos where license_class in (?, ?) order by photo_key", CC_CLASSES)]
        recipes = [r[0] for r in base.db.execute("select name from recipes order by name")]
        keys = {p["photo_key"] for p in photos}
        derived = [dict(r) for r in base.db.execute("select * from derived order by photo_key, recipe")
                   if r["photo_key"] in keys]
        splits = {r["split"]: r["rule"] for r in base.db.execute("select split, rule from splits")}
        manifest = base.manifest
    finally:
        base.close()
    for p in photos:
        for c in PUBLIC_DROP_PHOTO_COLUMNS:
            p[c] = 0
    has_cc = {p["record_key"] for p in photos if p["included"]}
    for r in records:
        if r["included"] and r["record_key"] not in has_cc:
            r.update(included=0, reason="no_cc_photos", split=None)
    built = Built(records, photos, [], [], {}, [], {
        "label_snapshot_at": manifest.get("label_snapshot_at"),
        "north_america_only": manifest.get("north_america_only")})
    rules = dict(manifest.get("splits", {}))
    rules.update({k: v for k, v in splits.items()})
    out = write_release(root, f"{rid}-cc", built, rules, recipes=recipes, derived=derived,
                        inputs={"public_variant_of": rid}, extra_info={
                            "public_variant_of": rid, "base_release_hash": manifest["hashes"]["release"],
                            "derived_complete": manifest.get("derived_complete"),
                            "freeze_approved": manifest.get("freeze_approved")}, log=log)
    check_public(root / f"{rid}-cc")
    return out


def check_public(folder: Path) -> None:
    """A public release holds no ARR photo and no coordinates (fails loudly if it does)."""
    db = _ro(folder / RELEASE_DB)
    try:
        arr = db.execute("select count(*) from photos where license_class not in (?, ?)",
                         CC_CLASSES).fetchone()[0]
        der = db.execute("select count(*) from derived d left join photos p using (photo_key) "
                         "where p.photo_key is null").fetchone()[0]
        cols = {c for t in ("records", "photos") for c in _columns(db, t)}
    finally:
        db.close()
    priv = _ro(folder / PRIVATE_DB)
    try:
        leaked = sum(priv.execute(f'select count(*) from "{t}"').fetchone()[0] for t in _tables(priv))
    finally:
        priv.close()
    if arr or der or leaked or cols & {"latitude", "longitude", "lat_micro", "lon_micro"}:
        raise ReleaseError(f"public variant {folder.name} holds non-CC photos ({arr}), stray "
                           f"derived rows ({der}) or private rows ({leaked})")


# --- diff and show --------------------------------------------------------------------------

def diff(a: str, b: str, root: Path | None = None, examples: int = 5) -> dict:
    ra = load_release(a, root, allow_draft=True)
    rb = load_release(b, root, allow_draft=True)
    try:
        def rows(rel, sql):
            return {r[0]: tuple(r[1:]) for r in rel.db.execute(sql)}
        rq = "select record_key, label, included, reason, split from records"
        pq = "select photo_key, original_sha256, license_class, included, reason from photos"
        A, B = rows(ra, rq), rows(rb, rq)
        PA, PB = rows(ra, pq), rows(rb, pq)
        out = {"a": a, "b": b, "hashes_changed": sorted(
            k for k in COMPONENTS + ("reference", "release")
            if ra.manifest["hashes"].get(k) != rb.manifest["hashes"].get(k))}

        def changes(x, y, idx, label):
            keys = sorted(k for k in set(x) & set(y) if x[k][idx] != y[k][idx])
            out[label] = {"n": len(keys), "examples": [(k, x[k][idx], y[k][idx])
                                                       for k in keys[:examples]]}
        for name, x, y in (("records", A, B), ("photos", PA, PB)):
            added, removed = sorted(set(y) - set(x)), sorted(set(x) - set(y))
            out[f"{name}_added"] = {"n": len(added), "examples": added[:examples]}
            out[f"{name}_removed"] = {"n": len(removed), "examples": removed[:examples]}
        changes(A, B, 0, "relabelled")
        changes(A, B, 1, "inclusion_changed")
        changes(A, B, 3, "split_changed")
        changes(PA, PB, 0, "photo_original_changed")
        changes(PA, PB, 1, "photo_licence_changed")
        changes(PA, PB, 2, "photo_inclusion_changed")
        return out
    finally:
        ra.close()
        rb.close()


def show(rid: str, root: Path | None = None) -> dict:
    rel = load_release(rid, root, allow_draft=True)
    try:
        return {"id": rid, "draft": rel.manifest["draft"], "hashes": rel.manifest["hashes"],
                "counts": rel.manifest["counts"], "splits": rel.manifest.get("splits")}
    finally:
        rel.close()
