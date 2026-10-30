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
                  "source_unknown", "held_out", "not_north_america", "not_fetched",
                  "missing_at_source",
                  "label_conflict", "no_label", "guest", "nonfungal_label", "review",
                  "crosspost_twin", "no_photos", "ok")
PHOTO_REASONS = ("record_excluded", "other_source", "no_original", "permission_withdrawn",
                 "mo_not_for_use", "review", "shared_other_label", "copy_in_train",
                 "copy_in_base", "ok")

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
# Tables only some releases hold, hashed (and part of release_hash) only where present, so
# a release made before one existed keeps its hashes: near_base, a test release's dHash
# flags against its base (2026-10-13).
OPTIONAL_COMPONENTS = ("near_base",)


def content_hashes(db: sqlite3.Connection, private: sqlite3.Connection | None,
                   private_hash: str | None = None) -> dict[str, str]:
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
    tables = _tables(db)
    for name in OPTIONAL_COMPONENTS:
        if name in tables:
            out[name] = stream_hash(name, table_rows(db, name, "photo_key"))
    if private is None:
        out["private"] = private_hash           # not shipped: as MANIFEST.json says
    else:
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
    names = COMPONENTS + tuple(n for n in OPTIONAL_COMPONENTS if n in hashes)
    return sha256_bytes("".join(f"{n}={hashes[n]}\n" for n in names).encode())


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
  benchmark text, benchmark_split text,
  vision_assisted integer, held_by_vision integer
);
create table photos (
  photo_key text primary key, manifest_photo_id integer not null, record_key text not null,
  position integer, source text not null, source_photo_id integer, source_url text,
  original_path text, original_sha256 text, original_bytes integer,
  license_code text, license_class text not null, license_checked_at text,
  owner_login text, owner_name text, attribution text,
  public_trainable integer not null,
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

# A test release only: each included photo's dHash and, when one is within dhash.NEAR, its
# nearest included base photo (a flag: nothing is dropped for it).
NEAR_BASE_SCHEMA = """
create table near_base (photo_key text primary key, dhash text not null,
                        base_photo_key text, distance integer);
"""

PRIVATE_SCHEMA = """
create table record_places (record_key text primary key, lat_micro integer, lon_micro integer);
create table photo_owners (photo_key text primary key, owner_user_id integer,
                           source_owner_id integer, permission text);
create table stores (store_id integer primary key, location text not null);
create table photo_stores (photo_key text primary key, store_id integer not null);
create table record_public_places (record_key text primary key, public_lat_micro integer,
                                   public_lon_micro integer, public_kind text not null,
                                   positional_accuracy_m integer, source text not null,
                                   read_at text, box_n_micro integer, box_s_micro integer,
                                   box_e_micro integer, box_w_micro integer);
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


PUBLIC_KINDS = ("inat_open", "inat_obscured", "mo_gps", "mo_box_centre", "none")


def _km(lat1, lon1, lat2, lon2) -> float:
    import math
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def public_place(key: str, source: str, inat: dict | None, mo: dict | None) -> dict | None:
    """The record's PUBLIC point (Steve, 2026-10-12, D1 of docs/atlas-environment.md): what
    the source itself shows, never .org's true point. iNat: the point the anonymous API
    gave (inat_observations: already randomised in its 0.2-degree cell when obscured, none
    when private). MO: its shown GPS, else the centre of its location box
    (mo_public_points, read anonymously by mopoints.py). None when the source's point was
    never read (the build's pre-flight refuses that for a frozen release)."""
    row = {"record_key": key, "public_lat_micro": None, "public_lon_micro": None,
           "public_kind": "none", "positional_accuracy_m": None, "source": source,
           "read_at": None, "box_n_micro": None, "box_s_micro": None, "box_e_micro": None,
           "box_w_micro": None}
    if source == "inat":
        if inat is None:
            return None
        row["read_at"] = inat.get("fetched_at")
        lat, lon = inat.get("latitude"), inat.get("longitude")
        if lat is None or lon is None:
            return row                      # private, or no location: nothing public
        obscured = bool(inat.get("obscured")) or "obscured" in (
            inat.get("geoprivacy"), inat.get("taxon_geoprivacy"))
        acc = inat.get("positional_accuracy")
        row.update(public_lat_micro=micro(lat), public_lon_micro=micro(lon),
                   public_kind="inat_obscured" if obscured else "inat_open",
                   positional_accuracy_m=None if acc is None else int(round(float(acc))))
        return row
    if source == "mo":
        if mo is None:
            return None
        row["read_at"] = mo.get("fetched_at")
        if mo.get("latitude") is not None and mo.get("longitude") is not None \
                and not mo.get("gps_hidden"):
            row.update(public_lat_micro=micro(mo["latitude"]),
                       public_lon_micro=micro(mo["longitude"]), public_kind="mo_gps")
            return row
        n, s, e, w = (mo.get(k) for k in ("box_north", "box_south", "box_east", "box_west"))
        if None in (n, s, e, w):
            return row
        lon_c = (e + w) / 2 if e >= w else ((e + w + 360) / 2 + 180) % 360 - 180
        lat_c = (n + s) / 2
        half_diag_m = int(round(_km(s, w, n, e if e >= w else e + 360) * 1000 / 2))
        row.update(public_lat_micro=micro(lat_c), public_lon_micro=micro(lon_c),
                   public_kind="mo_box_centre", positional_accuracy_m=half_diag_m,
                   box_n_micro=micro(n), box_s_micro=micro(s), box_e_micro=micro(e),
                   box_w_micro=micro(w))
        return row
    return None


def name_kind(name: str) -> str:
    from .names import parse_name
    words = (name or "").split()
    if len(words) <= 1:
        return "one-word"
    return "provisional" if parse_name(name).code else "formal"


REVIEW_COLUMNS = ["kind", "key", "reason", "note"]


def loo_gate(name: str, test_for: str | None, loo_exclusions: Path | None,
             no_loo_exclusions: str | None) -> dict:
    """The leave-one-out gate (Steve, 2026-10-11: "review first, then build"). A frozen
    release that can be trained on waits for the person-checked LOO exclusion list
    (scripts/loo_exclusions.py, written against the manifest this build freezes), or for
    Steve's explicit word that there are none (`no_loo_exclusions`: who, when). Returns
    what the release records about it: the list's file name, rows and sha256, or the word.
    Drafts and test releases (never trained on) need neither."""
    if loo_exclusions is not None and no_loo_exclusions:
        raise ReleaseError("give the LOO exclusion list or --no-loo-exclusions, not both")
    if loo_exclusions is not None:
        p = Path(loo_exclusions)
        if not p.is_file():
            raise ReleaseError(f"LOO exclusion list {p} not found")
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if not rows or rows[0] != REVIEW_COLUMNS:
            raise ReleaseError(f"{p.name} is not a review list (header kind, key, reason, "
                               "note; scripts/loo_exclusions.py writes one)")
        return {"file": p.name, "rows": len(rows) - 1, "sha256": sha256_file(p)}
    if no_loo_exclusions is not None:
        if not no_loo_exclusions.strip():
            raise ReleaseError("--no-loo-exclusions needs who said so, and when")
        return {"none": no_loo_exclusions.strip()}
    if is_draft(name) or test_for:
        return {"not_required": "draft" if is_draft(name) else "test release"}
    raise ReleaseError(
        f"{name} waits for the LOO exclusion list (Steve, 2026-10-11: review first, then "
        "build): pass --loo-exclusions <the TSV scripts/loo_exclusions.py wrote against this "
        "manifest>, or --no-loo-exclusions '<Steve, date>' only on his explicit word")


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
            if kind == "record" and key.startswith("inat:"):
                key = key[5:]            # review lanes write <source>:<id>; iNat keys are bare
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
    public_places: list[dict] = field(default_factory=list)
    public_missing: list[str] = field(default_factory=list)   # included, point never read
    near_base: list[dict] | None = None      # a test release with a dHash index: its flags


def public_trainable(license_class: str, answer: str | None) -> int:
    """Steve, 2026-10-09: the final public model trains only on CC photos (NC included)
    and photos whose photographer granted permission; internal benchmarking models may
    use every included photo."""
    return int(license_class in CC_CLASSES or answer == "granted")


def reason_class(reason: str | None) -> str:
    """A photo-exclusion reason, with drawing, drawing_micro and drawing_specimen as one:
    "drawing" (paper a3-spec §2)."""
    r = (reason or "").strip()
    return "drawing" if r.startswith("drawing") else r


def _photo_key(source: str, manifest_id: int, source_photo_id) -> str:
    if source == "inat":
        return f"inat:{manifest_id}"
    return f"{source}:{source_photo_id if source_photo_id is not None else manifest_id}"


def collect(conn: sqlite3.Connection, *, review_records: dict | None = None,
            review_photos: dict | None = None, north_america_only: bool = True,
            only_keys: set[str] | None = None, need_original: bool = True) -> Built:
    """Every candidate record and photo from a manifest (a private copy), with labels,
    inclusion reasons and photos' originals. Splits are assigned afterwards.

    `need_original` False (counting only, never a build: v1_test_count): a photo with no
    kept original that was downloaded (photos.status 'done'; the server box embeds and
    keeps no original) goes on through the other photo rules instead of `no_original`."""
    from . import guests, holdouts, names, taxonomy
    from .evaluate import clean
    from .permissions import EXCLUDED_FROM_USE_SQL
    from .permissions import SCHEMA as PERMISSIONS_SCHEMA
    try:                                  # MO's use rule (feat/record-sources-mo, on main)
        from .permissions import MO_EXCLUDED_FROM_USE_SQL
    except ImportError:
        MO_EXCLUDED_FROM_USE_SQL = "1"
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
    has_points = "mo_public_points" in tables
    pt_sel = ("p.latitude, p.longitude, p.gps_hidden, p.box_north, p.box_south, p.box_east, "
              "p.box_west, p.fetched_at, p.status" if has_points else ", ".join(["null"] * 9))
    pt_join = ("left join mo_public_points p on p.observation_id = r.observation_id"
               if has_points else "")
    # .org's DNA + Vision verdicts (records.VISION_ASSISTED_FIELD) and Vision holds, from the
    # green feed: 1/0, or null in a manifest without the column (not known, never "no").
    truth_sel = ", ".join(f"r.{c}" if c in rcols else "null"
                          for c in ("vision_assisted", "held_by_vision"))
    held = holdouts.held_out_ids(conn)
    bench: dict[str, tuple[str, str]] = {}
    if "heldout_records" in tables:
        for b, oid, split in conn.execute(
                "select benchmark, observation_id, split from heldout_records order by benchmark"):
            bench.setdefault(oid, (b, split))
    labels = names.manifest_labels(conn)
    tax = taxonomy.for_manifest(conn)

    records, places, keep, public_places = [], [], {}, []
    for row in conn.execute(f"""
        select r.observation_id, r.source, r.source_id, r.org_source, r.scientific_name, {mb},
               r.genus, r.family, r.exported_at, r.validated_on, r.green_projects,
               r.observed_on, r.country, r.state, r.north_america, r.label_conflict,
               r.latitude, r.longitude, o.status, o.user_login, o.uuid, {mo_sel},
               o.inat_latitude, o.inat_longitude, o.obscured, o.geoprivacy,
               o.taxon_geoprivacy, o.positional_accuracy, o.fetched_at, {pt_sel},
               {truth_sel}
        from records r
        left join inat_observations o on o.observation_id = r.observation_id
        {mo_join}
        {pt_join}
        order by r.observation_id"""):
        (key, source, source_id, org_source, name, mbn, genus, family, exported, vdate, projects,
         observed, country, state, na, conflict, lat, lon, inat_status, inat_login, uuid,
         mo_status, mo_login, *pub) = row
        (i_lat, i_lon, i_obs, i_geo, i_tgeo, i_acc, i_at, m_lat, m_lon, m_hidden, m_n, m_s,
         m_e, m_w, m_at, m_status, assisted, vision_held) = pub
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
        elif north_america_only and not na:
            reason = "not_north_america"        # never fetched: fetch-inat/fetch-mo read NA only
        elif status is None:
            reason = "not_fetched"              # no iNat/MO details read for it yet
        elif status != "ok":
            reason = "missing_at_source"        # deleted, private or not returned at the source
        elif conflict:
            reason = "label_conflict"
        elif not rank:
            reason = "no_label"
        elif guests.excluded(lab.genus):
            reason = "guest"
        elif taxonomy.nonfungal(tax, lab.genus):
            reason = "nonfungal_label"          # a bee, a host plant (Steve, 2026-10-13)
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
            # An MO observer is 'mo:<login>', never the iNat account of that name (evaluate.py).
            "observer_login": inat_login if source == "inat" else
            (f"mo:{mo_login}" if mo_login else None),
            "inat_uuid": uuid if source == "inat" else None,
            "included": 0, "reason": reason, "split": None,
            "benchmark": b[0], "benchmark_split": b[1],
            "vision_assisted": None if assisted is None else int(bool(assisted)),
            "held_by_vision": None if vision_held is None else int(bool(vision_held))})
        places.append({"record_key": key, "lat_micro": micro(lat), "lon_micro": micro(lon)})
        inat_pub = None if inat_status is None else {
            "latitude": i_lat, "longitude": i_lon, "obscured": i_obs, "geoprivacy": i_geo,
            "taxon_geoprivacy": i_tgeo, "positional_accuracy": i_acc, "fetched_at": i_at}
        # Read but not returned by MO's observations API (it shows no public point for
        # it): an answer, `none`, never .org's point.
        mo_pub = None if m_at is None else {
            "latitude": m_lat, "longitude": m_lon, "gps_hidden": m_hidden, "box_north": m_n,
            "box_south": m_s, "box_east": m_e, "box_west": m_w, "fetched_at": m_at}             if m_status == "ok" else {"fetched_at": m_at}
        pp = public_place(key, source, inat_pub, mo_pub)
        if pp is not None:
            public_places.append(pp)
        keep[key] = records[-1]

    # Photos: every photo linked to a candidate record, with its kept original (the
    # 'large' copy, S3 first) and the permission rule of evaluate.load_records.
    withdrawn = {r[0] for r in conn.execute(
        f"select op.photo_id from observation_photos op where not ({EXCLUDED_FROM_USE_SQL})")}
    mo_refused = {r[0] for r in conn.execute(
        f"select op.photo_id from observation_photos op where not ({MO_EXCLUDED_FROM_USE_SQL})")}
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
         url, checked, pstatus) in conn.execute(f"""
        select op.observation_id, op.photo_id, op.position, {psrc}, {pspid}, p.owner_user_id,
               {psoid}, p.owner_login, p.owner_name, p.license_code, p.license_class,
               p.attribution, p.source_url, p.license_checked_at, p.status
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
        elif psource != rec["source"]:
            reason = "other_source"             # a photo of another source's record
        elif copy is None and (need_original or pstatus != "done"):
            reason = "no_original"
        elif pid in withdrawn:
            reason = "permission_withdrawn"
        elif pid in mo_refused:
            reason = "mo_not_for_use"           # MO's ok_for_export / licence rule
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
            "public_trainable": public_trainable(lclass, answers.get(owner_id)),
            "included": int(reason == "ok"), "reason": reason})
        owners.append({"photo_key": pkey, "owner_user_id": owner_id, "source_owner_id": soid,
                       "permission": (answers.get(owner_id, "no_answer") if lclass == "arr"
                                      else None)})
        if copy:
            sid = stores.setdefault(copy[0], len(stores) + 1)
            photo_stores.append({"photo_key": pkey, "store_id": sid})

    drop_shared_other_label(photos, keep)
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
    from .placefallback import from_source_place
    info = {"label_snapshot_at": max(exported) if exported else None,
            "north_america_only": north_america_only,
            # Placeless on .org, North American by their source's own place (2026-10-12).
            "north_america_from_source_place": from_source_place(conn)}
    have_point = {p["record_key"] for p in public_places}
    missing = [r["record_key"] for r in records if r["included"] and r["record_key"]
               not in have_point]
    return Built(records, photos, places, owners, stores, photo_stores, info,
                 public_places, missing)


def drop_shared_other_label(photos: list[dict], records: dict[str, dict]) -> int:
    """Steve, 2026-10-12: the same image (identical bytes) on records with different labels
    is left out of every one of them (`shared_other_label`). It is one observer's photo of
    two organisms in one frame (a parasite and its host, two finds in one shot), and each
    copy would teach the model another name for the same image. The records keep their
    other photos; one left with none becomes `no_photos`. Returns the photos dropped."""
    groups: dict[str, list[dict]] = {}
    for p in photos:
        if p["included"] and p["original_sha256"]:
            groups.setdefault(p["original_sha256"], []).append(p)
    dropped = 0
    for group in groups.values():
        labels = {records[p["record_key"]]["label"] for p in group}
        if len({p["record_key"] for p in group}) > 1 and len(labels) > 1:
            for p in group:
                p.update(included=0, reason="shared_other_label")
                dropped += 1
    return dropped


def drop_val_copies_of_train(built: "Built") -> int:
    """Steve, 2026-10-12: an image (identical bytes) that is also in a `train` record is left
    out of its `val` records (`copy_in_train`), so the model is never scored on a photo it
    learnt from. A `val` record left with no photo leaves the release (`no_photos`).
    Returns the photos dropped."""
    split = {r["record_key"]: r["split"] for r in built.records if r["included"]}
    in_train = {p["original_sha256"] for p in built.photos
                if p["included"] and split.get(p["record_key"]) == "train"}
    dropped = 0
    for p in built.photos:
        if p["included"] and split.get(p["record_key"]) == "val" and \
                p["original_sha256"] in in_train:
            p.update(included=0, reason="copy_in_train")
            dropped += 1
    if dropped:
        has_photo = {p["record_key"] for p in built.photos if p["included"]}
        for r in built.records:
            if r["included"] and r["record_key"] not in has_photo:
                r.update(included=0, reason="no_photos", split=None)
    return dropped


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
                  extra_writer=None, log=print) -> dict:
    """Write <root>/<rid>/ (through <rid>.partial), MANIFEST.json last, files read-only.
    `extra_writer(folder, info, hashes, counts)` adds files before they are listed."""
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
    if built.near_base is not None:
        db.executescript(NEAR_BASE_SCHEMA)
        _insert(db, "near_base", built.near_base)
    _insert(priv, "record_places", built.places)
    _insert(priv, "photo_owners", built.owners)
    _insert(priv, "stores", [{"store_id": i, "location": loc} for loc, i in built.stores.items()])
    _insert(priv, "photo_stores", built.photo_stores)
    _insert(priv, "record_public_places", built.public_places)
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
    if extra_writer is not None:
        extra_writer(partial, info, hashes, counts)
    files = sorted(p for p in partial.rglob("*") if p.is_file())
    listed = [{"path": p.relative_to(partial).as_posix(), "bytes": p.stat().st_size,
               "sha256": sha256_file(p)} for p in files]
    (partial / SUMS).write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in listed),
                                encoding="utf-8")
    manifest = {**info, "dataset_release": rid, "reference_hash": hashes["reference"],
                "release_hash": hashes["release"],
                "hashes": hashes, "counts": counts, "files": listed,
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
    included_keys = {r["record_key"] for r in built.records if r["included"]}
    return {"records": len(built.records), "records_included": sum(r["included"] for r in built.records),
            "records_by_source_reason": by(built.records, "source", "reason"),
            "records_by_split": by([r for r in built.records if r["included"]], "split"),
            "records_by_source_split": by([r for r in built.records if r["included"]],
                                          "source", "split"),
            "photos": len(built.photos), "photos_included": len(inc_photos),
            "photos_by_source_reason": by(built.photos, "source", "reason"),
            "photos_included_by_licence": by(inc_photos, "source", "license_class"),
            "labels_included": len({r["label"] for r in built.records if r["included"]}),
            "public_places_by_kind": by([p for p in built.public_places
                                         if p["record_key"] in included_keys], "public_kind")}


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
          test_ids: Path | None = None, store_reader=None,
          loo_exclusions: Path | None = None, no_loo_exclusions: str | None = None,
          base_dhash: Path | None = None, twin_exclude_photos: Path | None = None,
          log=print) -> dict:
    """Cut a release (frozen ids need freeze_approved: who gave the trigger, when) or a
    draft. `derive_photos` 'all' computes every derived hash (reads every original);
    'none' or a number (a sample, drafts only) do less. A frozen trainable release also
    needs the LOO exclusion list or Steve's word that there is none (loo_gate); the list
    is read like any review list and recorded with its rows and sha256. A frozen test
    release (test_for) also needs `base_dhash`, its base's dHash index (dhash.py): each
    included test photo near a base photo is flagged in near_base, and a test record that is
    the cross-post twin of a base record (errata.match_twins: the same find posted to both
    MO and iNat) is left out as `crosspost_twin`; `twin_exclude_photos` lists base photos
    that never pair (the base's micrograph list).
    """
    root = Path(root or ROOT)
    check_id(name)
    if base_dhash is not None and not test_for:
        raise ReleaseError("--base-dhash is for a test release (--test-for)")
    if test_for and not is_draft(name) and base_dhash is None:
        raise ReleaseError(f"a frozen test release needs --base-dhash, the dHash index of "
                           f"{test_for} (`mv aws-launch-dhash-index {test_for}`)")
    if not is_draft(name) and not freeze_approved:
        raise ReleaseError(f"{name} is a frozen release id: cut it only with "
                           "--freeze-approved '<who, when>' after Steve's freeze trigger")
    if not is_draft(name) and derive_photos != "all":
        raise ReleaseError("a frozen release stores every derived hash (--derive all)")
    if (root / name).exists():
        raise ReleaseError(f"release {name} exists; a release is never rebuilt or edited")
    loo = loo_gate(name, test_for, loo_exclusions, no_loo_exclusions)
    review_files = list(review_files)
    if loo_exclusions is not None and Path(loo_exclusions) not in map(Path, review_files):
        review_files.append(Path(loo_exclusions))
    manifest_path = Path(manifest_path or config.MANIFEST_PATH)
    work = root / f".work-{name}"
    if work.exists():
        shutil.rmtree(work)
    try:
        snap = snapshot_manifest(manifest_path, work / "manifest.sqlite")
        # iNat's families per genus (taxonomy.py) shape the labels: copied beside the copy,
        # so taxonomy.for_manifest reads it, and its hash recorded as an input.
        from .taxonomy import CACHE as TAXONOMY_CACHE
        tax_src = manifest_path.parent / TAXONOMY_CACHE
        snap["taxonomy_sha256"] = None
        if tax_src.is_file():
            (work / TAXONOMY_CACHE).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tax_src, work / TAXONOMY_CACHE)
            snap["taxonomy_sha256"] = sha256_file(work / TAXONOMY_CACHE)
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
        extra = {"freeze_approved": freeze_approved, "loo_exclusions": loo}
        if base is not None:
            missing = only - {r["record_key"] for r in built.records}
            if missing:
                raise ReleaseError(f"{len(missing)} test ids are not in the manifest "
                                   f"(e.g. {sorted(missing)[:5]})")
            extra["photos_identical_to_base_dropped"] = separate_from_base(base, built)
            extra.update({"test_for": base.id, "test_for_release_hash": base.release_hash})
            if base_dhash is not None:
                from . import dhash
                index = dhash.load_index(base_dhash)
                dhash.check_index(index, base)
                index["file_sha256"] = sha256_file(Path(base_dhash))
                extra["near_base"] = flag_near_base(built, index, store_reader, log=log)
                excl, excl_cite = frozenset(), None
                if twin_exclude_photos is not None:
                    from .errata import read_photo_list
                    excl = read_photo_list(Path(twin_exclude_photos))
                    excl_cite = {"file": Path(twin_exclude_photos).name, "photos": len(excl),
                                 "sha256": sha256_file(Path(twin_exclude_photos))}
                extra["crosspost_twin"] = {**drop_crosspost_twins(built, base, index, excl,
                                                                  log=log),
                                           "exclude_photos": excl_cite}
            else:
                extra["near_base"] = None
                extra["crosspost_twin"] = None
                log(f"DRAFT: no --base-dhash: near-copies of {base.id} photos are not flagged, "
                    "and its cross-post twins are not left out")
        if built.public_missing:
            msg = (f"{len(built.public_missing):,} included records have no public point "
                   f"read (e.g. {built.public_missing[:5]}): run `mv fetch-mo-points` (MO) "
                   "or `mv fetch-inat` (iNat) first")
            if not is_draft(name):
                raise ReleaseError(msg)
            log(f"DRAFT: {msg}")
        extra["public_places_missing"] = len(built.public_missing)
        rules = assign_splits(built.records, val_cutoff=val_cutoff, val_weeks=val_weeks,
                              test_only=base is not None)
        extra["val_copies_of_train_dropped"] = drop_val_copies_of_train(built)
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


def separate_from_base(base: "Release", built: Built) -> int:
    """A test release shares no record with its base: one shared record refuses the build.
    A test photo byte-identical (original sha256) to an included base photo is left out
    (`copy_in_base`: one image cross-posted to iNat and MO, or re-posted, would be scored
    against its own copy); a record left with no photo becomes `no_photos` (paper plan §4,
    2026-10-13). Returns the photos dropped."""
    keys = {r["record_key"] for r in built.records}
    shared = keys & set(base.record_keys())
    if shared:
        raise SealedLeak(f"{len(shared)} test records are in {base.id} already "
                         f"(e.g. {sorted(shared)[:5]})")
    base_shas = base.photo_shas()
    dropped = 0
    for p in built.photos:
        if p["included"] and p["original_sha256"] in base_shas:
            p.update(included=0, reason="copy_in_base")
            dropped += 1
    if dropped:
        has_photo = {p["record_key"] for p in built.photos if p["included"]}
        for r in built.records:
            if r["included"] and r["record_key"] not in has_photo:
                r.update(included=0, reason="no_photos", split=None)
    return dropped


def drop_crosspost_twins(built: Built, base: "Release", index: dict,
                         exclude_photos=frozenset(), log=print) -> dict:
    """Leave out every test record that is the cross-post twin of an included base record:
    the same find posted to Mushroom Observer and iNaturalist, with the same label and a photo
    within errata.TWIN_MAX_BITS dHash bits (errata.match_twins; the rule of v1's erratum
    rev 4). Scored, it would meet its own photo in the reference. Template hashes (on the
    photos of errata.TEMPLATE_RECORDS or more records, base and test together) and base photos
    in `exclude_photos` never pair. Needs the test photos' hashes (flag_near_base first).
    The record becomes `crosspost_twin`, its photos `record_excluded`. Returns the summary
    MANIFEST records."""
    from . import errata
    test_hash = {r["photo_key"]: int(r["dhash"], 16) for r in built.near_base or []}
    test_recs = {r["record_key"]: r for r in built.records if r["included"]}
    base_recs = {k: (label, source) for k, label, source in base.db.execute(
        "select record_key, label, source from records where included = 1")}
    base_rec_of = dict(base.db.execute("select photo_key, record_key from photos "
                                       "where included = 1"))
    base_photos = [(k, base_rec_of[k], *base_recs[base_rec_of[k]], int(h))
                   for k, h in zip(index["photo_key"], index["dhash"])
                   if base_rec_of.get(k) in base_recs]
    test_photos = [(p["photo_key"], p["record_key"], test_recs[p["record_key"]]["label"],
                    test_recs[p["record_key"]]["source"], test_hash[p["photo_key"]])
                   for p in built.photos
                   if p["included"] and p["record_key"] in test_recs and p["photo_key"] in test_hash]
    templates = errata.template_hashes([(x[4], x[1]) for x in base_photos + test_photos])
    usable = [x for x in base_photos if x[0] not in exclude_photos] + test_photos
    pairs = errata.match_twins(usable, templates)
    drop: dict[str, str] = {}
    for (mo, inat), (mp, ip, bits) in sorted(pairs.items()):
        if (mo in test_recs) != (inat in test_recs):          # one side test, one side base
            drop.setdefault(mo if mo in test_recs else inat, inat if mo in test_recs else mo)
    for r in built.records:
        if r["record_key"] in drop:
            r.update(included=0, reason="crosspost_twin", split=None)
    photos_dropped = 0
    for p in built.photos:
        if p["included"] and p["record_key"] in drop:
            p.update(included=0, reason="record_excluded")
            photos_dropped += 1
    included = {p["photo_key"] for p in built.photos if p["included"]}
    if built.near_base is not None:
        built.near_base = [r for r in built.near_base if r["photo_key"] in included]
    log(f"cross-post twins: {len(drop):,} test records left out (their MO or iNat twin is "
        f"in {base.id})")
    return {"rule": f"an MO and an iNat record with the same label and a photo within "
                    f"{errata.TWIN_MAX_BITS} dHash bits, one in the test release and one "
                    f"included in {base.id}: the test record is left out (crosspost_twin)",
            "max_bits": errata.TWIN_MAX_BITS, "template_records": errata.TEMPLATE_RECORDS,
            "records_dropped": len(drop), "photos_dropped": photos_dropped,
            "by_source": {s: sum(1 for k in drop if test_recs[k]["source"] == s)
                          for s in sorted({test_recs[k]["source"] for k in drop})}}


def flag_near_base(built: Built, index: dict, store_reader, log=print) -> dict:
    """dHash each included test photo (read from the store, checked against its sha256)
    and flag the ones within dhash.NEAR of an included base photo (`index`, checked by
    dhash.check_index first). Sets built.near_base; drops nothing. Returns the summary
    MANIFEST records."""
    import numpy as np

    from . import dhash
    if store_reader is None:
        raise ReleaseError("the dHash check needs the photo store (store_reader)")
    inc = sorted((p for p in built.photos if p["included"]), key=lambda p: p["photo_key"])
    store_of = {ps["photo_key"]: ps["store_id"] for ps in built.photo_stores}
    location = {i: loc for loc, i in built.stores.items()}

    def one(p):
        body = store_reader(location[store_of[p["photo_key"]]], p["original_path"])
        if sha256_bytes(body) != p["original_sha256"]:
            raise ReleaseError(f"{p['photo_key']}: the original's sha256 differs from the manifest")
        return dhash.dhash64(body)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as ex:
        hashes = list(ex.map(one, inc))
    idx, dist = dhash.nearest(np.array(hashes, dtype=np.uint64), index["dhash"])
    built.near_base = [{"photo_key": p["photo_key"], "dhash": dhash.to_hex(h),
                        "base_photo_key": index["photo_key"][i] if i >= 0 else None,
                        "distance": int(d) if i >= 0 else None}
                       for p, h, i, d in zip(inc, hashes, idx.tolist(), dist.tolist())]
    flagged = {r["photo_key"] for r in built.near_base if r["base_photo_key"] is not None}
    log(f"dHash: {len(flagged):,} of {len(inc):,} test photos within {dhash.NEAR} of a "
        "base photo (flagged, not dropped)")
    return {"rule": f"dHash Hamming distance <= {dhash.NEAR} to an included base photo: "
                    "flagged in near_base, nothing dropped",
            "algorithm": index["algorithm"], "pillow": index["pillow"],
            "index_sha256": index.get("file_sha256"), "base_photos": index["photos"],
            "photos": len(inc), "photos_flagged": len(flagged),
            "records_flagged": len({p["record_key"] for p in inc if p["photo_key"] in flagged})}


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
    # Immutable and read-only, so one connection may serve the API's threads.
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class Release:
    id: str
    folder: Path
    manifest: dict
    db: sqlite3.Connection = field(repr=False)
    _private: sqlite3.Connection | None = field(default=None, repr=False)
    errata: object | None = field(default=None, repr=False)     # errata.Errata
    photo_exclusions: dict | None = field(default=None, repr=False)
    _photos_out: frozenset = field(default=frozenset(), repr=False)

    def use_errata(self, errata) -> None:
        """Leave out an erratum's records (errata.py) from every split this release
        reads from now on; cite() then names the erratum and its sha256."""
        from .errata import check
        check(errata, self)
        self.errata = errata

    def use_photo_exclusions(self, path, expect_sha256: str, reasons=None) -> dict:
        """Take the photos a person excluded in Human Feedback (`mv feedback
        export-photo-exclusions`: its exclude_photo rows; a keep_photo row takes nothing out;
        with `reasons`, only rows with one of those reasons: one arm of paper E35) out of what
        a model learns from and is compared against, from now on. They leave every read that
        includes the train split: the fine-tune's training photos, the train reference a val
        query is compared against, and the train + val reference of a test comparison or the
        served index. A read of val (or test) alone is the queries a model is judged on: it
        keeps every photo, so two models are compared on the same queries. A record left
        with no photo drops out of those reads. Not an erratum: no record or label changes.
        The file is pinned: refused unless its sha256 is `expect_sha256`; cite() then names
        it with its counts."""
        self._photos_out, self.photo_exclusions = self.photo_exclusion_set(
            path, expect_sha256, reasons)
        return self.photo_exclusions

    def photo_exclusion_set(self, path, expect_sha256: str, reasons=None,
                            merge_drawing: bool = False) -> tuple[frozenset, dict]:
        """(the manifest photo ids of a pinned photo-exclusion list's exclude_photo rows, with
        `reasons` only one arm's, in this release's train and val; what it counted), changing
        nothing: use_photo_exclusions for the whole release, paper-score per comparator.
        merge_drawing: drawing, drawing_micro and drawing_specimen are one reason, "drawing"
        (paper a3-spec §2)."""
        path = Path(path)
        body = path.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        if sha != str(expect_sha256 or "").lower():
            raise ReleaseError(f"{path.name}: sha256 {sha[:12]}, not the {str(expect_sha256)[:12]} "
                               "it was pinned to: not the photo-exclusion list the run named")
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8")), delimiter="\t"))
        if not rows or "photo_key" not in rows[0]:
            raise ReleaseError(f"{path.name}: needs a photo_key column and at least one row")
        def why(reason):
            return reason_class(reason) if merge_drawing else reason
        wanted = sorted({why(x) for x in reasons}) if reasons else None
        if wanted and "reason" not in rows[0]:
            raise ReleaseError(f"{path.name}: has no reason column to choose {wanted} from")
        keys = {r["photo_key"].strip() for r in rows
                if (r.get("decision") or "exclude_photo") == "exclude_photo"
                and (wanted is None or why(r["reason"]) in wanted)}
        if wanted and not keys:
            raise ReleaseError(f"{path.name}: no excluded photo has reason {wanted}")
        found = {}
        for key, pid, split in self.db.execute(
                "select p.photo_key, p.manifest_photo_id, r.split from photos p join records r "
                "using (record_key) where p.included = 1 and r.included = 1"):
            if key in keys:
                found[key] = (pid, split)
        out = frozenset(pid for pid, split in found.values() if split in TRAINABLE)
        left: dict[str, dict[str, int]] = {s: defaultdict(int) for s in TRAINABLE}
        q = ",".join("?" * len(TRAINABLE))
        for key, pid, split in self.db.execute(
                "select p.record_key, p.manifest_photo_id, r.split from photos p join records r "
                f"using (record_key) where p.included = 1 and r.included = 1 and r.split in ({q})",
                TRAINABLE):
            left[split][key] += pid not in out
        return out, {
            "file": path.name, "sha256": sha, "reasons": wanted or "all",
            "photos_listed": len(keys),
            "photos_left_out": {s: sum(sp == s for _, sp in found.values()) for s in TRAINABLE},
            "records_left_with_no_photo": {s: sum(not n for n in left[s].values())
                                           for s in TRAINABLE},
            "other_split_kept": sum(sp not in TRAINABLE for _, sp in found.values()),
            "not_in_release": len(keys) - len(found)}

    def _photos_out_of(self, splits) -> frozenset:
        """use_photo_exclusions' photos when `splits` is a learning or reference read (it
        holds train); none for a read of val or test alone (queries)."""
        return self._photos_out if "train" in splits else frozenset()

    def _not_errata(self, rows):
        if self.errata is None:
            return rows
        return [r for r in rows if r["record_key"] not in self.errata.keys]

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
    def snapshot_date(self) -> str:
        """The training-data snapshot date (YYYY-MM-DD): when the newest label in the
        release was read from .org, else when the manifest was copied for the build. A
        model trained on this release takes it as its version (Steve, 2026-10-11)."""
        row = self.db.execute("select max(label_exported_at) from records "
                              "where included = 1").fetchone()
        for when in (self.manifest.get("label_snapshot_at"), (row[0] if row else None),
                     self.manifest.get("inputs", {}).get("manifest", {}).get("copied_at"),
                     self.manifest["created"]):
            if when and re.match(r"\d{4}-\d{2}-\d{2}", str(when)):
                return str(when)[:10]
        raise ReleaseError(f"{self.id} records no snapshot date")

    @property
    def has_private(self) -> bool:
        return (self.folder / PRIVATE_DB).is_file()

    @property
    def private(self) -> sqlite3.Connection:
        if not self.has_private:
            raise ReleaseError(f"{self.id} was shipped without {PRIVATE_DB} (coordinates, "
                               "owners, stores); this needs the full release")
        if self._private is None:
            self._private = _ro(self.folder / PRIVATE_DB)
        return self._private

    def public_places(self) -> dict[str, dict]:
        """record_key -> its frozen PUBLIC point (D1; private.sqlite only: never shipped to
        a trainer, never public): public_lat_micro, public_lon_micro, public_kind,
        positional_accuracy_m, source, read_at and, for an MO box, its edges."""
        cur = self.private.execute("select * from record_public_places")
        cols = [d[0] for d in cur.description]
        return {r[0]: dict(zip(cols, r)) for r in cur}

    def near_base(self) -> dict[str, dict]:
        """A test release: photo_key -> {dhash, base_photo_key, distance} for every included
        photo (base_photo_key None unless flagged). Empty when the release has no flags."""
        if "near_base" not in _tables(self.db):
            return {}
        return {r[0]: {"dhash": r[1], "base_photo_key": r[2], "distance": r[3]}
                for r in self.db.execute("select photo_key, dhash, base_photo_key, distance "
                                         "from near_base")}

    def cite(self) -> dict:
        """What an experiment writes into its outputs (and the registry entry)."""
        out = {"dataset_release": self.id, "release_hash": self.release_hash,
               "reference_hash": self.reference_hash, "labels_hash": self.labels_hash,
               "release_code_commit": self.code_commit, "draft": self.manifest["draft"]}
        if self.errata is not None:
            out["errata"] = self.errata.cite()
        if self.photo_exclusions is not None:
            out["photo_exclusions"] = dict(self.photo_exclusions)
        return out

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
        return self._not_errata(self.db.execute(
            f"select * from records where included = 1 and split in ({q}) "
            "order by record_key", splits).fetchall())

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

    def photo_ids(self, splits=TRAINABLE, public_trainable_only: bool = False) -> list[int]:
        q = ",".join("?" * len(splits))
        pt = " and p.public_trainable = 1" if public_trainable_only else ""
        skip = self.errata.keys if self.errata is not None else frozenset()
        drop = self._photos_out_of(splits)
        return [r[0] for r in self.db.execute(
            f"select p.manifest_photo_id, r.record_key from photos p join records r "
            f"using (record_key) where p.included = 1 and r.included = 1 and r.split in ({q}){pt}",
            tuple(splits)) if r[1] not in skip and r[0] not in drop]

    def as_records(self, splits=TRAINABLE, photo_row: dict[int, int] | None = None,
                   public_trainable_only: bool = False, exclude: frozenset = frozenset()
                   ) -> list:
        """evaluate.Record objects for these splits: labels, dates and photos from the
        release, coordinates from private.sqlite (None when it was not shipped). photo_rows
        hold photo_row[photo id] for photos with a vector (all included photos when
        photo_row is None). public_trainable_only: the photos the final public model may
        learn from (CC or permission granted); a record left with none is left out. A read
        that holds train (learning, a reference) leaves out use_photo_exclusions' photos;
        `exclude` (photo_exclusion_set) leaves out more, for this read only."""
        from .evaluate import Record, real_date
        splits = (splits,) if isinstance(splits, str) else tuple(splits)
        places = {r[0]: (r[1], r[2]) for r in self.private.execute(
            "select record_key, lat_micro, lon_micro from record_places")} \
            if self.has_private else {}
        photos: dict[str, list[int]] = defaultdict(list)
        pt = " and public_trainable = 1" if public_trainable_only else ""
        drop = self._photos_out_of(splits) | frozenset(exclude)
        for key, pid in self.db.execute(
                f"select record_key, manifest_photo_id from photos where included = 1{pt} "
                "order by record_key, position, photo_key"):
            if pid not in drop:
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
                uuid=r["inat_uuid"] or "", **_source_fields(r)))
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

    def derive(self, photo_key: str, recipe: str, store_reader=None,
               store_location: str | None = None) -> Derived:
        """The derived image from our original, checked against the stored hashes.
        `store_location`: where the originals are when the release has no private part
        (a public variant on the server: MV_DATASET_PHOTO_STORE)."""
        p = self.db.execute("select * from photos where photo_key = ?", (photo_key,)).fetchone()
        if p is None:
            raise ReleaseError(f"{photo_key} is not in {self.id}")
        r = self.db.execute("select recipe_json from recipes where name = ?", (recipe,)).fetchone()
        if r is None:
            raise ReleaseError(f"{self.id} has no recipe {recipe!r}")
        stored_recipe = json.loads(r[0])
        loc = None
        if store_location is None and self.has_private:
            loc = self.private.execute(
                "select s.location from photo_stores ps join stores s using (store_id) "
                "where ps.photo_key = ?", (photo_key,)).fetchone()
        location = store_location or (loc[0] if loc else None)
        if not location or not p["original_path"] or not p["original_sha256"]:
            raise ReleaseError(f"{photo_key}: no kept original")
        body = (store_reader or default_store_reader())(location, p["original_path"])
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


def _source_fields(r) -> dict:
    """evaluate.Record's source fields, when this checkout's Record has them."""
    from dataclasses import fields

    from .evaluate import Record
    names = {f.name for f in fields(Record)}
    return {k: v for k, v in (("source", r["source"]), ("source_id", r["source_id"] or ""))
            if k in names}


def load_release(rid: str, root: Path | None = None, *, allow_draft: bool = False,
                 check_files: bool = True, without_private: bool = False) -> Release:
    """Open a finished release read-only. Checks MANIFEST.json and (by default) every
    file's sha256. Drafts are refused unless allow_draft. `without_private`: a copy
    shipped without private.sqlite (an AWS trainer needs no coordinates or owners); every
    other file is still checked, and anything that needs the private part says so."""
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
            if without_private and f["path"] == PRIVATE_DB:
                continue
            p = folder / f["path"]
            if not p.is_file() or p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
                raise ReleaseError(f"{rid}: {f['path']} does not match {MANIFEST}")
    return Release(rid, folder, manifest, _ro(folder / RELEASE_DB))


def verify(rid: str, root: Path | None = None, *, photos: str = "0", store_reader=None,
           allow_draft: bool = True, without_private: bool = False) -> dict:
    """Recompute every file hash and content hash; with photos (a number or 'all'),
    re-read originals from the store and check their sha256. `without_private`: a copy
    shipped without private.sqlite; its hash is taken from MANIFEST.json (unchecked)."""
    rel = load_release(rid, root, allow_draft=allow_draft, without_private=without_private)
    try:
        hashes = content_hashes(rel.db, None if without_private else rel.private,
                                private_hash=rel.manifest["hashes"]["private"]
                                if without_private else None)
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
                "originals_checked": checked, "private_checked": not without_private, "ok": True}
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
                            "freeze_approved": manifest.get("freeze_approved")},
                        extra_writer=write_zenodo_bundle, log=log)
    check_public(root / f"{rid}-cc")
    return out


# --- the Zenodo bundle (Steve, 2026-10-09: public files on Zenodo with a DOI) -------------

ZENODO_DIR = "zenodo"
TABLE_LICENCE = "cc-by-4.0"          # the tables, recipes and card; photos carry their own
CODE_REPO = "https://github.com/BiodiverseLabs/mycomap-vision"
RECORD_CSV_COLUMNS = ("record_key", "source", "source_id", "label", "label_rank", "species",
                      "genus", "family", "name_kind", "mycobank_number", "validated_on",
                      "observed_on", "country", "state", "included", "reason", "split")
PHOTO_CSV_COLUMNS = ("photo_key", "record_key", "position", "source", "source_photo_id",
                     "source_url", "license_code", "license_class", "owner_login",
                     "owner_name", "attribution", "original_sha256", "original_bytes",
                     "included", "reason")


def zenodo_metadata(info: dict, hashes: dict, counts: dict) -> dict:
    """Zenodo's deposit metadata. Creators beyond Steve are to be confirmed (TBC)."""
    rid = info["id"]
    return {"metadata": {
        "title": f"MycoMap Vision dataset release {rid}: DNA-validated fungal observations "
                 "with CC-licensed photos",
        "upload_type": "dataset",
        "version": rid,
        "description": (
            "DNA-validated (green in a mycomap.org project) iNaturalist and Mushroom Observer "
            "records from North America with their Creative Commons photos: record ids, "
            "labels with provenance, splits by validation date, the inclusion list with a "
            "reason per record and photo, photo licences and attribution, the sha256 of each "
            "original and of each image derived by a published recipe. No all-rights-reserved "
            "photos and no coordinates. Photos are not included as files: each row names the "
            "photo's public URL and licence; derived images can be regenerated and checked. "
            f"release_hash {hashes['release']}."),
        "creators": [{"name": "Russell, Stephen"}, {"name": "MycoMap contributors (TBC)"}],
        "keywords": ["fungi", "mycology", "DNA barcoding", "image classification",
                     "iNaturalist", "Mushroom Observer", "MycoMap"],
        "license": TABLE_LICENCE,
        "related_identifiers": [{"identifier": CODE_REPO, "relation": "isSupplementedBy",
                                 "scheme": "url"}],
        "notes": ("Photo licences are per photo (photos.csv license_code); the record-level "
                  f"licence ({TABLE_LICENCE}) covers the tables, recipes and dataset card. "
                  "Zenodo lists md5; sha256 in files.json and SHA256SUMS is the release's "
                  "primary hash."),
    }}


def write_zenodo_bundle(folder: Path, info: dict, hashes: dict, counts: dict) -> None:
    """zenodo/: metadata.json (deposit schema), records.csv, photos.csv (a licence per
    photo), recipes.json and files.json (every file of the upload with sha256, size and
    licence). CC photos only: the public variant holds no other."""
    z = folder / ZENODO_DIR
    z.mkdir()
    db = sqlite3.connect(folder / RELEASE_DB)
    try:
        recipes = [r[0] for r in db.execute("select name from recipes order by name")]
        derived = defaultdict(dict)
        for key, recipe, sha in db.execute("select photo_key, recipe, derived_sha256 from derived"):
            derived[key][recipe] = sha
        with open(z / "records.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(RECORD_CSV_COLUMNS)
            w.writerows(db.execute(f"select {', '.join(RECORD_CSV_COLUMNS)} from records "
                                   "order by record_key"))
        with open(z / "photos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(PHOTO_CSV_COLUMNS + tuple(f"{r}_sha256" for r in recipes))
            for row in db.execute(f"select {', '.join(PHOTO_CSV_COLUMNS)} from photos "
                                  "order by photo_key"):
                w.writerow(tuple(row) + tuple(derived[row[0]].get(r, "") for r in recipes))
        (z / "recipes.json").write_text(json.dumps(
            [json.loads(r[0]) for r in db.execute("select recipe_json from recipes order by name")],
            indent=2), encoding="utf-8")
    finally:
        db.close()
    (z / "metadata.json").write_text(json.dumps(zenodo_metadata(info, hashes, counts), indent=2),
                                     encoding="utf-8")
    listed = [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size,
               "sha256": sha256_file(p),
               "licence": ("per photo: photos.csv license_code" if p.name == "photos.csv"
                           else TABLE_LICENCE)}
              for p in sorted(folder.rglob("*")) if p.is_file() and p.name != PRIVATE_DB]
    (z / "files.json").write_text(json.dumps(
        {"note": "MANIFEST.json and SHA256SUMS are written after this list; MANIFEST.json "
                 "lists every file including this one.", "files": listed}, indent=2),
        encoding="utf-8")


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
