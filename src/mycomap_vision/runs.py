"""Run links: every identification on the site has its own link (Steve, 2026-10-20: "Each
vision run should just open the results with the link already made, so people can just
share their link immediately. Nobody else would ever have the link unless they or an
admin shared it.").

When the Identify page gets an answer (an upload, or an iNat/MO link), the run is kept
under an id and the page's address becomes /r/<id>. Anyone with the link may open it,
signed in or not: the three GET routes signin.run_path names, noindex, listed nowhere. The
id is the only way in: 96 random bits, never sequential.

What a run keeps (RunStore), and never anything else:
- the answer as anyone may see it: no one's own collections (`yours` is dropped wherever
  it is), plus the spare specimens and the photos' vectors, so the person who ran it gets
  their "Your collections" panel made afresh when they open it signed in. The vectors are
  never served;
- for uploads, copies at most COPY_SIDE px on the long side, re-encoded from the pixels:
  no EXIF, GPS, camera or time survives, and the original is never kept. An iNat or MO
  run keeps no photos, only the observation's own photo addresses at its source, each
  shown one's licence code and who took it (photos_credit), so its card can credit them;
- its link-preview card (linkcards.py) once made, as `card.jpg` beside it: an iNat or MO
  run's is drawn from its CC0 / CC BY photos, read from the source once;
- the place at 0.1 degree and the date, as the page shows them; never the exact point;
- the reference photos each specimen could show, so a photo whose photographer withdrew
  since is never shown (answercache.reshow_photos);
- who ran it: their mycomap.org id and name as the signed session gives them, so they can
  delete it and can find their past runs later (Steve, 2026-10-20: "keep their .org
  usernames associated with it so they might be able to review their past searches").
  Never served. OwnerIndex finds a person's runs without listing the bucket: rows on the
  box (state/site-runs.sqlite), and one list per person in the bucket
  (site-runs/owners/<sha256 of their id>.json) so losing the box loses nothing.

Kept until the person who ran it, or a mycomap.org admin, deletes it (Steve, 2026-10-20:
"No expiry. In S3 bucket."). The store of record is the Vision bucket's `site-runs/`
prefix (S3Objects; the box's IAM user may Put, Get and Delete there and nothing more, see
deploy/aws/box-policy.template.json); a laptop or a test uses a folder (DirObjects).
Only the site's runs are kept: API v1 and the app's sessions keep nothing new.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from .guards import check_image_size, photo_formats

ID_BYTES = 12                              # 16 url-safe characters
ID_PATTERN = r"[A-Za-z0-9_-]{16}"
_ID = re.compile(rf"^{ID_PATTERN}$")
PREFIX = "site-runs/"
COPY_SIDE = 1024
COPY_QUALITY = 80
KINDS = ("photos", "observation")
CARD = "card.jpg"                          # the run's link-preview image, beside it


def new_id() -> str:
    return secrets.token_urlsafe(ID_BYTES)


def valid_id(value: str | None) -> bool:
    return bool(_ID.match(value or ""))


def observation_run_id(secret: bytes, answer_key: str, answered_at: str) -> str:
    """One id per kept observation answer (answercache), the same each time it is served:
    keyed with the server's secret, so it can't be worked out from the observation."""
    mac = hmac.new(secret, f"run|{answer_key}|{answered_at}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac[:ID_BYTES]).decode("ascii")


def small_copy(body: bytes, side: int = COPY_SIDE) -> bytes:
    """A JPEG at most `side` px on its long side, drawn from the photo's pixels (turned the
    way its EXIF says first): nothing of the file's metadata is written."""
    with Image.open(io.BytesIO(body), formats=photo_formats()) as img:
        check_image_size(img)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((side, side), Image.LANCZOS)
        clean = Image.new("RGB", img.size)          # a fresh image carries no info
        clean.paste(img)
        out = io.BytesIO()
        clean.save(out, "JPEG", quality=COPY_QUALITY, optimize=True)
        return out.getvalue()


def without(value, keys: set):
    """A copy of JSON-like `value` with every `keys` entry removed, at any depth."""
    if isinstance(value, dict):
        return {k: without(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [without(v, keys) for v in value]
    return value


def coarse_context(used: dict) -> dict:
    """The place and date the answer used, as the page shows them: the place at 0.1 degree."""
    lat, lng = used.get("latitude"), used.get("longitude")
    return {"latitude": None if lat is None else round(float(lat), 1),
            "longitude": None if lng is None else round(float(lng), 1),
            "observed_on": used.get("observed_on"),
            "place_from": used.get("place_from"), "date_from": used.get("date_from")}


# --- where runs are kept ---------------------------------------------------------------

class DirObjects:
    """Objects as files under `root` (a laptop, a test)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(key)
        return path

    def put(self, key: str, body: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
        try:
            self._path(key).parent.rmdir()
        except OSError:
            pass


class S3Objects:
    """Objects in the Vision bucket. The box may not list it (no s3:ListBucket), so S3
    answers a missing key with 403 as well as 404: both are "no such object"."""

    MISSING = ("NoSuchKey", "404", "403", "AccessDenied")

    def __init__(self, bucket: str, client):
        self.bucket, self.client = bucket, client

    def put(self, key: str, body: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def get(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in self.MISSING:
                return None
            raise

    def exists(self, key: str) -> bool:
        """HEAD, which needs only s3:GetObject (and answers a missing key 403 or 404)."""
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in self.MISSING:
                return False
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)




class OwnerIndex:
    """Whose runs are whose: owner id and name, run id, kind, when (a person may also have
    opened an iNat/MO run someone else made first), and every run kept with its photo
    copies' bytes, for the box's health check. Thread-safe; `path=None` keeps it in memory.
    Nothing serves the owners yet: they are for a "your past runs" page later."""

    SCHEMA = """
    create table if not exists site_runs (
        owner_id   text not null,
        owner_name text,
        run_id     text not null,
        kind       text not null,
        created_at real not null,
        primary key (owner_id, run_id)
    );
    create index if not exists site_runs_by_run on site_runs (run_id);
    create table if not exists runs (
        run_id      text primary key,
        kind        text not null,
        created_at  real not null,
        photo_bytes integer not null default 0
    );
    """

    def __init__(self, path: Path | None):
        self._lock = threading.Lock()
        if path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self._conn.execute("pragma journal_mode = wal")
        with self._lock:
            self._conn.executescript(self.SCHEMA)

    def add(self, owner_id: str, owner_name: str | None, run_id: str, kind: str,
            created_at: float) -> bool:
        """True when the row is new (opening the same run again adds nothing)."""
        with self._lock:
            n = self._conn.execute(
                "insert or ignore into site_runs (owner_id, owner_name, run_id, kind, "
                "created_at) values (?, ?, ?, ?, ?)",
                (owner_id, owner_name, run_id, kind, created_at)).rowcount
            self._conn.commit()
        return bool(n)

    def runs_of(self, owner_id: str) -> list[dict]:
        """This person's runs, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "select run_id, kind, created_at from site_runs where owner_id = ? "
                "order by created_at desc", (owner_id,)).fetchall()
        return [{"run_id": r[0], "kind": r[1], "created_at": r[2]} for r in rows]

    def owners_of(self, run_id: str) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute(
                "select owner_id from site_runs where run_id = ?", (run_id,))]

    def remove_run(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("delete from site_runs where run_id = ?", (run_id,))
            self._conn.execute("delete from runs where run_id = ?", (run_id,))
            self._conn.commit()

    def add_run(self, run_id: str, kind: str, created_at: float, photo_bytes: int) -> None:
        """Every run kept, whoever ran it (signed out too), and its photo copies' bytes."""
        with self._lock:
            self._conn.execute(
                "insert into runs (run_id, kind, created_at, photo_bytes) values (?, ?, ?, ?) "
                "on conflict(run_id) do update set photo_bytes = excluded.photo_bytes",
                (run_id, kind, created_at, photo_bytes))
            self._conn.commit()

    def totals(self) -> dict:
        """Runs kept (by kind) and their photo copies' bytes, from this box's index."""
        with self._lock:
            rows = self._conn.execute(
                "select kind, count(*), coalesce(sum(photo_bytes), 0) from runs group by kind"
            ).fetchall()
        by_kind = {k: int(n) for k, n, _ in rows}
        return {"runs": sum(by_kind.values()), "by_kind": by_kind,
                "photo_bytes": int(sum(b for _, _, b in rows))}


def owner_list_key(owner_id: str) -> str:
    """The person's run list in the bucket (OwnerIndex's copy, for rebuilding it)."""
    return f"{PREFIX}owners/{hashlib.sha256(owner_id.encode()).hexdigest()[:32]}.json"


class RunStore:
    """Runs as `site-runs/<id>/run.json` and `site-runs/<id>/<n>.jpg`. Writes go on a
    background thread (`write`, by default a small pool), so the answer never waits for
    the bucket; until a write lands the run is answered from memory."""

    def __init__(self, objects, write: Callable[[Callable[[], None]], object] | None = None,
                 clock=time.time, note=print, cache_size: int = 256,
                 index: OwnerIndex | None = None):
        self.objects, self.clock, self.note = objects, clock, note
        self.index = index if index is not None else OwnerIndex(None)
        if write is None:
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="run-links")
            write = pool.submit
        self._write = write
        self._lock = threading.Lock()
        self._lists_lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._cache: OrderedDict[str, dict | None] = OrderedDict()
        self._cache_size = cache_size

    @staticmethod
    def _key(rid: str, name: str) -> str:
        return f"{PREFIX}{rid}/{name}"

    def keep(self, run: dict, photo_bodies: list[bytes] = (), rid: str | None = None) -> str:
        """Keeps `run` (made by run_record) and copies of these photos; its id. A run is
        stored with `yours` removed, whatever the caller passed, and goes on its owner's
        list."""
        if run.get("kind") not in KINDS:
            raise ValueError(f"unknown kind {run.get('kind')!r}")
        rid = rid or new_id()
        run = {**without(run, {"yours"}), "id": rid, "created": self.clock(),
               "photos": len(photo_bodies)}
        bodies = list(photo_bodies)
        owner = run.get("owner")
        fresh = self._index(owner, rid, run["kind"], run["created"])
        with self._lock:
            self._pending[rid] = run

        def job():
            try:
                size = 0
                for n, body in enumerate(bodies):
                    copy = small_copy(body)
                    size += len(copy)
                    self.objects.put(self._key(rid, f"{n}.jpg"), copy, "image/jpeg")
                self.objects.put(self._key(rid, "run.json"),
                                 json.dumps(run, separators=(",", ":")).encode(),
                                 "application/json")
                self.index.add_run(rid, run["kind"], run["created"], size)
                with self._lock:
                    self._remember(rid, run)
            except Exception as e:  # noqa: BLE001 - the link then answers 404; reported
                self.note(f"[runs] keeping {rid} failed: {type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._pending.pop(rid, None)
            if fresh:
                self._owner_list(owner, rid, run["kind"], run["created"])
        self._write(job)
        return rid

    def opened_by(self, owner: dict | None, rid: str, kind: str) -> None:
        """Adds an existing run to this person's list (an iNat/MO run they opened that
        someone else made first)."""
        if self._index(owner, rid, kind, self.clock()):
            created = self.clock()
            self._write(lambda: self._owner_list(owner, rid, kind, created))

    def _index(self, owner: dict | None, rid: str, kind: str, created: float) -> bool:
        if not owner or not owner.get("id"):
            return False
        return self.index.add(str(owner["id"]), owner.get("name"), rid, kind, created)

    def _owner_list(self, owner: dict | None, rid: str, kind: str, created: float) -> None:
        """The person's list in the bucket, so the box's index can be rebuilt."""
        key = owner_list_key(str(owner["id"]))
        try:
            with self._lists_lock:
                body = self.objects.get(key)
                doc = json.loads(body) if body else {"runs": []}
                doc["owner"] = {"id": str(owner["id"]), "name": owner.get("name")}
                if all(r.get("run_id") != rid for r in doc["runs"]):
                    doc["runs"].append({"run_id": rid, "kind": kind, "created_at": created})
                self.objects.put(key, json.dumps(doc, separators=(",", ":")).encode(),
                                 "application/json")
        except Exception as e:  # noqa: BLE001 - the box's index still has it
            self.note(f"[runs] owner list for {rid} failed: {type(e).__name__}: {e}")

    def _drop_from_lists(self, rid: str, owners: list[str]) -> None:
        for owner_id in owners:
            key = owner_list_key(owner_id)
            try:
                with self._lists_lock:
                    body = self.objects.get(key)
                    if not body:
                        continue
                    doc = json.loads(body)
                    doc["runs"] = [r for r in doc.get("runs") or [] if r.get("run_id") != rid]
                    self.objects.put(key, json.dumps(doc, separators=(",", ":")).encode(),
                                     "application/json")
            except Exception as e:  # noqa: BLE001 - the run itself is gone
                self.note(f"[runs] owner list after deleting {rid}: {type(e).__name__}: {e}")

    def _remember(self, rid: str, run: dict | None) -> None:
        self._cache[rid] = run
        self._cache.move_to_end(rid)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def get(self, rid: str) -> dict | None:
        """The run, or None when there is none (or it was deleted)."""
        if not valid_id(rid):
            return None
        with self._lock:
            if rid in self._pending:
                return self._pending[rid]
            if rid in self._cache:
                return self._cache[rid]
        body = self.objects.get(self._key(rid, "run.json"))
        try:
            run = json.loads(body) if body is not None else None
        except ValueError:
            run = None
        with self._lock:
            self._remember(rid, run)
        return run

    def photo(self, rid: str, n: int) -> bytes | None:
        run = self.get(rid)
        if run is None or run.get("kind") != "photos" or not 0 <= n < int(run.get("photos") or 0):
            return None
        return self.objects.get(self._key(rid, f"{n}.jpg"))

    def card(self, rid: str) -> bytes | None:
        """The run's link-preview card as kept beside it, or None (not made yet, no run)."""
        if self.get(rid) is None:
            return None
        return self.objects.get(self._key(rid, CARD))

    def has_card(self, rid: str) -> bool:
        """Whether the run's card is kept beside it (without reading it)."""
        if self.get(rid) is None:
            return False
        exists = getattr(self.objects, "exists", None)
        if exists is None:
            return self.objects.get(self._key(rid, CARD)) is not None
        return bool(exists(self._key(rid, CARD)))

    def keep_card(self, rid: str, body: bytes) -> bool:
        """Keeps the run's card beside it; False (and nothing kept) when the run is gone."""
        if self.get(rid) is None:
            return False
        self.objects.put(self._key(rid, CARD), body, "image/jpeg")
        if self.get(rid) is None:           # deleted while the card was being kept
            self.objects.delete(self._key(rid, CARD))
            return False
        return True

    def delete(self, rid: str) -> bool:
        """Removes the run, its photo copies, its card and its place on every owner's list;
        False when there was none."""
        run = self.get(rid)
        if run is None:
            return False
        for n in range(int(run.get("photos") or 0)):
            self.objects.delete(self._key(rid, f"{n}.jpg"))
        self.objects.delete(self._key(rid, CARD))
        self.objects.delete(self._key(rid, "run.json"))
        with self._lock:
            self._remember(rid, None)
        owners = self.index.owners_of(rid)
        self.index.remove_run(rid)
        self._drop_from_lists(rid, owners)
        return True

    def summary(self) -> dict:
        """For the box's health check (mycomap.org's server key only): runs kept since the
        index began, and how much room their photo copies take."""
        return self.index.totals()


def from_setting(value: str | None, state_dir: Path, note=print) -> RunStore | None:
    """MV_RUN_LINKS: `s3` keeps runs in the Vision bucket's site-runs/ (the box), a folder
    keeps them there (a laptop), and off or unset keeps none: no run gets a link. The owner
    index goes in `state_dir`."""
    where = (value or "").strip()
    if where.lower() in ("", "off", "0", "false", "no"):
        return None
    if where.lower() == "s3":
        from . import aws
        objects = S3Objects(aws.bucket(), aws.s3_client())
    else:
        objects = DirObjects(Path(where))
    return RunStore(objects, note=note, index=OwnerIndex(Path(state_dir) / "site-runs.sqlite"))


def run_record(kind: str, owner: dict | None, models: list[dict], context_used: dict,
               answered_at: str, release: str | None, observation: dict | None = None) -> dict:
    """What a run keeps. `owner`: {"id", "name"} from the signed session (None: nobody
    signed in). `models`: per model, {"model", "result", "spare", "vectors", "hidden",
    "photo_ids"} (api.site_kept's parts plus each specimen's reference photos)."""
    return {"kind": kind, "owner": owner, "models": models,
            "context_used": coarse_context(context_used), "answered_at": answered_at,
            "release": release, "observation": observation}


def public_observation(observation: dict | None) -> dict | None:
    """An iNat/MO run's observation as the page shows it: the source's own photo
    addresses (Creative Commons only, else null) and, beside them, each shown photo's
    licence code and who took it (photos_credit); never the place, never anything kept
    here."""
    if not observation:
        return None
    keep = ("source", "id", "url", "photos", "photos_used", "name_at_source",
            "rank_at_source", "photos_shown", "photos_credit", "excluded_reference",
            "left_out_photos")
    return {k: observation.get(k) for k in keep}
