"""iNaturalist's suggestions under MycoMap Vision's answers (Steve, 2026-10-25).

Each photo, as soon as it arrives, is forwarded to mycomap.org, server to server
(POST /api/vision/inat-suggestions/photos with the server key, naming the member), while
Vision's own answer runs. mycomap.org asks iNat's computer vision with that member's own
linked iNat account, applies the thresholds (shown at 70, "not a fungus" at 80) and logs
iNat's answer on its side, joined by run id / client_ref. Vision never holds iNat tokens
and never puts iNat's answer in its own tables (identify log, early answers).

What is shown, under Vision's answer and never before it:
- at most three names iNat suggests that Vision's answer doesn't name (`lines`), each
  "iNaturalist suggests · 82%";
- "iNaturalist suggests this may not be a fungus · 92%" when every photo says so;
- on the website, iNat's full answer in a folded dropdown.
An iNaturalist link on the website ("from a link") gets the same (Steve, 2026-10-26): the
observation's photos as Vision read them, at most as many as an upload may have, with its
public place rounded to 0.1 degree and none when iNat obscures it. A kept answer reads the
photos again, off the request; a repeat costs no iNat call (mycomap.org's 90-day score
cache is keyed on the photo's bytes, which are the same each time: jpeg_of is
deterministic).

When Vision fails or times out on the website, the lines still show, with a Retry (Steve,
2026-10-25): the page names the find (`inat_ref`) when it sends the photos, and asks for
them once it knows Vision failed. Never while Vision is still checking.
The place goes with a website find only rounded to 0.1 degree (~10 km), never exact: the
place the person entered, else the first photo's EXIF one (Steve, 2026-10-25: people
forget to set the map). An app find sends its place only when its geoprivacy is open.
Members without a linked iNat account and anonymous visitors get none: mycomap.org says
so and nothing is shown. Any failure or a slow mycomap.org leaves the lines out quietly;
Vision's answer is never held back or broken by iNat.

Off unless MV_INAT_SUGGESTIONS=1: showing iNat's suggestions needs iNat's written
permission (requested 2026-10-25).

The rules match mycomap.org's for the app (services/inatSuggestions.ts there, contract in
its docs/inat-suggestions.md); tests/test_inat_suggest.py runs its test vectors.
"""

from __future__ import annotations

import io
import math
import re
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import config

ENDPOINT = "/api/vision/inat-suggestions/photos"
WAIT_SECONDS = 12.0             # from the photo's arrival; after it, the lines are dropped
POST_TIMEOUT = (3.0, 11.0)      # mycomap.org gives itself 10 s for iNat
MAX_LINES = 3
SHOW_AT, NOT_FUNGUS_AT = 70.0, 80.0     # Steve's thresholds (2026-10-25), unless .org says
JPEG_SIDE = 1024                # longest side sent; mycomap.org takes up to 4 MB
# The page's name for a find (inat_ref): with "site-" it is the client_ref, which
# mycomap.org keeps to 64 characters.
_REF = re.compile(r"^[A-Za-z0-9_-]{16,59}$")
MAX_PHOTOS = 10                 # per find, as an upload may have (guards.MAX_PHOTOS): each
                                # photo is one iNat call from mycomap.org's daily budget
MAX_IN_FLIGHT = 32              # forwards waiting or running; more are skipped
KEEP_SECONDS = 600              # how long a find's answers are kept in memory

Post = Callable[[dict, bytes], "dict | None"]


def enabled() -> bool:
    return (config.setting("MV_INAT_SUGGESTIONS") or "").strip().lower() in ("1", "true", "on")


# --- what is shown (pure) -------------------------------------------------------------

_QUOTES = re.compile(r"[\"'‘’“”]")
_QUALIFIER = re.compile(r"^(cf|aff)\.?\s+")
_INNER_QUALIFIER = re.compile(r"\s(cf|aff)\.?\s")


def name_key(name: str | None) -> str:
    """A name as names are compared: case and spacing ignored, quotes and a "cf." /
    "aff." removed (mycomap.org's nameKey, rule for rule)."""
    text = " ".join(_QUOTES.sub("", name or "").split()).lower()
    return _INNER_QUALIFIER.sub(" ", _QUALIFIER.sub("", text), count=1)


GIVEN_AT = 0.05     # a name Vision shows below 5% isn't one it gives (Steve, 2026-10-25)


def given(c: dict) -> bool:
    """Whether Vision gives this name: shown at 5% or more, or shown with its chance
    withheld (no number to go by)."""
    conf = c.get("confidence")
    return bool(c.get("name")) and (not isinstance(conf, (int, float)) or conf >= GIVEN_AT)


def vision_names(results: list[dict]) -> set[str]:
    """The names Vision's answer gives, at any rank, in every model shown: its ranked
    lists and its likely sets, each name at 5% or more. A name Vision lists lower (iNat
    said Hypholoma capnoides 85% where Vision listed it at 0%) leaves iNat's line shown."""
    out: set[str] = set()
    for r in results or []:
        for cands in (r.get("ranks") or {}).values():
            out.update(name_key(c["name"]) for c in cands or [] if given(c))
        for s in (r.get("likely") or {}).values():
            out.update(name_key(n["name"]) for n in (s or {}).get("names") or [] if given(n))
    out.discard("")
    return out


def merge(bodies: list[dict | None]) -> dict | None:
    """One find's answer from each photo's: per iNat taxon, its best score over the photos,
    a genus giving way to a species of it kept from another photo; "not a fungus" only when
    every photo iNat scored says so (a habitat shot can't flag the find), at the weakest
    share. None when no photo answered "ok"."""
    ok = [b for b in bodies if isinstance(b, dict) and b.get("status") == "ok"]
    if not ok:
        return None
    best: dict = {}
    for b in ok:
        for c in b.get("candidates") or []:
            tid, score = c.get("taxon_id"), c.get("score")
            if tid is None or not isinstance(score, (int, float)):
                continue
            held = best.get(tid)
            names = {*(held or {}).get("match_names", []), *(c.get("match_names") or [])}
            if held is None or score > held["score"]:
                held = dict(c)
            held["match_names"] = sorted(names)
            best[tid] = held
    kept = list(best.values())
    species = {name_key(c.get("name")) for c in kept if c.get("rank") != "genus"}
    kept = [c for c in kept if not (
        c.get("rank") == "genus" and (
            any(s.startswith(name_key(c.get("name")) + " ") for s in species)
            or any(c["taxon_id"] in (o.get("ancestor_ids") or []) for o in kept if o is not c)))]
    flags = [b.get("not_fungus") for b in ok]
    not_fungus = None
    if all(isinstance(f, dict) for f in flags):
        not_fungus = min(flags, key=lambda f: f.get("share") or 0)
    full = [b["full_response"] for b in ok if isinstance(b.get("full_response"), dict)]
    return {"candidates": sorted(kept, key=lambda c: -c["score"]),
            "not_fungus": not_fungus, "thresholds": ok[0].get("thresholds"),
            "full": full or None, "photos": len(bodies)}


def thresholds(merged: dict | None) -> tuple[float, float]:
    """(show at, "not a fungus" at): mycomap.org's settings as it sent them back, else
    Steve's. mycomap.org applies them; they are applied here again, so a line below them
    is never shown whatever is sent."""
    sent = (merged or {}).get("thresholds") or {}

    def number(key, default):
        v = sent.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default
    not_fungus = "notFungus" if "notFungus" in sent else "not_fungus"   # .org sends camelCase
    return number("show", SHOW_AT), number(not_fungus, NOT_FUNGUS_AT)


def public_place(obs) -> tuple[float, float] | None:
    """An iNat observation's place as a link find sends it: its public coordinates rounded
    to 0.1 degree; none when iNat obscures them or there are none (byobservation's
    `place`: "coordinates" / "obscured" / None). Never true coordinates: Vision only ever
    reads the public ones for a link."""
    if getattr(obs, "place", None) != "coordinates":
        return None
    return rounded_place(obs.latitude, obs.longitude)


def rounded_place(lat: float | None, lng: float | None) -> tuple[float, float] | None:
    """A place as a website find sends it: to 0.1 degree (~10 km); None without one."""
    if lat is None or lng is None:
        return None
    return round(float(lat), 1), round(float(lng), 1)


def percent(value: float) -> int:
    """Rounded as the app rounds (half up)."""
    return int(math.floor(value + 0.5))


def lines(merged: dict | None, names: set[str], max_lines: int = MAX_LINES) -> list[dict]:
    """The names to show: iNat's candidates Vision doesn't already give (any of a
    candidate's names, its current name or MycoMap's synonyms, matching any of Vision's),
    a genus dropped when Vision names a species in it; best first, at most three."""
    out = []
    show_at, _ = thresholds(merged)
    for c in (merged or {}).get("candidates") or []:
        if c["score"] < show_at:
            continue
        keys = {name_key(n) for n in [c.get("name"), *(c.get("match_names") or [])]} - {""}
        if keys & names:
            continue
        if c.get("rank") == "genus" and any(n.startswith(k + " ") for k in keys for n in names):
            continue
        out.append({"taxon_id": c.get("taxon_id"), "name": c.get("name"), "rank": c.get("rank"),
                    "score": c["score"],
                    "label": f"iNaturalist suggests · {percent(c['score'])}%"})
        if len(out) == max_lines:
            break
    return out


def shown(merged: dict | None, names: set[str], full: bool = True) -> dict | None:
    """What the page shows under Vision's answer, or None for nothing."""
    if merged is None:
        return None
    nf = merged.get("not_fungus")
    if isinstance(nf, dict) and not (nf.get("share") or 0) >= thresholds(merged)[1]:
        nf = None
    view = {"lines": lines(merged, names),
            "not_fungus": ({**nf, "label": "iNaturalist suggests this may not be a fungus · "
                                           f"{percent(nf.get('share') or 0)}%"}
                           if isinstance(nf, dict) else None),
            "full": merged.get("full") if full else None}
    if not view["lines"] and not view["not_fungus"] and not view["full"]:
        return None
    return view


# --- the photo sent ---------------------------------------------------------------------

def jpeg_of(image) -> bytes:
    """The decoded photo as mycomap.org takes it: a JPEG, its longest side at most 1024."""
    from PIL import Image
    img = image.convert("RGB")
    img.thumbnail((JPEG_SIDE, JPEG_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def org_poster(base_url: str, key: str, timeout=POST_TIMEOUT) -> Post:
    """POSTs one photo to mycomap.org; its JSON answer, or None for anything but a 200."""
    import requests
    url = base_url.rstrip("/") + ENDPOINT
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "User-Agent": config.USER_AGENT})

    def post(fields: dict, jpeg: bytes) -> dict | None:
        r = session.post(url, data={k: str(v) for k, v in fields.items()},
                         files={"photo": ("photo.jpg", jpeg, "image/jpeg")}, timeout=timeout)
        if r.status_code != 200:
            return None
        body = r.json()
        return body if isinstance(body, dict) else None
    return post


# --- forwarding, off the request ------------------------------------------------------

@dataclass
class Find:
    """One find's forwarded photos: whose, when, and each photo's answer as it comes."""
    key: str
    owner: str
    started: float
    futures: list[Future]
    names: set[str] | None = None           # Vision's names, once its answer is out
    full: bool = True
    done: Callable[[str, dict | None], None] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class Suggestions:
    """Forwards photos to mycomap.org on a few threads of its own, so no request waits on
    it, and keeps each website find's answers for a few minutes."""

    def __init__(self, post: Post, clock: Callable[[], float] = time.monotonic,
                 workers: int = 4, wait: float = WAIT_SECONDS, note=print):
        self.post = post
        self.clock = clock
        self.wait = wait
        self.note = note
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inat-suggest")
        # Link finds whose photos must be read first (a kept answer): their own threads,
        # so a slow read never holds up the forwards.
        self.readers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inat-suggest-read")
        self.room = threading.BoundedSemaphore(MAX_IN_FLIGHT)
        self.finds: dict[str, Find] = {}
        self.by_run: dict[str, str] = {}
        self.lock = threading.Lock()

    @classmethod
    def from_settings(cls, note=print) -> "Suggestions | None":
        """On when MV_INAT_SUGGESTIONS is set and mycomap.org's address and key are."""
        if not enabled():
            return None
        from .permissions import org_key
        base_url, key = config.setting("MV_ORG_BASE_URL"), org_key()
        if not base_url or not key:
            note("[inat-suggest] MV_INAT_SUGGESTIONS is on but MV_ORG_BASE_URL / "
                 "MV_ORG_VISION_KEY(_FILE) are not set: off")
            return None
        return cls(org_poster(base_url, key), note=note)

    def _send(self, fields: dict, image) -> dict | None:
        try:
            return self.post(fields, jpeg_of(image))
        except Exception as e:  # noqa: BLE001 - no lines, never an error for the person
            self.note(f"[inat-suggest] forward failed: {type(e).__name__}")
            return None
        finally:
            self.room.release()

    def forward(self, org_user_id: str, image, *, run_id: str | None = None,
                client_ref: str | None = None, photo_ref: str | None = None,
                location: tuple[float, float] | None = None, full: bool = False) -> Future | None:
        """Sends one photo, without waiting. None when it isn't sent (no join key, or too
        many already on their way)."""
        if not org_user_id or not (run_id or client_ref):
            return None
        if not self.room.acquire(blocking=False):
            self.note("[inat-suggest] too many photos on their way: one skipped")
            return None
        fields = {"org_user_id": org_user_id, "run_id": run_id, "client_ref": client_ref,
                  "photo_ref": photo_ref, "full": "1" if full else None}
        if location is not None:
            fields["lat"], fields["lng"] = location
        fields = {k: v for k, v in fields.items() if v is not None}
        try:
            return self.pool.submit(self._send, fields, image.copy())
        except Exception:  # noqa: BLE001 - shutting down
            self.room.release()
            return None

    def _key_for(self, owner: str, ref: str | None) -> str:
        """The find's key: site-<the page's ref> when well formed and not another member's,
        else a fresh one."""
        key = "site-" + (ref if ref and _REF.match(ref) else secrets.token_urlsafe(12))
        held = self.finds.get(key)
        if held is not None and held.owner != owner:
            key = "site-" + secrets.token_urlsafe(12)
        return key

    def _forward_all(self, owner: str, images: list, key: str, run_id: str | None,
                     place, full: bool) -> list[Future]:
        place = rounded_place(*place) if place is not None else None
        return [f for i, img in enumerate(images)
                if (f := self.forward(owner, img, run_id=run_id, client_ref=key,
                                      photo_ref=str(i + 1), location=place,
                                      full=full)) is not None]

    def _track(self, key: str, owner: str, futures: list[Future], done, full: bool) -> str:
        # The run's lines are found by link_run, which the caller makes only for the run's
        # own member: an observation's run is shared, so its id alone says nothing of whose.
        find = Find(key, owner, self.clock(), futures, full=full, done=done)
        with self.lock:
            self.finds[key] = find
        left = [len(futures)]

        def one_done(_):
            with find.lock:
                left[0] -= 1
                last = left[0] == 0
            if last and find.done is not None:
                try:
                    find.done(key, self._merged(find))
                except Exception as e:  # noqa: BLE001 - only the run's copy is lost
                    self.note(f"[inat-suggest] keeping it with the run failed: {type(e).__name__}")
        for f in futures:
            f.add_done_callback(one_done)
        return key

    def start(self, owner: str, images: list, *, run_id: str | None,
              done: Callable[[str, dict | None], None] | None = None, full: bool = True,
              ref: str | None = None, place: tuple[float, float] | None = None) -> str | None:
        """Forwards a website find's photos, with `place` rounded to 0.1 degree whatever
        precision it came in (rounded_place). Its key, for asking how it stands; None when
        nothing was sent. `done(key, merged)` once every photo has answered.
        `ref`: the page's own name for the find, so it can ask even when Vision's answer
        never reaches it; refused (a fresh key instead) when malformed or another
        member's."""
        self._forget_old()
        key = self._key_for(owner, ref)
        futures = self._forward_all(owner, images[:MAX_PHOTOS], key, run_id, place, full)
        if not futures:
            return None
        return self._track(key, owner, futures, done, full)

    def start_later(self, owner: str, read: Callable[[], "tuple[list, tuple | None] | None"], *,
                    run_id: str | None, done: Callable[[str, dict | None], None] | None = None,
                    full: bool = True, ref: str | None = None) -> str | None:
        """As start, for photos still to be read (a link whose answer was kept, so Vision
        read nothing): `read()` -> (images, place) or None, run off the request. Its key."""
        self._forget_old()
        key = self._key_for(owner, ref)

        def job() -> list:
            try:
                got = read()
            except Exception as e:  # noqa: BLE001 - no lines; the answer is unaffected
                self.note(f"[inat-suggest] reading the photos failed: {type(e).__name__}")
                return []
            if not got:
                return []
            images, place = got
            futures = self._forward_all(owner, images[:MAX_PHOTOS], key, run_id, place, full)
            finished, _ = wait(futures, timeout=self.wait + 5)
            return [f.result() if f in finished else None for f in futures]
        try:
            future = self.readers.submit(job)
        except Exception:  # noqa: BLE001 - shutting down
            return None
        return self._track(key, owner, [future], done, full)

    def link_run(self, key: str, run_id: str) -> None:
        """The run is this member's: asking for the run's lines now finds this find."""
        with self.lock:
            if key in self.finds:
                self.by_run[run_id] = key

    def answered(self, key: str, names: set[str]) -> None:
        """Vision's answer is out: from now on the lines may be shown, against its names."""
        find = self.finds.get(key)
        if find is not None:
            find.names = set(names)

    def key_of_run(self, run_id: str) -> str | None:
        return self.by_run.get(run_id)

    @staticmethod
    def _merged(find: Find) -> dict | None:
        bodies: list = []
        for f in find.futures:
            got = f.result() if f.done() and not f.cancelled() and f.exception() is None else None
            bodies.extend(got if isinstance(got, list) else [got])   # a read job: its photos'
        return merge(bodies)

    def view(self, key: str | None, owner: str | None, vision_failed: bool = False) -> dict | None:
        """How a find stands for the person who sent it, once Vision has answered (or, with
        `vision_failed`, the page knows it failed): {"pending": True, "ref"} while photos
        are out within the wait, then what is shown, or None (nothing to show, not theirs,
        unknown, past the wait, or Vision still checking)."""
        find = self.finds.get(key or "")
        if find is None or not owner or find.owner != owner:
            return None
        names = find.names if find.names is not None else (set() if vision_failed else None)
        if names is None:
            return None
        if not all(f.done() for f in find.futures):
            if self.clock() - find.started < self.wait:
                return {"pending": True, "ref": find.key}
            if not any(f.done() for f in find.futures):
                return None
        return shown(self._merged(find), names, find.full)

    def _forget_old(self) -> None:
        now = self.clock()
        with self.lock:
            for key in [k for k, f in self.finds.items() if now - f.started > KEEP_SECONDS]:
                self.finds.pop(key, None)
            for rid in [r for r, k in self.by_run.items() if k not in self.finds]:
                self.by_run.pop(rid, None)
