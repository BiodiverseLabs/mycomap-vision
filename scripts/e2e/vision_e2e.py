"""End-to-end contract checks for the public API as the MycoMap app calls it (docs/api.md).

Drives a Vision server exactly as the app does: app tokens minted by mycomap.org's own
signer (mint.mts), photos prepared as the app prepares them (1024 px, JPEG q85, no EXIF).
Session checks are SKIPPED, not failed, while the server has no /sessions routes.

Usage: python vision_e2e.py <base> --key-file K [--private-pem P --aud A] [--quotas]
       --photos a.jpg|https://... b.jpg ... [--not-fungus duck.jpg|URL] [--json out.json]

Without --private-pem (e.g. on the server box, with its own server key): every call uses
the server key, and the checks that need app tokens (token refusals, member isolation,
usage, quotas) are SKIPPED. Photos may be local paths or https URLs (iNat open-data photos).
With --private-pem, set VISION_E2E_MINT_CWD to mycomap.org's artifacts/api-server folder
(mint.mts imports its signer from there). A server-key run is never counted against any
member's quota; each run opens and ends a few sessions.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

HERE = Path(__file__).parent
def api_server() -> Path:
    """mycomap.org's artifacts/api-server folder, for minting app tokens (mint.mts)."""
    where = os.environ.get("VISION_E2E_MINT_CWD")
    if not where:
        raise SystemExit("--private-pem needs VISION_E2E_MINT_CWD: mycomap.org's "
                         "artifacts/api-server folder")
    return Path(where)
BANNED = re.compile(r"bioclip|dinov|nearest\+|nearest-mix|-ft-\d|species-mean|fine-?tuned", re.I)
RESULTS: list[dict] = []


def record(name: str, ok: bool | None, detail: str = "") -> None:
    RESULTS.append({"check": name, "result": "PASS" if ok else "SKIP" if ok is None else "FAIL",
                    "detail": detail})
    print(f"{RESULTS[-1]['result']:4}  {name}{'  - ' + detail if detail else ''}", flush=True)


def app_copy(path) -> bytes:
    src = str(path)
    if src.startswith(("http://", "https://")):
        r = requests.get(src, timeout=60)
        r.raise_for_status()
        path = io.BytesIO(r.content)
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        img = img.resize((1024, round(h * 1024 / w)) if w >= h else (round(w * 1024 / h), 1024),
                         Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


class Mint:
    def __init__(self, pem: str, aud: str):
        self.pem, self.aud = str(Path(pem).resolve()), aud

    def __call__(self, sub: str, plan="free", limit="100", offset=0, aud=None) -> str:
        return subprocess.run(
            ["node", "--import", "tsx", str(HERE / "mint.mts"), self.pem, aud or self.aud, sub,
             plan, str(limit), str(offset)],
            cwd=api_server(), capture_output=True, text=True, check=True, shell=os.name == "nt",
        ).stdout.strip()


def top(result: dict, rank: str, n: int = 5) -> list[tuple[str, float | None]]:
    return [(c["name"], c.get("confidence")) for c in (result.get("ranks") or {}).get(rank, [])[:n]]


def same_answer(a: dict, b: dict, tol: float = 1e-3) -> tuple[bool, str]:
    for rank in ("species", "genus", "family"):
        ta, tb = top(a, rank), top(b, rank)
        if [x[0] for x in ta] != [x[0] for x in tb]:
            return False, f"{rank} names differ: {ta[:3]} vs {tb[:3]}"
        for (_, ca), (_, cb) in zip(ta, tb):
            if (ca is None) != (cb is None) or (ca is not None and abs(ca - cb) > tol):
                return False, f"{rank} confidences differ: {ta[:3]} vs {tb[:3]}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--private-pem")
    ap.add_argument("--aud", default="https://vision.mycomap.org")
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--not-fungus")
    ap.add_argument("--json")
    ap.add_argument("--quotas", action="store_true")
    a = ap.parse_args()
    v1 = a.base.rstrip("/") + "/api/v1"
    server = {"Authorization": f"Bearer {Path(a.key_file).read_text().strip()}"}
    tokens = bool(a.private_pem)
    mint = Mint(a.private_pem, a.aud) if tokens else None
    tok_a = {"Authorization": f"Bearer {mint('u_e2e_member_a')}"} if tokens else server
    tok_b = {"Authorization": f"Bearer {mint('u_e2e_member_b')}"} if tokens else None
    if not tokens and a.quotas:
        ap.error("--quotas needs app tokens (--private-pem)")
    photos = [app_copy(p) for p in a.photos]
    seen_bodies: list[str] = []

    def call(method, path, headers=None, **kw):
        r = requests.request(method, v1 + path, headers=headers or {}, timeout=300, **kw)
        seen_bodies.append(r.text)
        try:
            body = r.json()
        except ValueError:
            body = None
        return r, body

    def identify(headers, idx, data=None):
        files = [("photos", (f"p{i}.jpg", photos[i], "image/jpeg")) for i in idx]
        return call("POST", "/identify", headers, files=files, data={"top_k": "5", **(data or {})})

    # --- identify (the app's path for records saved offline) -------------------------
    r, b = identify(tok_a, [0, 1, 2])
    record("identify answers 200 (" + ("an .org-minted app token" if tokens else "the server key") + ")", r.status_code == 200,
           f"{r.status_code} {str(b)[:160]}" if r.status_code != 200 else "")
    base3 = b if r.status_code == 200 else None
    if base3:
        m = base3.get("model") or {}
        record("model.name is 'MycoMap Vision'", m.get("name") == "MycoMap Vision", str(m))
        record("model.version present and opaque ([A-Za-z0-9._:-]{1,40})",
               isinstance(m.get("version"), str) and bool(re.fullmatch(r"[A-Za-z0-9._:-]{1,40}", m["version"])),
               f"version={m.get('version')!r}")
        record("model.data {snapshot, records, species, genera}",
               isinstance(m.get("data"), dict) and {"snapshot", "species"} <= set(m["data"]),
               f"data={m.get('data')!r}")
        record("Cache-Control: no-store", "no-store" in r.headers.get("Cache-Control", ""),
               r.headers.get("Cache-Control", ""))
        record("Server-Timing header and timing block",
               bool(r.headers.get("Server-Timing")) and {"queue_ms", "embed_ms", "total_ms"} <= set(base3.get("timing") or {}))
        record("ranks species/genus/family/order/class/phylum",
               {"species", "genus", "family", "order", "class", "phylum"} <= set(base3.get("ranks") or {}),
               f"ranks={sorted(base3.get('ranks') or {})}")
        record("`answer` key present (headline the app shows)", "answer" in base3)
        record("gate verdict per photo", len(((base3.get("gate") or {}).get("photos") or [])) == 3)
        record("usage block filled for an app token (§4.1; needs MV_QUOTAS count|enforce)",
               isinstance(base3.get("usage"), dict) if tokens else None,
               f"usage={base3.get('usage')!r}")
        ctx = base3.get("context_used") or {}
        record("no EXIF place/date taken from app copies (the app strips EXIF)",
               ctx.get("latitude") is None and ctx.get("observed_on") is None, str(ctx))

    r, b = call("POST", "/identify", files=[("photos", ("p.jpg", photos[0], "image/jpeg"))])
    record("no credential -> 401 invalid_token", r.status_code == 401 and (b or {}).get("code") == "invalid_token",
           f"{r.status_code} {b}")
    if not tokens:
        for name in ("tampered app token -> 401", "expired app token -> 401 token_expired",
                     "token for another audience -> 401"):
            record(name, None, "no app tokens (run with --private-pem)")
    bad = (tok_a["Authorization"][:-4] if tokens else "Bearer mvt.x.y") + ("AAAA" if not tok_a["Authorization"].endswith("AAAA") else "BBBB")
    r, b = identify({"Authorization": bad}, [0]) if tokens else (None, None)
    if tokens:
        record("tampered app token -> 401 invalid_token",
               r.status_code == 401 and (b or {}).get("code") == "invalid_token", f"{r.status_code} {b}")
        r, b = identify({"Authorization": f"Bearer {mint('u_e2e_old', offset=-3600)}"}, [0])
        record("expired app token -> 401 token_expired (app re-mints once)",
               r.status_code == 401 and (b or {}).get("code") == "token_expired", f"{r.status_code} {b}")
        r, b = identify({"Authorization": f"Bearer {mint('u_e2e_x', aud='https://evil.example')}"}, [0])
        record("token for another audience -> 401", r.status_code == 401, f"{r.status_code} {b}")
    files = [("photos", (f"p{i}.jpg", photos[i % len(photos)], "image/jpeg")) for i in range(11)]
    r, b = call("POST", "/identify", tok_a, files=files)
    record("11 photos -> 400 too_many_photos", r.status_code == 400 and (b or {}).get("code") == "too_many_photos",
           f"{r.status_code} {b}")
    r, b = identify(tok_a, [0], {"observed_on": "11/10/2026"})
    record("bad date -> 400 bad_request", r.status_code == 400 and (b or {}).get("code") == "bad_request", f"{r.status_code}")
    r, b = identify(tok_a, [0], {"latitude": "40.1"})
    record("latitude without longitude -> 400 bad_request", r.status_code == 400, f"{r.status_code}")
    r, b = call("POST", "/identify", tok_a, files=[("photos", ("x.jpg", b"not an image", "image/jpeg"))])
    record("not an image -> 415 unsupported_image", r.status_code == 415 and (b or {}).get("code") == "unsupported_image",
           f"{r.status_code} {b}")
    r, b = call("GET", "/no-such-route", tok_a)
    record("unknown v1 path -> JSON 404 code not_found (app: 'unavailable', no retry loop)",
           r.status_code == 404 and (b or {}).get("code") == "not_found", f"{r.status_code} {str(b)[:120]}")
    if a.not_fungus:
        nf = app_copy(a.not_fungus)
        r, b = call("POST", "/identify", tok_a, files=[("photos", ("duck.jpg", nf, "image/jpeg"))])
        ok = r.status_code == 200 and (b.get("gate") or {}).get("verdict") == "not_fungus" and not b.get("answer")
        record("non-fungus photo -> 200, gate not_fungus, no headline answer", ok,
               f"{r.status_code} gate={(b or {}).get('gate', {}).get('verdict') if b else None} answer={(b or {}).get('answer')}")

    # --- sessions (the app's live path while photos are taken) ------------------------
    # A fresh member, so the 20 photos/min member limit isn't spent by the checks above;
    # the reference identifies use the server key (no member limit).
    tok_a = {"Authorization": f"Bearer {mint('u_e2e_member_c')}"} if tokens else server
    r, b = call("POST", "/sessions", tok_a, json={"client_ref": "e2e-1"})
    if r.status_code in (404, 405) and (b or {}).get("code") != "session_gone":
        record("sessions routes exist", False, f"POST /sessions -> {r.status_code} {str(b)[:120]}")
        for name in ("session create/add/resend/patch/get/delete-photo/end", "session == identify",
                     "session isolation between members", "session photo cap"):
            record(name, None, "no /sessions routes on this server")
    else:
        record("POST /sessions -> 201 with session_id, expires_at, max_photos, model",
               r.status_code == 201 and {"session_id", "expires_at", "max_photos", "model"} <= set(b or {}),
               f"{r.status_code} {str(b)[:200]}")
        sid = (b or {}).get("session_id")
        record("session max_photos is 10", (b or {}).get("max_photos") == 10, str((b or {}).get("max_photos")))
        answers = {}
        for i, ref in enumerate(["p1", "p2", "p3"]):
            r, b = call("POST", f"/sessions/{sid}/photos", tok_a,
                        files={"photo": (f"{ref}.jpg", photos[i], "image/jpeg")}, data={"photo_ref": ref})
            answers[ref] = b
            good = (r.status_code == 200 and {"photo", "combined", "photos_scored", "timing", "session"} <= set(b or {})
                    and b["photos_scored"] == i + 1 and (b.get("photo") or {}).get("photo_ref") == ref)
            record(f"add photo {ref} -> 200, photos_scored {i + 1}, photo_ref echoed", good,
                   f"{r.status_code} {str(b)[:200]}" if not good else "")
        r, b = call("POST", f"/sessions/{sid}/photos", tok_a,
                    files={"photo": ("p2.jpg", photos[1], "image/jpeg")}, data={"photo_ref": "p2"})
        t = (b or {}).get("timing") or {}
        record("resend p2 (dropped response) -> same answer, nothing re-embedded, still 3 photos",
               r.status_code == 200 and (b or {}).get("photos_scored") == 3 and t.get("photos_embedded", 1) == 0
               and same_answer(b["photo"], answers["p2"]["photo"])[0],
               f"{r.status_code} scored={(b or {}).get('photos_scored')} timing={t}")
        if base3:
            ok, why = same_answer(answers["p3"]["combined"], identify(server, [0, 1, 2])[1])
            record("session combined after 3 photos == POST /identify of the same 3", ok, why)
        place = {"latitude": 40.1234, "longitude": -82.9876, "observed_on": "2026-10-11"}
        r, b = call("PATCH", f"/sessions/{sid}", tok_a, json=place)
        cu = (b or {}).get("context_used") or {}
        record("PATCH place/date -> 200 {context_used, combined}, place rounded to 0.1",
               r.status_code == 200 and "combined" in (b or {}) and cu.get("latitude") in (40.1, 40.12, 40.123),
               f"{r.status_code} context_used={cu}")
        r2, b2 = identify(server, [0, 1, 2], {k: str(v) for k, v in place.items()})
        if r.status_code == 200 and r2.status_code == 200:
            ok, why = same_answer(b["combined"], b2)
            record("re-scored session == identify with the same place/date", ok, why)
        r, b = call("GET", f"/sessions/{sid}", tok_a)
        record("GET session -> combined with per_photo for 3 photos",
               r.status_code == 200 and len(((b or {}).get("combined") or {}).get("per_photo") or []) == 3,
               f"{r.status_code}")
        r, b = call("GET", f"/sessions/{sid}", tok_b) if tokens else (None, None)
        if not tokens:
            record("another member's token -> 404 session_gone (no leak)", None, "no app tokens")
        else:
            record("another member's token -> 404 session_gone (no leak)",
                   r.status_code == 404 and (b or {}).get("code") == "session_gone", f"{r.status_code} {b}")
        r, b = call("DELETE", f"/sessions/{sid}/photos/p2", tok_a)
        r2, b2 = identify(server, [0, 2], {k: str(v) for k, v in place.items()})
        if r.status_code == 200 and r2.status_code == 200:
            ok, why = same_answer(b["combined"], b2)
            record("DELETE photo p2 -> combined == identify of p1+p3", ok, why)
        else:
            record("DELETE photo p2 -> 200", False, f"{r.status_code} {b}")
        for i in range(4, 13):   # p2 was removed: p1, p3 + 8 = 10, then one more
            r, b = call("POST", f"/sessions/{sid}/photos", tok_a,
                        files={"photo": (f"p{i}.jpg", photos[i % len(photos)], "image/jpeg")},
                        data={"photo_ref": f"p{i}"})
        record("photo past the cap -> 400 too_many_photos",
               r.status_code == 400 and (b or {}).get("code") == "too_many_photos", f"{r.status_code} {b}")
        r, _ = call("DELETE", f"/sessions/{sid}", tok_a)
        record("DELETE session -> 204", r.status_code == 204, str(r.status_code))
        r, b = call("GET", f"/sessions/{sid}", tok_a)
        record("ended session -> 404 session_gone (app reopens and resends)",
               r.status_code == 404 and (b or {}).get("code") == "session_gone", f"{r.status_code} {b}")
        r, b = call("POST", "/sessions/s_doesnotexist/photos", tok_a,
                    files={"photo": ("p.jpg", photos[0], "image/jpeg")}, data={"photo_ref": "p1"})
        record("photo to an unknown session -> 404 session_gone",
               r.status_code == 404 and (b or {}).get("code") == "session_gone", f"{r.status_code} {b}")


    # --- monthly quotas (MV_QUOTAS=enforce; docs/api.md §6.4) ---------------------------
    if a.quotas:
        run = str(int(time.time()))[-6:]
        lim1 = {"Authorization": f"Bearer {mint('u_e2equota1_' + run, limit=1)}"}
        r, b = identify(lim1, [0])
        u = (b or {}).get("usage") or {}
        record("quota: 1st identify on a limit-1 token -> 200, usage used 1 limit 1",
               r.status_code == 200 and u.get("used") == 1 and u.get("limit") == 1
               and re.fullmatch(r"\d{4}-\d{2}", str(u.get("period"))) is not None and bool(u.get("resets_at")),
               f"{r.status_code} usage={u}")
        r, b = identify(lim1, [1])
        record("quota: 2nd identify -> 429 quota_exceeded {limit 1, used 1, resets_at} + Retry-After",
               r.status_code == 429 and (b or {}).get("code") == "quota_exceeded" and b.get("limit") == 1
               and b.get("used") == 1 and bool(b.get("resets_at")) and bool(r.headers.get("Retry-After")),
               f"{r.status_code} {b} retry-after={r.headers.get('Retry-After')}")
        lim2 = {"Authorization": f"Bearer {mint('u_e2equota2_' + run, limit=2)}"}
        r, b = call("POST", "/sessions", lim2, json={})
        sid = (b or {}).get("session_id")
        used = []
        for i in range(3):
            r, b = call("POST", f"/sessions/{sid}/photos", lim2,
                        files={"photo": (f"q{i}.jpg", photos[i], "image/jpeg")}, data={"photo_ref": f"q{i}"})
            used.append(((b or {}).get("usage") or {}).get("used"))
        cu = (((b or {}).get("combined") or {}).get("usage") or {}).get("used")
        record("quota: a session counts once, at its first photo (3 photos -> used 1, also in combined)",
               used == [1, 1, 1] and cu == 1, f"used per photo {used}, combined {cu}")
        r, b = identify(lim2, [0])
        record("quota: identify after the session -> used 2 (one pool per member)",
               r.status_code == 200 and ((b or {}).get("usage") or {}).get("used") == 2, f"{r.status_code} {(b or {}).get('usage')}")
        r, b = call("POST", f"/sessions/{sid}/photos", lim2,
                    files={"photo": ("q3.jpg", photos[3], "image/jpeg")}, data={"photo_ref": "q3"})
        record("quota: the already-counted session keeps taking photos at the limit",
               r.status_code == 200, f"{r.status_code} {str(b)[:160]}")
        r, b = call("POST", "/sessions", lim2, json={})
        sid2 = (b or {}).get("session_id")
        r, b = call("POST", f"/sessions/{sid2}/photos", lim2,
                    files={"photo": ("z.jpg", photos[0], "image/jpeg")}, data={"photo_ref": "z"})
        record("quota: a NEW session past the limit -> 429 quota_exceeded at its first photo",
               r.status_code == 429 and (b or {}).get("code") == "quota_exceeded", f"{r.status_code} {b}")
        # Steve, 2026-10-12: a session re-opened for the SAME find (same pool + client_ref) within
        # 24 h of the counted one is free; a new find counts; no client_ref always counts.
        one = {"Authorization": f"Bearer {mint('u_e2equota4_' + run, limit=1)}"}
        def first_photo(ref):
            r0, b0 = call("POST", "/sessions", one, json={"client_ref": ref} if ref else {})
            sid_ = (b0 or {}).get("session_id")
            r1, b1 = call("POST", f"/sessions/{sid_}/photos", one,
                          files={"photo": ("f.jpg", photos[0], "image/jpeg")}, data={"photo_ref": "f1"})
            return sid_, r1, b1
        s1, r, b = first_photo("find-1")
        call("DELETE", f"/sessions/{s1}", one)
        s2, r, b = first_photo("find-1")
        record("quota: the same find re-opened (session lost) is free at the limit, used stays 1",
               r.status_code == 200 and ((b or {}).get("usage") or {}).get("used") == 1, f"{r.status_code} {(b or {}).get('usage') or b}")
        _, r, b = first_photo("find-2")
        record("quota: a new find (new client_ref) at the limit -> 429 quota_exceeded",
               r.status_code == 429 and (b or {}).get("code") == "quota_exceeded", f"{r.status_code} {b}")
        _, r, b = first_photo(None)
        record("quota: a session without client_ref is counted (refused at the limit)",
               r.status_code == 429 and (b or {}).get("code") == "quota_exceeded", f"{r.status_code} {b}")
        # /identify takes client_ref too (the app's background finisher), same 24 h rule both ways.
        five = {"Authorization": f"Bearer {mint('u_e2equota5_' + run, limit=1)}"}
        r0, b0 = call("POST", "/sessions", five, json={"client_ref": "find-bg"})
        sid5 = (b0 or {}).get("session_id")
        call("POST", f"/sessions/{sid5}/photos", five, files={"photo": ("f.jpg", photos[0], "image/jpeg")}, data={"photo_ref": "f1"})
        r, b = identify(five, [0, 1, 2], {"client_ref": "find-bg"})
        record("quota: background /identify of a find its session already counted -> free, used 1",
               r.status_code == 200 and ((b or {}).get("usage") or {}).get("used") == 1, f"{r.status_code} {(b or {}).get('usage') or b}")
        r, b = identify(five, [0], {"client_ref": "find-other"})
        record("quota: /identify of another find at the limit -> 429", r.status_code == 429 and (b or {}).get("code") == "quota_exceeded", f"{r.status_code}")
        r, b = identify(five, [0])
        record("quota: /identify without client_ref at the limit -> 429", r.status_code == 429 and (b or {}).get("code") == "quota_exceeded", f"{r.status_code}")
        six = {"Authorization": f"Bearer {mint('u_e2equota6_' + run, limit=1)}"}
        r, b = identify(six, [0], {"client_ref": "find-rev"})
        r0, b0 = call("POST", "/sessions", six, json={"client_ref": "find-rev"})
        sid6 = (b0 or {}).get("session_id")
        r, b = call("POST", f"/sessions/{sid6}/photos", six, files={"photo": ("f.jpg", photos[1], "image/jpeg")}, data={"photo_ref": "f1"})
        record("quota: a session for a find /identify already counted -> free, used 1",
               r.status_code == 200 and ((b or {}).get("usage") or {}).get("used") == 1, f"{r.status_code} {(b or {}).get('usage') or b}")
        r, b = identify(tok_a, [0], {"client_ref": "x" * 65})
        record("/identify client_ref over 64 chars -> 400 bad_request", r.status_code == 400 and (b or {}).get("code") == "bad_request", f"{r.status_code} {b}")
        unl = {"Authorization": f"Bearer {mint('u_e2equota3_' + run, limit='null')}"}
        r, b = identify(unl, [0])
        u = (b or {}).get("usage") or {}
        record("quota: limit null (staff) -> unlimited, usage.limit null", r.status_code == 200 and "limit" in u and u["limit"] is None,
               f"{r.status_code} usage={u}")
        r, b = identify(server, [0])
        record("quota: the .org server key is never counted (usage null)", r.status_code == 200 and b.get("usage") is None,
               f"usage={(b or {}).get('usage')}")

    leaks = sorted({m.group(0) for t in seen_bodies for m in BANNED.finditer(t)})
    record("no backbone/method/fine-tune names in any v1 answer or refusal", not leaks, ", ".join(leaks))
    fails = sum(r["result"] == "FAIL" for r in RESULTS)
    print(f"\n{sum(r['result'] == 'PASS' for r in RESULTS)} pass, {fails} fail, "
          f"{sum(r['result'] == 'SKIP' for r in RESULTS)} skipped")
    if a.json:
        Path(a.json).write_text(json.dumps(RESULTS, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
