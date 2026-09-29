"""Photographers' answers from mycomap.org, and what may be shown or used because of them."""

import threading
from datetime import datetime, timedelta, timezone

import pytest
from test_api import app_with_model, post_photos
from test_models_and_scoreboard import seed_two_species

from mycomap_vision import api, evaluate, inat, permissions
from mycomap_vision.permissions import PermissionSyncError, PermissionView

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
KEY = "k" * 64


def answer(*rows, generated="2026-10-01T12:00:00.000Z"):
    return {"generatedAt": generated, "currentTermsVersion": "v1",
            "permissions": [{"inatUserId": uid, "inatLogin": f"u{uid}", "status": status,
                             "termsVersion": "v1", "grantedAt": None, "withdrawnAt": None,
                             "updatedAt": None} for uid, status in rows]}


def set_photo(conn, photo_id, license_class, owner):
    conn.execute("update photos set license_class = ?, license_code = ?, owner_user_id = ? "
                 "where photo_id = ?", (license_class, "" if license_class == "arr" else "cc0",
                                        owner, photo_id))
    conn.commit()


# --- pulling the answers ---------------------------------------------------------

class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_fetch_asks_the_vision_endpoint_with_the_bearer_key():
    s = FakeSession(FakeResponse(200, answer((1, "granted"))))
    body = permissions.fetch("https://org.example/", KEY, session=s)
    assert body["permissions"][0]["inatUserId"] == 1
    url, headers = s.calls[0]
    assert url == "https://org.example/api/vision/photo-permissions"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert headers["User-Agent"].startswith("MycoMap-Vision/")


@pytest.mark.parametrize("response, reason", [
    (FakeResponse(401), "refused the key"),
    (FakeResponse(503), "no VISION_API_KEY"),
    (FakeResponse(500), "answered 500"),
    (FakeResponse(200, None), "not JSON"),
    (OSError("boom " + KEY), "could not reach"),
])
def test_fetch_failures_say_why_and_never_repeat_the_key(response, reason):
    import requests
    if isinstance(response, OSError):
        response = requests.ConnectionError(str(response))
    with pytest.raises(PermissionSyncError) as e:
        permissions.fetch("https://org.example", KEY, session=FakeSession(response))
    assert reason in str(e.value)
    assert KEY not in str(e.value)


def test_a_sync_replaces_the_copy_and_records_the_attempt(conn):
    permissions.sync(conn, "https://org", KEY, fetcher=lambda b, k: answer((1, "granted"), (2, "withdrawn")),
                     now=lambda: NOW)
    stats = permissions.sync(conn, "https://org", KEY, fetcher=lambda b, k: answer((2, "granted")),
                             now=lambda: NOW + timedelta(minutes=5))
    assert stats == {"people": 1, "granted": 1, "withdrawn": 0}
    assert [tuple(r) for r in conn.execute("select inat_user_id, status from photo_permissions")] == [(2, "granted")]
    assert permissions.last_good_sync(conn) == NOW + timedelta(minutes=5)


def test_a_failed_or_malformed_sync_keeps_the_previous_copy_and_is_recorded(conn):
    permissions.sync(conn, "https://org", KEY, fetcher=lambda b, k: answer((1, "granted")), now=lambda: NOW)

    def down(b, k):
        raise PermissionSyncError("mycomap.org answered 500")
    for bad in (down, lambda b, k: answer((1, "maybe")), lambda b, k: answer((1, "granted"), (1, "granted")),
                lambda b, k: {"nope": True}):
        with pytest.raises(PermissionSyncError):
            permissions.sync(conn, "https://org", KEY, fetcher=bad, now=lambda: NOW + timedelta(hours=1))
    assert [tuple(r) for r in conn.execute("select inat_user_id, status from photo_permissions")] == [(1, "granted")]
    assert permissions.last_good_sync(conn) == NOW
    report = permissions.status_report(conn)
    assert report["last_attempt"]["ok"] is False and report["granted"] == 1


# --- the showing rule ----------------------------------------------------------

def test_cc_photos_follow_their_licence_and_arr_photos_need_a_fresh_grant():
    view = PermissionView(frozenset({42}), frozenset({43}), NOW, timedelta(hours=1))
    at = NOW + timedelta(minutes=10)
    assert view.may_show("open", 43, at) and view.may_show("nc", None, at)
    assert view.may_show("arr", 42, at)
    assert not view.may_show("arr", 43, at), "withdrawn"
    assert not view.may_show("arr", 44, at), "no answer yet"
    assert not view.may_show("arr", None, at)
    assert not view.may_show(None, 42 + 100, at), "unknown licence counts as all rights reserved"
    assert not view.may_show("arr", 42, NOW + timedelta(hours=2)), "a grant read too long ago"
    assert not PermissionView(frozenset({42})).may_show("arr", 42, at), "never synced"


def test_a_record_shows_its_best_photo_that_is_cc_or_granted():
    from mycomap_vision.identify import PhotoInfo, pick_best
    view = PermissionView(frozenset({42}), frozenset({43}), NOW, timedelta(hours=1))
    photos = [PhotoInfo("u", "arr", "a", 43), PhotoInfo("u", "arr", "b", 42),
              PhotoInfo("u", "nc", "c", 43), PhotoInfo("u", "arr", "d", 44)]
    ok = [view.may_show(p.license_class, p.owner_user_id, NOW) for p in photos]
    assert pick_best([0.99, 0.9, 0.8, 0.95], ok) == 1, "the granted photo beats a CC one it outscores"
    assert pick_best([0.99, 0.7, 0.8, 0.95], ok) == 2
    assert pick_best([0.9], [False]) is None


def test_the_key_can_come_from_a_file_like_the_other_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("MV_ORG_VISION_KEY", raising=False)
    monkeypatch.delenv("MV_ORG_VISION_KEY_FILE", raising=False)
    assert permissions.org_key() is None
    f = tmp_path / "org-vision-key"
    f.write_text(KEY + "\n")
    monkeypatch.setenv("MV_ORG_VISION_KEY_FILE", str(f))
    assert permissions.org_key() == KEY
    monkeypatch.setenv("MV_ORG_VISION_KEY", "from-env")
    assert permissions.org_key() == "from-env", "a key in the environment wins"
    monkeypatch.delenv("MV_ORG_VISION_KEY")
    monkeypatch.setenv("MV_ORG_VISION_KEY_FILE", str(tmp_path / "missing"))
    assert permissions.org_key() is None


def test_the_fingerprint_follows_who_said_no_only():
    a = PermissionView(frozenset({1}), frozenset({5, 6}))
    assert a.fingerprint() == PermissionView(frozenset({2, 3}), frozenset({6, 5})).fingerprint()
    assert a.fingerprint() != PermissionView(frozenset({1}), frozenset({5})).fingerprint()


# --- the using rule ------------------------------------------------------------

def test_records_leave_out_arr_photos_of_people_who_said_no_and_keep_everyone_else(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    set_photo(conn, 1000, "arr", 42)   # granted
    set_photo(conn, 1001, "arr", 43)   # withdrawn
    set_photo(conn, 1002, "arr", 44)   # no answer yet: stays while permission is sought
    set_photo(conn, 1003, "open", 43)  # a CC photo of the person who said no: its licence governs
    permissions.save_snapshot(conn, answer((42, "granted"), (43, "withdrawn")), NOW)
    photo_ids = {p: p for p in range(1000, 1008)}
    kept = {pid for r in evaluate.load_records(conn, photo_ids) for pid in r.photo_rows}
    assert kept == set(range(1000, 1008)) - {1001}


def test_load_records_works_on_a_manifest_copy_made_before_permissions_existed(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    conn.executescript("drop table photo_permissions; drop table permission_syncs;")
    assert len(evaluate.load_records(conn, {p: p for p in range(1000, 1008)})) == 8


# --- the identifier ----------------------------------------------------------

def specimens(client):
    res = post_photos(client, [247], "m1/nearest")
    assert res.status_code == 200, res.text
    return {s["observation_id"]: s for s in res.json()["results"][0]["specimens"]}


def test_identify_shows_arr_photos_only_with_a_fresh_grant_and_drops_refusals(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    set_photo(conn, 1006, "arr", 42)   # obs 106, the closest match to red 247
    set_photo(conn, 1002, "arr", 43)   # obs 102
    set_photo(conn, 1000, "open", 44)  # obs 100

    s = specimens(client)
    assert s["106"]["photo_url"] is None and s["106"]["photo_withheld"] and s["106"]["photo_owner"] is None
    assert s["102"]["photo_url"] is None, "no answers synced yet: not shown"
    assert s["100"]["photo_url"].endswith("/medium.jpg") and not s["100"]["photo_withheld"]

    now = datetime.now(timezone.utc)
    permissions.save_snapshot(conn, answer((42, "granted"), (43, "withdrawn")), now)
    s = specimens(client)
    assert s["106"]["photo_url"].endswith("/medium.jpg") and s["106"]["photo_owner"] == "alice"
    assert "102" not in s, "a refusal takes the photo out of the reference set at once"

    conn.execute("update permission_syncs set attempted_at = ?",
                 ((now - timedelta(hours=3)).isoformat(timespec="seconds"),))
    conn.commit()
    assert specimens(client)["106"]["photo_url"] is None, "answers not refreshed lately: fail closed"


def test_a_licence_change_applies_to_the_next_identification(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    assert specimens(client)["106"]["photo_url"] is not None
    set_photo(conn, 1006, "arr", 42)
    assert specimens(client)["106"]["photo_url"] is None
    set_photo(conn, 1006, "nc", 42)
    assert specimens(client)["106"]["photo_url"] is not None


def test_stats_report_where_the_answers_stand(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    permissions.save_snapshot(conn, answer((1, "granted"), (2, "withdrawn"), (3, "granted")),
                              datetime.now(timezone.utc))
    p = client.get("/api/stats").json()["permissions"]
    assert (p["granted"], p["withdrawn"], p["fresh_for_showing"]) == (2, 1, True)


# --- background work ----------------------------------------------------------

def test_background_work_keeps_running_after_a_failure():
    calls, done = [], threading.Event()

    def work():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("iNat down")
        done.set()
    logged = []
    api.start_background("t", 0.01, work, note=logged.append)
    assert done.wait(5)
    assert logged and "iNat down" in logged[0]


def test_serve_says_when_answers_are_not_configured(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("MV_ORG_BASE_URL", raising=False)
    monkeypatch.delenv("MV_ORG_VISION_KEY", raising=False)
    monkeypatch.delenv("MV_LICENSE_REFRESH_HOURS", raising=False)
    started = []
    monkeypatch.setattr(api, "start_background", lambda name, *a, **k: started.append(name))
    logged = []
    api.create_app(tmp_path / "m.sqlite", web_dist=None, note=logged.append)
    assert started == [] and any("not shown" in m for m in logged)
    monkeypatch.setenv("MV_ORG_BASE_URL", "https://org.example")
    monkeypatch.setenv("MV_ORG_VISION_KEY", KEY)
    monkeypatch.setenv("MV_LICENSE_REFRESH_HOURS", "24")
    api.create_app(tmp_path / "m.sqlite", web_dist=None, note=logged.append)
    assert started == ["permissions", "licences"]


# --- licence refresh ----------------------------------------------------------

def test_licence_refresh_rereads_the_stalest_records_and_records_changes(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    conn.execute("update inat_observations set fetched_at = '2026-09-01T00:00:00+00:00'")
    conn.execute("update inat_observations set fetched_at = '2026-09-30T00:00:00+00:00' "
                 "where observation_id = '107'")
    conn.execute("update inat_observations set fetched_at = '2026-10-01T11:00:00+00:00' "
                 "where observation_id = '106'")
    conn.commit()
    from conftest import inat_obs
    asked = []

    def fake_fetch(session, ids, limiter):
        asked.append(list(ids))
        # obs 100's photo became all rights reserved on iNat.
        return [inat_obs(int(i), photos=[(0, 900 + int(i), "" if i == "100" else "cc0",
                                          "inaturalist-open-data.s3.amazonaws.com")]) for i in ids]
    stats = inat.refresh_licenses(conn, older_than_hours=24, fetch=fake_fetch, session=object(),
                                  now=lambda: NOW, log=lambda s: None)
    # Oldest first: 100-105 (a month old), then 107 (a day and a half); 106 is fresh.
    assert asked == [["100", "101", "102", "103", "104", "105", "107"]]
    assert stats["requested"] == 7 and stats["license_changes"] == 1
    assert conn.execute("select license_class from photos where photo_id = 1000").fetchone()[0] == "arr"
    assert conn.execute("select count(*) from license_history where photo_id = 1000").fetchone()[0] == 2
    assert inat.refresh_licenses(conn, older_than_hours=24, fetch=fake_fetch, session=object(),
                                 now=lambda: NOW, log=lambda s: None)["requested"] == 0
