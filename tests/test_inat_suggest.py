"""iNaturalist's suggestions under MycoMap Vision's answers (inat_suggest.py; Steve,
2026-10-25): asked through mycomap.org with the member's own iNat account, shown only
under Vision's answer and never before it, never to anonymous visitors or members without
iNat, never a name Vision already gives, and never at the cost of Vision's answer."""

import io
import json
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_api import open_limits, png
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer
from test_models_and_scoreboard import Const, seed_two_species
from test_run_links import identify, jpeg_with_gps, person, visitor
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import apiv1, identifylog, inat_suggest, runs
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig


def cand(tid, name, score, rank="species", match=()):
    return {"taxon_id": tid, "name": name, "rank": rank, "score": score, "vision_score": score,
            "seen_nearby": True, "match_names": list(match) or [name]}


def ok(*cands, not_fungus=None, full=None):
    body = {"status": "ok", "thresholds": {"show": 70, "notFungus": 80},
            "candidates": list(cands), "not_fungus": not_fungus}
    if full is not None:
        body["full_response"] = full
    return body


NOT_FUNGUS = {"share": 92.4, "top": {"taxon_id": 1, "name": "Carya ovata", "rank": "species",
                                     "score": 88}, "common_ancestor": None}


class FakeOrgSuggest:
    """mycomap.org's iNat-suggestions endpoint, played in the test: records every photo
    forwarded, answers `answer` (a body, None for a refusal, or an exception), and holds
    each answer until `release` is set when one is given."""

    def __init__(self, answer=None, release: threading.Event | None = None):
        self.answer = answer if answer is not None else ok()
        self.release = release
        self.calls: list[tuple[dict, bytes]] = []
        self.lock = threading.Lock()

    def __call__(self, fields: dict, jpeg: bytes):
        with self.lock:
            self.calls.append((dict(fields), jpeg))
        if self.release is not None:
            self.release.wait(30)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer(fields) if callable(self.answer) else self.answer


def suggestions(org, **kw):
    return inat_suggest.Suggestions(org, note=lambda s: None, **kw)


def site(conn, tmp_path, org=None, mode="all", links=True, inat=True):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1", log=lambda s: None)
    conn.commit()
    run_store = runs.RunStore(runs.DirObjects(tmp_path / "bucket"), write=lambda job: job(),
                              note=lambda s: None, index=runs.OwnerIndex(tmp_path / "idx.sqlite"))
    signin = SigninConfig(mode=mode, origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    sugg = suggestions(org) if inat and org is not None else None
    chosen = sugg if sugg is not None else (True if inat == "settings" else False)
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), signin=signin, background=False, fungus_gate=None,
                     run_store=run_store if links else None, note=lambda s: None,
                     api_v1=True, sessions=True, server_key=SERVER_KEY,
                     app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
                     inat_suggestions=chosen)
    return app, sugg


def settle(sugg, timeout=10.0):
    """Waits for every forwarded photo's answer."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if all(f.done() for find in sugg.finds.values() for f in find.futures):
            return
        time.sleep(0.02)
    raise AssertionError("forwards never finished")


def poll(client, ref, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        view = client.get(f"/api/inat-suggestions/{ref}").json()["inat"]
        if not (view or {}).get("pending"):
            return view
        time.sleep(0.05)
    raise AssertionError("still pending")


def vision_species(body) -> list[str]:
    return [c["name"] for c in body["results"][0]["ranks"]["species"]]


# --- what is shown (the rules, pure) ----------------------------------------------------

def test_a_name_vision_already_gives_is_never_repeated_however_it_is_written():
    names = inat_suggest.vision_names([{"ranks": {"species": [{"name": "Amanita muscaria"}],
                                                   "genus": [{"name": "Amanita"}]},
                                        "likely": {"species": {"names": [{"name": "Boletus edulis"}]}}}])
    merged = inat_suggest.merge([ok(
        cand(1, "amanita  MUSCARIA", 95),                                   # case, spacing
        cand(2, "Cf. 'Boletus edulis'", 90),                                # cf., quotes; a likely name
        cand(3, "Fly agaric renamed", 88, match=["Fly agaric renamed", "Amanita muscaria"]),  # synonym
        cand(4, "Amanita", 85, rank="genus"),                                # Vision names the genus
        cand(5, "Russula emetica", 80))])
    assert [l["name"] for l in inat_suggest.lines(merged, names)] == ["Russula emetica"]


def test_a_genus_is_dropped_when_vision_names_a_species_in_it():
    names = inat_suggest.vision_names([{"ranks": {"species": [{"name": "Cortinarius alboviolaceus"}]}}])
    merged = inat_suggest.merge([ok(cand(1, "Cortinarius", 90, rank="genus"),
                                    cand(2, "Cortinarius alboviolaceus", 85),
                                    cand(3, "Corticium", 80, rank="genus"))])
    assert [l["name"] for l in inat_suggest.lines(merged, names)] == ["Corticium"]


def test_at_most_three_lines_best_first_labelled_with_inats_rounded_score():
    merged = inat_suggest.merge([ok(*[cand(i, f"Species {i}", s)
                                      for i, s in enumerate([71.5, 99.4, 80.5, 75])])])
    shown = inat_suggest.lines(merged, set())
    assert [l["name"] for l in shown] == ["Species 1", "Species 2", "Species 3"]
    assert [l["label"] for l in shown] == ["iNaturalist suggests · 99%",
                                           "iNaturalist suggests · 81%",       # half up
                                           "iNaturalist suggests · 75%"]


def test_the_threshold_is_applied_even_to_a_line_mycomap_org_sent_below_it():
    merged = inat_suggest.merge([ok(cand(1, "Above", 70.0), cand(2, "Below", 69.9))])
    assert [l["name"] for l in inat_suggest.lines(merged, set())] == ["Above"]
    raised = inat_suggest.merge([{**ok(cand(1, "Above", 72)), "thresholds": {"show": 75,
                                                                              "notFungus": 80}}])
    assert inat_suggest.lines(raised, set()) == []
    weak = inat_suggest.merge([ok(not_fungus={**NOT_FUNGUS, "share": 79.9})])
    assert inat_suggest.shown(weak, set()) is None


def test_not_a_fungus_is_shown_when_inat_is_confident_on_every_photo():
    view = inat_suggest.shown(inat_suggest.merge([ok(not_fungus=NOT_FUNGUS),
                                                  ok(not_fungus={**NOT_FUNGUS, "share": 85})]), set())
    assert view["not_fungus"]["label"] == "iNaturalist suggests this may not be a fungus · 85%"
    # One photo of the fungus itself (a habitat shot elsewhere) keeps the find a fungus.
    mixed = inat_suggest.merge([ok(not_fungus=NOT_FUNGUS), ok(cand(1, "Mycena galericulata", 90))])
    assert mixed["not_fungus"] is None
    # A photo iNat couldn't score is skipped, as mycomap.org skips it for the app.
    assert inat_suggest.merge([ok(not_fungus=NOT_FUNGUS), None])["not_fungus"]["share"] == 92.4


def test_several_photos_keep_each_taxons_best_score_and_a_genus_gives_way_to_its_species():
    merged = inat_suggest.merge([ok(cand(1, "Mycena", 72, rank="genus"), cand(3, "Russula", 80, rank="genus")),
                                 ok(cand(1, "Mycena", 91, rank="genus"), cand(2, "Mycena pura", 75),
                                    cand(3, "Russula", 85, rank="genus"))])
    assert [(c["name"], c["score"]) for c in merged["candidates"]] == [("Russula", 85),
                                                                      ("Mycena pura", 75)]


@pytest.mark.parametrize("status", ["no_inat_account", "inat_login_only", "inat_reconnect",
                                    "inat_unavailable"])
def test_members_without_a_usable_inat_account_get_nothing(status):
    assert inat_suggest.merge([{"status": status}]) is None
    assert inat_suggest.shown(inat_suggest.merge([{"status": status}]), set()) is None


def test_inats_answer_is_never_shown_before_visions_even_when_it_comes_first():
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 90)))
    sugg = suggestions(org)
    key = sugg.start("member-1", [Image.new("RGB", (64, 64))], run_id=None)
    settle(sugg)
    assert sugg.view(key, "member-1") is None           # back, but Vision hasn't answered
    sugg.answered(key, set())
    assert [l["name"] for l in sugg.view(key, "member-1")["lines"]] == ["Russula emetica"]
    assert sugg.view(key, "member-2") is None           # someone else's find


def test_a_slow_mycomap_org_is_dropped_quietly_after_the_wait():
    release = threading.Event()
    now = [100.0]
    sugg = suggestions(FakeOrgSuggest(ok(cand(1, "Russula emetica", 90)), release=release),
                       clock=lambda: now[0])
    key = sugg.start("member-1", [Image.new("RGB", (64, 64))], run_id=None)
    sugg.answered(key, set())
    assert sugg.view(key, "member-1") == {"pending": True, "ref": key}
    now[0] += inat_suggest.WAIT_SECONDS + 0.1
    assert sugg.view(key, "member-1") is None
    release.set()


def test_the_photo_sent_is_a_jpeg_no_longer_than_1024_px():
    body = inat_suggest.jpeg_of(Image.new("RGB", (3000, 2000), (200, 10, 10)))
    sent = Image.open(io.BytesIO(body))
    assert sent.format == "JPEG" and max(sent.size) == 1024 and len(body) < 4 * 2**20


def test_off_unless_mv_inat_suggestions_is_set(monkeypatch):
    monkeypatch.setenv("MV_ORG_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("MV_ORG_VISION_KEY", "k" * 40)
    monkeypatch.delenv("MV_INAT_SUGGESTIONS", raising=False)
    assert inat_suggest.Suggestions.from_settings(note=lambda s: None) is None
    monkeypatch.setenv("MV_INAT_SUGGESTIONS", "0")
    assert inat_suggest.Suggestions.from_settings(note=lambda s: None) is None
    monkeypatch.setenv("MV_INAT_SUGGESTIONS", "1")
    assert inat_suggest.Suggestions.from_settings(note=lambda s: None) is not None
    monkeypatch.delenv("MV_ORG_VISION_KEY")
    assert inat_suggest.Suggestions.from_settings(note=lambda s: None) is None


# --- the website ------------------------------------------------------------------------

def test_a_member_gets_inats_names_under_visions_answer_without_visions_own(conn, tmp_path):
    def answer(fields):
        return ok(cand(1, "Russula emetica", 88), cand(2, "Mycena pura", 76),
                  full={"common_ancestor": None, "results": [{"name": "Russula emetica"}]})
    org = FakeOrgSuggest(answer)
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    body = identify(client, lat="49.2345", lng="-123.1789").json()
    view = body.get("inat") or {}
    if view.get("pending"):
        view = poll(client, view["ref"])
    taken = set(vision_species(body))
    assert view["lines"] and not {l["name"] for l in view["lines"]} & taken
    assert view["full"] == [{"common_ancestor": None, "results": [{"name": "Russula emetica"}]}]
    [(fields, jpeg)] = org.calls
    assert fields["org_user_id"] == "member-1"
    assert fields["run_id"] == body["run"]["id"]
    assert fields["client_ref"].startswith("site-") and fields["photo_ref"] == "1"
    assert fields["full"] == "1"
    assert (fields["lat"], fields["lng"]) == (49.2, -123.2)  # the entered place, rounded
    assert jpeg[:2] == b"\xff\xd8"


def test_names_vision_gives_are_dropped_on_the_page_and_names_it_lists_below_5_percent_are_not(
        conn, tmp_path):
    seen = {}

    def answer(fields):
        return ok(*[cand(i, n, 90 - i) for i, n in enumerate(seen["names"])])
    org = FakeOrgSuggest(answer)
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    first = identify(client).json()
    species = first["results"][0]["ranks"]["species"]
    seen["names"] = [c["name"] for c in species]           # iNat says what Vision lists
    body = identify(client).json()
    view = body.get("inat")
    if view and view.get("pending"):
        view = poll(client, view["ref"])
    below = [c["name"] for c in species if (c.get("confidence") or 0) < 0.05]
    assert [l["name"] for l in (view or {}).get("lines") or []] == below[:3]
    assert len(below) < len(species)                       # Vision gave at least one name


def test_anonymous_visitors_get_no_inat_and_nothing_is_sent(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, _ = site(conn, tmp_path, org, mode="off")
    body = identify(visitor(app)).json()
    assert body["results"] and "inat" not in body
    assert org.calls == []


def test_a_member_without_inat_gets_nothing(conn, tmp_path):
    org = FakeOrgSuggest({"status": "no_inat_account"})
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    body = identify(client).json()
    view = body.get("inat")
    if view and view.get("pending"):
        view = poll(client, view["ref"])
    assert view is None and len(org.calls) == 1


@pytest.mark.parametrize("failure", [RuntimeError("connection refused"), {"status": "weird"},
                                     ["not", "an", "object"]])
def test_a_mycomap_org_failure_leaves_visions_answer_whole_and_shows_nothing(conn, tmp_path,
                                                                             failure):
    org = FakeOrgSuggest(failure)
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    res = identify(client)
    assert res.status_code == 200 and res.json()["results"][0]["ranks"]["species"]
    view = res.json().get("inat")
    if view and view.get("pending"):
        view = poll(client, view["ref"])
    assert view is None


def test_visions_answer_never_waits_for_inat(conn, tmp_path):
    release = threading.Event()
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)), release=release)
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    try:
        start = time.monotonic()
        body = identify(client).json()
        took = time.monotonic() - start
        assert body["results"][0]["ranks"]["species"]
        assert body["inat"] == {"pending": True, "ref": body["inat"]["ref"]}
        assert took < 10                                   # the fake holds iNat for 30 s
        release.set()
        assert [l["name"] for l in poll(client, body["inat"]["ref"])["lines"]] == ["Russula emetica"]
    finally:
        release.set()


def test_only_the_member_who_sent_the_photos_can_read_their_suggestions(conn, tmp_path):
    release = threading.Event()
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)), release=release)
    app, sugg = site(conn, tmp_path, org)
    try:
        ref = identify(person(app, "member-1")).json()["inat"]["ref"]
        assert person(app, "member-2").get(f"/api/inat-suggestions/{ref}").json() == {"inat": None}
    finally:
        release.set()


def test_off_by_default_the_site_sends_nothing_and_shows_nothing(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("MV_INAT_SUGGESTIONS", raising=False)
    app, _ = site(conn, tmp_path, org=None, inat="settings")  # the setting decides: unset
    client = person(app, "member-1")
    body = identify(client, inat_ref=REF).json()
    assert "inat" not in body
    assert client.get("/api/inat-suggestions/site-anything").json() == {"inat": None}
    assert client.get(f"/api/inat-suggestions/site-{REF}?vision=failed").json() == {"inat": None}


REF = "pageref-0123456789"


def test_when_vision_fails_the_members_inat_lines_still_show_for_a_retry(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    res = identify(client, models="nope/x", inat_ref=REF)      # fails after the photos arrived
    assert res.status_code == 400
    settle(sugg)
    failed = client.get(f"/api/inat-suggestions/site-{REF}?vision=failed").json()["inat"]
    assert [l["name"] for l in failed["lines"]] == ["Russula emetica"]
    [(fields, _)] = org.calls
    assert fields["client_ref"] == f"site-{REF}"


def test_while_vision_is_still_checking_no_inat_lines_are_shown(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    sugg = suggestions(org)
    key = sugg.start("member-1", [Image.new("RGB", (64, 64))], run_id=None, ref=REF)
    settle(sugg)
    assert key == f"site-{REF}"
    assert sugg.view(key, "member-1") is None                  # iNat is back, Vision isn't
    assert sugg.view(key, "member-1", vision_failed=True)["lines"]
    # Over HTTP too: without the page saying Vision failed, nothing.
    app, s2 = site(conn, tmp_path, FakeOrgSuggest(ok(cand(1, "Russula emetica", 88))))
    client = person(app, "member-1")
    identify(client, models="nope/x", inat_ref=REF)
    settle(s2)
    assert client.get(f"/api/inat-suggestions/site-{REF}").json() == {"inat": None}


def test_a_member_cannot_take_over_another_members_find_by_its_ref():
    sugg = suggestions(FakeOrgSuggest(ok(cand(1, "Russula emetica", 88))))
    first = sugg.start("member-1", [Image.new("RGB", (64, 64))], run_id=None, ref=REF)
    second = sugg.start("member-2", [Image.new("RGB", (64, 64))], run_id=None, ref=REF)
    settle(sugg)
    assert first == f"site-{REF}" and second != first
    assert sugg.view(first, "member-2", vision_failed=True) is None
    assert sugg.start("member-1", [Image.new("RGB", (8, 8))], run_id=None,
                      ref="bad ref!").startswith("site-")      # malformed: a fresh key


def test_a_runs_page_shows_its_owner_their_suggestions_and_no_one_else(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, sugg = site(conn, tmp_path, org)
    owner = person(app, "member-1")
    rid = identify(owner).json()["run"]["id"]
    settle(sugg)
    sugg.finds.clear()                                       # a restart: only the bucket keeps it
    sugg.by_run.clear()
    mine = owner.get(f"/api/runs/{rid}").json()
    assert [l["name"] for l in mine["inat"]["lines"]] == ["Russula emetica"]
    assert "inat" not in person(app, "member-2").get(f"/api/runs/{rid}").json()
    assert "inat" not in visitor(app).get(f"/api/runs/{rid}").json()


def test_inats_answer_never_goes_in_visions_own_identify_log(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Inatonly suggestedname", 88)))
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    body = identify(client).json()
    view = body.get("inat") or {}
    if view.get("pending"):
        view = poll(client, view["ref"])
    assert view["lines"][0]["name"] == "Inatonly suggestedname"
    assert app.state.identify_log.flush()
    db = sqlite3.connect(identifylog.default_path(tmp_path / "manifest.sqlite"))
    try:
        tables = [r[0] for r in db.execute("select name from sqlite_master where type = 'table'")]
        assert db.execute("select count(*) from identify_log").fetchone()[0] >= 1  # Vision's row
        dump = "\n".join(str(row) for t in tables for row in db.execute(f'select * from "{t}"'))
    finally:
        db.close()
    assert "Inatonly" not in dump and "inaturalist suggests" not in dump.lower()


def test_deleting_a_run_deletes_its_inat_answer(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, sugg = site(conn, tmp_path, org)
    owner = person(app, "member-1")
    rid = identify(owner).json()["run"]["id"]
    settle(sugg)
    assert list((tmp_path / "bucket").rglob("inat.json"))
    assert owner.delete(f"/api/runs/{rid}").status_code == 204
    assert not list((tmp_path / "bucket").rglob("inat.json"))


# --- the app ----------------------------------------------------------------------------

def app_headers(**claims):
    return bearer(app_token(**claims))


def test_an_app_members_photos_go_to_mycomap_org_with_their_find(conn, tmp_path):
    org = FakeOrgSuggest()
    app, sugg = site(conn, tmp_path, org, mode="off")
    client = TestClient(app)
    headers = app_headers(uid="org-user-7")
    sid = client.post("/api/v1/sessions", json={"client_ref": "find-1", "latitude": 49.2,
                                                "longitude": -123.1}, headers=headers).json()["session_id"]
    res = client.post(f"/api/v1/sessions/{sid}/photos", headers=headers,
                      files={"photo": ("a.png", png(247), "image/png")}, data={"photo_ref": "p1"})
    assert res.status_code == 200
    deadline = time.monotonic() + 10
    while not org.calls and time.monotonic() < deadline:
        time.sleep(0.02)
    [(fields, _)] = org.calls
    assert (fields["org_user_id"], fields["client_ref"], fields["photo_ref"]) == ("org-user-7",
                                                                                 "find-1", "p1")
    assert "lat" not in fields                               # not said to be open
    assert "full" not in fields                              # the app shows no dropdown


def test_an_app_find_sends_its_place_only_when_its_geoprivacy_is_open(conn, tmp_path):
    org = FakeOrgSuggest()
    app, _ = site(conn, tmp_path, org, mode="off")
    client = TestClient(app)
    headers = app_headers(uid="org-user-7")
    sid = client.post("/api/v1/sessions", json={"client_ref": "find-2", "latitude": 49.2,
                                                "longitude": -123.1, "inat_location": True},
                      headers=headers).json()["session_id"]
    client.post(f"/api/v1/sessions/{sid}/photos", headers=headers,
                files={"photo": ("a.png", png(247), "image/png")}, data={"photo_ref": "p1"})
    deadline = time.monotonic() + 10
    while not org.calls and time.monotonic() < deadline:
        time.sleep(0.02)
    [(fields, _)] = org.calls
    assert (fields["lat"], fields["lng"]) == (49.2, -123.1)
    assert client.post("/api/v1/sessions", json={"inat_location": "yes"},
                       headers=headers).status_code == 400


def test_app_identify_forwards_each_photo_with_its_client_ref(conn, tmp_path):
    org = FakeOrgSuggest()
    app, _ = site(conn, tmp_path, org, mode="off")
    client = TestClient(app)
    res = client.post("/api/v1/identify", headers=app_headers(uid="org-user-7"),
                      files=[("photos", ("a.png", png(247), "image/png")),
                             ("photos", ("b.png", png(10), "image/png"))],
                      data={"client_ref": "find-3"})
    assert res.status_code == 200
    deadline = time.monotonic() + 10
    while len(org.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sorted(f["photo_ref"] for f, _ in org.calls) == ["1", "2"]
    assert {f["client_ref"] for f, _ in org.calls} == {"find-3"}


@pytest.mark.parametrize("who", ["no uid", "server key", "no client_ref"])
def test_nothing_is_sent_for_a_token_naming_no_member_or_a_find_with_no_ref(conn, tmp_path, who):
    org = FakeOrgSuggest()
    app, _ = site(conn, tmp_path, org, mode="off")
    client = TestClient(app)
    headers = {"no uid": app_headers(), "server key": bearer(SERVER_KEY),
               "no client_ref": app_headers(uid="org-user-7")}[who]
    data = {} if who == "no client_ref" else {"client_ref": "find-4"}
    res = client.post("/api/v1/identify", headers=headers,
                      files=[("photos", ("a.png", png(247), "image/png"))], data=data)
    assert res.status_code == 200
    time.sleep(0.3)
    assert org.calls == []


# --- the place a website find sends (Steve, 2026-10-25) -------------------------------

def sent_place(conn, tmp_path, photos, **data):
    org = FakeOrgSuggest()
    app, sugg = site(conn, tmp_path, org)
    assert identify(person(app, "member-1"), photos=photos, **data).status_code == 200
    settle(sugg)
    [(fields, _)] = org.calls
    return (fields.get("lat"), fields.get("lng"))


GPS_PHOTO = (("gps.jpg", jpeg_with_gps(), "image/jpeg"),)      # EXIF 40.1234, -82.98
PLAIN_PHOTO = (("a.png", png(247), "image/png"),)


def test_a_website_find_sends_the_place_the_person_entered_over_the_photos_own(conn, tmp_path):
    assert sent_place(conn, tmp_path, GPS_PHOTO, lat="49.2345", lng="-123.1789") == (49.2, -123.2)


def test_a_website_find_sends_the_photos_exif_place_when_none_was_entered(conn, tmp_path):
    assert sent_place(conn, tmp_path, GPS_PHOTO) == (40.1, -83.0)


def test_a_website_find_sends_no_place_when_there_is_none(conn, tmp_path):
    assert sent_place(conn, tmp_path, PLAIN_PHOTO) == (None, None)


def test_a_place_is_never_sent_finer_than_a_tenth_of_a_degree():
    assert inat_suggest.rounded_place(40.123456, -82.98) == (40.1, -83.0)
    assert inat_suggest.rounded_place(None, -82.98) is None
    org = FakeOrgSuggest()
    sugg = suggestions(org)
    sugg.start("member-1", [Image.new("RGB", (32, 32))], run_id=None,
               place=(49.287654, -123.116543))                 # however it is handed over
    settle(sugg)
    [(fields, _)] = org.calls
    assert (fields["lat"], fields["lng"]) == (49.3, -123.1)


# --- mycomap.org's test vectors (services/inatSuggestions.test.ts, lane AA) ------------
# The same cases, so the website and the app show the same lines.

def test_org_vector_names_compare_ignoring_case_spacing_quotes_and_cf():
    assert inat_suggest.name_key("  Amanita   cf. Muscaria ") == "amanita muscaria"
    for vision in (["amanita  MUSCARIA"], ["Amanita cf. muscaria"]):
        assert inat_suggest.lines(inat_suggest.merge([ok(cand(1, "Amanita muscaria", 80))]),
                                  inat_suggest.vision_names([{"ranks": {"species": [
                                      {"name": vision[0]}]}}])) == []
    assert inat_suggest.lines(inat_suggest.merge([ok(cand(1, "Amanita muscaria", 80))]),
                              {inat_suggest.name_key("Amanita pantherina")})


def test_org_vector_synonyms_and_retired_names_match():
    names = ["Boletus badius", "Imleria badia", "Xerocomus badius"]   # buildMatchNames there
    merged = inat_suggest.merge([ok(cand(1, "Imleria badia", 80, match=names))])
    for vision in ("Xerocomus badius", "Boletus badius"):
        assert inat_suggest.lines(merged, {inat_suggest.name_key(vision)}) == []


def test_org_vector_a_genus_vision_names_a_species_of_but_a_species_under_visions_genus_is_news():
    genus = inat_suggest.merge([ok(cand(1, "Amanita", 80, rank="genus"))])
    assert inat_suggest.lines(genus, {"amanita muscaria"}) == []
    assert inat_suggest.lines(genus, {"amanitopsis vaginata"})
    species = inat_suggest.merge([ok(cand(2, "Amanita muscaria", 80))])
    assert inat_suggest.lines(species, {"amanita"})


def test_org_vector_lines_disagreements_only_best_first_at_most_three_labelled():
    merged = inat_suggest.merge([ok(*[cand(i, n, s) for i, (n, s) in enumerate(
        [("A a", 95), ("B b", 90), ("C c", 85), ("D d", 80), ("E e", 75)])])])
    shown = inat_suggest.lines(merged, {"b b"})
    assert [l["name"] for l in shown] == ["A a", "C c", "D d"]
    assert shown[0]["label"] == "iNaturalist suggests · 95%"
    assert inat_suggest.lines(inat_suggest.merge([ok(cand(1, "X", 81.6))]), set())[0]["label"] \
        == "iNaturalist suggests · 82%"
    view = inat_suggest.shown(inat_suggest.merge([ok(not_fungus={**NOT_FUNGUS, "share": 91.5})]), set())
    assert view["not_fungus"]["label"] == "iNaturalist suggests this may not be a fungus · 92%"


def test_org_vector_a_find_keeps_each_taxon_at_its_best_photo_and_a_species_replaces_its_genus():
    best = inat_suggest.merge([ok(cand(1, "Amanita muscaria", 72)), ok(cand(1, "Amanita muscaria", 88))])
    assert [(c["name"], c["score"]) for c in best["candidates"]] == [("Amanita muscaria", 88)]
    replaced = inat_suggest.merge([ok(cand(2, "Amanita", 90, rank="genus")),
                                   ok(cand(1, "Amanita muscaria", 80))])
    assert [c["name"] for c in replaced["candidates"]] == ["Amanita muscaria"]


def test_org_vector_not_a_fungus_only_when_every_photo_says_so_at_the_weakest_share():
    one = {**NOT_FUNGUS, "share": 90}
    two = {**NOT_FUNGUS, "share": 84}
    assert inat_suggest.merge([ok(not_fungus=one), ok()])["not_fungus"] is None
    assert inat_suggest.merge([ok(not_fungus=one), ok(not_fungus=two)])["not_fungus"]["share"] == 84


def test_mycomap_orgs_not_a_fungus_setting_is_read_as_it_sends_it():
    body = {**ok(not_fungus={**NOT_FUNGUS, "share": 85}), "thresholds": {"show": 70, "notFungus": 90}}
    assert inat_suggest.shown(inat_suggest.merge([body]), set()) is None


def test_a_pages_ref_never_makes_a_client_ref_longer_than_mycomap_org_keeps():
    sugg = suggestions(FakeOrgSuggest())
    img = [Image.new("RGB", (8, 8))]
    longest = sugg.start("member-1", img, run_id=None, ref="r" * 59)
    assert longest == "site-" + "r" * 59 and len(longest) == 64
    too_long = sugg.start("member-1", img, run_id=None, ref="r" * 60)
    assert too_long != "site-" + "r" * 60 and len(too_long) <= 64
    settle(sugg)
    assert all(len(f["client_ref"]) <= 64 for f, _ in sugg.post.calls)


# --- iNat's daily budget (iNat's approval, 2026-10-25: ~60 a minute, under 10,000 a day) --

@pytest.mark.parametrize("answer", [{**ok(cand(1, "Russula emetica", 88)), "status": s}   # even with names
                                    for s in ("inat_daily_budget", "daily_budget_reached",
                                              "inat_rate_limited")])
def test_when_mycomap_orgs_inat_budget_is_spent_nothing_shows_and_visions_answer_is_untouched(
        conn, tmp_path, answer):
    org = FakeOrgSuggest(answer)
    app, sugg = site(conn, tmp_path, org)
    client = person(app, "member-1")
    res = identify(client, inat_ref=REF)
    assert res.status_code == 200 and res.json()["results"][0]["ranks"]["species"]
    view = res.json().get("inat")
    if view and view.get("pending"):
        view = poll(client, view["ref"])
    assert view is None
    assert client.get(f"/api/inat-suggestions/site-{REF}?vision=failed").json() == {"inat": None}


@pytest.mark.parametrize("status", [429, 503, 500, 404])
def test_a_refusal_from_mycomap_org_is_no_answer_never_an_error(status):
    import http.server
    import threading as th

    class Refuse(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.dumps({"code": "inat_daily_budget"}).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    th.Thread(target=server.serve_forever, daemon=True).start()
    try:
        post = inat_suggest.org_poster(f"http://127.0.0.1:{server.server_port}", "k" * 40)
        assert post({"org_user_id": "m", "client_ref": "site-x"}, b"\xff\xd8") is None
    finally:
        server.shutdown()


# --- what counts as a name Vision gives (Steve, 2026-10-25) ---------------------------

def test_a_name_vision_lists_below_5_percent_leaves_inats_line_shown():
    vision = [{"ranks": {"species": [{"name": "Hypholoma lateritium", "confidence": 0.999},
                                     {"name": "Hypholoma capnoides", "confidence": 0.0},
                                     {"name": "Hypholoma fasciculare", "confidence": 0.049}]}}]
    merged = inat_suggest.merge([ok(cand(1, "Hypholoma capnoides", 85.3),
                                    cand(2, "Hypholoma fasciculare", 80),
                                    cand(3, "Hypholoma lateritium", 75))])
    shown = inat_suggest.lines(merged, inat_suggest.vision_names(vision))
    assert [l["name"] for l in shown] == ["Hypholoma capnoides", "Hypholoma fasciculare"]


def test_a_name_vision_shows_at_5_percent_or_without_a_chance_counts_as_given():
    vision = [{"ranks": {"species": [{"name": "Amanita muscaria", "confidence": 0.05},
                                     {"name": "Amanita pantherina"}]},          # chance withheld
               "likely": {"species": {"names": [{"name": "Boletus edulis", "confidence": 0.2},
                                                {"name": "Boletus pinophilus", "confidence": 0.01}]}}}]
    assert inat_suggest.vision_names(vision) == {"amanita muscaria", "amanita pantherina",
                                                 "boletus edulis"}
