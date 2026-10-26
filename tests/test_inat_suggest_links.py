"""iNaturalist's suggestions for an iNaturalist link on the Identify page ("from a link";
Steve, 2026-10-26): the observation's photos as Vision read them go to mycomap.org for the
signed-in member, with its public place rounded to 0.1 degree (none when iNat obscures it),
joined to the observation's run; a kept answer reads the photos again so a repeat comes
from mycomap.org's cache. Mushroom Observer links send nothing."""

import hashlib
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_api import open_limits, png
from test_api_v1_by_observation import fetcher, inat_obs
from test_api_v1_identify import FakeOrg, SERVER_KEY
from test_inat_suggest import REF, FakeOrgSuggest, cand, ok, poll, settle, suggestions
from test_models_and_scoreboard import Const, seed_two_species
from test_run_links import person, visitor
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import api as api_module
from mycomap_vision import apiv1, inat_suggest, runs
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig


def site(conn, tmp_path, org, mode="all"):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1", log=lambda s: None)
    conn.commit()
    run_store = runs.RunStore(runs.DirObjects(tmp_path / "bucket"), write=lambda job: job(),
                              note=lambda s: None, index=runs.OwnerIndex(tmp_path / "idx.sqlite"))
    f = fetcher()
    sugg = suggestions(org) if org is not None else None
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), background=False, fungus_gate=None,
                     signin=SigninConfig(mode=mode, origin=HERE, issuer=ISSUER,
                                         public_key=PUBLIC_KEY, secret=SECRET),
                     run_store=run_store, note=lambda s: None, by_observation=True,
                     observation_fetcher=f, api_v1=True, server_key=SERVER_KEY,
                     app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
                     inat_suggestions=sugg if sugg is not None else False)
    return app, sugg, f.session


def by_link(client, source="inat", number=555, **body):
    return client.post("/api/identify/observation",
                       json={"source": source, "id": number, "inat_ref": REF, **body})


def wait_calls(org, n, timeout=10.0):
    end = time.monotonic() + timeout
    while len(org.calls) < n and time.monotonic() < end:
        time.sleep(0.02)
    return org.calls


def lines_of(client, body):
    view = body.get("inat")
    if view and view.get("pending"):
        view = poll(client, view["ref"])
    return [l["name"] for l in (view or {}).get("lines") or []]


def test_an_inat_link_sends_its_photos_for_a_signed_in_member_and_shows_the_lines(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, sugg, _ = site(conn, tmp_path, org)
    client = person(app, "member-1")
    res = by_link(client)
    assert res.status_code == 200, res.text
    assert lines_of(client, res.json()) == ["Russula emetica"]
    calls = wait_calls(org, 2)
    assert sorted(f["photo_ref"] for f, _ in calls) == ["1", "2"]
    assert {f["org_user_id"] for f, _ in calls} == {"member-1"}
    assert {f["client_ref"] for f, _ in calls} == {f"site-{REF}"}
    assert {(f["lat"], f["lng"]) for f, _ in calls} == {(40.1, -83.0)}  # public "40.12,-82.98"
    assert {f["full"] for f, _ in calls} == {"1"}


@pytest.mark.parametrize("who", ["anonymous", "no inat account"])
def test_no_lines_for_anonymous_visitors_or_members_without_inat(conn, tmp_path, who):
    org = FakeOrgSuggest({"status": "no_inat_account"} if who != "anonymous" else ok(cand(1, "R e", 88)))
    app, _, _ = site(conn, tmp_path, org, mode="off" if who == "anonymous" else "all")
    client = visitor(app) if who == "anonymous" else person(app, "member-1")
    body = by_link(client).json()
    assert body["results"] and lines_of(client, body) == []
    if who == "anonymous":
        time.sleep(0.2)
        assert org.calls == []


def test_a_mushroom_observer_link_sends_nothing(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, _, _ = site(conn, tmp_path, org)
    client = person(app, "member-1")
    body = by_link(client, "mo", 777).json()
    assert body["results"] and "inat" not in body
    by_link(client, "mo", 777)                                   # kept: still nothing
    time.sleep(0.3)
    assert org.calls == []


@pytest.mark.parametrize("case", ["obscured", "no location", "place unticked"])
def test_an_obscured_or_missing_or_unwanted_place_is_never_sent(conn, tmp_path, case):
    org = FakeOrgSuggest()
    app, _, inat = site(conn, tmp_path, org)
    if case == "obscured":
        inat.inat[555] = inat_obs(555, [9001, 9002], location="40.1234,-82.9876", obscured=True)
    elif case == "no location":
        inat.inat[555] = inat_obs(555, [9001, 9002], location=None)
    extra = {"use_place": False} if case == "place unticked" else {}
    assert by_link(person(app, "member-1"), **extra).status_code == 200
    calls = wait_calls(org, 2)
    assert calls and all("lat" not in f and "lng" not in f for f, _ in calls)


def test_the_public_place_is_rounded_to_a_tenth_of_a_degree():
    class Obs:
        place, latitude, longitude = "coordinates", 49.287654, -123.116543
    assert inat_suggest.public_place(Obs) == (49.3, -123.1)
    Obs.place = "obscured"
    assert inat_suggest.public_place(Obs) is None
    Obs.place, Obs.latitude = None, None
    assert inat_suggest.public_place(Obs) is None


def test_a_find_sends_at_most_as_many_photos_as_an_upload_may_have():
    org = FakeOrgSuggest()
    sugg = suggestions(org)
    sugg.start("member-1", [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(14)], run_id=None)
    settle(sugg)
    assert len(org.calls) == inat_suggest.MAX_PHOTOS == 10


def test_a_kept_answer_reads_the_photos_again_joined_to_its_run_with_the_same_bytes(conn, tmp_path):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, sugg, inat = site(conn, tmp_path, org)
    client = person(app, "member-1")
    first = by_link(client).json()
    assert first["cached"] is False
    wait_calls(org, 2)
    fresh = {f["photo_ref"]: hashlib.sha256(j).hexdigest() for f, j in org.calls}
    org.calls.clear()
    again = client.post("/api/identify/observation",
                        json={"source": "inat", "id": 555, "inat_ref": "pageref-again-000001"}).json()
    assert again["cached"] is True
    assert lines_of(client, again) == ["Russula emetica"]
    calls = wait_calls(org, 2)
    assert {f["run_id"] for f, _ in calls} == {again["run"]["id"]} == {first["run"]["id"]}
    # The same bytes each time, so mycomap.org's score cache (keyed on them) answers a repeat.
    assert {f["photo_ref"]: hashlib.sha256(j).hexdigest() for f, j in calls} == fresh


def test_a_links_run_page_shows_its_owner_the_lines_and_no_one_else(conn, tmp_path):
    org = FakeOrgSuggest(lambda fields: ok(cand(1, "Russula emetica", 88)) if fields["org_user_id"]
                         == "member-1" else ok(cand(2, "Mycena pura", 90)))
    app, sugg, _ = site(conn, tmp_path, org)
    owner = person(app, "member-1")
    rid = by_link(owner).json()["run"]["id"]
    settle(sugg)
    sugg.finds.clear()                                           # a restart: the bucket keeps it
    sugg.by_run.clear()
    assert [l["name"] for l in owner.get(f"/api/runs/{rid}").json()["inat"]["lines"]] \
        == ["Russula emetica"]
    other = person(app, "member-2")
    assert by_link(other).json()["run"]["id"] == rid             # the observation's one link
    settle(sugg)
    assert "inat" not in other.get(f"/api/runs/{rid}").json()
    # Another member opening the same observation's run never replaces its owner's lines.
    assert [l["name"] for l in owner.get(f"/api/runs/{rid}").json()["inat"]["lines"]]         == ["Russula emetica"]
    assert "inat" not in visitor(app).get(f"/api/runs/{rid}").json()


def test_when_vision_fails_on_a_link_the_lines_still_show_for_a_retry(conn, tmp_path, monkeypatch):
    org = FakeOrgSuggest(ok(cand(1, "Russula emetica", 88)))
    app, sugg, _ = site(conn, tmp_path, org)
    client = person(app, "member-1")

    def fails(*a, **k):
        from fastapi import HTTPException
        raise HTTPException(503, "busy identifying other photos")
    monkeypatch.setattr(api_module, "fill_context", fails)      # after the photos are read
    assert by_link(client).status_code == 503
    settle(sugg)
    assert client.get(f"/api/inat-suggestions/site-{REF}").json() == {"inat": None}
    failed = client.get(f"/api/inat-suggestions/site-{REF}?vision=failed").json()["inat"]
    assert [l["name"] for l in failed["lines"]] == ["Russula emetica"]


def test_the_photo_sent_is_the_same_bytes_for_the_same_photo():
    img = Image.open(__import__("io").BytesIO(png(247))).convert("RGB")
    assert inat_suggest.jpeg_of(img) == inat_suggest.jpeg_of(img.copy())


def test_another_members_answer_never_lands_on_a_run_whose_owners_answer_is_still_out(
        conn, tmp_path):
    import threading
    owners_turn = threading.Event()

    def answer(fields):
        if fields["org_user_id"] == "member-1":
            owners_turn.wait(20)                                 # the owner's iNat is slow
            return ok(cand(1, "Russula emetica", 88))
        return ok(cand(2, "Mycena pura", 90))
    org = FakeOrgSuggest(answer)
    app, sugg, _ = site(conn, tmp_path, org)
    owner = person(app, "member-1")
    rid = by_link(owner).json()["run"]["id"]                    # kept; its iNat answer still out
    sugg.finds.clear()                                           # forgotten (a restart, 10 min)
    other = person(app, "member-2")
    assert by_link(other).json()["run"]["id"] == rid             # the same ref, the same run
    deadline = time.monotonic() + 10
    while sum(f["org_user_id"] == "member-2" for f, _ in org.calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)                                              # member-2's answers are back
    owners_turn.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        inat = owner.get(f"/api/runs/{rid}").json().get("inat")
        if inat and not inat.get("pending"):
            break
        time.sleep(0.05)
    assert [l["name"] for l in inat["lines"]] == ["Russula emetica"]
