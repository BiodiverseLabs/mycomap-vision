"""Run links (runs.py): every identification on the site is kept under its own link
/r/<id>, which anyone who has it may open, and only the person who ran it or a mycomap.org
admin may delete. Steve, 2026-10-20."""

import io
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_api import open_limits, png
from test_api_v1_by_observation import fetcher
from test_api_v1_identify import FakeOrg as Introspect, SERVER_KEY
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET, mint

from mycomap_vision import apiv1, runs
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig, run_path


class Roles:
    """mycomap.org's roles, as feedback_api.OrgClient.roles answers them."""

    def __init__(self, admins=()):
        self.admins = set(admins)

    def roles(self, user_id):
        return {"admin": user_id in self.admins, "staff": False, "projects": []}


def site(conn, tmp_path, admins=("admin-1",), links=True, by_observation=False, web_dist=None,
         observation_fetcher=None, card_writer=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1", log=lambda s: None)
    conn.commit()
    bucket = runs.DirObjects(tmp_path / "bucket")
    run_store = runs.RunStore(bucket, write=lambda job: job(), note=lambda s: None,
                              index=runs.OwnerIndex(tmp_path / "site-runs.sqlite"))
    signin = SigninConfig(mode="all", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    extra = dict(api_v1=True, server_key=SERVER_KEY, by_observation=True,
                 observation_fetcher=observation_fetcher or fetcher(),
                 app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE),
                 introspect=Introspect()) if by_observation else {}
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), signin=signin, background=False, fungus_gate=None,
                     run_store=run_store if links else None, feedback_org=Roles(admins),
                     web_dist=web_dist, note=lambda s: None, card_writer=card_writer, **extra)
    return app, run_store, tmp_path / "bucket"


def person(app, sub, name="Mycena Fan"):
    """A browser signed in as `sub`, through the whole sign-in round trip."""
    client = TestClient(app, follow_redirects=False)
    start = client.get("/auth/signin", params={"returnTo": "/"})
    nonce = parse_qs(urlsplit(start.headers["location"]).query)["nonce"][0]
    done = client.get("/auth/dev-bridge/callback", params={"token": mint(
        {"sub": sub, "aud": HERE, "nonce": nonce, "name": name})})
    assert done.status_code == 302
    return client


def visitor(app):
    return TestClient(app, follow_redirects=False)


def jpeg_with_gps(size=(2000, 1500), colour=(247, 0, 0)) -> bytes:
    """A photo as a phone saves it: EXIF with the camera, the time and the exact place."""
    img = Image.new("RGB", size, colour)
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"                       # Make
    exif[0x0132] = "2026:10:11 09:30:00"              # DateTime
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (40.0, 7.0, 24.44), 3: "W", 4: (82.0, 58.0, 48.0)})
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def identify(client, photos=(("a.png", png(247), "image/png"),), **data):
    return client.post("/api/identify", files=[("photos", p) for p in photos], data=data)


# --- every run gets its own link ---------------------------------------------------------

def test_an_identification_on_the_site_comes_back_with_its_own_link(conn, tmp_path):
    app, store, _ = site(conn, tmp_path)
    res = identify(person(app, "user-1"))
    assert res.status_code == 200, res.text
    run = res.json()["run"]
    assert runs.valid_id(run["id"]) and run["url"] == f"/r/{run['id']}"
    assert store.get(run["id"])["kind"] == "photos"


def test_with_run_links_off_nothing_is_kept_and_no_link_is_given(conn, tmp_path):
    app, _, bucket = site(conn, tmp_path, links=False)
    res = identify(person(app, "user-1"))
    assert res.status_code == 200
    assert "run" not in res.json()
    assert not bucket.exists()


def test_run_ids_are_16_random_url_safe_characters_never_repeated():
    ids = {runs.new_id() for _ in range(10_000)}
    assert len(ids) == 10_000
    assert all(runs.valid_id(i) for i in ids)
    assert not runs.valid_id("abc") and not runs.valid_id("../../etc/passwd/x")
    assert not runs.valid_id("A" * 15 + "!")


def test_an_id_that_was_never_given_out_finds_nothing(conn, tmp_path):
    app, _, _ = site(conn, tmp_path)
    anyone = visitor(app)
    assert anyone.get(f"/api/runs/{runs.new_id()}").status_code == 404
    assert anyone.get("/api/runs/not-an-id").status_code in (401, 404)


# --- what is kept: never EXIF, the exact place or anyone's identity -------------------------

def test_a_kept_photo_is_a_small_copy_with_no_exif_and_no_gps(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    body = jpeg_with_gps()
    res = identify(person(app, "user-1"), photos=(("phone.jpg", body, "image/jpeg"),))
    assert res.status_code == 200, res.text
    rid = res.json()["run"]["id"]
    copy = visitor(app).get(f"/api/runs/{rid}/photos/0.jpg")
    assert copy.status_code == 200 and copy.headers["content-type"] == "image/jpeg"
    on_disk = (bucket / runs.PREFIX / rid / "0.jpg").read_bytes()
    assert copy.content == on_disk
    for kept in (copy.content, on_disk):
        img = Image.open(io.BytesIO(kept))
        assert max(img.size) == runs.COPY_SIDE
        assert not dict(img.getexif()) and "exif" not in img.info
        assert b"Exif" not in kept and b"PhoneMaker" not in kept and b"2026:10:11" not in kept
    # Nothing else of the original was kept beside it.
    assert sorted(p.name for p in (bucket / runs.PREFIX / rid).iterdir()) == ["0.jpg", "run.json"]


def test_the_exact_place_is_never_kept_or_shown_only_the_tenth_of_a_degree(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    res = identify(person(app, "user-1"), lat="40.12345", lng="-82.98765",
                   observed_on="2026-10-11")
    rid = res.json()["run"]["id"]
    raw = (bucket / runs.PREFIX / rid / "run.json").read_text()
    assert "40.12345" not in raw and "82.98765" not in raw and "40.123" not in raw
    shown = visitor(app).get(f"/api/runs/{rid}").json()
    assert shown["context_used"] == {"latitude": 40.1, "longitude": -83.0,
                                     "observed_on": "2026-10-11", "place_from": "entered",
                                     "date_from": "entered"}
    assert "40.12345" not in json.dumps(shown)


def test_a_run_keeps_a_place_to_a_tenth_of_a_degree_whatever_it_was_handed():
    record = runs.run_record("photos", None, [], {
        "latitude": 40.12345, "longitude": -82.98765, "observed_on": "2026-10-11",
        "place_from": "photo", "date_from": "photo"}, "2026-10-20T10:00:00Z", None)
    assert record["context_used"]["latitude"] == 40.1
    assert record["context_used"]["longitude"] == -83.0
    assert "40.12345" not in json.dumps(record)


def test_run_links_are_off_unless_mv_run_links_names_where(tmp_path):
    for off in (None, "", "off", "OFF", "0", "no"):
        assert runs.from_setting(off, tmp_path) is None, off
    store = runs.from_setting(str(tmp_path / "runs"), tmp_path)
    assert isinstance(store.objects, runs.DirObjects)
    assert (tmp_path / "site-runs.sqlite").is_file()


def test_a_place_read_from_the_photos_exif_is_kept_only_to_a_tenth_of_a_degree(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    res = identify(person(app, "user-1"), photos=(("phone.jpg", jpeg_with_gps(), "image/jpeg"),))
    rid = res.json()["run"]["id"]
    run = json.loads((bucket / runs.PREFIX / rid / "run.json").read_text())
    assert run["context_used"]["place_from"] == "photo"
    assert (run["context_used"]["latitude"], run["context_used"]["longitude"]) == (40.1, -83.0)


def test_who_ran_it_is_kept_but_never_shown_to_anyone(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    rid = identify(person(app, "user-1", "Mycena Fan")).json()["run"]["id"]
    run = json.loads((bucket / runs.PREFIX / rid / "run.json").read_text())
    assert run["owner"] == {"id": "user-1", "name": "Mycena Fan"}
    for client in (visitor(app), person(app, "user-2"), person(app, "user-1", "Mycena Fan")):
        text = client.get(f"/api/runs/{rid}").text
        assert "user-1" not in text and "Mycena Fan" not in text
        assert '"owner"' not in text and '"vectors"' not in text and '"photo_ids"' not in text


def test_no_ones_own_collections_are_kept_in_a_run():
    store = runs.RunStore(runs.DirObjects.__new__(runs.DirObjects), write=lambda job: None)
    kept = store.keep({"kind": "photos", "owner": None, "models": [
        {"result": {"yours": {"linked": True, "records": 2, "specimens": [{"x": 1}]},
                    "per_photo": [{"specimens": [{"observation_id": "1", "yours": True}]}]}}]})
    held = store.get(kept)
    assert "yours" not in json.dumps(held)


# --- who may open a link -------------------------------------------------------------------

def test_anyone_with_the_link_may_open_it_and_its_photos_signed_in_or_not(conn, tmp_path):
    app, _, _ = site(conn, tmp_path)
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    for client in (visitor(app), person(app, "user-2")):
        res = client.get(f"/api/runs/{rid}")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["kind"] == "photos" and body["photos"] == 1 and body["mine"] is False
        assert body["results"][0]["ranks"]["species"][0]["name"] == "A x"
        assert client.get(f"/api/runs/{rid}/photos/0.jpg").status_code == 200


def test_a_links_page_answer_and_photos_are_never_cached_or_indexed(conn, tmp_path):
    app, _, _ = site(conn, tmp_path)
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    anyone = visitor(app)
    for path in (f"/api/runs/{rid}", f"/api/runs/{rid}/photos/0.jpg",
                 f"/api/runs/{runs.new_id()}"):
        res = anyone.get(path)
        assert res.headers["x-robots-tag"] == "noindex, nofollow", path
        assert "no-store" in res.headers["cache-control"], path


def test_only_a_runs_own_three_addresses_are_open_while_the_site_needs_sign_in(conn, tmp_path):
    app, _, _ = site(conn, tmp_path)
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    anyone = visitor(app)
    # The page itself (the shell; the site's catch-all) is not sent to sign in.
    assert anyone.get(f"/r/{rid}").status_code != 302
    # Everything else still is: other pages, identifying, deleting, other API routes.
    assert anyone.get("/research").status_code == 302
    assert anyone.get(f"/r/{rid}/x").status_code == 302
    assert identify(anyone).status_code == 401
    assert anyone.delete(f"/api/runs/{rid}").status_code == 401
    assert anyone.get("/api/runs").status_code == 401
    assert anyone.get(f"/api/runs/{rid}/photos/0.png").status_code == 401
    assert run_path(f"/r/{rid}") and run_path(f"/api/runs/{rid}/photos/3.jpg")
    for path in ("/r/short", f"/r/{rid}/", f"/api/runs/{rid}/vectors", "/api/runs",
                 f"/api/runs/{rid}/photos/0.jpg/x", f"/r/{rid}?x"):
        assert not run_path(path), path


def test_a_links_page_loads_signed_out_and_tells_search_engines_to_stay_away(conn, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>site</title>")
    app, _, _ = site(conn, tmp_path, web_dist=dist)
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    page = visitor(app).get(f"/r/{rid}")
    assert page.status_code == 200 and "<title>site</title>" in page.text
    assert page.headers["x-robots-tag"] == "noindex, nofollow"
    assert "x-robots-tag" not in person(app, "user-1").get("/about").headers


def test_the_person_who_ran_it_gets_their_own_part_and_no_one_else_does(conn, tmp_path):
    app, _, _ = site(conn, tmp_path)
    owner = person(app, "user-1")
    rid = identify(owner).json()["run"]["id"]
    mine = owner.get(f"/api/runs/{rid}").json()
    assert mine["mine"] is True and mine["can_delete"] is True
    other = person(app, "user-2").get(f"/api/runs/{rid}").json()
    assert other["mine"] is False and other["can_delete"] is False
    assert all("yours" not in r for r in other["results"])


def test_a_specimen_photo_whose_photographer_withdrew_since_is_not_shown(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    before = visitor(app).get(f"/api/runs/{rid}").json()
    shown = [s for s in before["results"][0]["specimens"] if s["photo_url"]]
    assert shown
    conn.execute("update photos set license_class = 'arr'")       # nobody's may be shown now
    conn.commit()
    after = visitor(app).get(f"/api/runs/{rid}").json()
    assert all(s["photo_url"] is None and s["photo_withheld"]
               for s in after["results"][0]["specimens"])


# --- deleting ---------------------------------------------------------------------------------

def test_the_person_who_ran_it_can_delete_it_and_then_it_is_gone(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    owner = person(app, "user-1")
    rid = identify(owner, photos=(("a.jpg", jpeg_with_gps(), "image/jpeg"),)).json()["run"]["id"]
    assert (bucket / runs.PREFIX / rid / "0.jpg").is_file()
    assert owner.delete(f"/api/runs/{rid}").status_code == 204
    anyone = visitor(app)
    assert anyone.get(f"/api/runs/{rid}").status_code == 404
    assert anyone.get(f"/api/runs/{rid}/photos/0.jpg").status_code == 404
    assert not (bucket / runs.PREFIX / rid).exists()
    assert store.index.runs_of("user-1") == []
    assert owner.delete(f"/api/runs/{rid}").status_code == 404


def test_an_admin_can_delete_anyones_run_and_another_member_cannot(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, admins=("admin-1",))
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    other = person(app, "user-2")
    assert other.get(f"/api/runs/{rid}").json()["can_delete"] is False
    assert other.delete(f"/api/runs/{rid}").status_code == 403
    admin = person(app, "admin-1")
    assert admin.get(f"/api/runs/{rid}").json()["can_delete"] is True
    assert admin.delete(f"/api/runs/{rid}").status_code == 204
    assert visitor(app).get(f"/api/runs/{rid}").status_code == 404


# --- a person's runs, findable later -------------------------------------------------------

def test_a_persons_runs_are_found_by_them_and_never_by_another(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path)
    a, b = person(app, "user-1", "Ann"), person(app, "user-2", "Bo")
    first = identify(a).json()["run"]["id"]
    second = identify(a).json()["run"]["id"]
    theirs = identify(b).json()["run"]["id"]
    assert {r["run_id"] for r in store.index.runs_of("user-1")} == {first, second}
    assert {r["run_id"] for r in store.index.runs_of("user-2")} == {theirs}
    # The bucket's copy of each list, to rebuild the box's index from.
    listed = json.loads((bucket / runs.owner_list_key("user-1")).read_text())
    assert listed["owner"] == {"id": "user-1", "name": "Ann"}
    assert {r["run_id"] for r in listed["runs"]} == {first, second}
    other = json.loads((bucket / runs.owner_list_key("user-2")).read_text())
    assert {r["run_id"] for r in other["runs"]} == {theirs}


# --- iNat and MO links ---------------------------------------------------------------------

def by_link(client, source="inat", number=555, **body):
    return client.post("/api/identify/observation", json={"source": source, "id": number, **body})


def test_an_observation_run_keeps_no_photos_only_their_addresses_at_the_source(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path, by_observation=True)
    res = by_link(person(app, "user-1"))
    assert res.status_code == 200, res.text
    rid = res.json()["run"]["id"]
    assert sorted(p.name for p in (bucket / runs.PREFIX / rid).iterdir()) == ["run.json"]
    shown = visitor(app).get(f"/api/runs/{rid}").json()
    assert shown["kind"] == "observation" and shown["photos"] == 0
    assert shown["observation"]["source"] == "inat" and shown["observation"]["id"] == "555"
    assert shown["observation"]["photos_shown"] == res.json()["observation"]["photos_shown"]
    assert visitor(app).get(f"/api/runs/{rid}/photos/0.jpg").status_code == 404


def test_an_observation_answered_once_has_one_link_for_everyone_who_opens_it(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, by_observation=True)
    first = by_link(person(app, "user-1")).json()
    again = by_link(person(app, "user-2")).json()
    assert again["cached"] is True
    assert again["run"]["id"] == first["run"]["id"]
    # Both find it among their runs.
    assert [r["run_id"] for r in store.index.runs_of("user-1")] == [first["run"]["id"]]
    assert [r["run_id"] for r in store.index.runs_of("user-2")] == [first["run"]["id"]]


def test_run_again_gets_a_new_link_and_the_old_one_keeps_the_old_answer(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, by_observation=True)
    client = person(app, "user-1")
    first = by_link(client).json()["run"]["id"]
    again = by_link(client, rerun=True).json()["run"]["id"]
    assert again != first
    assert visitor(app).get(f"/api/runs/{first}").status_code == 200


def test_kept_observation_answers_are_untouched_by_links_and_deleting_them(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, by_observation=True)
    client = person(app, "user-1")
    rid = by_link(client).json()["run"]["id"]
    cache = app.state.answer_cache

    def rows():
        return cache._conn.execute("select * from observation_answers order by 1").fetchall()
    before = rows()
    assert before
    visitor(app).get(f"/api/runs/{rid}")
    assert client.delete(f"/api/runs/{rid}").status_code == 204
    assert rows() == before
    # The observation is still answered from what is kept, without reading it again.
    assert by_link(person(app, "user-2")).json()["cached"] is True


def test_an_observation_runs_id_cannot_be_worked_out_without_the_servers_secret():
    key, at = "site:inat:555:m1/nearest:date=True,place=True", "2026-10-20T10:00:00Z"
    a = runs.observation_run_id(b"s" * 32, key, at)
    assert runs.valid_id(a) and a == runs.observation_run_id(b"s" * 32, key, at)
    assert a != runs.observation_run_id(b"t" * 32, key, at)
    assert a != runs.observation_run_id(b"s" * 32, key, "2026-10-20T10:00:01Z")


def test_an_answer_is_never_lost_when_its_link_cant_be_made(conn, tmp_path, monkeypatch):
    app, store, _ = site(conn, tmp_path, by_observation=True)

    def down(*a, **k):
        raise OSError("the bucket is unreachable")
    monkeypatch.setattr(store, "keep", down)
    monkeypatch.setattr(store, "get", down)
    client = person(app, "user-1")
    res = identify(client)
    assert res.status_code == 200 and res.json()["results"] and "run" not in res.json()
    res = by_link(client)
    assert res.status_code == 200 and res.json()["results"] and "run" not in res.json()


# --- the store ------------------------------------------------------------------------------

def test_a_missing_object_in_the_bucket_is_no_run_even_when_s3_says_403():
    from botocore.exceptions import ClientError

    class Client:
        def get_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    assert runs.S3Objects("b", Client()).get("site-runs/x/run.json") is None

    class Broken:
        def get_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "SlowDown"}}, "GetObject")
    with pytest.raises(ClientError):
        runs.S3Objects("b", Broken()).get("site-runs/x/run.json")


def test_every_object_a_run_writes_is_under_site_runs():
    written = []

    class Objects:
        def put(self, key, body, content_type):
            written.append(key)

        def get(self, key):
            return None

        def delete(self, key):
            written.append(key)
    store = runs.RunStore(Objects(), write=lambda job: job())
    rid = store.keep({"kind": "photos", "owner": {"id": "u", "name": "n"}, "models": []},
                     [jpeg_with_gps((40, 30))])
    store.delete(rid)
    assert written and all(k.startswith("site-runs/") for k in written)


def test_a_run_is_answered_from_memory_until_the_bucket_has_it():
    jobs = []
    store = runs.RunStore(runs.DirObjects.__new__(runs.DirObjects), write=jobs.append)
    rid = store.keep({"kind": "photos", "owner": None, "models": []})
    assert store.get(rid)["kind"] == "photos"         # still waiting to be written
    assert len(jobs) == 1


def test_a_folder_store_never_writes_outside_its_folder(tmp_path):
    objects = runs.DirObjects(tmp_path / "bucket")
    with pytest.raises(ValueError):
        objects.put("../escape.txt", b"x", "text/plain")


# --- the box's health check ----------------------------------------------------------------

def test_the_box_reports_its_runs_and_their_photos_size_to_the_server_key_only(conn, tmp_path):
    app, store, bucket = site(conn, tmp_path, by_observation=True)
    client = person(app, "user-1")
    upload = identify(client, photos=(("a.jpg", jpeg_with_gps((800, 600)), "image/jpeg"),))
    by_link(client)
    size = (bucket / runs.PREFIX / upload.json()["run"]["id"] / "0.jpg").stat().st_size
    full = visitor(app).get("/api/health", headers={"Authorization": f"Bearer {SERVER_KEY}"})
    assert full.json()["run_links"] == {"runs": 2, "by_kind": {"photos": 1, "observation": 1},
                                        "photo_bytes": size}
    # A deleted run leaves the count.
    assert client.delete(f"/api/runs/{upload.json()['run']['id']}").status_code == 204
    after = visitor(app).get("/api/health", headers={"Authorization": f"Bearer {SERVER_KEY}"})
    assert after.json()["run_links"] == {"runs": 1, "by_kind": {"observation": 1}, "photo_bytes": 0}
    # Never in the public answer.
    assert "run_links" not in visitor(app).get("/api/health").json()


def test_with_run_links_off_the_health_check_says_so(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, links=False, by_observation=True)
    full = visitor(app).get("/api/health", headers={"Authorization": f"Bearer {SERVER_KEY}"})
    assert full.json()["run_links"] is None
