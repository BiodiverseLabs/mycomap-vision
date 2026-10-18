"""The Identify page's "from a link" (POST /api/identify/observation): an iNat or MO
observation's own photos, place and date, read as identify-by-observation reads them and
answered in the site's shape; the page may show only the Creative Commons or
public-domain photos among them."""

import pytest
from test_api import png
from test_api_v1_by_observation import INAT_PHOTO, MO_PHOTO, app, inat_obs, mo_obs

from mycomap_vision import byobservation as bo
from mycomap_vision.signin import SigninConfig
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET


def by_link(client, source, number, **body):
    return client.post("/api/identify/observation", json={"source": source, "id": number, **body})


def with_licences(obs: dict, codes: list) -> dict:
    for op, code in zip(obs["observation_photos"], codes):
        op["photo"]["license_code"] = code
    return obs


def test_a_link_gets_the_answer_its_photos_place_and_date_would_get_on_the_site(conn, tmp_path):
    client, site = app(conn, tmp_path)
    res = by_link(client, "inat", 555)
    assert res.status_code == 200, res.text
    body = res.json()
    uploaded = client.post("/api/identify", files=[("photos", ("a.png", png(247), "image/png")),
                                                   ("photos", ("b.png", png(245), "image/png"))],
                           data={"lat": "40.12", "lng": "-82.98", "observed_on": "2026-10-11"})
    assert uploaded.status_code == 200, uploaded.text
    a, b = body["results"][0], uploaded.json()["results"][0]
    assert a["ranks"] == b["ranks"]
    assert [p["ranks"] for p in a["per_photo"]] == [p["ranks"] for p in b["per_photo"]]
    assert body["context_used"]["place_from"] == "observation"
    assert body["context_used"]["date_from"] == "observation"
    obs = body["observation"]
    assert (obs["source"], obs["id"], obs["photos"], obs["photos_used"]) == ("inat", "555", 2, 2)
    assert obs["url"] == "https://www.inaturalist.org/observations/555"
    assert obs["excluded_reference"] is False and obs["left_out_photos"] == 0
    assert body["timing"]["fetch_ms"] is not None
    # Read from iNat's API, then its photo host at large size.
    assert site.urls()[1:] == [INAT_PHOTO.format(id=9001, size="large"),
                               INAT_PHOTO.format(id=9002, size="large")]


def test_only_creative_commons_photos_are_offered_to_the_page(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[555] = with_licences(inat_obs(555, [9001, 9002]), [None, "cc-by-nc"])
    shown = by_link(client, "inat", 555).json()["observation"]["photos_shown"]
    assert shown == [None, INAT_PHOTO.format(id=9002, size="large")]


def test_photos_shown_follow_per_photo_when_a_photo_cant_be_read(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[560] = with_licences(inat_obs(560, [9001, 9003, 9002]), ["cc0", "cc0", "cc-by"])
    body = by_link(client, "inat", 560).json()
    assert body["observation"]["photos"] == 3 and body["observation"]["photos_used"] == 2
    assert body["observation"]["photos_shown"] == [INAT_PHOTO.format(id=9001, size="large"),
                                                   INAT_PHOTO.format(id=9002, size="large")]
    assert [p["photo"] for p in body["results"][0]["per_photo"]] == [0, 1]


def test_an_mo_link_reads_its_licences_too(conn, tmp_path):
    client, site = app(conn, tmp_path)
    obs = mo_obs(777, 5001, [5002])
    obs["primary_image"]["license"] = "Creative Commons Wikipedia Compatible v3.0"
    obs["images"][0]["license"] = "All Rights Reserved"
    site.mo[777] = obs
    body = by_link(client, "mo", 777).json()
    assert body["results"][0]["ranks"]["species"][0]["name"] == "B y"
    assert body["observation"]["photos_shown"] == [MO_PHOTO.format(id=5001, size="960"), None]


def test_a_reference_record_never_finds_itself_and_its_place_is_not_used(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[100] = inat_obs(100, [9001])           # seeded reference record "A x"
    body = by_link(client, "inat", 100).json()
    obs = body["observation"]
    assert obs["excluded_reference"] is True and obs["left_out_photos"] >= 1
    assert "100" not in {s["observation_id"] for s in body["results"][0]["specimens"]}
    assert body["context_used"]["place_from"] is None and body["context_used"]["date_from"] is None


def test_a_photo_filed_under_a_reference_record_is_left_out(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    body = by_link(client, "inat", 556).json()       # its photo 1003 is record 103's
    assert body["observation"]["excluded_reference"] is False
    assert body["observation"]["left_out_photos"] == 1
    assert "103" not in {s["observation_id"] for s in body["results"][0]["specimens"]}
    assert body["context_used"]["place_from"] is None


@pytest.mark.parametrize("source,number,status", [
    ("inat", 404040, 404), ("mo", 404040, 404), ("inat", 557, 422), ("inat", 558, 422),
    ("gbif", 1, 400), ("inat", "12a", 400), ("inat", "", 400),
])
def test_what_cant_be_identified_is_refused(conn, tmp_path, source, number, status):
    client, _ = app(conn, tmp_path)
    assert by_link(client, source, number).status_code == status


def test_use_place_and_use_date_false_leave_them_out(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    used = by_link(client, "inat", 555, use_place=False, use_date=False).json()["context_used"]
    assert used["place_from"] is None and used["date_from"] is None


def test_the_route_is_off_with_identify_by_observation(conn, tmp_path):
    off, site = app(conn, tmp_path, by_observation=False)
    assert by_link(off, "inat", 555).status_code in (404, 405)
    assert site.log == []


def test_the_link_needs_a_signed_in_person_whenever_identify_does():
    for mode in ("identify", "all"):
        signin = SigninConfig(mode=mode, origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY, secret=SECRET)
        assert signin.needs_session("POST", "/api/identify/observation")


@pytest.mark.parametrize("licence,shown", [
    ("cc0", True), ("cc-by-nc-nd", True), ("CC-BY", True),
    ("Creative Commons Wikipedia Compatible v3.0", True), ("Public Domain", True),
    (None, False), ("", False), ("All Rights Reserved", False), ("Copyright", False),
])
def test_open_licences_are_creative_commons_or_public_domain(licence, shown):
    assert bo.open_licence(licence) is shown


def test_a_photo_without_a_licence_list_is_not_shown():
    obs = bo.Observation("inat", "1", ["https://x/1.jpg"], [1])
    assert bo.shown_photo_url(obs, 0) is None and bo.shown_photo_url(obs, 5) is None


# --- answers kept until what Vision answers from changes --------------------------------

from test_api_v1_by_observation import SERVER, by_number  # noqa: E402

from mycomap_vision.answercache import AnswerCache, answer_key, answers_version  # noqa: E402


def test_a_link_answered_before_comes_back_without_reading_the_observation_again(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_link(client, "inat", 555).json()
    reads = len(site.log)
    again = by_link(client, "inat", 555)
    assert again.status_code == 200, again.text
    body = again.json()
    assert len(site.log) == reads                   # nothing read from iNat
    assert (first["cached"], body["cached"]) == (False, True)
    assert body["results"][0]["ranks"] == first["results"][0]["ranks"]
    assert body["observation"] == first["observation"]
    assert body["context_used"] == first["context_used"]


def test_other_options_are_answered_on_their_own(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_link(client, "inat", 555)
    reads = len(site.log)
    body = by_link(client, "inat", 555, use_place=False).json()
    assert body["cached"] is False and len(site.log) > reads


def test_an_answer_from_another_version_is_never_served(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_link(client, "inat", 555)
    cache = client.app.state.answer_cache
    cache._conn.execute("update observation_answers set version = 'older'")
    reads = len(site.log)
    body = by_link(client, "inat", 555).json()
    assert body["cached"] is False and len(site.log) > reads


def test_identify_by_observation_keeps_its_answer_too_and_says_the_version(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_number(client, "inat", 555).json()
    reads = len(site.log)
    again = by_number(client, "inat", 555)
    assert again.status_code == 200, again.text
    body = again.json()
    assert len(site.log) == reads
    assert (first["cached"], body["cached"]) == (False, True)
    assert body["request_id"] != first["request_id"]
    for key in ("ranks", "likely", "specimens", "answer", "gate", "observation", "context_used"):
        assert body[key] == first[key], key
    version = first["model"]["answers_version"]
    assert body["model"]["answers_version"] == version
    models = client.get("/api/v1/models", headers=SERVER).json()
    assert models["models"][0]["answers_version"] == version


def test_kept_answers_follow_the_version():
    cache = AnswerCache(None)
    key = answer_key("v1", "inat", "1", "m1", "nearest", {"top_k": 5})
    cache.put(key, "v1", {"a": 1})
    assert cache.get(key, "v1") == {"a": 1}
    assert cache.get(key, "v2") is None
    cache.put(key, "v2", {"a": 2})
    assert cache.get(key, "v1") is None and cache.get(key, "v2") == {"a": 2}
    other = answer_key("v1", "inat", "1", "m1", "nearest", {"top_k": 3})
    assert other != key and cache.get(other, "v2") is None


def test_the_answers_version_moves_with_the_release_and_the_reference():
    base = answers_version("2026-10", (100, "run-1", "fp", 0))
    assert base == answers_version("2026-10", (100, "run-1", "fp", 0))
    for other in (answers_version("2026-11", (100, "run-1", "fp", 0)),   # a new release
                  answers_version("2026-10", (101, "run-1", "fp", 0)),   # photos added
                  answers_version("2026-10", (100, "run-2", "fp", 0)),   # recalibrated
                  answers_version("2026-10", (100, "run-1", "fp2", 0)),  # someone said no
                  answers_version("2026-10", (100, "run-1", "fp", 7))):  # a night's changes
        assert other != base
