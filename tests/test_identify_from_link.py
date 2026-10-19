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


# --- answers kept until someone runs them again (Steve, 2026-10-19) --------------------

from test_api_v1_by_observation import SERVER, by_number  # noqa: E402

from mycomap_vision import answercache  # noqa: E402
from mycomap_vision.answercache import AnswerCache, answer_key, answers_version  # noqa: E402
from mycomap_vision.identify import PhotoInfo  # noqa: E402


def age_every_kept_answer(client):
    """What a night's new references, a recalibration or a new release does to the
    answers kept before it: their answers version is no longer the current one."""
    client.app.state.answer_cache._conn.execute("update observation_answers set version = 'older'")


def site_rows(client) -> list:
    return client.app.state.answer_cache._conn.execute(
        "select answer_key, asked_by, payload from observation_answers "
        "where answer_key like 'site:%'").fetchall()


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
    assert body["answered"] == first["answered"]
    assert body["answered"]["by"] == "site" and body["answered"]["current"] is True
    assert body["answered"]["at"].endswith("Z")


def test_other_options_are_answered_on_their_own(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_link(client, "inat", 555)
    reads = len(site.log)
    body = by_link(client, "inat", 555, use_place=False).json()
    assert body["cached"] is False and len(site.log) > reads


def test_an_older_answer_is_served_until_someone_runs_it_again(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_link(client, "inat", 555).json()
    age_every_kept_answer(client)
    reads = len(site.log)
    kept = by_link(client, "inat", 555).json()
    assert len(site.log) == reads                   # never rerun on its own
    assert kept["cached"] is True
    assert kept["results"][0]["ranks"] == first["results"][0]["ranks"]
    assert kept["answered"]["current"] is False     # the page offers to run it again
    assert kept["answered"]["at"] == first["answered"]["at"]
    fresh = by_link(client, "inat", 555, rerun=True).json()
    assert fresh["cached"] is False and len(site.log) > reads
    assert fresh["answered"]["current"] is True
    reads = len(site.log)
    after = by_link(client, "inat", 555).json()
    assert after["cached"] is True and after["answered"]["current"] is True
    assert len(site.log) == reads


def test_run_again_on_the_site_runs_even_a_current_answer(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_link(client, "inat", 555)
    reads = len(site.log)
    assert by_link(client, "inat", 555, rerun=True).json()["cached"] is False
    assert len(site.log) > reads


def test_a_validate_records_run_opens_on_the_site_without_running_again(conn, tmp_path):
    client, site = app(conn, tmp_path)
    org = by_number(client, "inat", 555).json()
    assert org["cached"] is False and org["answered"]["by"] == "mycomap.org"
    reads = len(site.log)
    opened = by_link(client, "inat", 555)
    assert opened.status_code == 200, opened.text
    body = opened.json()
    assert len(site.log) == reads                   # nothing read, nothing scored
    assert body["cached"] is True
    assert body["answered"]["by"] == "mycomap.org"
    assert body["answered"]["at"] == org["answered"]["at"]
    # The same answer the site would have got by running it itself.
    own = by_link(client, "inat", 555, rerun=True).json()
    assert body["results"][0]["ranks"] == own["results"][0]["ranks"]
    assert [p["ranks"] for p in body["results"][0]["per_photo"]] == [
        p["ranks"] for p in own["results"][0]["per_photo"]]
    assert ([s["observation_id"] for s in body["results"][0]["specimens"]]
            == [s["observation_id"] for s in own["results"][0]["specimens"]])
    assert body["context_used"] == own["context_used"]
    assert body["observation"] == own["observation"]


def test_the_site_copy_of_an_api_run_keeps_the_vectors_for_the_persons_own_part(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    by_number(client, "inat", 555)
    [(key, who, payload)] = site_rows(client)
    kept = answercache.json.loads(payload)
    assert who == "mycomap.org" and key.startswith("site:inat:555:")
    assert kept["vectors"] is not None and "release" in kept
    assert '"yours"' not in payload


def test_a_reference_record_run_through_the_api_opens_on_the_site_too(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[100] = inat_obs(100, [9001])           # seeded reference record "A x"
    by_number(client, "inat", 100)
    reads = len(site.log)
    body = by_link(client, "inat", 100).json()
    assert body["cached"] is True and len(site.log) == reads
    obs = body["observation"]
    assert obs["excluded_reference"] is True and obs["left_out_photos"] >= 1
    assert len(obs["photos_shown"]) == obs["photos_used"] == obs["photos"]
    assert "100" not in {s["observation_id"] for s in body["results"][0]["specimens"]}


def test_a_site_run_answers_identify_by_observation_without_running_again(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_link(client, "inat", 555)
    reads = len(site.log)
    body = by_number(client, "inat", 555).json()
    assert len(site.log) == reads
    assert (body["cached"], body["kept_from"]) == (True, "site")
    assert body["answered"]["by"] == "site"


def test_identify_by_observation_keeps_its_answer_too_and_says_the_version(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_number(client, "inat", 555).json()
    reads = len(site.log)
    again = by_number(client, "inat", 555)
    assert again.status_code == 200, again.text
    body = again.json()
    assert len(site.log) == reads
    assert (first["cached"], body["cached"]) == (False, True)
    assert body["kept_from"] == "api"
    assert body["request_id"] != first["request_id"]
    for key in ("ranks", "likely", "specimens", "answer", "gate", "observation", "context_used",
                "answered"):
        assert body[key] == first[key], key
    version = first["model"]["answers_version"]
    assert body["model"]["answers_version"] == version
    models = client.get("/api/v1/models", headers=SERVER).json()
    assert models["models"][0]["answers_version"] == version


def test_identify_by_observation_serves_an_older_answer_until_rerun(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_number(client, "inat", 555).json()
    age_every_kept_answer(client)
    reads = len(site.log)
    kept = by_number(client, "inat", 555).json()
    assert len(site.log) == reads and kept["cached"] is True
    assert kept["model"]["answers_version"] == "older"
    assert kept["answered"]["current"] is False
    assert kept["ranks"] == first["ranks"]
    fresh = by_number(client, "inat", 555, rerun=True).json()
    assert fresh["cached"] is False and len(site.log) > reads
    assert fresh["model"]["answers_version"] == first["model"]["answers_version"]
    assert fresh["answered"]["current"] is True


def test_rerun_is_true_or_false(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    res = by_number(client, "inat", 555, rerun="yes")
    assert (res.status_code, res.json()["code"]) == (400, "bad_request")


def test_the_newer_of_the_two_kept_answers_is_held_for_org(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_number(client, "inat", 555)
    assert kept_answer(client, "inat", 555).json()["kept_from"] == "api"
    by_link(client, "inat", 555, rerun=True)          # run again on the site, later
    body = kept_answer(client, "inat", 555).json()
    assert body["kept_from"] == "site" and body["answered"]["by"] == "site"


# --- an older answer's photos follow who may be shown now ------------------------------

def specimen(obs_id, url):
    return {"observation_id": obs_id, "photo_url": url, "photo_owner": "old",
            "photo_withheld": url is None}


def test_an_older_answers_photos_follow_who_may_be_shown_now():
    infos = {1: PhotoInfo("u1", "arr", "ann", 11), 2: PhotoInfo("u2", "open", "bo", 12),
             3: PhotoInfo("u3", "arr", "cy", 13), 4: PhotoInfo("u4", "open", "di", 14)}
    record_photos = {"r1": [1, 2], "r2": [3], "r4": [4]}
    answer = {"specimens": [specimen("r1", "url:u1"), specimen("r2", "url:u3"),
                            specimen("gone", "url:u9")],
              "per_photo": [{"specimens": [specimen("r4", "url:u4")]}],
              "spare": [[specimen("r2", None)]]}
    granted = {13}                                   # ann withdrew; cy still grants
    answercache.reshow_photos(
        answer, lambda o: record_photos.get(o, []), lambda ids: {i: infos[i] for i in ids},
        lambda i: i.license_class == "open" or i.owner_user_id in granted,
        lambda i: "url:" + i.source_url)
    r1, r2, gone = answer["specimens"]
    # Ann's photo is no longer shown: another of the record's photos that may be is.
    assert (r1["photo_url"], r1["photo_owner"], r1["photo_withheld"]) == ("url:u2", "bo", False)
    assert r2["photo_url"] == "url:u3" and r2["photo_owner"] == "old"   # unchanged
    # A record no longer in the reference set shows no photo.
    assert (gone["photo_url"], gone["photo_owner"], gone["photo_withheld"]) == (None, None, True)
    assert answer["per_photo"][0]["specimens"][0]["photo_url"] == "url:u4"
    assert answer["spare"][0][0]["photo_url"] == "url:u3"


def test_an_older_answer_shows_no_photo_that_may_no_longer_be_shown(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_link(client, "inat", 555).json()["results"][0]
    assert any(s["photo_url"] for s in first["specimens"])
    age_every_kept_answer(client)
    # Every reference photo becomes all rights reserved, and nobody has granted showing it.
    conn.execute("update photos set license_class = 'arr'")
    conn.commit()
    for kept in (by_link(client, "inat", 555).json()["results"][0],
                 kept_answer(client, "inat", 555).json()):
        assert all(s["photo_url"] is None and s["photo_withheld"] for s in kept["specimens"])


def test_a_kept_answer_says_when_who_and_under_which_version():
    cache = AnswerCache(None, clock=lambda: 1_760_000_000.0)
    key = answer_key("site", "inat", "1", "m1", "nearest", {"place": True})
    cache.put(key, "v1", {"a": 1}, "mycomap.org")
    kept = cache.held(key)
    assert (kept.payload, kept.version, kept.asked_by) == ({"a": 1}, "v1", "mycomap.org")
    assert kept.about("v1") == {"at": "2025-10-09T08:53:20Z", "by": "mycomap.org",
                                "answers_version": "v1", "current": True}
    assert kept.about("v2")["current"] is False
    assert cache.held(answer_key("site", "inat", "2", "m1", "nearest", {"place": True})) is None


def test_a_store_made_before_asked_by_gets_the_column(tmp_path):
    import sqlite3
    path = tmp_path / "answers.sqlite"
    old = sqlite3.connect(path)
    old.execute("create table observation_answers (answer_key text primary key, version text "
                "not null, payload text not null, answered_at real not null)")
    old.execute("insert into observation_answers values ('k', 'v', '{}', 1.0)")
    old.commit()
    old.close()
    kept = AnswerCache(path).held("k")
    assert kept is not None and kept.asked_by is None and kept.payload == {}


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


def test_identify_by_observation_says_which_photos_may_be_shown(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[555] = with_licences(inat_obs(555, [9001, 9002]), ["cc-by", None])
    body = by_number(client, "inat", 555).json()
    assert body["observation"]["photos_shown"] == [INAT_PHOTO.format(id=9001, size="large"), None]


# --- GET /api/v1/answers: what Vision holds, for .org, without reading or scoring --------

def kept_answer(client, source, number, headers=SERVER, **params):
    return client.get(f"/api/v1/answers/{source}/{number}", params=params, headers=headers)


def test_an_answer_got_on_the_site_is_held_for_org_in_the_v1_shape(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[555] = with_licences(inat_obs(555, [9001, 9002]), ["cc-by", None])
    on_site = by_link(client, "inat", 555).json()
    reads = len(site.log)
    res = kept_answer(client, "inat", 555)
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(site.log) == reads                   # nothing read from iNat, nothing scored
    assert (body["cached"], body["kept_from"]) == (True, "site")
    result = on_site["results"][0]
    def names(ranks):                               # v1 adds its own `confident` marks
        return {r: [(c["name"], c["confidence"]) for c in ranks.get(r, [])]
                for r in ("species", "genus", "family")}
    assert names(body["ranks"]) == names(result["ranks"])
    assert [names(p["ranks"]) for p in body["per_photo"]] == [names(p["ranks"])
                                                              for p in result["per_photo"]]
    assert body["context_used"] == on_site["context_used"]
    obs = body["observation"]
    assert (obs["source"], obs["id"], obs["photos"], obs["photos_used"]) == ("inat", "555", 2, 2)
    assert obs["photos_shown"] == [INAT_PHOTO.format(id=9001, size="large"), None]
    assert "excluded_reference" not in obs and "left_out_photos" not in obs
    assert body["excluded_reference"] is None and body["left_out_photos"] == 0
    models = client.get("/api/v1/models", headers=SERVER).json()
    assert body["model"]["answers_version"] == models["models"][0]["answers_version"]
    assert "backbone" not in body["model"] and res.headers["cache-control"] == "no-store"


def test_a_reference_record_answered_on_the_site_says_it_was_left_out(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.inat[100] = inat_obs(100, [9001])           # seeded reference record "A x"
    by_link(client, "inat", 100)
    body = kept_answer(client, "inat", 100).json()
    assert body["excluded_reference"] == {"source": "inat", "source_id": "100"}


def test_an_identify_by_observation_answer_is_held_too(conn, tmp_path):
    client, site = app(conn, tmp_path)
    first = by_number(client, "inat", 555).json()
    reads = len(site.log)
    body = kept_answer(client, "inat", 555).json()
    assert len(site.log) == reads
    assert (body["cached"], body["kept_from"]) == (True, "api")
    assert body["request_id"] != first["request_id"]
    for key in ("ranks", "answer", "gate", "observation", "context_used", "model", "answered"):
        assert body[key] == first[key], key


def test_nothing_held_is_404_not_answered_and_reads_nothing(conn, tmp_path):
    client, site = app(conn, tmp_path)
    res = kept_answer(client, "inat", 555)
    assert (res.status_code, res.json()["code"]) == (404, "not_answered")
    assert site.log == []
    # An answer under other options, or another version, is not this one.
    by_link(client, "inat", 555, use_place=False)
    assert kept_answer(client, "inat", 555).status_code == 404
    assert kept_answer(client, "inat", 555, use_place="false").status_code == 200
    # Another version's answer is still held, saying it is not current.
    age_every_kept_answer(client)
    older = kept_answer(client, "inat", 555, use_place="false")
    assert older.status_code == 200 and older.json()["answered"]["current"] is False


def test_reading_a_held_answer_counts_nothing(conn, tmp_path):
    client, site = app(conn, tmp_path)
    by_link(client, "inat", 555)
    before = client.get("/api/v1/usage", headers=SERVER).json()
    kept_answer(client, "inat", 555)
    kept_answer(client, "inat", 555)
    assert client.get("/api/v1/usage", headers=SERVER).json() == before


def test_the_held_answer_needs_a_credential_and_a_real_observation(conn, tmp_path):
    client, site = app(conn, tmp_path)
    assert kept_answer(client, "inat", 555, headers={}).status_code == 401
    res = kept_answer(client, "gbif", 555)
    assert (res.status_code, res.json()["code"]) == (404, "not_found")
    assert kept_answer(client, "inat", "12ab").status_code == 400
    off, _ = app(conn, tmp_path / "off", by_observation=False)
    assert kept_answer(off, "inat", 555).json()["code"] == "not_found"
