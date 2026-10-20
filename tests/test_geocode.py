"""Place search for the Identify page's map: Nominatim through Vision, within its policy."""

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from mycomap_vision import config, geocode
from mycomap_vision.api import create_app
from mycomap_vision.geocode import GeocodeError, PlaceSearch
from mycomap_vision.signin import SigninConfig

CORVALLIS = [{"display_name": "Corvallis, Benton County, Oregon, United States",
              "lat": "44.5645659", "lon": "-123.2620435", "importance": 0.6}]


class Clock:
    """A clock the test moves; sleeping moves it too, and is recorded."""

    def __init__(self):
        self.now, self.slept = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.now += s


def searcher(answer=CORVALLIS, **kw):
    clock, calls = Clock(), []

    def fetch(url):
        calls.append(url)
        if isinstance(answer, Exception):
            raise answer
        return answer
    return PlaceSearch(fetch, clock=clock, sleep=clock.sleep, **kw), calls, clock


def test_a_place_name_gives_points_in_mycomap_orgs_shape():
    s, calls, _ = searcher()
    assert s.search("  Corvallis   Oregon ") == [
        {"display_name": "Corvallis, Benton County, Oregon, United States",
         "lat": 44.5645659, "lon": -123.2620435}]
    q = parse_qs(urlparse(calls[0]).query)
    assert calls[0].startswith(geocode.NOMINATIM)
    assert q["q"] == ["Corvallis Oregon"] and q["limit"] == ["5"] and q["format"] == ["json"]


def test_nominatim_is_asked_with_our_user_agent_and_nothing_about_the_person(monkeypatch):
    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return R()
    import requests
    monkeypatch.setattr(requests, "get", get)
    geocode.nominatim_fetch("https://nominatim.openstreetmap.org/search?q=x")
    assert seen["headers"]["User-Agent"] == config.USER_AGENT
    assert set(seen["headers"]) == {"User-Agent", "Accept"}
    assert seen["timeout"]


def test_the_same_search_is_answered_from_the_cache_for_a_day():
    s, calls, clock = searcher()
    s.search("Corvallis")
    clock.now += 23 * 3600
    s.search("corvallis")                       # same words, any case
    assert len(calls) == 1
    clock.now += 2 * 3600                        # past 24 h: asked again
    s.search("Corvallis")
    assert len(calls) == 2


def test_the_cache_keeps_only_the_newest_searches():
    s, calls, clock = searcher(max_entries=2)
    for q in ("Alpha", "Bravo", "Charlie"):
        clock.now += 2
        s.search(q)
    clock.now += 2
    s.search("Charlie")
    assert len(calls) == 3                      # still cached
    clock.now += 2
    s.search("Alpha")
    assert len(calls) == 4                      # dropped as the oldest


def test_nominatim_is_never_asked_more_than_once_a_second():
    s, calls, clock = searcher()
    s.search("Alpha")
    s.search("Bravo")                           # at once: waits its turn
    assert clock.slept == [1.1]
    clock.now += 5
    s.search("Charlie")                         # long after: no wait
    assert clock.slept == [1.1]
    assert len(calls) == 3


def test_a_search_that_would_wait_too_long_is_refused_with_retry_after():
    s, calls, clock = searcher(max_wait=2.0)
    s.search("Alpha")
    s._next_slot = clock.now + 4.2               # a queue of searches ahead of this one
    with pytest.raises(GeocodeError) as e:
        s.search("Bravo")
    assert e.value.status == 429 and e.value.retry_after == 5
    assert len(calls) == 1


def test_short_long_and_failed_searches_are_refused_plainly():
    s, calls, _ = searcher()
    assert s.search("ab") == [] and s.search("   ") == []
    with pytest.raises(GeocodeError) as e:
        s.search("x" * 201)
    assert e.value.status == 400
    assert calls == []
    down, _, _ = searcher(answer=TimeoutError("slow"))
    with pytest.raises(GeocodeError) as e:
        down.search("Corvallis")
    assert e.value.status == 503 and "try again" in e.value.detail


def test_a_failure_is_not_cached_and_odd_rows_are_skipped():
    s, calls, clock = searcher(answer=RuntimeError("down"))
    with pytest.raises(GeocodeError):
        s.search("Corvallis")
    s.fetch = lambda url: [{"lat": "x", "lon": "1"}, {"lat": "95", "lon": "0"}, "junk",
                           {"lat": "1.5", "lon": "2.5"}]
    clock.now += 2
    assert s.search("Corvallis") == [{"display_name": "", "lat": 1.5, "lon": 2.5}]


def app(tmp_path, place_search, signin=None):
    return TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                 place_search=place_search, web_dist=None, background=False,
                                 signin=signin, fungus_gate=None, server_key=None,
                                 app_tokens=None, introspect=None))


def test_the_page_gets_places_and_refusals_as_json(conn, tmp_path):
    s, _, _ = searcher(max_wait=2.0)
    client = app(tmp_path, s)
    r = client.get("/api/geocode/search", params={"q": "Corvallis"})
    assert r.status_code == 200 and r.json()[0]["lat"] == 44.5645659
    assert client.get("/api/geocode/search", params={"q": "x" * 201}).json() == {
        "detail": "Search for a shorter place name."}
    s._next_slot = s.clock() + 10
    r = client.get("/api/geocode/search", params={"q": "Bravo"})
    assert r.status_code == 429 and r.headers["Retry-After"] == "10"


def test_place_search_is_behind_the_sign_in_like_the_rest_of_the_api(conn, tmp_path):
    from test_signin import HERE, PUBLIC_KEY, SECRET
    s, calls, _ = searcher()
    client = app(tmp_path, s, signin=SigninConfig(mode="all", origin=HERE, public_key=PUBLIC_KEY,
                                                  secret=SECRET))
    r = client.get("/api/geocode/search", params={"q": "Corvallis"})
    assert r.status_code == 401
    assert calls == []


# --- Suggestions as you type: Photon, never Nominatim -------------------------------------

from mycomap_vision.geocode import PlaceSuggest  # noqa: E402

PERRY = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-82.24, 39.74]},
     "properties": {"type": "county", "osm_key": "place", "osm_value": "county", "name": "Perry",
                    "state": "Ohio", "country": "United States", "countrycode": "US",
                    "extent": [-82.47, 39.93, -82.02, 39.55]}},
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-82.55, 39.45]},
     "properties": {"type": "other", "osm_key": "leisure", "osm_value": "nature_reserve",
                    "name": "Hocking Hills State Park", "county": "Hocking", "state": "Ohio",
                    "country": "United States", "countrycode": "US"}},
]}


def suggester(answer=PERRY, **kw):
    clock, calls = Clock(), []

    def fetch(url):
        calls.append(url)
        if isinstance(answer, Exception):
            raise answer
        return answer
    return PlaceSuggest(fetch, clock=clock, **kw), calls, clock


def test_suggestions_come_from_photon_in_the_place_search_shape():
    s, calls, _ = suggester()
    places, status = s.suggest("  Perry   County, OH ", "1.2.3.4")
    assert status == "ok"
    assert places == [
        {"display_name": "Perry County, Ohio, United States", "lat": 39.74, "lon": -82.24},
        {"display_name": "Hocking Hills State Park, Hocking County, Ohio, United States",
         "lat": 39.45, "lon": -82.55}]
    assert calls[0].startswith(geocode.PHOTON) and "nominatim" not in calls[0]
    q = parse_qs(urlparse(calls[0]).query)
    assert q["q"] == ["Perry County, OH"]
    assert q["layer"] == ["county"]                     # the word "county" asks for counties
    assert set(q) == {"q", "limit", "lang", "layer"}    # no location bias, nothing about the person


def test_a_query_without_the_word_county_is_not_held_to_counties():
    s, calls, _ = suggester()
    s.suggest("Hocking Hills")
    s.suggest("Perry Cou")                              # a half-typed word isn't the word
    assert len(calls) == 2
    assert all("layer" not in parse_qs(urlparse(u).query) for u in calls)


def test_photon_is_asked_with_our_user_agent_and_a_contact_and_nothing_else(monkeypatch):
    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"features": []}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return R()
    import requests
    monkeypatch.setattr(requests, "get", get)
    geocode.photon_fetch("https://photon.komoot.io/api/?q=x")
    assert seen["headers"]["User-Agent"] == config.USER_AGENT
    assert seen["headers"]["From"] == geocode.CONTACT
    assert set(seen["headers"]) == {"User-Agent", "From", "Accept"}
    assert seen["timeout"]


def test_fewer_than_three_letters_never_reach_photon():
    s, calls, _ = suggester()
    assert s.suggest("Pe") == ([], "short")
    assert s.suggest("  P  e ") == ([], "short")
    assert s.suggest("x" * 201) == ([], "long")
    assert calls == []


def test_the_same_words_are_answered_from_the_cache():
    s, calls, clock = suggester()
    first = s.suggest("Perry County", "a")
    clock.now += 3600
    assert s.suggest("perry  county", "b") == first          # any case, any spacing
    assert len(calls) == 1
    clock.now += 24 * 3600                                    # a day on: asked again
    s.suggest("Perry County", "a")
    assert len(calls) == 2


def test_one_address_typing_too_fast_is_held_back_but_others_are_not():
    s, calls, clock = suggester(per_ip=(3, 60.0))
    for q in ("Per", "Perr", "Perry"):
        assert s.suggest(q, "1.1.1.1")[1] == "ok"
    assert s.suggest("Perry C", "1.1.1.1") == ([], "limited")
    assert s.suggest("Perry C", "2.2.2.2")[1] == "ok"         # someone else
    assert len(calls) == 4
    clock.now += 61                                           # a minute on
    assert s.suggest("Perry Co", "1.1.1.1")[1] == "ok"


def test_the_whole_server_asks_photon_at_most_its_share():
    s, calls, clock = suggester(upstream=(2, 60.0))
    assert s.suggest("Alpha", "a")[1] == "ok"
    assert s.suggest("Bravo", "b")[1] == "ok"
    assert s.suggest("Charlie", "c") == ([], "limited")
    assert s.suggest("alpha", "d")[1] == "ok"                 # the cache still answers
    assert len(calls) == 2
    clock.now += 61
    assert s.suggest("Charlie", "c")[1] == "ok"


def test_photon_failing_is_a_clean_empty_answer_and_it_is_left_alone_a_while():
    s, calls, clock = suggester(answer=TimeoutError("slow"), backoff=60)
    assert s.suggest("Perry County", "a") == ([], "unavailable")
    assert s.suggest("Hocking Hills", "a") == ([], "unavailable")
    assert len(calls) == 1                                    # not asked again while paused
    s.fetch = lambda url: (calls.append(url), PERRY)[1]
    clock.now += 61
    assert s.suggest("Perry County", "a")[1] == "ok"          # the failure wasn't cached
    assert len(calls) == 2


def test_photon_saying_too_many_is_left_alone_as_long_as_it_asks():
    import requests
    e = requests.HTTPError("429 Too Many Requests")
    e.response = type("R", (), {"headers": {"Retry-After": "300"}})()
    s, calls, clock = suggester(answer=e, backoff=60)
    assert s.suggest("Perry County") == ([], "unavailable")
    clock.now += 120
    assert s.suggest("Hocking Hills") == ([], "unavailable")
    assert len(calls) == 1
    clock.now += 200
    s.suggest("Hocking Hills")
    assert len(calls) == 2


def test_odd_features_are_skipped_and_repeats_shown_once():
    junk = {"features": [
        "junk", {"properties": {"name": "No point"}},
        {"geometry": {"coordinates": [0, 95]}, "properties": {"name": "Off the globe"}},
        {"geometry": {"coordinates": [1, 2]}, "properties": {"type": "house"}},       # no name
        {"geometry": {"coordinates": [-82.5, 39.4]}, "properties": {"name": "Ash Cave", "state": "Ohio"}},
        {"geometry": {"coordinates": [-82.6, 39.5]}, "properties": {"name": "Ash Cave", "state": "Ohio"}},
    ]}
    s, _, _ = suggester(answer=junk)
    assert s.suggest("Ash Cave") == ([{"display_name": "Ash Cave, Ohio", "lat": 39.4, "lon": -82.5}], "ok")
    s2, _, _ = suggester(answer=["not", "geojson"])
    assert s2.suggest("Ash Cave") == ([], "ok")


def test_a_us_county_reads_as_people_write_it_and_other_places_keep_their_names():
    f = {"geometry": {"coordinates": [-90, 30]}, "properties": {
        "type": "county", "name": "Orleans Parish", "state": "Louisiana", "countrycode": "US",
        "country": "United States"}}
    assert geocode._suggestion(f)["display_name"] == "Orleans Parish, Louisiana, United States"
    f["properties"].update(name="Kent", state="England", country="United Kingdom", countrycode="GB")
    assert geocode._suggestion(f)["display_name"] == "Kent, England, United Kingdom"


def test_typed_place_names_stay_out_of_the_access_log():
    import logging
    fmt = '%s - "%s %s HTTP/%s" %d'
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, fmt,
                            ("1.2.3.4:5", "GET", "/api/geocode/suggest?q=Perry%20County", "1.1", 200), None)
    geocode.GeocodeQueryRedactor().filter(rec)
    assert "Perry" not in rec.getMessage() and "/api/geocode/suggest" in rec.getMessage()
    other = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, fmt,
                              ("1.2.3.4:5", "GET", "/api/stats?x=1", "1.1", 200), None)
    geocode.GeocodeQueryRedactor().filter(other)
    assert "/api/stats?x=1" in other.getMessage()
    geocode.redact_access_log()
    geocode.redact_access_log()
    assert sum(isinstance(f, geocode.GeocodeQueryRedactor)
               for f in logging.getLogger("uvicorn.access").filters) == 1


def suggest_app(tmp_path, place_suggest, signin=None):
    return TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                 place_search=searcher()[0], place_suggest=place_suggest,
                                 web_dist=None, background=False, signin=signin, fungus_gate=None,
                                 server_key=None, app_tokens=None, introspect=None))


def test_the_page_gets_suggestions_as_a_list_and_why_one_is_empty(conn, tmp_path):
    s, calls, _ = suggester(per_ip=(2, 60.0))
    client = suggest_app(tmp_path, s)
    r = client.get("/api/geocode/suggest", params={"q": "Perry County"})
    assert r.status_code == 200 and r.headers["X-Suggest-Status"] == "ok"
    assert r.json()[0] == {"display_name": "Perry County, Ohio, United States", "lat": 39.74, "lon": -82.24}
    r = client.get("/api/geocode/suggest", params={"q": "Pe"})
    assert (r.status_code, r.json(), r.headers["X-Suggest-Status"]) == (200, [], "short")
    client.get("/api/geocode/suggest", params={"q": "Hocking"})
    r = client.get("/api/geocode/suggest", params={"q": "Hocking Hills"})   # 3rd from this address
    assert (r.status_code, r.json(), r.headers["X-Suggest-Status"]) == (200, [], "limited")


def test_photon_down_is_an_empty_list_for_the_page_not_an_error(conn, tmp_path):
    s, _, _ = suggester(answer=ConnectionError("down"))
    r = suggest_app(tmp_path, s).get("/api/geocode/suggest", params={"q": "Perry County"})
    assert (r.status_code, r.json(), r.headers["X-Suggest-Status"]) == (200, [], "unavailable")


def test_suggestions_are_behind_the_sign_in_like_the_rest_of_the_api(conn, tmp_path):
    from test_signin import HERE, PUBLIC_KEY, SECRET
    s, calls, _ = suggester()
    client = suggest_app(tmp_path, s, signin=SigninConfig(mode="all", origin=HERE, public_key=PUBLIC_KEY,
                                                           secret=SECRET))
    r = client.get("/api/geocode/suggest", params={"q": "Perry County"})
    assert r.status_code == 401
    assert calls == []
