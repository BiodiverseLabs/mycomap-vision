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
