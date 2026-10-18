"""Identify by observation number (docs/api.md §3.2): an iNat or MO observation's own photos,
place and date, read politely and kept in memory only; an observation that is itself a DNA
reference record never matches itself."""

import sys

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from test_api import open_limits, png
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer, identify
from test_api_v1_sessions import same_answer
from test_loo import reference
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, byobservation as bo, quotas as q
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.evaluate import build_index
from mycomap_vision.methods import NearestAndMean

SERVER = bearer(SERVER_KEY)
INAT_PHOTO = "https://static.inaturalist.org/photos/{id}/{size}.jpg"
MO_PHOTO = "https://mushroomobserver.org/images/{size}/{id}.jpg"


# --- a fake iNat and MO ---------------------------------------------------------------

class Clock:
    """Time that only moves when the fetcher sleeps: pacing is checked, never waited."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Response:
    def __init__(self, status=200, json=None, content=b"", headers=None):
        self.status_code, self._json, self.content = status, json, content
        self.headers = headers or {}

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

    def close(self):
        pass


def inat_obs(oid, photos, location="40.12,-82.98", observed_on="2026-10-11", obscured=False):
    return {"id": oid, "observed_on": observed_on, "location": location, "obscured": obscured,
            "taxon": {"name": "A x", "rank": "species"},
            "observation_photos": [{"position": i, "photo": {
                "id": pid, "url": INAT_PHOTO.format(id=pid, size="square")}}
                for i, pid in enumerate(photos)]}


def mo_obs(oid, primary, others=(), lat=40.12, lng=-82.98, gps_hidden=False, box=None):
    def image(i):
        return {"id": i, "original_url": MO_PHOTO.format(id=i, size="orig") + "?1717"}
    return {"id": oid, "date": "2026-10-11", "latitude": lat, "longitude": lng,
            "gps_hidden": gps_hidden, "location": box or {},
            "consensus": {"name": "B y", "rank": "species"},
            "primary_image": image(primary), "images": [image(i) for i in others]}


class Site:
    """iNat's API and photo host and MO's, as a requests.Session. `photos` maps a photo
    id to its bytes; every request is logged with the (fake) time it was sent."""

    def __init__(self, clock):
        self.clock = clock
        self.headers = {}
        self.inat = {555: inat_obs(555, [9001, 9002]),
                     556: inat_obs(556, [1003]),             # a reference record's photo
                     557: inat_obs(557, []),
                     558: inat_obs(558, [9003])}
        self.mo = {777: mo_obs(777, 5001, [5002])}
        self.photos = {9001: png(247), 9002: png(245), 1003: png(12), 9003: b"not an image",
                       5001: png(12), 5002: png(15)}
        self.redirects: dict[str, str] = {}
        self.fail: dict[str, Response] = {}
        self.log: list[tuple[float, str]] = []

    def get(self, url, params=None, timeout=None, stream=False, allow_redirects=True):
        assert allow_redirects is False or params is not None   # photos: no auto-redirects
        assert timeout is not None
        self.log.append((self.clock(), url))
        if url in self.fail:
            return self.fail[url]
        if url == bo.INAT_API:
            obs = self.inat.get(int(params["id"]))
            return Response(json={"results": [obs] if obs else []})
        if url == bo.MO_API:
            obs = self.mo.get(int(params["id"]))
            return Response(json={"run_time": 0.2, "results": [obs] if obs else []})
        if url in self.redirects:
            return Response(302, headers={"Location": self.redirects[url]})
        for pattern in (INAT_PHOTO, MO_PHOTO):
            for pid, body in self.photos.items():
                if url in (pattern.format(id=pid, size="large"), pattern.format(id=pid, size="960")):
                    return Response(content=body)
        return Response(404)

    def urls(self):
        return [u for _, u in self.log]


def fetcher(clock=None, **kw):
    clock = clock or Clock()
    return bo.Fetcher(session=Site(clock), clock=clock, sleep=clock.sleep, **kw)


def app(conn, tmp_path, by_observation=True, quotas=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    f = fetcher()
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(), background=False, server_key=SERVER_KEY, api_v1=True,
        by_observation=by_observation, observation_fetcher=f,
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
        fungus_gate=None, quotas=quotas, note=lambda s: None))
    return client, f.session


def by_number(client, source, number, headers=SERVER, **body):
    return client.post(f"/api/v1/identify/{source}/{number}", json=body or None,
                       headers=headers)


# --- through the API --------------------------------------------------------------------

def test_an_inat_observation_gets_the_answer_its_photos_place_and_date_would_get(conn, tmp_path):
    client, site = app(conn, tmp_path)
    res = by_number(client, "inat", 555)
    assert res.status_code == 200, res.text
    body = res.json()
    same_answer(body, identify(client, SERVER, reds=(247, 245), latitude=40.12,
                               longitude=-82.98, observed_on="2026-10-11").json())
    assert body["observation"] == {
        "source": "inat", "id": "555", "url": "https://www.inaturalist.org/observations/555",
        "photos": 2, "photos_used": 2, "place": "coordinates",
        "name_at_source": "A x", "rank_at_source": "species",
        "photos_shown": [None, None]}
    assert body["context_used"] == {"latitude": 40.1, "longitude": -83.0,
                                    "observed_on": "2026-10-11", "place_from": "observation",
                                    "date_from": "observation"}
    assert body["excluded_reference"] is None and body["left_out_photos"] == 0
    t = body["timing"]
    assert t["fetch_ms"] is not None and t["photos_embedded"] == 2 and t["photos_reused"] == 0
    assert res.headers["cache-control"] == "no-store" and "fetch;dur=" in res.headers["server-timing"]
    # Large photos, from iNat's photo host, after one API read.
    assert site.urls()[1:] == [INAT_PHOTO.format(id=9001, size="large"),
                               INAT_PHOTO.format(id=9002, size="large")]


def test_use_place_and_use_date_false_leave_them_out(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    body = by_number(client, "inat", 555, use_place=False, use_date=False).json()
    assert body["context_used"]["place_from"] is None and body["context_used"]["date_from"] is None
    assert body["observation"]["place"] is None
    same_answer(body, identify(client, SERVER, reds=(247, 245)).json())


def test_a_reference_record_is_answered_without_itself_and_without_fetching(conn, tmp_path):
    client, site = app(conn, tmp_path)
    res = by_number(client, "inat", 100)            # seeded: "A x", photo 1000
    assert res.status_code == 200, res.text
    body = res.json()
    assert site.log == []                           # nothing read from iNat
    assert body["excluded_reference"] == {"source": "inat", "source_id": "100"}
    assert body["left_out_photos"] == 1
    assert (body["timing"]["photos_embedded"], body["timing"]["photos_reused"]) == (0, 1)
    assert body["timing"]["fetch_ms"] == 0
    # Its own record is no specimen and no place or date is used: it is part of the prior.
    assert "100" not in {s["observation_id"] for s in body["specimens"]}
    assert body["context_used"]["place_from"] is None and body["context_used"]["date_from"] is None
    assert body["ranks"]["species"][0]["name"] == "A x"     # the other A records still match
    assert body["observation"]["photos_used"] == 1


def test_a_photo_filed_under_a_reference_record_is_left_out_too(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    body = by_number(client, "inat", 556).json()    # its photo 1003 is record 103's
    assert body["excluded_reference"] is None and body["left_out_photos"] == 1
    assert "103" not in {s["observation_id"] for s in body["specimens"]}
    assert body["context_used"]["place_from"] is None


def test_an_mo_observation_reads_primary_image_first_at_mo_pace(conn, tmp_path):
    client, site = app(conn, tmp_path)
    body = by_number(client, "mo", 777).json()
    assert body["ranks"]["species"][0]["name"] == "B y"
    assert body["observation"]["url"] == "https://mushroomobserver.org/obs/777"
    assert body["observation"]["photos_used"] == 2
    times = [t for t, _ in site.log]
    assert site.urls()[1:] == [MO_PHOTO.format(id=5001, size="960"),
                               MO_PHOTO.format(id=5002, size="960")]
    assert all(b - a >= 5.0 for a, b in zip(times, times[1:]))   # 20 requests a minute


@pytest.mark.parametrize("source,number,status,code", [
    ("inat", 404040, 404, "observation_not_found"),
    ("mo", 404040, 404, "observation_not_found"),
    ("inat", 557, 422, "no_usable_photos"),
    ("inat", 558, 422, "no_usable_photos"),
    ("gbif", 1, 404, "not_found"),
    ("inat", "12a", 400, "bad_request"),
])
def test_what_cant_be_identified_is_refused_with_a_code(conn, tmp_path, source, number,
                                                       status, code):
    client, _ = app(conn, tmp_path)
    res = by_number(client, source, number)
    assert (res.status_code, res.json()["code"]) == (status, code)


@pytest.mark.parametrize("body", [{"top_k": 0}, {"top_k": True}, {"use_place": "yes"},
                                  {"model": 3}, {"client_ref": "x" * 65}])
def test_bad_options_are_400(conn, tmp_path, body):
    client, _ = app(conn, tmp_path)
    res = by_number(client, "inat", 555, **body)
    assert (res.status_code, res.json()["code"]) == (400, "bad_request")


def test_when_inat_fails_it_is_502_with_retry_after(conn, tmp_path):
    client, site = app(conn, tmp_path)
    site.fail[bo.INAT_API] = Response(429)
    res = by_number(client, "inat", 555)
    assert (res.status_code, res.json()["code"]) == (502, "source_unavailable")
    assert res.headers["retry-after"] == "60"


def test_it_needs_a_credential_and_is_off_unless_turned_on(conn, tmp_path):
    client, site = app(conn, tmp_path)
    assert by_number(client, "inat", 555, headers={}).status_code == 401
    off, site = app(conn, tmp_path / "off", by_observation=False)
    res = by_number(off, "inat", 555)
    assert (res.status_code, res.json()["code"]) == (404, "not_found")
    assert site.log == []


def test_a_caller_over_its_photo_window_costs_inat_no_photo_reads(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", "1")
    client, site = app(conn, tmp_path)
    res = by_number(client, "inat", 555, headers=bearer(app_token()))   # 2 photos
    assert (res.status_code, res.json()["code"]) == (429, "rate_limited")
    assert site.urls() == [bo.INAT_API]


def test_the_same_observation_within_a_day_counts_once(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", "20")
    client, _ = app(conn, tmp_path, quotas=q.Quotas(tmp_path / "usage.sqlite", mode="count"))
    member = bearer(app_token(sub="u_member0042"))
    used = [by_number(client, "inat", 555, headers=member).json()["usage"]["used"]
            for _ in range(2)]
    assert used == [1, 1]
    assert by_number(client, "inat", 100, headers=member).json()["usage"]["used"] == 2


def test_the_outline_marks_it_live_only_when_on(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    off, _ = app(conn, tmp_path / "off", by_observation=False)

    def status(c):
        parts = [p for s in c.get("/api/docs/v1.json").json()["sections"]
                 for p in s.get("blocks") or []
                 if isinstance(p, dict) and p.get("path") == "/identify/inat/{observation_id}"]
        return parts[0]["status"]
    assert (status(client), status(off)) == ("available", "planned")


# --- the fetcher ------------------------------------------------------------------------

def test_only_one_observation_is_read_at_a_time_and_a_long_queue_is_busy():
    f = fetcher(max_waiting=1)
    f.waiting = 2                       # one reading, one queued
    with pytest.raises(bo.Busy) as e:
        f.fetch("inat", "555", 10)
    assert e.value.retry_after > 0 and f.session.log == []


def test_a_refused_caller_reads_no_photos():
    f = fetcher()

    def refuse(obs):
        raise PermissionError("over the limit")
    with pytest.raises(PermissionError):
        f.fetch("inat", "555", 10, admit=refuse)
    assert f.session.urls() == [bo.INAT_API]


def test_inat_api_reads_are_a_second_apart():
    clock = Clock()
    f = fetcher(clock)
    for _ in range(3):
        f.fetch("inat", "557", 10, admit=lambda obs: 0)
    times = [t for t, u in f.session.log if u == bo.INAT_API]
    assert [round(b - a, 3) for a, b in zip(times, times[1:])] == [1.0, 1.0]


def test_an_mo_redirect_is_followed_only_to_mos_own_image_host():
    f = fetcher()
    site = f.session
    away = MO_PHOTO.format(id=5001, size="960")
    site.redirects[away] = "https://elsewhere.example/5001.jpg"
    site.redirects[MO_PHOTO.format(id=5002, size="960")] = MO_PHOTO.format(id=5001, size="960")
    obs, bodies = f.fetch("mo", "777", 10)
    assert "https://elsewhere.example/5001.jpg" not in site.urls()
    assert len(bodies) == 0             # 5001 redirects away; 5002 to 5001, which redirects
    site.redirects.clear()
    site.redirects[MO_PHOTO.format(id=5002, size="960")] = \
        "https://images.mushroomobserver.org/960/5002.jpg"
    site.photos[5002] = png(15)
    _, bodies = f.fetch("mo", "777", 10)
    assert "https://images.mushroomobserver.org/960/5002.jpg" in site.urls()


def test_a_photo_too_big_is_skipped(monkeypatch):
    monkeypatch.setattr(bo, "MAX_PHOTO_BYTES", 10)
    f = fetcher()
    _, bodies = f.fetch("inat", "555", 10)
    assert bodies == []


def test_a_network_failure_is_unavailable():
    f = fetcher()

    def down(*a, **kw):
        raise requests.ConnectionError("no route")
    f.session.get = down
    with pytest.raises(bo.Unavailable):
        f.fetch("inat", "555", 10)


def test_mo_place_is_gps_unless_hidden_then_a_small_named_place():
    small = {"latitude_north": "40.2", "latitude_south": "40.0", "longitude_east": "-82.9",
             "longitude_west": "-83.1"}
    wide = {"latitude_north": "42", "latitude_south": "38", "longitude_east": "-80",
            "longitude_west": "-85"}
    gps = bo.parse_mo(mo_obs(1, 10, box=small))
    assert (gps.latitude, gps.longitude, gps.place) == (40.12, -82.98, "coordinates")
    hidden = bo.parse_mo(mo_obs(1, 10, gps_hidden=True, box=small))
    assert (round(hidden.latitude, 2), round(hidden.longitude, 2), hidden.place) == (
        40.1, -83.0, "area")
    none = bo.parse_mo(mo_obs(1, 10, lat=None, lng=None, box=wide))
    assert (none.latitude, none.place) == (None, None)
    assert gps.photo_ids == [10 ** 12 + 10]          # as Vision's photos table numbers MO's


def test_inat_obscured_location_is_used_as_published_and_said_so():
    obs = bo.parse_inat(inat_obs(1, [5], location="40.3,-83.2", obscured=True))
    assert (obs.latitude, obs.longitude, obs.place) == (40.3, -83.2, "obscured")
    assert obs.photo_urls == [INAT_PHOTO.format(id=5, size="large")]
    assert bo.parse_inat(inat_obs(1, [5], location=None)).place is None


@pytest.mark.parametrize("raw,ok", [("123", "123"), ("0", None), ("-1", None), ("1e3", None),
                                    ("", None), ("1" * 13, None)])
def test_observation_numbers_are_digits(raw, ok):
    assert bo.valid_id(raw) == ok


# --- leaving a record out ---------------------------------------------------------------

def test_what_an_observation_leaves_out_is_its_record_and_every_copy_of_its_photos():
    from types import SimpleNamespace

    from mycomap_vision.identify import Identifier
    ident = Identifier.__new__(Identifier)
    # Columns: iNat 100 (photos 1, 2) | iNat 200 (photo 2 again, and 4) | MO 100 (photo 3)
    ident.col_record = SimpleNamespace(source=np.array(["inat", "inat", "mo"]),
                                       source_id=np.array(["100", "200", "100"]),
                                       rec=np.array([0, 0, 1, 1, 2]))
    ident.col_photo = np.array([1, 2, 2, 4, 3])
    hidden, own = ident.left_out("inat", "100")
    assert own.tolist() == [0, 1] and hidden.tolist() == [0, 1, 2]
    hidden, own = ident.left_out("mo", "100")              # never iNat's 100
    assert own.tolist() == [4] and hidden.tolist() == [4]
    hidden, own = ident.left_out("inat", "555", photo_ids=[4])
    assert own.tolist() == [] and hidden.tolist() == [3]
    hidden, own = ident.left_out("inat", "556")
    assert hidden.tolist() == [] and own.tolist() == []


def test_leaving_a_record_out_scores_as_if_it_was_never_in_the_reference(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)          # numpy, as the box serves it
    vecs, recs = reference()
    index = build_index(recs)
    model = NearestAndMean()
    model.fit(vecs, index)
    rng = np.random.default_rng(7)
    query = rng.normal(size=(2, vecs.shape[1])).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    sims = model.photo_sims(query)
    everything = model.per_photo_scores(sims, query)
    for rec in recs:
        hidden = np.flatnonzero(np.isin(index.cols, rec.photo_rows))
        got = model.leave_out(everything, sims, query, hidden)
        rest = [r for r in recs if r is not rec]
        index2 = build_index(rest)
        model2 = NearestAndMean()
        model2.fit(vecs, index2)
        want = dict(zip(index2.species, model2.per_photo_scores(model2.photo_sims(query),
                                                                 query).T))
        for u, name in enumerate(index.species):
            if name in want:
                assert got[:, u] == pytest.approx(want[name], abs=2e-3), (rec.observation_id, name)
            else:
                assert np.all(got[:, u] == -np.inf), (rec.observation_id, name)
