"""Records validated on mycomap.org since the last nightly read and not held yet: the How
it works page's "waiting for tonight's update" (nightly.fetch_green_since, nightly.waiting,
/api/stats "waiting").

Steve, 2026-11-06: the page said +154 for the week while he had validated far more. The
weekly figure counts only what the set holds; whatever was validated after the last read
was invisible until the next night."""

import pytest
from fastapi.testclient import TestClient

from test_api import open_limits
from test_models_and_scoreboard import Const
from test_nightly import QUIET, RID, FakeOutside, box, night, no_retry_pauses, row, seed_rows  # noqa: F401

from mycomap_vision import manifest, nightly
from mycomap_vision.api import create_app

READ_AT = "2026-10-08T07:00:00.000Z"       # test_nightly.night's mycomap.org clock


def since_answer(rows, since=READ_AT, without_record=0, truncated=False):
    return {"generatedAt": "2026-10-08T20:00:00.000Z", "since": since, "count": len(rows),
            "truncated": truncated, "greenWithoutRecord": without_record, "records": rows}


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Session:
    def __init__(self, resp):
        self.resp, self.asked = resp, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.asked.append((url, params, headers))
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


# --- reading mycomap.org ---------------------------------------------------------------

def test_the_records_since_the_last_read_are_asked_for_with_the_key_and_the_moment():
    s = Session(Resp(200, since_answer([row(300, "C z")])))
    out = nightly.fetch_green_since("https://org.example/", "k" * 40, READ_AT, session=s)
    assert out["count"] == 1
    url, params, headers = s.asked[0]
    assert url == "https://org.example/api/vision/green-records/since"
    assert params == {"since": READ_AT}
    assert headers["Authorization"] == "Bearer " + "k" * 40


@pytest.mark.parametrize("resp", [
    Resp(401, {}), Resp(503, {"code": "not_configured"}), Resp(500, {}), Resp(200),
    Resp(200, {"records": "x", "count": 0, "generatedAt": "t"}),
    Resp(200, {"records": [row(1, "A x")], "count": 2, "generatedAt": "t"}),
    Resp(200, {"records": [row(1, "A x")], "count": 1}),
    Resp(200, {"records": [{"scientific_name": "A x"}], "count": 1, "generatedAt": "t"}),
])
def test_anything_but_a_well_formed_answer_is_refused(resp):
    with pytest.raises(nightly.NightlyRefused):
        nightly.fetch_green_since("https://org.example", "k" * 40, READ_AT, session=Session(resp))


def test_a_network_failure_is_refused_without_the_key():
    import requests
    with pytest.raises(nightly.NightlyRefused) as e:
        nightly.fetch_green_since("https://org.example", "secret-key-" + "k" * 30, READ_AT,
                                  session=Session(requests.ConnectionError("down")))
    assert "secret-key" not in str(e.value)


# --- counting --------------------------------------------------------------------------

def test_the_last_read_is_the_last_good_nights_clock(box):
    layer = nightly.prepare(box, **QUIET)
    conn = manifest.connect(layer.manifest)
    try:
        assert nightly.last_read_at(conn) is None
    finally:
        conn.close()
    assert night(layer, seed_rows())["ok"]
    conn = manifest.connect(layer.manifest)
    try:
        assert nightly.last_read_at(conn) == READ_AT
    finally:
        conn.close()


def test_only_records_not_held_yet_count_as_the_weekly_figure_counts_them(box):
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows())
    held = seed_rows()[0]                                    # re-validated: already here
    rows = [dict(held, scientific_name="A renamed"),
            row(300, "C z"),                                  # new, North America: counts
            row(301, "C z"),                                  # new: counts
            dict(row(302, "C z"), continent="Europe", country="DK"),     # outside: not counted
            row(303, "D w"), row(303, "E v"),                 # two names: a label conflict
            ]
    conn = manifest.connect(layer.manifest)
    try:
        out = nightly.waiting(conn, since_answer(rows, without_record=7))
    finally:
        conn.close()
    assert out == {"records": 2, "since": READ_AT, "as_of": "2026-10-08T20:00:00.000Z",
                   "truncated": False, "green_without_record": 7}


def test_records_held_under_another_source_are_still_new(box):
    # An MO and an iNat observation can share a number (sources.py): the key decides.
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows())
    held = seed_rows()[0]
    conn = manifest.connect(layer.manifest)
    try:
        out = nightly.waiting(conn, since_answer([dict(held, source="MO Observations")]))
    finally:
        conn.close()
    assert out["records"] == 1


# --- the page's figure -----------------------------------------------------------------

def site(box, layer, fetch, monkeypatch, org=True, notes=None):
    if org:
        monkeypatch.setenv("MV_ORG_BASE_URL", "https://org.example")
        monkeypatch.setenv("MV_ORG_VISION_KEY", "k" * 40)
    else:
        monkeypatch.delenv("MV_ORG_BASE_URL", raising=False)
        monkeypatch.delenv("MV_ORG_VISION_KEY", raising=False)
        monkeypatch.delenv("MV_ORG_VISION_KEY_FILE", raising=False)
    return TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        layer=layer, nightly_fetch=lambda: (seed_rows(), READ_AT),
        nightly_steps=FakeOutside({}).steps(), waiting_fetch=fetch,
        note=(notes.append if notes is not None else (lambda s: None))))


def test_the_stats_say_how_many_records_wait_for_tonight(box, monkeypatch):
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows())
    asked = []

    def fetch(since):
        asked.append(since)
        return since_answer([row(300, "C z"), row(301, "C z"), seed_rows()[0]], without_record=3)

    notes = []
    stats = site(box, layer, fetch, monkeypatch, notes=notes).get("/api/stats").json()
    assert asked == [READ_AT]
    assert stats["waiting"] == {"records": 2, "since": READ_AT, "as_of": "2026-10-08T20:00:00.000Z"}
    assert "green_without_record" not in stats["waiting"]       # the log's, not the page's
    assert any("3 records turned green" in n for n in notes)


def test_no_figure_before_the_first_good_night(box, monkeypatch):
    layer = nightly.prepare(box, **QUIET)
    asked = []
    stats = site(box, layer, lambda s: asked.append(s), monkeypatch).get("/api/stats").json()
    assert stats["waiting"] is None and asked == []


def test_no_figure_and_no_call_without_mycomap_orgs_address_and_key(box, monkeypatch):
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows())
    asked = []
    stats = site(box, layer, lambda s: asked.append(s), monkeypatch, org=False).get("/api/stats").json()
    assert stats["waiting"] is None and asked == []


def test_a_failed_read_gives_no_figure_and_the_page_still_answers(box, monkeypatch):
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows())

    def fetch(since):
        raise nightly.NightlyRefused("mycomap.org answered 500; nothing was changed")

    notes = []
    res = site(box, layer, fetch, monkeypatch, notes=notes).get("/api/stats")
    assert res.status_code == 200 and res.json()["waiting"] is None
    assert any(n.startswith("[waiting] not counted") for n in notes)


def test_without_the_nightly_update_there_is_no_figure(box, monkeypatch):
    monkeypatch.setenv("MV_ORG_BASE_URL", "https://org.example")
    monkeypatch.setenv("MV_ORG_VISION_KEY", "k" * 40)
    client = TestClient(create_app(box / "releases" / RID / "manifest.sqlite",
                                   box / "releases" / RID / "embeddings",
                                   backbone_loader=lambda name: Const(name), limits=open_limits(),
                                   background=False, note=lambda s: None,
                                   waiting_fetch=lambda s: pytest.fail("asked .org")))
    assert client.get("/api/stats").json()["waiting"] is None
