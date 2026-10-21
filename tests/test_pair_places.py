"""Where each copy of a record pair is, as MO and iNat list it: the place text, never a
coordinate; Vision's MO cache before MO's API; an hour in memory; a failure not kept."""

import requests

from mycomap_vision.pair_places import INAT_API, MO_API, PairPlaces


class Sites:
    def __init__(self, mo=None, inat=None, fail=False):
        self.mo, self.inat, self.fail, self.asked = mo, inat, fail, []

    def __call__(self, url, params):
        self.asked.append((url, params["id"]))
        if self.fail:
            raise requests.ConnectionError("down")
        if url == MO_API:
            return {"results": [{"id": 5, "latitude": 1.5, "longitude": 2.5,
                                 "location": {"name": self.mo} if self.mo else None}]}
        return {"results": [{"id": 7, "place_guess": self.inat, "location": "1.5,2.5"}]}


def places(sites, cached=None, clock=lambda: 0.0):
    return PairPlaces(cached_mo=cached, get=sites, clock=clock, mo_interval=0, inat_interval=0)


def test_each_copy_says_where_its_own_site_lists_it_and_never_a_coordinate():
    s = Sites(mo="Mt. Rainier NP, Washington, USA", inat="Tasmania, AU")
    got = places(s).of(["mo:5", "7"])
    assert got == {"mo:5": {"site": "MO", "text": "Mt. Rainier NP, Washington, USA", "status": "ok"},
                   "7": {"site": "iNat", "text": "Tasmania, AU", "status": "ok"}}
    assert "1.5" not in repr(got) and "2.5" not in repr(got)
    assert s.asked == [(MO_API, "5"), (INAT_API, "7")]


def test_vision_s_cache_of_mo_names_is_read_before_mo_is_asked():
    s = Sites(mo="from MO")
    p = places(s, cached=lambda key: "Cached place, Oregon, USA" if key == "mo:5" else None)
    assert p.one("mo:5")["text"] == "Cached place, Oregon, USA"
    assert p.one("mo:6")["text"] == "from MO"
    assert s.asked == [(MO_API, "6")]


def test_a_site_that_lists_no_place_says_none():
    assert places(Sites(mo=None)).one("mo:5") == {"site": "MO", "text": None, "status": "none"}
    assert places(Sites(inat="  ")).one("7") == {"site": "iNat", "text": None, "status": "none"}


def test_an_answer_is_kept_an_hour_and_a_failure_not_at_all():
    now = [0.0]
    s = Sites(inat="Here")
    p = places(s, clock=lambda: now[0])
    p.one("7"); p.one("7")
    assert len(s.asked) == 1
    now[0] = 3601
    p.one("7")
    assert len(s.asked) == 2
    down = Sites(fail=True)
    q = places(down)
    assert q.one("7") == {"site": "iNat", "text": None, "status": "unavailable"}
    q.one("7")
    assert len(down.asked) == 2                 # asked again: the failure was not kept
