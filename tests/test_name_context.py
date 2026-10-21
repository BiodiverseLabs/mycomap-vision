"""The names behind a Human Feedback name item (Steve, 2026-10-21): each name linked to its
iNat taxon, whether iNat still uses it and what it points to now, the same name written
another way on iNat, and MycoMap's synonym list around it."""

import requests

from mycomap_vision import name_context as nc

LIFE, FUNGI_ANC = 48460, [48460, 47170, 47169]


def taxon(id, name, active=True, current=None, obs=10, changes=1, rank="species",
          ancestors=FUNGI_ANC):
    return {"id": id, "name": name, "rank": rank, "is_active": active,
            "current_synonymous_taxon_ids": current, "observations_count": obs,
            "taxon_changes_count": changes, "ancestor_ids": ancestors}


# iNat as of 2026-10-21 for the names on the page.
LUTEO_INACTIVE = taxon(1287172, "Callistosporium luteoolivaceum", active=False, current=[],
                       obs=0, changes=0)
LUTEO_ACTIVE = taxon(118354, "Callistosporium luteo-olivaceum", obs=1993)
OLD = taxon(5, "Cortinarius oldus", active=False, current=[6])
NEW = taxon(6, "Cortinarius newus")
SPLIT = taxon(7, "Russula splitus", active=False, current=[8, 9])
PLANT = taxon(11, "Russula plantae", ancestors=[48460, 47126])


class Inat:
    def __init__(self, fail=False):
        self.asked, self.fail = [], fail
        self.by_q = {
            "Callistosporium luteoolivaceum": [LUTEO_INACTIVE],
            "Callistosporium luteo": [LUTEO_ACTIVE, LUTEO_INACTIVE],
            "Cortinarius oldus": [OLD],
            "Cortinarius newus": [NEW],
            "Russula splitus": [SPLIT],
            "Russula plantae": [PLANT],
            "Serpula lacrymans shastensis": [taxon(143863, "Serpula lacrymans shastensis",
                                                   active=False, current=[], rank="variety")],
        }
        self.by_id = {6: NEW, 8: taxon(8, "Russula a"), 9: taxon(9, "Russula b")}

    def __call__(self, url, params):
        self.asked.append((url, params.get("q") or url.rsplit("/", 1)[-1]))
        if self.fail:
            raise requests.ConnectionError("down")
        if url == nc.AUTOCOMPLETE:
            got = self.by_q.get(params["q"], [])
            if params.get("is_active") == "true":
                got = [t for t in got if t["is_active"]]
            return {"results": got}
        ids = [int(i) for i in url.rsplit("/", 1)[-1].split(",")]
        return {"results": [self.by_id[i] for i in ids]}


def names(inat=None, clock=lambda: 0.0):
    return nc.InatNames(get=inat or Inat(), clock=clock, interval=0)


def test_an_inactive_name_with_no_current_name_says_so_and_finds_inats_other_writing():
    """Callistosporium luteoolivaceum: iNat's taxon of that spelling is inactive, points
    nowhere and was never used; iNat's active one is spelled luteo-olivaceum."""
    got = names().one("Callistosporium luteoolivaceum")
    assert got["status"] == "inactive" and got["current"] == []
    assert got["taxon"]["url"] == "https://www.inaturalist.org/taxa/1287172"
    assert [t["name"] for t in got["near"]] == ["Callistosporium luteo-olivaceum"]
    assert got["near"][0]["url"] == "https://www.inaturalist.org/taxa/118354"
    assert "no current name" in nc.describe(got) and "never used" in nc.describe(got)


def test_an_inactive_name_shows_the_current_name_it_points_to_linked():
    got = names().one("Cortinarius oldus")
    assert got["status"] == "inactive"
    assert got["current"] == [nc.taxon_view(NEW)]
    assert got["near"] == []                                   # a current name: no guessing
    assert nc.describe(got) == "Inactive on iNaturalist; its current name is Cortinarius newus."


def test_a_current_name_that_is_itself_inactive_says_so():
    """Boletus pallidoroseus -> Lanmaoa pallidorosea, which iNat later made inactive too."""
    i = Inat()
    i.by_q["Boletus pallidoroseus"] = [taxon(500028, "Boletus pallidoroseus", active=False,
                                             current=[1068083])]
    i.by_id[1068083] = taxon(1068083, "Lanmaoa pallidorosea", active=False, current=[1687451])
    got = names(i).one("Boletus pallidoroseus")
    assert nc.describe(got) == ("Inactive on iNaturalist; its current name is Lanmaoa "
                                "pallidorosea (itself inactive on iNat).")


def test_a_split_lists_every_current_name():
    got = names().one("Russula splitus")
    assert [t["name"] for t in got["current"]] == ["Russula a", "Russula b"]
    assert "split into Russula a, Russula b" in nc.describe(got)


def test_an_infraspecific_name_is_asked_as_inat_writes_it():
    i = Inat()
    got = names(i).one("Serpula lacrymans var. shastensis")
    assert got["asked"] == "Serpula lacrymans shastensis" and got["status"] == "inactive"
    assert i.asked[0] == (nc.AUTOCOMPLETE, "Serpula lacrymans shastensis")


def test_a_taxon_outside_fungi_is_not_the_name():
    got = names().one("Russula plantae")
    assert got["status"] == "not_found" and got["taxon"] is None
    assert got["search_url"] == "https://www.inaturalist.org/taxa/search?q=Russula%20plantae"


def test_an_answer_is_kept_a_day_and_a_failure_not_at_all():
    now = [0.0]
    i = Inat()
    n = names(i, clock=lambda: now[0])
    n.one("Cortinarius newus"); n.one("Cortinarius newus")
    assert len(i.asked) == 1
    now[0] = nc.TTL + 1
    n.one("Cortinarius newus")
    assert len(i.asked) == 2
    down = names(Inat(fail=True))
    assert down.one("Cortinarius newus")["status"] == "unavailable"
    assert down.cached("Cortinarius newus") is None


def test_two_writings_of_one_name_stay_two_names():
    item = {"current": "Callistosporium luteoolivaceum", "proposed": ""}
    rows = [{"retired": "Callistosporium luteo-olivaceum",
             "accepted": "Callistosporium luteoolivaceum"}]
    got = nc.context(item, names(), lambda n: {"name": n, "rows": rows})
    shown = {e["name"]: e for e in got["names"]}
    assert set(shown) == {"Callistosporium luteoolivaceum", "Callistosporium luteo-olivaceum"}
    assert shown["Callistosporium luteoolivaceum"]["roles"] == ["records", "synonym list"]
    assert shown["Callistosporium luteo-olivaceum"]["roles"] == ["synonym list"]
    assert got["synonyms"] == {"status": "ok", "rows": rows, "more": 0}


def test_the_items_names_are_always_asked_synonyms_only_while_the_budget_lasts():
    i = Inat()
    item = {"current": "Cortinarius oldus", "proposed": "Cortinarius newus"}
    rows = [{"retired": f"Cortinarius s{k}", "accepted": "Cortinarius newus"} for k in range(5)]
    got = nc.context(item, names(i), lambda n: {"rows": rows}, budget=2)
    status = {e["name"]: e["inat"]["status"] for e in got["names"]}
    assert status["Cortinarius oldus"] == "inactive" and status["Cortinarius newus"] == "active"
    asked_syn = [n for n, s in status.items() if n.startswith("Cortinarius s") and s != "not_asked"]
    assert len(asked_syn) == 2
    unasked = next(e for e in got["names"] if e["inat"]["status"] == "not_asked")
    assert unasked["inat"]["search_url"].startswith("https://www.inaturalist.org/taxa/search?q=")
    assert unasked["about"] == "Not looked up yet."


def test_synonyms_unavailable_still_shows_the_items_names():
    def down(name):
        raise requests.ConnectionError("org down")
    got = nc.context({"current": "Cortinarius oldus", "proposed": "Cortinarius newus"},
                     names(), down)
    assert got["synonyms"]["status"] == "unavailable"
    assert [e["name"] for e in got["names"]] == ["Cortinarius oldus", "Cortinarius newus"]
    none_yet = nc.context({"current": "Cortinarius oldus"}, names(), lambda n: None)
    assert none_yet["synonyms"]["status"] == "unavailable"


def test_a_split_proposal_lists_each_proposed_name():
    assert nc.item_names({"current": "A x", "proposed": "B y; C z"}) == [
        ("A x", "records"), ("B y", "proposed"), ("C z", "proposed")]
