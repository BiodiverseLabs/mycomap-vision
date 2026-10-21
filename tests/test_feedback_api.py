"""The Human Feedback page's API over the real app: who gets in (mycomap.org admins every
item, project leaders and contributors their projects' items, everyone else nothing), the live
record from mycomap.org with each sequence's BLAST link chosen by .org's staff rule (staff:
the admin BLAST results form; others: the public job page; opened in a new window by the page),
decisions, refresh (re-read from .org, Vision re-asked with the record left out), and the
change list for the admins who apply it."""

from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET, good_claims, mint

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_api
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig

ORG = "https://www.mycomap.org"


class FakeOrg:
    """mycomap.org's two Human Feedback endpoints, as feedback_api.OrgClient calls them."""

    def __init__(self):
        self.roles_by_user = {
            "admin-1": {"admin": True, "staff": True, "projects": []},
            "leader-1": {"admin": False, "staff": True, "projects": [{"id": "1", "name": "p1", "role": "leader"}]},
            "member-1": {"admin": False, "staff": False,
                         "projects": [{"id": "2", "name": "p2", "role": "contributor"}]},
            "nobody-1": {"admin": False, "staff": False, "projects": []},
        }
        self.names = {"100": "A x", "104": "B y"}
        self.calls = []

    def roles(self, user_id):
        self.calls.append(("roles", user_id))
        return self.roles_by_user.get(user_id)

    def record(self, key):
        self.calls.append(("record", key))
        if key not in self.names:
            return None
        return {"observation": f"inat:{key}", "name": self.names[key], "source": "iNaturalist",
                "projects": [{"id": "p1", "status": "yes", "name": self.names[key]}],
                "links": {"validate": [f"{ORG}/admin/validate-records/p1?record={key}"],
                          "name_update": f"{ORG}/admin/name-update"},
                "sequences": [
                    {"id": 77, "name": self.names[key], "trash": False, "url": f"{ORG}/sequences/77",
                     "blast": {"job_id": 9, "url": f"{ORG}/mycoblast/9",
                               "admin_results_url": f"{ORG}/admin/blast-results/{key}/77",
                               "results_url": f"{ORG}/blast-results/{key}/77", "saved_at": "x"}},
                    {"id": 78, "name": "A x", "trash": True, "url": f"{ORG}/sequences/78",
                     "blast": {"job_id": None, "url": None,
                               "admin_results_url": f"{ORG}/admin/blast-results/{key}/78",
                               "results_url": f"{ORG}/blast-results/{key}/78", "saved_at": None}}]}


def make_site(conn, tmp_path, places=None, inat_names=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1", log=lambda s: None)
    conn.commit()
    fstore = fb.Store(tmp_path / "state" / "human-feedback.sqlite")
    fstore.seed([
        {"item_type": "photos_unlike_label", "subject": "100", "rule": "M1",
         "record_keys": ["100"], "current": "A x", "projects": ["p1"]},
        {"item_type": "group_label", "subject": "Laccaria laccata group", "rule": "R2",
         "records": 3, "projects": ["p2"]},
    ], "test")
    org = FakeOrg()
    signin = SigninConfig(mode="identify", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), signin=signin, feedback_store=fstore,
                     feedback_org=org, feedback_places=places, feedback_inat_names=inat_names)
    return TestClient(app, follow_redirects=False), fstore, org


@pytest.fixture
def site(conn, tmp_path):
    return make_site(conn, tmp_path)


def as_user(client, sub):
    client.cookies.clear()
    start = client.get("/auth/signin", params={"returnTo": "/feedback"})
    nonce = parse_qs(urlsplit(start.headers["location"]).query)["nonce"][0]
    client.get("/auth/dev-bridge/callback",
               params={"token": mint({**good_claims(nonce=nonce), "sub": sub})})
    return client


def item_id(fstore, item_type):
    return fstore.items(item_type)["items"][0]["id"]


def test_signed_out_and_people_without_a_role_get_nothing(site):
    c, _, _ = site
    assert c.get("/api/feedback/summary").status_code == 401
    as_user(c, "nobody-1")
    r = c.get("/api/feedback/summary")
    assert r.status_code == 403 and "admins, project leaders and contributors" in r.json()["detail"]


def test_admins_see_every_item_and_others_only_their_projects(site):
    c, fstore, _ = site
    as_user(c, "admin-1")
    me = c.get("/api/feedback/me").json()
    assert me["admin"] is True and me["staff"] is True
    by_type = {s["type"]: s["open"] for s in c.get("/api/feedback/summary").json()}
    assert by_type["photos_unlike_label"] == 1 and by_type["group_label"] == 1
    as_user(c, "member-1")
    by_type = {s["type"]: s["open"] for s in c.get("/api/feedback/summary").json()}
    assert by_type["photos_unlike_label"] == 0 and by_type["group_label"] == 1
    assert c.get(f"/api/feedback/items/{item_id(fstore, 'photos_unlike_label')}").status_code == 404


def test_an_allowlisted_admin_gets_in_before_org_answers_roles(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_FEEDBACK_ADMINS", "ghost-9, other-3")
    c, _, org = make_site(conn, tmp_path)
    org.roles_by_user.clear()                     # .org does not answer roles yet
    as_user(c, "ghost-9")
    me = c.get("/api/feedback/me").json()
    assert me["admin"] is True and me["staff"] is True
    as_user(c, "stranger-5")
    assert c.get("/api/feedback/me").status_code == 403


def test_the_blast_link_follows_orgs_staff_rule_admin_form_or_public_job(site):
    c, fstore, _ = site
    iid = item_id(fstore, "photos_unlike_label")
    as_user(c, "leader-1")                                     # staff on .org
    seqs = c.get(f"/api/feedback/items/{iid}").json()["live"][0]["sequences"]
    assert seqs[0]["blast_link"] == {"href": f"{ORG}/admin/blast-results/100/77", "kind": "admin"}
    assert seqs[1]["blast_link"] == {"href": f"{ORG}/admin/blast-results/100/78", "kind": "admin"}
    fstore.conn.execute("update items set projects = '[\"p1\",\"p2\"]'")
    fstore.conn.commit()
    as_user(c, "member-1")                                     # not staff
    seqs = c.get(f"/api/feedback/items/{iid}").json()["live"][0]["sequences"]
    assert seqs[0]["blast_link"] == {"href": f"{ORG}/mycoblast/9", "kind": "public"}
    assert seqs[1]["blast_link"] is None                       # no job, no admin form
    assert "admin" not in str([s["blast_link"] for s in seqs])


def test_blast_link_rule_directly():
    seq = {"blast": {"url": "u", "admin_results_url": "a"}}
    assert feedback_api.blast_link(seq, True) == {"href": "a", "kind": "admin"}
    assert feedback_api.blast_link(seq, False) == {"href": "u", "kind": "public"}
    assert feedback_api.blast_link({"blast": None}, True) is None


def test_a_decision_is_stored_with_who_made_it_and_bad_ones_are_refused(site):
    c, fstore, _ = site
    iid = item_id(fstore, "photos_unlike_label")
    as_user(c, "admin-1")
    r = c.post(f"/api/feedback/items/{iid}/decision", json={"decision": "rename"})
    assert r.status_code == 400 and "needs a value" in r.json()["detail"]
    r = c.post(f"/api/feedback/items/{iid}/decision",
               json={"decision": "exclude_photos", "photo_keys": ["inat:1000"], "note": "slip"})
    assert r.status_code == 200 and r.json()["status"] == "decided"
    assert r.json()["decisions"][0]["by_user"] == "admin-1"
    assert fstore.excluded_photos() == {"inat:1000"}
    assert c.post(f"/api/feedback/items/{iid}/reopen").json()["status"] == "open"


def test_refresh_rereads_org_and_reasks_vision_with_the_record_left_out(site):
    c, fstore, org = site
    iid = item_id(fstore, "photos_unlike_label")
    as_user(c, "admin-1")                      # a label check: decided first, then refreshed
    c.post(f"/api/feedback/items/{iid}/decision", json={"decision": "rename", "value": "A x"})
    out = c.post(f"/api/feedback/items/{iid}/refresh").json()
    r = out["refreshed"][0]
    assert r["name_now"] == "A x" and ("record", "100") in org.calls
    assert r["vision"]["available"] is True and r["vision"]["top5"][0][0] == "A x"
    assert r["still_disagrees"] is False
    org.names["100"] = "B y"                                   # renamed on .org meanwhile
    r = c.post(f"/api/feedback/items/{iid}/refresh").json()["refreshed"][0]
    assert r["name_now"] == "B y" and r["still_disagrees"] is True
    assert len(out["item"]["refreshes"]) == 1 and len(fstore.get(iid)["refreshes"]) == 2


def test_the_change_list_is_for_admins_and_marks_applied(site):
    c, fstore, _ = site
    gid = item_id(fstore, "group_label")
    as_user(c, "member-1")
    c.post(f"/api/feedback/items/{gid}/decision",
           json={"decision": "rename", "value": "Laccaria sp. 'laccata-PNW01'"})
    assert c.get("/api/feedback/changes.tsv").status_code == 403
    as_user(c, "admin-1")
    tsv = c.get("/api/feedback/changes.tsv").text.splitlines()
    assert tsv[0].startswith("decision_id\titem_type") and len(tsv) == 2
    assert "Laccaria sp. 'laccata-PNW01'" in tsv[1] and "\torg\t" in tsv[1]
    did = int(tsv[1].split("\t")[0])
    assert c.post(f"/api/feedback/decisions/{did}/applied").status_code == 200
    assert len(c.get("/api/feedback/changes.tsv").text.splitlines()) == 1


def test_blind_items_look_like_the_others_and_label_checks_hide_visions_answer(site):
    """Finds Vision got right are mixed blind into the label-check queues: nothing in the
    JSON tells them apart, and Vision's answer, the rule and the outcome evidence stay
    hidden until the reviewer decides (refresh, which asks Vision, waits too)."""
    c, fstore, _ = site
    fstore.seed([{"item_type": "photos_unlike_label", "subject": "104", "rule": "blind sample "
                  "of finds Vision answered right (seed 3)", "blind": True, "model_flagged": True,
                  "record_keys": ["104"], "current": "B y", "projects": ["p1"]}], "blind-x")
    fstore.conn.execute("update items set proposed = 'Xylodon ovisporus', evidence = "
                        "'{\"v1_val_outcome\": \"miss\", \"evidence\": \"Vision: Xylodon\"}' "
                        "where subject = '100'")
    fstore.conn.commit()
    as_user(c, "admin-1")
    items = c.get("/api/feedback/items", params={"type": "photos_unlike_label"}).json()["items"]
    assert len(items) == 2
    for i in items:
        assert not {"blind", "seed", "hidden_until_decided"} & set(i)
        assert i["masked"] is True and i["rule"] == "label check" and i["proposed"] == ""
        assert "v1_val_outcome" not in i["evidence"] and "evidence" not in i["evidence"]
    blind_id = next(i["id"] for i in items if i["subject"] == "104")
    assert set(items[0]) == set(items[1])                       # the same fields, both ways
    assert c.post(f"/api/feedback/items/{blind_id}/refresh").status_code == 409
    done = c.post(f"/api/feedback/items/{blind_id}/decision",
                  json={"decision": "hard_example"}).json()
    assert done["masked"] is False and "blind" not in done
    assert "check" not in done["rule"]                          # revealed after deciding


def test_a_record_pair_shows_both_copies_photos_and_other_items_none(conn, tmp_path, monkeypatch):
    """One find, two places (Steve, 2026-10-21): the item carries each copy's photos, so a
    reviewer sees at once whether the two records are one find. A copy Vision holds no photos
    of gets an empty list; a single-record item gets no photos here."""
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")   # mo:999 has no date; the seal: test_feedback_seal
    c, fstore, _ = make_site(conn, tmp_path)
    fstore.seed([{"item_type": "twin_location_conflict", "subject": "mo:999 + 101",
                  "granularity": "record pair", "rule": "twins", "record_keys": ["mo:999", "101"],
                  "records": 2, "projects": ["p1"]}], "twins")
    as_user(c, "leader-1")
    got = c.get(f"/api/feedback/items/{item_id(fstore, 'twin_location_conflict')}").json()
    assert got["photos"]["mo:999"] == []
    [ph] = got["photos"]["101"]
    assert ph["photo_key"] == "inat:1001" and ph["shown"] == "public"
    assert ph["thumb"].startswith("https://")
    single = c.get(f"/api/feedback/items/{item_id(fstore, 'photos_unlike_label')}").json()
    assert "photos" not in single


def test_a_record_pair_lists_each_copys_place_as_its_site_does_and_no_coordinate(conn, tmp_path,
                                                                                monkeypatch):
    """Beside the photos, where MO and iNat each say the find is (Steve, 2026-10-21): the
    place text only. Asked apart from the item; a single-record item gets none; a reviewer
    who cannot open the item gets nothing."""
    from mycomap_vision.pair_places import PairPlaces
    asked = []

    def get(url, params):
        asked.append(params["id"])
        if "mushroomobserver" in url:
            return {"results": [{"id": 999, "latitude": 47.6, "longitude": -122.3,
                                 "location": {"name": "Seattle, King Co., Washington, USA",
                                              "latitude_north": 47.7}}]}
        return {"results": [{"id": 101, "place_guess": "Hobart, Tasmania, Australia",
                             "location": "-42.9,147.3"}]}

    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    c, fstore, _ = make_site(conn, tmp_path, places=PairPlaces(get=get, mo_interval=0, inat_interval=0))
    fstore.seed([{"item_type": "twin_location_conflict", "subject": "mo:999 + 101",
                  "granularity": "record pair", "rule": "twins", "record_keys": ["mo:999", "101"],
                  "records": 2, "projects": ["p1"]}], "twins")
    pair = item_id(fstore, "twin_location_conflict")
    assert c.get(f"/api/feedback/items/{pair}/places").status_code == 401
    as_user(c, "member-1")                                     # p2 only
    assert c.get(f"/api/feedback/items/{pair}/places").status_code == 404
    assert asked == []
    as_user(c, "leader-1")
    got = c.get(f"/api/feedback/items/{pair}/places").json()["places"]
    assert got == {"mo:999": {"site": "MO", "text": "Seattle, King Co., Washington, USA", "status": "ok"},
                   "101": {"site": "iNat", "text": "Hobart, Tasmania, Australia", "status": "ok"}}
    single = item_id(fstore, "photos_unlike_label")
    assert c.get(f"/api/feedback/items/{single}/places").json() == {"places": {}}
    assert asked == ["999", "101"]


def test_a_name_item_lists_its_names_with_inats_taxa_and_mycomaps_synonyms(conn, tmp_path,
                                                                           monkeypatch):
    """Names iNaturalist no longer uses (Steve, 2026-10-21): the item's names, each linked to
    its iNat taxon, and MycoMap's synonym list around them. Asked apart from the item; other
    item types get none; a reviewer who cannot open the item gets nothing."""
    from mycomap_vision.name_context import AUTOCOMPLETE, InatNames
    asked = []

    def get(url, params):
        asked.append(params.get("q"))
        if url == AUTOCOMPLETE and params["q"] == "Callistosporium luteoolivaceum":
            return {"results": [{"id": 1287172, "name": "Callistosporium luteoolivaceum",
                                 "rank": "species", "is_active": False,
                                 "current_synonymous_taxon_ids": [], "observations_count": 0,
                                 "taxon_changes_count": 0, "ancestor_ids": [48460, 47170]}]}
        if url == AUTOCOMPLETE and params["q"] == "Callistosporium luteo":
            return {"results": [{"id": 118354, "name": "Callistosporium luteo-olivaceum",
                                 "rank": "species", "is_active": True,
                                 "ancestor_ids": [48460, 47170]}]}
        return {"results": []}

    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    c, fstore, org = make_site(conn, tmp_path, inat_names=InatNames(get=get, interval=0))
    org.synonyms = lambda name: {"name": name, "rows": [
        {"retired": "Callistosporium luteo-olivaceum", "accepted": "Callistosporium luteoolivaceum"}]}
    fstore.seed([{"item_type": "name_inactive_on_inat", "subject": "Callistosporium luteoolivaceum",
                  "granularity": "name", "rule": "R4", "current": "Callistosporium luteoolivaceum",
                  "proposed": "", "records": 110, "projects": ["p1"]}], "r4")
    nid = item_id(fstore, "name_inactive_on_inat")
    assert c.get(f"/api/feedback/items/{nid}/names").status_code == 401
    as_user(c, "member-1")                                     # p2 only
    assert c.get(f"/api/feedback/items/{nid}/names").status_code == 404
    assert asked == []
    as_user(c, "leader-1")
    got = c.get(f"/api/feedback/items/{nid}/names").json()
    by_name = {e["name"]: e for e in got["names"]}
    on_records = by_name["Callistosporium luteoolivaceum"]
    assert on_records["roles"] == ["records", "synonym list"]
    assert on_records["inat"]["taxon"]["url"] == "https://www.inaturalist.org/taxa/1287172"
    assert on_records["inat"]["near"][0]["url"] == "https://www.inaturalist.org/taxa/118354"
    assert "no current name" in on_records["about"]
    assert got["synonyms"]["rows"] == [{"retired": "Callistosporium luteo-olivaceum",
                                        "accepted": "Callistosporium luteoolivaceum"}]
    other = item_id(fstore, "group_label")
    as_user(c, "admin-1")
    assert c.get(f"/api/feedback/items/{other}/names").json()["names"] == []
