"""Your collections on the Identify result (Steve, 2026-10-18; yours.py): a signed-in
person's own DNA-verified records among the references, shown to that person only, and the
closest-specimens list without them so nothing shows twice.

The seeded reference: "A x" records 100, 101, 102, 106 (red photos) and "B y" records 103,
104, 105, 107 (dark photos). Red photos (247, 245) are a find most like "A x". Anna owns 104
and 106 on iNat; Ben owns 100; everyone else is another iNat user.
"""

import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png, post_photos
from test_api_v1_by_observation import fetcher, inat_obs
from test_api_v1_identify import FakeOrg, SERVER_KEY
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET, good_claims, mint
from urllib.parse import parse_qs, urlsplit

from mycomap_vision import answercache, apiv1, yours
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig

ANNA, BEN, CARA, DORA = "user-anna", "user-ben", "user-cara", "user-dora"
ANNA_INAT, BEN_INAT, OTHERS_INAT = 11, 22, 99
OWNERS = {"100": BEN_INAT, "104": ANNA_INAT, "106": ANNA_INAT}
ACCOUNTS = {
    ANNA: yours.Linked(inat_ids=frozenset({ANNA_INAT})),
    BEN: yours.Linked(inat_ids=frozenset({BEN_INAT})),
    CARA: yours.Linked(unverified_inat=True),                  # nothing verified
    DORA: yours.Linked(inat_ids=frozenset({123456})),          # verified, no records here
}


class FakeOrgAccounts:
    """mycomap.org's GET /api/vision/linked-accounts, counted per user."""

    def __init__(self, accounts=ACCOUNTS):
        self.accounts, self.calls, self.fail, self.fail_for = dict(accounts), [], False, set()

    def __call__(self, sub):
        self.calls.append(sub)
        if self.fail or sub in self.fail_for:
            raise yours.LinkedError("HTTP 503")
        return self.accounts.get(sub, yours.Linked())


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def seed(conn, tmp_path):
    """The reference above, embedded once; any number of apps may serve it."""
    if (tmp_path / "emb" / "m1").exists():
        return
    store = seed_two_species(conn, tmp_path)
    conn.executemany("update inat_observations set user_id = ? where observation_id = ?",
                     [(OWNERS.get(str(o), OTHERS_INAT), str(o)) for o in range(100, 108)])
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()


def build(conn, tmp_path, mode="identify", org=None, clock=None, cache=None):
    seed(conn, tmp_path)
    root = tmp_path / "emb"
    org = org or FakeOrgAccounts()
    cache = cache or answercache.AnswerCache(None)
    f = fetcher()
    signin = SigninConfig(mode=mode, origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    app = create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(), background=False, signin=signin, server_key=SERVER_KEY,
        api_v1=True, by_observation=True, observation_fetcher=f,
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(), fungus_gate=None,
        note=lambda s: None, answer_cache=cache,
        linked_accounts=yours.LinkedCache(org, clock=clock or Clock()))
    client = TestClient(app, follow_redirects=False)
    client.site = f.session                      # the fake iNat and MO (test_api_v1_by_observation)
    return client, org, cache


def sign_in_as(client, sub):
    start = client.get("/auth/signin", params={"returnTo": "/"})
    nonce = parse_qs(urlsplit(start.headers["location"]).query)["nonce"][0]
    done = client.get("/auth/dev-bridge/callback",
                      params={"token": mint({**good_claims(nonce=nonce), "sub": sub})})
    assert done.status_code == 302, done.text
    return client


def another(client, sub):
    """A second browser on the same site, signed in as `sub`."""
    return sign_in_as(TestClient(client.app, follow_redirects=False), sub)


def upload(client, reds=(247, 245)):
    res = post_photos(client, list(reds))
    assert res.status_code == 200, res.text
    return res.json()["results"][0]


def by_link(client, number=555):
    res = client.post("/api/identify/observation", json={"source": "inat", "id": number})
    assert res.status_code == 200, res.text
    return res.json()


def ids(specimens):
    return [s["observation_id"] for s in specimens]


# --- who sees it -----------------------------------------------------------------------

def test_signed_out_there_is_no_panel_and_mycomap_org_is_never_asked(conn, tmp_path):
    client, org, _ = build(conn, tmp_path, mode="off")
    r = upload(client)
    assert "yours" not in r
    assert len(r["specimens"]) == yours.SHOWN
    assert not any(s.get("yours") for p in r["per_photo"] for s in p["specimens"])
    assert org.calls == []


def test_a_forged_session_cookie_gets_no_panel(conn, tmp_path):
    client, org, _ = build(conn, tmp_path, mode="identify")
    client.cookies.set("mv_session", "forged.cookie")
    assert post_photos(client, [247]).status_code == 401
    assert org.calls == []


def test_signed_in_the_panel_holds_only_your_own_records_most_alike_first(conn, tmp_path):
    client, org, _ = build(conn, tmp_path)
    r = upload(sign_in_as(client, ANNA))
    mine = r["yours"]
    assert mine["linked"] is True and mine["records"] == 2
    assert set(ids(mine["specimens"])) == {"104", "106"}
    sims = [s["similarity"] for s in mine["specimens"]]
    assert sims == sorted(sims, reverse=True)
    assert ids(mine["specimens"])[0] == "106"         # the "A x" record, like the red find
    assert org.calls == [ANNA]


def test_never_another_users_records(conn, tmp_path):
    client, _, _ = build(conn, tmp_path)
    r = upload(sign_in_as(client, BEN))
    assert ids(r["yours"]["specimens"]) == ["100"]
    assert not {"104", "106"} & set(ids(r["yours"]["specimens"]))


def test_an_answer_from_mycomap_org_about_someone_else_is_refused():
    with pytest.raises(yours.LinkedError):
        yours.parse_linked({"user": BEN, "inat": [{"user_id": BEN_INAT, "login": "ben"}],
                            "mo": []}, ANNA)


# --- nothing shows twice -----------------------------------------------------------------

def test_the_closest_list_leaves_your_records_out_and_still_shows_six(conn, tmp_path):
    client, _, _ = build(conn, tmp_path)
    plain = upload(another(client, CARA))              # nothing connected: what anyone sees
    assert "106" in ids(plain["specimens"])            # someone else sees Anna's record there
    r = upload(sign_in_as(client, ANNA))
    assert not set(ids(r["specimens"])) & set(ids(r["yours"]["specimens"]))
    assert len(r["specimens"]) == yours.SHOWN
    sims = [s["similarity"] for s in r["specimens"]]
    assert sims == sorted(sims, reverse=True)


def test_a_per_photo_match_that_is_yours_is_marked_and_no_other(conn, tmp_path):
    client, _, _ = build(conn, tmp_path)
    r = upload(sign_in_as(client, ANNA))
    for p in r["per_photo"]:
        for s in p["specimens"]:
            assert bool(s.get("yours")) == (s["observation_id"] in {"104", "106"})


# --- empty states --------------------------------------------------------------------------

def test_signed_in_with_nothing_connected_says_so_and_changes_nothing_else(conn, tmp_path):
    client, _, _ = build(conn, tmp_path)
    r = upload(sign_in_as(client, CARA))
    assert r["yours"] == {"linked": False, "records": 0, "specimens": []}
    assert len(r["specimens"]) == yours.SHOWN and "106" in ids(r["specimens"])
    assert not any(s.get("yours") for p in r["per_photo"] for s in p["specimens"])


def test_nothing_connected_leaves_the_list_exactly_as_anyone_gets_it():
    spec = lambda i: {"observation_id": str(i), "similarity": 1 - i / 100}
    result = {"specimens": [spec(i) for i in range(6)], "per_photo": []}
    spare = [spec(i) for i in range(6, 12)]
    plain = yours.personalize(result, spare, None, set(), 0, [])
    unlinked = yours.personalize(result, spare, ACCOUNTS[CARA], set(), 0, [])
    assert "yours" not in plain
    assert unlinked["specimens"] == plain["specimens"] == result["specimens"]


def test_connected_but_none_of_your_records_in_the_reference_shows_nothing(conn, tmp_path):
    client, _, _ = build(conn, tmp_path)
    r = upload(sign_in_as(client, DORA))
    assert r["yours"] == {"linked": True, "records": 0, "specimens": []}
    assert len(r["specimens"]) == yours.SHOWN


def test_when_mycomap_org_cant_be_read_the_answer_is_as_for_anyone(conn, tmp_path):
    org = FakeOrgAccounts()
    org.fail_for = {ANNA}
    client, _, _ = build(conn, tmp_path, org=org)
    plain = upload(another(client, CARA))
    r = upload(sign_in_as(client, ANNA))
    assert "yours" not in r
    assert ids(r["specimens"]) == ids(plain["specimens"])


# --- the per-person cache ------------------------------------------------------------------

def test_accounts_are_kept_ten_minutes_per_person_and_failures_one(conn, tmp_path):
    clock, org = Clock(), FakeOrgAccounts()
    cache = yours.LinkedCache(org, clock=clock)
    cache.get(ANNA), cache.get(ANNA), cache.get(BEN)
    assert org.calls == [ANNA, BEN]
    clock.now += yours.TTL_SECONDS + 1
    cache.get(ANNA)
    assert org.calls == [ANNA, BEN, ANNA]
    org.fail = True
    clock.now += yours.TTL_SECONDS + 1
    assert cache.get(ANNA) is None and cache.get(ANNA) is None
    assert org.calls.count(ANNA) == 3                   # the failure is kept a while too
    org.fail = False
    clock.now += yours.FAILED_TTL_SECONDS + 1
    assert cache.get(ANNA) == ACCOUNTS[ANNA]


def test_the_cache_holds_a_bounded_number_of_people():
    org = FakeOrgAccounts()
    cache = yours.LinkedCache(org, max_people=2)
    for sub in (ANNA, BEN, CARA, ANNA):
        cache.get(sub)
    assert org.calls == [ANNA, BEN, CARA, ANNA]         # Anna was let go, then read again


# --- kept answers are everyone's -----------------------------------------------------------

def test_a_kept_answer_never_holds_anyones_own_part(conn, tmp_path):
    client, _, cache = build(conn, tmp_path)
    first = by_link(sign_in_as(client, ANNA))["results"][0]
    assert set(ids(first["yours"]["specimens"])) <= {"104", "106"}
    rows = cache._conn.execute("select payload from observation_answers").fetchall()
    assert rows
    for (payload,) in rows:
        assert '"yours"' not in payload and ANNA not in payload
    kept = [answercache.json.loads(p) for (p,) in rows][0]
    assert "106" in ids(kept["result"]["specimens"] + kept["spare"])   # not de-duplicated
    assert len(kept["result"]["specimens"]) == yours.SHOWN


def test_a_kept_answer_is_made_personal_per_request(conn, tmp_path):
    client, _, cache = build(conn, tmp_path)
    by_link(sign_in_as(client, ANNA))                  # answered and kept
    ben = by_link(another(client, BEN))
    assert ben["cached"] is True
    r = ben["results"][0]
    assert ids(r["yours"]["specimens"]) == ["100"]
    assert "106" in ids(r["specimens"])                # Anna's record is just a specimen to Ben
    again = by_link(client)
    assert again["cached"] is True
    assert set(ids(again["results"][0]["yours"]["specimens"])) == {"104", "106"}
    assert "106" not in ids(again["results"][0]["specimens"])


def test_your_own_reference_record_is_left_out_when_you_identify_it(conn, tmp_path):
    client, org, _ = build(conn, tmp_path)
    org.accounts[ANNA] = yours.Linked(inat_ids=frozenset({ANNA_INAT, BEN_INAT}))   # 100 is "hers"
    client.site.inat[100] = inat_obs(100, [9001])     # the seeded "A x" reference record
    r = by_link(sign_in_as(client, ANNA), 100)
    assert r["observation"]["excluded_reference"] is True
    assert "100" not in ids(r["results"][0]["yours"]["specimens"])


# --- the rules on their own ------------------------------------------------------------------

def test_a_find_that_is_not_a_fungus_gets_no_panel():
    result = {"specimens": [], "per_photo": [], "gate": {"verdict": "not_fungus"}}
    assert "yours" not in yours.personalize(result, [], ACCOUNTS[ANNA], {"1"}, 1,
                                            [{"observation_id": "1"}])


def test_personalising_never_changes_the_kept_answer():
    spec = lambda i: {"observation_id": str(i), "similarity": 1 - i / 100}
    result = {"specimens": [spec(i) for i in range(6)],
              "per_photo": [{"photo": 0, "specimens": [spec(0)]}]}
    spare = [spec(i) for i in range(6, 12)]
    before = answercache.json.dumps([result, spare])
    out = yours.personalize(result, spare, ACCOUNTS[ANNA], {"0", "2"}, 2, [spec(0), spec(2)])
    assert answercache.json.dumps([result, spare]) == before
    assert ids(out["specimens"]) == ["1", "3", "4", "5", "6", "7"]
    assert out["per_photo"][0]["specimens"][0]["yours"] is True


def test_owners_match_by_id_so_a_renamed_login_still_matches():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        create table inat_observations (observation_id text, user_id integer, user_login text);
        create table mo_observations (observation_id text, owner_id integer, owner_login text);
        insert into inat_observations values ('1', 11, 'old_name'), ('2', 22, 'ben');
        insert into mo_observations values ('mo:7', 5, 'mo_anna'), ('mo:8', null, 'mo_only');
    """)
    keys = np.array(["1", "2", "mo:7", "mo:8"])
    sources = np.array(["inat", "inat", "mo", "mo"])
    owners = yours.OwnerIndex.from_manifest(conn, keys, sources)
    renamed = yours.parse_linked({"user": ANNA, "inat": [{"user_id": 11, "login": "new_name"}],
                                  "mo": [{"user_id": 5, "login": "mo_anna"}]}, ANNA)
    assert owners.records(renamed).tolist() == [0, 2]
    by_login = yours.parse_linked({"user": ANNA, "inat": [],
                                   "mo": [{"user_id": None, "login": "MO_Only"}]}, ANNA)
    assert owners.records(by_login).tolist() == [3]
    assert owners.records(yours.Linked()).tolist() == []


def test_kept_vectors_come_back_exactly():
    v = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
    assert np.array_equal(yours.decode_vectors(yours.encode_vectors(v)), v)
    assert yours.decode_vectors(None) is None and yours.decode_vectors({"b64": "!"}) is None


class Reply:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class OneReply:
    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.asked.append((url, params, headers))
        return self.reply


def test_reading_mycomap_org_sends_the_key_and_only_this_persons_id():
    s = OneReply(Reply(200, {"user": ANNA, "inat": [{"user_id": 11, "login": "a"}], "mo": [],
                             "unverified": {"inat": False, "mo": False}}))
    got = yours.fetch_linked("https://org.example", "k" * 40, ANNA, session=s)
    assert got.inat_ids == frozenset({11})
    url, params, headers = s.asked[0]
    assert url == "https://org.example/api/vision/linked-accounts" and params == {"user": ANNA}
    assert headers["Authorization"] == "Bearer " + "k" * 40


def test_no_such_user_is_no_accounts_but_a_missing_route_is_a_failure():
    gone = OneReply(Reply(404, {"code": "not_found", "message": "No such active user."}))
    assert yours.fetch_linked("https://org.example", "k", ANNA, session=gone) == yours.Linked()
    for reply in (Reply(404, {"message": "Not found"}), Reply(404), Reply(401, {}), Reply(503, {}),
                  Reply(200, None), Reply(200, {"user": BEN, "inat": [], "mo": []})):
        with pytest.raises(yours.LinkedError):
            yours.fetch_linked("https://org.example", "k", ANNA, session=OneReply(reply))
