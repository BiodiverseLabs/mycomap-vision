"""Research > Excluded photos (api.py, excluded_looked.py; Steve, 2026-10-26): the photos people
left out of Vision in Human Feedback, by why, and "where Vision looked" on any of them. Who may
list, read and ask, what is refused and with which status, which photos are listed (a person's
latest decision only; People only to reviewers), what is shown of them (a read-only Human
Feedback viewer's rules), the paper's seal, and that a map leaves the photo's own record out."""

import numpy as np
import pytest
from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species
from test_photo_review import photo_item
from test_run_links import Roles, person, visitor
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import excluded_looked as xl
from mycomap_vision import feedback as fb
from mycomap_vision import looked, signin as signin_rules
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig

LIST = "/api/research/excluded-photos"
LOOKED = "/api/research/excluded-photos/looked"
SEAL = "2026-10-12T13:17:51+00:00"
ANN = {"sub": "admin-1", "name": "Reviewer Ann"}
PERSONAL = ('"by_user"', '"by_name"', '"note"', "admin-1", "Reviewer Ann", "my private note")


class Maker:
    """Stands in for the model: records what each map was made from, and checks the target it
    would explain leaves the photo's own record out."""

    def __init__(self):
        self.calls = []

    def __call__(self, loaded, backbone, ident, name, body, hidden):
        whole = looked.target_for(ident, name)
        left = looked.target_for(ident, name, hidden) if len(hidden) else whole
        self.calls.append({"backbone": backbone, "name": name, "photo": body,
                           "hidden_records": {ident.col_record[int(c)].observation_id
                                              for c in hidden},
                           "photos_whole": len(whole.vectors), "photos_left": len(left.vectors),
                           # The real rank: another record's photo of the name, alone.
                           "rank": xl.rank_of(ident, left.vectors[0], name, hidden)})
        return {"crop": {"x": 0.0, "y": 0.125, "w": 1.0, "h": 0.75}, "grid": 2,
                "map": [0, 85, 170, 255], "rank": 3}


def add_photo(conn, photo_id, record, like, license_class="open", code="cc0", owner=None):
    """Another photo on `record`, its copy the same file as photo `like`'s."""
    url = conn.execute("select source_url from photos where photo_id = ?", (like,)).fetchone()[0]
    conn.execute("insert into photos (photo_id, source, license_class, license_code, owner_user_id, "
                 "owner_login, source_url, first_seen_at, license_checked_at, status) "
                 "values (?, 'inat', ?, ?, ?, 'someone', ?, 'now', 'now', 'done')",
                 (photo_id, license_class, code, owner, url.replace(str(like), str(photo_id))))
    conn.execute("insert into observation_photos values (?, ?, 2)", (record, photo_id))
    store, path = conn.execute("select store, path from photo_copies where photo_id = ?",
                               (like,)).fetchone()
    conn.execute("insert into photo_copies (photo_id, store, size, path) values (?, ?, 'large', ?)",
                 (photo_id, store, path))


def site(conn, tmp_path, admins=("admin-1",), unsealed=None, base="m2", sealed=("105",),
         maker=None, excluded=True):
    """seed_two_species (records 100-107, A x and B y, one CC photo each, 1000-1007) plus a
    second CC photo on 103 (1103) and an all-rights-reserved one on 100 (1100); m1 serves, m2 is
    the base. Excluded by a person: 1000 micrograph, 1003 slip, 1004 people, 1100 habitat,
    1005 slip (record 105, past the seal). Not excluded: 1001 kept, 1002 only flagged, 1006
    excluded and then kept."""
    store = seed_two_species(conn, tmp_path)
    add_photo(conn, 1103, "103", 1003)
    add_photo(conn, 1100, "100", 1000, license_class="arr", code="", owner=55)
    conn.commit()
    root = tmp_path / "emb"
    for name in ("m1", "m2"):
        embed_photos(conn, store, Const(name, twist=0.3 if name == "m2" else 0.0),
                     photos_to_embed(conn, name, "large", store.location, "local"),
                     root / name, log=lambda s: None)
    conn.commit()
    fstore = fb.Store(tmp_path / "state" / "human-feedback.sqlite")
    keys = {"inat:1000": "100", "inat:1001": "101", "inat:1002": "102", "inat:1003": "103",
            "inat:1004": "104", "inat:1005": "105", "inat:1006": "106", "inat:1100": "100"}
    fstore.seed([photo_item(k, r, "R9 v2 slip p=0.83 (high)", projects=()) for k, r in keys.items()],
                "test")
    ids = {i["subject"]: i["id"] for i in fstore.items_of("non_specimen_photo")}
    for key, value in (("inat:1000", "micrograph"), ("inat:1003", "slip"), ("inat:1004", "people"),
                       ("inat:1100", "habitat"), ("inat:1005", "slip"), ("inat:1006", "micrograph")):
        fstore.decide(ids[key], "exclude_photo", ANN, value=value, note="my private note")
    fstore.decide(ids["inat:1001"], "keep_photo", ANN)
    fstore.decide(ids["inat:1006"], "keep_photo", ANN)
    # The paper's seal: record 105 was first green after it.
    fstore.gate = fb.SealGate(lambda k: "2026-10-20" if k in sealed else "2026-01-01", seal=SEAL)
    signin = SigninConfig(mode="all", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    maker = maker or Maker()
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), signin=signin, background=False, fungus_gate=None,
                     feedback_store=fstore, feedback_org=Roles(admins), note=lambda s: None,
                     where_looked=None, excluded_looked=maker if excluded else None,
                     excluded_base=base, excluded_unsealed=unsealed)
    return app, maker, tmp_path


@pytest.fixture
def made(conn, tmp_path):
    return site(conn, tmp_path)


def keys_of(body):
    return [t["photo_key"] for t in body["tiles"]]


# --- which photos are listed, and to whom ------------------------------------------------------

def test_a_photo_is_listed_only_when_a_persons_latest_decision_left_it_out(made):
    app, _, _ = made
    body = visitor(app).get(LIST).json()
    assert sorted(keys_of(body)) == ["inat:1000", "inat:1003", "inat:1005", "inat:1100"]
    # Kept (1001), only flagged by the tagger (1002), excluded and then kept (1006): never here.
    assert {"inat:1001", "inat:1002", "inat:1006"}.isdisjoint(keys_of(body))
    assert {t["photo_key"]: t["category"] for t in body["tiles"]}["inat:1003"] == "slip"


def test_people_photos_are_listed_only_to_human_feedback_reviewers(made):
    app, _, _ = made
    for client in (visitor(app), person(app, "member-1")):          # signed out; no role
        body = client.get(LIST).json()
        assert "inat:1004" not in keys_of(body)
        assert "people" not in [c["code"] for c in body["categories"]]
        assert client.get(LIST, params={"category": "people"}).json()["tiles"] == []
        assert client.get(LOOKED, params={"photo": "inat:1004"}).status_code == 404
    admin = person(app, "admin-1")
    body = admin.get(LIST).json()
    assert "inat:1004" in keys_of(body)
    assert {"code": "people", "label": "People", "count": 1} in body["categories"]


def test_categories_count_what_this_viewer_may_list_most_photos_first(made):
    app, _, _ = made
    cats = visitor(app).get(LIST).json()["categories"]
    assert cats == [{"code": "slip", "label": "Field slips and labels", "count": 2},
                    {"code": "habitat", "label": "Habitat and scenes", "count": 1},
                    {"code": "micrograph", "label": "Micrographs", "count": 1}]
    one = visitor(app).get(LIST, params={"category": "slip"}).json()
    assert sorted(keys_of(one)) == ["inat:1003", "inat:1005"] and one["total"] == 2
    assert visitor(app).get(LIST, params={"category": "nonsense"}).status_code == 400


def test_the_list_and_kept_maps_are_open_to_anyone_and_asking_needs_sign_in(made):
    app, maker, _ = made
    assert signin_rules.research_path("GET", LIST) and signin_rules.research_path("GET", LOOKED)
    assert not signin_rules.research_path("POST", LOOKED)
    assert visitor(app).get(LIST).status_code == 200
    res = visitor(app).post(LOOKED, params={"photo": "inat:1003"})
    assert res.status_code == 401 and not maker.calls


# --- what is shown of each photo -----------------------------------------------------------------

def test_only_photos_anyone_may_see_are_shown_and_never_who_decided(made):
    app, _, _ = made
    res = visitor(app).get(LIST)
    tiles = {t["photo_key"]: t for t in res.json()["tiles"]}
    cc, arr = tiles["inat:1003"], tiles["inat:1100"]
    assert cc["shown"] == "public" and cc["photo_url"] and cc["thumb"] and cc["license_code"] == "cc0"
    assert cc["name"] == "B y" and cc["record_url"] == "https://www.inaturalist.org/observations/103"
    # An all-rights-reserved photo: a placeholder, no address, no photographer.
    assert arr["shown"] == "hidden"
    assert arr["photo_url"] is None and arr["thumb"] is None and arr["owner_login"] is None
    for word in PERSONAL:
        assert word not in res.text, word
    assert res.headers["cache-control"] == "private, no-store"


def test_a_photo_not_shown_publicly_has_no_map_to_read_or_ask_for(made):
    app, maker, _ = made
    assert visitor(app).get(LOOKED, params={"photo": "inat:1100"}).status_code == 404
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1100"}).status_code == 404
    assert not maker.calls


def test_a_photo_whose_owner_withdrew_is_no_longer_mapped_or_read(conn, made):
    app, maker, _ = made
    member = person(app, "m")
    assert member.post(LOOKED, params={"photo": "inat:1003"}).status_code == 200
    # The owner of a CC photo can't withdraw it, but an MO owner can mark it not for export;
    # here: the photo's licence turns all rights reserved, read again on every request.
    conn.execute("update photos set license_class = 'arr' where photo_id = 1003")
    conn.commit()
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003"}).status_code == 404


# --- asking for a map, and reading it ------------------------------------------------------------

def test_a_signed_in_member_asks_once_and_anyone_reads_it_after(made):
    app, maker, _ = made
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003"}).json() == {"status": "none"}
    res = person(app, "member-1").post(LOOKED, params={"photo": "inat:1003"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "ready" and body["name"] == "B y" and body["model"] == "v1"
    assert body["map"] == {"crop": {"x": 0.0, "y": 0.125, "w": 1.0, "h": 0.75}, "grid": 2,
                           "map": [0, 85, 170, 255]}
    assert body["rank"] == 3 and body["top"] == xl.RANK_DEPTH and body["backbone"] == "m1"
    again = visitor(app).get(LOOKED, params={"photo": "inat:1003"})
    assert again.status_code == 200 and again.json() == body
    # Asked again by anyone: the kept map, never made again; the list marks it.
    assert person(app, "other").post(LOOKED, params={"photo": "inat:1003"}).status_code == 200
    assert len(maker.calls) == 1
    tiles = {t["photo_key"]: t for t in visitor(app).get(LIST).json()["tiles"]}
    assert tiles["inat:1003"]["maps"] == ["v1"] and tiles["inat:1000"]["maps"] == []
    mapped = visitor(app).get(LIST, params={"maps": 1}).json()
    assert keys_of(mapped) == ["inat:1003"]


def test_the_map_explains_the_records_name_with_the_record_itself_left_out(made):
    app, maker, _ = made
    person(app, "m").post(LOOKED, params={"photo": "inat:1003"})
    [call] = maker.calls
    assert call["name"] == "B y" and call["backbone"] == "m1"
    # Record 103 (its other photo, 1103, is a reference photo of B y) is left out whole.
    assert call["hidden_records"] == {"103"}
    assert call["photos_left"] < call["photos_whole"]
    assert call["rank"] == 1


def test_who_asked_is_never_kept(made):
    app, _, tmp = made
    person(app, "member-77", name="Asker Name").post(LOOKED, params={"photo": "inat:1003"})
    kept = (tmp / xl.FILE).read_bytes()
    assert b"member-77" not in kept and b"Asker Name" not in kept


def test_the_base_model_has_its_own_map(made):
    app, maker, _ = made
    assert visitor(app).get(LIST).json()["models"] == ["v1", "base"]
    res = person(app, "m").post(LOOKED, params={"photo": "inat:1003", "model": "base"})
    assert res.status_code == 200 and res.json()["model"] == "base"
    assert res.json()["backbone"] == "m2" and maker.calls[-1]["backbone"] == "m2"
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003"}).json() == {"status": "none"}


def test_without_the_base_models_vectors_only_v1_is_offered(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, base="not-here")
    assert visitor(app).get(LIST).json()["models"] == ["v1"]
    res = person(app, "m").post(LOOKED, params={"photo": "inat:1003", "model": "base"})
    assert res.status_code == 404


def test_a_photo_that_is_not_excluded_has_no_map_here(made):
    app, maker, _ = made
    for key in ("inat:1001", "inat:1002", "inat:1006", "inat:999999"):
        assert person(app, "m").post(LOOKED, params={"photo": key}).status_code == 404, key
    assert not maker.calls


# --- the paper's seal ------------------------------------------------------------------------------

def test_maps_of_records_past_the_seal_are_held_back_while_the_switch_is_off(made):
    app, maker, _ = made
    tile = {t["photo_key"]: t for t in visitor(app).get(LIST).json()["tiles"]}["inat:1005"]
    assert tile["held"] is True and tile["shown"] == "public"      # the photo still shows
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1005"}).status_code == 403
    assert visitor(app).get(LOOKED, params={"photo": "inat:1005"}).status_code == 403
    assert not maker.calls


def test_a_record_on_the_seals_own_day_is_held_back_too(conn, tmp_path):
    app, maker, _ = site(conn, tmp_path, sealed=())
    app.state.feedback.gate = fb.SealGate(lambda k: "2026-10-12" if k == "103" else "2026-01-01",
                                          seal=SEAL)
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003"}).status_code == 403


def test_the_switch_lifts_the_seal(conn, tmp_path):
    app, maker, _ = site(conn, tmp_path, unsealed=True)
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1005"}).status_code == 200
    assert len(maker.calls) == 1


# --- limits and failures ---------------------------------------------------------------------------

@pytest.mark.parametrize("which, asks", [(0, 20), (1, 200)])
def test_a_member_asks_at_most_twenty_times_in_ten_minutes_and_two_hundred_a_day(
        conn, tmp_path, which, asks):
    app, maker, _ = site(conn, tmp_path)
    limiter = app.state.excluded_limits[which]      # 0: shared with the run maps; 1: the day's
    for _ in range(asks):
        assert limiter.check("busy-1") == 0
    res = person(app, "busy-1").post(LOOKED, params={"photo": "inat:1003"})
    assert res.status_code == 429 and int(res.headers["retry-after"]) > 0
    assert not maker.calls
    assert person(app, "someone-else").post(LOOKED, params={"photo": "inat:1003"}).status_code == 200
    # A map kept already costs nothing to ask for again.
    assert person(app, "busy-1").post(LOOKED, params={"photo": "inat:1003"}).status_code == 200


def test_a_map_that_cannot_be_made_says_so(conn, tmp_path):
    def broken(*args):
        raise looked.NoMap("nothing to compare with")
    app, _, _ = site(conn, tmp_path, maker=broken)
    res = person(app, "m").post(LOOKED, params={"photo": "inat:1003"})
    assert res.status_code == 409
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003"}).status_code == 503


def test_switched_off_the_page_finds_nothing(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, excluded=False)
    assert visitor(app).get(LIST).status_code == 404
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003"}).status_code == 404


# --- the leave-out itself ----------------------------------------------------------------------------

def test_leaving_a_record_out_takes_its_photos_out_of_the_names_score_and_average():
    vecs = np.eye(4, dtype=np.float32)
    index = type("I", (), {"species": ["A x"], "starts": np.array([0]), "cols": np.arange(4),
                           "rec_starts": np.array([0, 2, 3])})()
    ident = type("Id", (), {"index": index, "model": object(),
                            "reference_vectors": lambda self, cols: vecs[cols]})()
    whole = looked.target_for(ident, "A x")
    assert whole.record.tolist() == [0, 0, 1, 2]
    left = looked.target_for(ident, "A x", np.array([0, 1]))      # record 0's two photos
    assert left.vectors.tolist() == vecs[2:].tolist() and left.record.tolist() == [0, 1]
    assert np.allclose(left.mean, np.array([0, 0, 1, 1]) / np.sqrt(2))
    with pytest.raises(looked.NoMap):
        looked.target_for(ident, "A x", np.arange(4))


def test_the_list_follows_new_decisions_within_its_check_interval(tmp_path):
    fstore = fb.Store(tmp_path / "human-feedback.sqlite")
    fstore.seed([photo_item("inat:1", "10", "R9 v2 slip p=0.9 (high)", projects=())], "test")
    [item] = fstore.items_of("non_specimen_photo")
    now = [0.0]
    listing = xl.Listing(fstore, every=30.0, clock=lambda: now[0])
    assert listing.rows() == []
    fstore.decide(item["id"], "exclude_photo", ANN, value="slip")
    assert listing.rows() == []                                   # read again only after 30 s
    now[0] = 31.0
    assert listing.rows() == [xl.Excluded("inat:1", "10", "slip")]
    fstore.decide(item["id"], "keep_photo", ANN)                  # a later decision wins
    now[0] = 62.0
    assert listing.rows() == []


# --- the base model's vectors: rows alone don't make it usable --------------------------------------

def test_manifest_rows_without_the_base_models_files_offer_v1_only(conn, tmp_path):
    # A release's manifest keeps every backbone the laptop embedded; the files are what count.
    app, maker, _ = site(conn, tmp_path)
    (tmp_path / "emb" / "m2" / "shard-00000.npy").unlink()
    assert visitor(app).get(LIST).json()["models"] == ["v1"]
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003", "model": "base"}).status_code == 404
    assert not maker.calls


def test_a_folder_shipped_as_the_manifest_names_it_passes_the_check(conn, tmp_path):
    site(conn, tmp_path)
    got = xl.check_vectors(conn, "m2", tmp_path / "emb" / "m2")
    assert got["ok"], got["problems"]
    assert got["vectors"] == conn.execute("select count(*) from embeddings where backbone = 'm2'"
                                          ).fetchone()[0]
    assert xl.check_vectors(conn, "m2", tmp_path / "emb" / "m2", dim=got["dim"])["ok"]


def test_the_same_folder_as_int8_codes_with_their_scales_passes_too(conn, tmp_path):
    from mycomap_vision.quantize import write_int8_shards
    site(conn, tmp_path)
    write_int8_shards(tmp_path / "emb" / "m2", tmp_path / "int8" / "m2")
    got = xl.check_vectors(conn, "m2", tmp_path / "int8" / "m2")
    assert got["ok"] and got["dtype"] == "int8", got["problems"]
    (tmp_path / "int8" / "m2" / "shard-00000.scale.npy").unlink()
    assert not xl.check_vectors(conn, "m2", tmp_path / "int8" / "m2")["ok"]


def test_the_check_refuses_a_folder_the_server_would_misread(conn, tmp_path):
    site(conn, tmp_path)
    folder = tmp_path / "emb" / "m2"
    ids = np.load(folder / "shard-00000.ids.npy")
    vecs = np.load(folder / "shard-00000.npy")
    bad = lambda: xl.check_vectors(conn, "m2", folder)["problems"]
    np.save(folder / "shard-00000.ids.npy", ids[::-1].copy())            # rows in another order
    assert any("name another photo" in p for p in bad())
    np.save(folder / "shard-00000.ids.npy", ids)
    np.save(folder / "shard-00000.npy", vecs * 3)                         # not unit length
    assert any("unit length" in p for p in bad())
    np.save(folder / "shard-00000.npy", vecs[:-1])                        # a row short
    assert any("vectors of shape" in p for p in bad())
    np.save(folder / "shard-00000.npy", vecs)
    assert xl.check_vectors(conn, "m2", folder, dim=vecs.shape[1] + 1)["problems"]
    (folder / "shard-00000.ids.npy").unlink()
    assert any("missing" in p for p in bad())
    assert not xl.vectors_shipped(xl.manifest_shards(conn, "m2"), folder)
    assert xl.check_vectors(conn, "nothing", folder)["problems"] == [
        "the manifest has no embeddings rows for nothing"]


def test_a_base_map_leaves_the_served_models_fast_search_packed(conn, tmp_path, monkeypatch):
    from mycomap_vision import fastsearch
    pytest.importorskip("torch")
    monkeypatch.setenv("MV_SEARCH_INT8", "1")
    monkeypatch.setattr(fastsearch, "SLOT", fastsearch.Slot())
    app, _, _ = site(conn, tmp_path)
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003"}).status_code == 200
    served = fastsearch.SLOT.owner
    if served is None:
        pytest.skip(f"no int8 engine here: {fastsearch.SLOT.last_error}")
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003", "model": "base"}
                                 ).status_code == 200
    assert fastsearch.SLOT.owner is served and served.packed is not None
