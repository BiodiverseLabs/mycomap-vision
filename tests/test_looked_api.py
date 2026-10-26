"""Where Vision looked on a run's link (api.py, looked.py; Steve, 2026-10-26): who may ask for
the maps and who may read them, what is refused and with which status, what is read to make
them (only the photos the link shows), and that they are kept beside the run."""

import json

import pytest
from test_api import open_limits, png
from test_models_and_scoreboard import Const, seed_two_species
from test_run_links import Roles, identify, person, visitor
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import looked, runs
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig, run_path


class Maker:
    """Stands in for the model: records each photo it maps, returns a fixed small map."""

    def __init__(self):
        self.calls = []

    def __call__(self, loaded, backbone, ident, name, body, specimen):
        self.calls.append({"backbone": backbone, "name": name, "photo": body,
                           "specimen": specimen})
        return {"crop": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}, "grid": 2,
                "map": [0, 85, 170, 255], "pairs": [[0.25, 0.25, 0.75, 0.75]],
                "specimen_crop": {"x": 0.1, "y": 0.0, "w": 0.8, "h": 1.0}}


class Source:
    """The polite source fetcher, without the network: answers any address."""

    def __init__(self):
        self.asked = []

    def photos_at(self, source, urls):
        self.asked.append((source, list(urls)))
        return [(i, b"specimen-photo") for i in range(len(urls))]


def site(conn, tmp_path, where_looked="maker", monkeypatch=None, loader=None, source=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1", log=lambda s: None)
    conn.commit()
    run_store = runs.RunStore(runs.DirObjects(tmp_path / "bucket"), write=lambda job: job(),
                              note=lambda s: None, index=runs.OwnerIndex(tmp_path / "idx.sqlite"))
    signin = SigninConfig(mode="all", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    maker, source = Maker(), source or Source()
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=loader or (lambda n: Const(n)),
                     limits=open_limits(), signin=signin, background=False, fungus_gate=None,
                     run_store=run_store, feedback_org=Roles(), note=lambda s: None,
                     observation_fetcher=source,
                     where_looked=maker if where_looked == "maker" else where_looked)
    return app, run_store, maker, source, tmp_path / "bucket"


def a_run(app, who="owner-1"):
    """An identification on the site, kept under its link: its id."""
    res = identify(person(app, who), photos=(("a.png", png(247), "image/png"),
                                             ("b.png", png(240), "image/png")))
    assert res.status_code == 200, res.text
    return res.json()["run"]["id"]


# --- who may ask, who may read ------------------------------------------------------------------

def test_anyone_signed_in_with_the_link_may_ask_and_anyone_with_the_link_may_read(conn, tmp_path):
    app, store, maker, _, _ = site(conn, tmp_path)
    rid = a_run(app, "owner-1")
    member = person(app, "someone-else")                  # not the person who ran it
    res = member.post(f"/api/runs/{rid}/looked")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "ready" and body["name"] == "A x"
    assert [p["photo"] for p in body["photos"]] == [0, 1]
    assert body["photos"][0]["map"] == [0, 85, 170, 255] and body["photos"][0]["grid"] == 2
    signed_out = visitor(app).get(f"/api/runs/{rid}/looked")
    assert signed_out.status_code == 200 and signed_out.json()["photos"] == body["photos"]
    assert signed_out.headers["cache-control"] == "private, no-store"
    assert run_path(f"/api/runs/{rid}/looked") and not run_path(f"/api/runs/{rid}/looked/x")


def test_the_person_who_ran_it_may_ask_too(conn, tmp_path):
    app, _, _, _, _ = site(conn, tmp_path)
    rid = a_run(app, "owner-1")
    assert person(app, "owner-1").post(f"/api/runs/{rid}/looked").status_code == 200


def test_asking_needs_a_sign_in(conn, tmp_path):
    app, _, maker, _, _ = site(conn, tmp_path)
    rid = a_run(app)
    res = visitor(app).post(f"/api/runs/{rid}/looked")
    assert res.status_code == 401
    assert not maker.calls


def test_before_anyone_asks_there_is_nothing_to_read(conn, tmp_path):
    app, _, _, _, _ = site(conn, tmp_path)
    rid = a_run(app)
    res = visitor(app).get(f"/api/runs/{rid}/looked")
    assert res.status_code == 404 and res.json() == {"status": "none"}


def test_a_link_that_was_never_given_out_finds_nothing(conn, tmp_path):
    app, _, maker, _, _ = site(conn, tmp_path)
    member = person(app, "m")
    assert member.post(f"/api/runs/{runs.new_id()}/looked").status_code == 404
    assert visitor(app).get(f"/api/runs/{runs.new_id()}/looked").status_code == 404
    assert not maker.calls


# --- made once, kept beside the run, gone with it -------------------------------------------------

def test_the_maps_are_made_once_and_kept_beside_the_run(conn, tmp_path):
    app, store, maker, _, bucket = site(conn, tmp_path)
    rid = a_run(app)
    member = person(app, "m")
    member.post(f"/api/runs/{rid}/looked")
    assert len(maker.calls) == 2                          # one per photo
    kept = json.loads((bucket / runs.PREFIX / rid / runs.LOOKED).read_bytes())
    assert kept["version"] == looked.VERSION and kept["name"] == "A x"
    assert kept["model"]["backbone"] == "m1"
    # Asked again, by anyone: read from the kept maps, never made again.
    assert person(app, "other").post(f"/api/runs/{rid}/looked").status_code == 200
    assert visitor(app).get(f"/api/runs/{rid}/looked").status_code == 200
    assert len(maker.calls) == 2


def test_the_maps_are_made_from_the_runs_own_photo_copies(conn, tmp_path):
    app, store, maker, _, _ = site(conn, tmp_path)
    rid = a_run(app)
    person(app, "m").post(f"/api/runs/{rid}/looked")
    assert [c["photo"] for c in maker.calls] == [store.photo(rid, 0), store.photo(rid, 1)]
    assert all(c["name"] == "A x" and c["backbone"] == "m1" for c in maker.calls)


def test_deleting_a_run_deletes_its_maps(conn, tmp_path):
    app, store, _, _, bucket = site(conn, tmp_path)
    rid = a_run(app, "owner-1")
    person(app, "m").post(f"/api/runs/{rid}/looked")
    assert (bucket / runs.PREFIX / rid / runs.LOOKED).exists()
    assert person(app, "owner-1").delete(f"/api/runs/{rid}").status_code == 204
    assert not (bucket / runs.PREFIX / rid / runs.LOOKED).exists()
    assert visitor(app).get(f"/api/runs/{rid}/looked").status_code == 404


# --- the specimen match: only a photo the link shows ----------------------------------------------

def test_each_photo_is_matched_with_the_specimen_photo_the_link_shows_for_it(conn, tmp_path):
    app, _, maker, source, _ = site(conn, tmp_path)
    rid = a_run(app)
    shown = visitor(app).get(f"/api/runs/{rid}").json()["results"][0]["per_photo"]
    body = person(app, "m").post(f"/api/runs/{rid}/looked").json()
    for entry in body["photos"]:
        spec = next(p for p in shown if p["photo"] == entry["photo"])["specimens"][0]
        assert entry["match"]["specimen"]["photo_url"] == spec["photo_url"]
        assert entry["match"]["specimen"]["observation_id"] == spec["observation_id"]
        assert entry["match"]["pairs"] == [[0.25, 0.25, 0.75, 0.75]]
        assert entry["match"]["crop"] == {"x": 0.1, "y": 0.0, "w": 0.8, "h": 1.0}
    assert [urls for _, urls in source.asked] == [[s["photo_url"]] for s in (
        next(p for p in shown if p["photo"] == e["photo"])["specimens"][0] for e in body["photos"])]
    assert all(c["specimen"] == b"specimen-photo" for c in maker.calls)


def test_a_match_is_shown_only_while_the_link_still_shows_that_specimen_photo(conn, tmp_path):
    app, store, _, _, bucket = site(conn, tmp_path)
    rid = a_run(app)
    person(app, "m").post(f"/api/runs/{rid}/looked")
    path = bucket / runs.PREFIX / rid / runs.LOOKED
    kept = json.loads(path.read_bytes())
    kept["photos"][0]["match"]["photo_url"] = "https://example.org/a-photo-no-longer-shown.jpg"
    path.write_text(json.dumps(kept))
    photos = visitor(app).get(f"/api/runs/{rid}/looked").json()["photos"]
    assert photos[0]["match"] is None
    assert photos[1]["match"] is not None


def test_a_specimen_photo_that_cant_be_read_leaves_the_map_without_a_match(conn, tmp_path):
    app, _, maker, source, _ = site(conn, tmp_path)
    source.photos_at = lambda s, urls: []                  # the source didn't answer
    rid = a_run(app)
    body = person(app, "m").post(f"/api/runs/{rid}/looked").json()
    assert all(p["match"] is None and p["map"] for p in body["photos"])
    assert all(c["specimen"] is None for c in maker.calls)


def test_an_inat_runs_maps_read_only_the_photos_its_link_shows(conn, tmp_path):
    """An iNat or MO run keeps no photos, only the addresses its link shows (Creative Commons
    only, else null): a photo whose address the link doesn't show is never read or mapped."""
    app, store, maker, source, bucket = site(conn, tmp_path)
    rid = a_run(app)
    path = bucket / runs.PREFIX / rid / "run.json"
    run = json.loads(path.read_bytes())
    run["kind"], run["photos"] = "observation", 0
    shown = "https://inaturalist-open-data.s3.amazonaws.com/photos/77/large.jpg"
    run["observation"] = {"source": "inat", "id": "555", "photos": 2, "photos_used": 2,
                          "photos_shown": [shown, None],
                          "photos_credit": [{"licence": "cc-by", "by": "A Photographer"}, None]}
    path.write_text(json.dumps(run))
    store._cache.clear()
    body = person(app, "m").post(f"/api/runs/{rid}/looked").json()
    assert [p["photo"] for p in body["photos"]] == [0]
    assert source.asked[0] == ("inat", [shown])
    assert maker.calls[0]["photo"] == b"specimen-photo"   # what the stand-in source answers
    assert len(maker.calls) == 1


# --- refusals --------------------------------------------------------------------------------------

def test_an_answer_with_no_species_has_nothing_to_map(conn, tmp_path):
    app, store, maker, _, bucket = site(conn, tmp_path)
    rid = a_run(app)
    path = bucket / runs.PREFIX / rid / "run.json"
    run = json.loads(path.read_bytes())
    run["models"][0]["result"]["ranks"]["species"] = []
    path.write_text(json.dumps(run))
    store._cache.clear()
    res = person(app, "m").post(f"/api/runs/{rid}/looked")
    assert res.status_code == 409 and "no species" in res.json()["detail"]
    assert not maker.calls


def test_too_many_maps_waiting_is_refused_with_when_to_try_again(conn, tmp_path, monkeypatch):
    app, _, maker, _, _ = site(conn, tmp_path)
    rid = a_run(app)

    def busy(self, job):
        raise looked.Busy(41.5)
    monkeypatch.setattr(looked.Worker, "submit", busy)
    res = person(app, "m").post(f"/api/runs/{rid}/looked")
    assert res.status_code == 429 and res.headers["retry-after"] == "42"


def test_one_person_asking_for_many_runs_is_slowed_down(conn, tmp_path):
    app, _, maker, _, _ = site(conn, tmp_path)
    rids = [a_run(app) for _ in range(21)]
    member = person(app, "busy-member")
    codes = [member.post(f"/api/runs/{rid}/looked").status_code for rid in rids]
    assert codes[:20] == [200] * 20 and codes[20] == 429
    # Someone else is not held up by it.
    assert person(app, "other").post(f"/api/runs/{rids[20]}/looked").status_code == 200


def test_a_failed_map_says_so_and_can_be_asked_for_again(conn, tmp_path):
    app, _, maker, _, _ = site(conn, tmp_path)
    rid = a_run(app)
    real = maker.__call__
    calls = {"n": 0}

    def flaky(*a):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("out of memory")
        return real(*a)
    app_maker = flaky
    # The site's maker is the Maker instance; make it fail once.
    maker.__class__ = type("Flaky", (Maker,), {"__call__": lambda self, *a: app_maker(*a)})
    member = person(app, "m")
    res = member.post(f"/api/runs/{rid}/looked")
    assert res.status_code == 503 and res.json()["status"] == "failed"
    assert visitor(app).get(f"/api/runs/{rid}/looked").status_code == 503
    assert member.post(f"/api/runs/{rid}/looked").status_code == 200


# --- the real maps, from the served model and its reference -----------------------------------------

def test_the_real_maps_come_from_the_served_model_and_its_reference_set(conn, tmp_path):
    """The maker the site runs (no stand-in): the served backbone's ViT tower, the served index's
    reference for the answer's species, and the specimen photo the link shows."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    from open_clip.transformer import VisionTransformer
    from test_looked import jpeg
    from torchvision import transforms as T
    torch.manual_seed(0)
    vit = VisionTransformer(image_size=56, patch_size=14, width=64, layers=6, heads=4, mlp_ratio=2.0,
                            output_dim=2).eval()        # the reference vectors here are 2-d
    tf = T.Compose([T.Resize(56), T.CenterCrop(56), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    class Served(Const):
        """Identifies as the stand-in model does, and carries a real tower for the maps."""
        transform = tf

        class base:
            model = vit

    class Photos(Source):
        def photos_at(self, source, urls):
            self.asked.append((source, list(urls)))
            return [(i, jpeg((64, 64), seed=i)) for i in range(len(urls))]

    app, store, _, source, _ = site(conn, tmp_path, where_looked=True,
                                    loader=lambda n: Served(n), source=Photos())
    rid = a_run(app)
    res = person(app, "m").post(f"/api/runs/{rid}/looked")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "A x" and len(body["photos"]) == 2
    for entry in body["photos"]:
        assert entry["grid"] == 4 and len(entry["map"]) == 16
        assert min(entry["map"]) >= 0 and max(entry["map"]) == 255
        assert entry["crop"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        if entry["match"]:
            assert all(0 < v < 1 for pair in entry["match"]["pairs"] for v in pair)
    assert source.asked                                    # the specimen photos were read


# --- on and off -------------------------------------------------------------------------------------

def test_the_page_is_told_whether_the_button_is_there(conn, tmp_path):
    app, _, _, _, _ = site(conn, tmp_path)
    assert visitor(app).get("/api/me").json()["features"] == {"where_looked": True}


@pytest.mark.parametrize("off", ["argument", "setting"])
def test_switched_off_there_is_no_button_and_no_maps(conn, tmp_path, monkeypatch, off):
    if off == "setting":
        monkeypatch.setenv("MV_WHERE_LOOKED", "0")
        app, _, maker, _, _ = site(conn, tmp_path, where_looked=True)
    else:
        app, _, maker, _, _ = site(conn, tmp_path, where_looked=False)
    rid = a_run(app)
    assert visitor(app).get("/api/me").json()["features"] == {"where_looked": False}
    assert person(app, "m").post(f"/api/runs/{rid}/looked").status_code == 404
    assert visitor(app).get(f"/api/runs/{rid}/looked").status_code == 404
