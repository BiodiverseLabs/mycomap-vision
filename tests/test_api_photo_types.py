"""Photo-type tags on Identify, through the API (phototypes.py; Steve, 2026-10-22): a
micrograph, label or habitat photo tagged in the high band is left out of the species
answer and reported with its tag; "Include anyway" counts it; other kinds are tagged only;
without the tagger file nothing changes; kept answers and run links key on the set of
photos counted."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png
from test_api_v1_by_observation import fetcher
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer
from test_api_v1_sessions import FakeGate
from test_identify_log import rows
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, phototypes, runs
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.phototypes import HIGH, LOW, MEDIUM, PhotoTagger

SERVER = bearer(SERVER_KEY)
CLASSES = ("specimen", "micrograph", "slip", "habitat", "uv", "chemical", "spore_print",
           "dried", "drawing", "tree", "people", "not_fungus")
BANDS = {"micrograph": [[0.35, HIGH]], "slip": [[0.35, HIGH]],
         "habitat": [[0.5, HIGH], [0.35, MEDIUM]], "uv": [[0.5, MEDIUM], [0.35, LOW]]}

HABITAT, UV, MICRO = 240, 245, 250     # test photos by their red value (Const's vectors)


class StubTagger(PhotoTagger):
    """The tagger's rules with a stubbed head: a photo's kind follows its red value (the
    stand-in backbone's vectors are its angle), everything else is a specimen photo."""

    def __init__(self, kinds=None):
        super().__init__(np.zeros((2, len(CLASSES)), np.float32), np.zeros(len(CLASSES)),
                         CLASSES, BANDS, 0.35, "m1", "stub")
        self.kinds = kinds or {HABITAT: ("habitat", 0.9), UV: ("uv", 0.9),
                               MICRO: ("micrograph", 0.9)}
        self.tagged = 0

    def probs(self, base_vectors):
        out = []
        for v in np.asarray(base_vectors, np.float64):
            red = int(round(np.arctan2(v[1], v[0]) / 3 * 255))
            kind, p = self.kinds.get(red, ("specimen", 0.95))
            row = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1))
            row[CLASSES.index(kind)] = p
            out.append(row)
        self.tagged += len(out)
        return np.asarray(out)


def observations(photos=None):
    """The fake iNat and MO, with some photos replaced ({photo id: bytes})."""
    f = fetcher()
    f.session.photos.update(photos or {})
    return f


def make_app(conn, tmp_path, tagger, seeded, gate=None, run_store=None, site=None):
    if not seeded:
        store = seed_two_species(conn, tmp_path)
        embed_photos(conn, store, Const("m1"),
                     photos_to_embed(conn, "m1", "large", store.location, "local"),
                     tmp_path / "emb" / "m1", log=lambda s: None)
        conn.commit()
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", tmp_path / "emb", backbone_loader=lambda n: Const(n),
        limits=open_limits(), background=False, server_key=SERVER_KEY, api_v1=True,
        sessions=True, by_observation=True, observation_fetcher=observations(site),
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
        fungus_gate=gate, photo_tagger=tagger, run_store=run_store, note=lambda s: None))


@pytest.fixture
def apps(conn, tmp_path):
    """(the server with a stubbed tagger, the same server without one)."""
    tagged = make_app(conn, tmp_path, StubTagger(), seeded=False)
    plain = make_app(conn, tmp_path, None, seeded=True)
    return tagged, plain


def upload(client, reds, **data):
    files = [("photos", (f"p{i}.png", png(r), "image/png")) for i, r in enumerate(reds)]
    res = client.post("/api/identify", files=files, data={"models": "m1/nearest", **data})
    assert res.status_code == 200, res.text
    [result] = res.json()["results"]
    return result


def v1(client, reds, **data):
    files = [("photos", (f"p{i}.png", png(r), "image/png")) for i, r in enumerate(reds)]
    return client.post("/api/v1/identify", files=files, data=data, headers=SERVER)


def answer(result):
    """What the species answer is made of: the combined ranks and the specimens matched."""
    return result["ranks"], [(s["observation_id"], s["matched_query_photo"])
                             for s in result["specimens"]]


# --- left out, included anyway, tagged only ---------------------------------------------

def test_a_habitat_photo_is_left_out_of_the_species_answer_and_reported_with_its_tag(apps):
    tagged, plain = apps
    r = upload(tagged, [247, HABITAT])
    assert r["photo_types"]["left_out"] == [1] and r["photo_types"]["kept_all"] is False
    tag = r["per_photo"][1]["photo_type"]
    assert (tag["type"], tag["band"], tag["label"], tag["left_out"], tag["included"]) == (
        "habitat", "high", "Habitat", True, False)
    assert r["per_photo"][0]["photo_type"]["type"] == "specimen"
    assert r["per_photo"][0]["photo_type"]["label"] is None          # no chip
    # The answer is the one the first photo alone gets, and the habitat photo matches nothing.
    assert answer(r)[0] == answer(upload(plain, [247]))[0]
    assert all(m == 0 for _, m in answer(r)[1])
    assert r["per_photo"][1]["specimens"] == [] and r["per_photo"][1]["ranks"]["species"] == []
    assert "Photo 2 looks like a habitat photo, so it was left out of the answer." in r["hints"]


def test_include_anyway_counts_the_left_out_photo(apps):
    tagged, plain = apps
    r = upload(tagged, [247, HABITAT], include="1")
    assert r["photo_types"]["left_out"] == [] and r["photo_types"]["included"] == [1]
    tag = r["per_photo"][1]["photo_type"]
    assert tag["left_out"] is False and tag["included"] is True and tag["label"] == "Habitat"
    assert answer(r) == answer(upload(plain, [247, HABITAT]))
    assert not any("left out" in h for h in r["hints"])


def test_a_uv_photo_is_tagged_but_still_counts(apps):
    tagged, plain = apps
    r = upload(tagged, [247, UV])
    tag = r["per_photo"][1]["photo_type"]
    assert (tag["type"], tag["band"], tag["label"], tag["left_out"]) == ("uv", "medium", "UV",
                                                                        False)
    assert r["photo_types"]["left_out"] == []
    assert answer(r) == answer(upload(plain, [247, UV]))


def test_a_habitat_tag_below_the_high_band_is_shown_but_does_not_leave_the_photo_out(
        conn, tmp_path):
    client = make_app(conn, tmp_path, StubTagger({HABITAT: ("habitat", 0.45)}), seeded=False)
    r = upload(client, [247, HABITAT])
    tag = r["per_photo"][1]["photo_type"]
    assert (tag["band"], tag["label"], tag["left_out"]) == ("medium", "Habitat", False)
    assert r["photo_types"]["left_out"] == []


def test_when_every_photo_would_be_left_out_the_answer_is_from_all_of_them_and_says_so(apps):
    tagged, plain = apps
    r = upload(tagged, [HABITAT, MICRO])
    assert r["photo_types"] == {**r["photo_types"], "left_out": [], "kept_all": True}
    assert answer(r) == answer(upload(plain, [HABITAT, MICRO]))
    assert any(h.startswith("Every photo looks like") for h in r["hints"])
    assert r["ranks"]["species"], "an answer, not the gate's refusal"


def test_without_the_tagger_file_nothing_changes(conn, tmp_path, monkeypatch):
    monkeypatch.delenv(phototypes.SETTING, raising=False)
    plain = make_app(conn, tmp_path, True, seeded=False)      # loads from the setting: none
    r = upload(plain, [247, HABITAT], include="1")
    assert "photo_types" not in r
    assert all("photo_type" not in p for p in r["per_photo"])
    assert r == upload(plain, [247, HABITAT])
    body = v1(plain, [247, HABITAT], include="1").json()
    assert "photo_types" not in body and all("photo_type" not in p for p in body["per_photo"])
    assert plain.get("/api/health").json()["photo_tagger"] is None
    # Not even a malformed include is read: the request is answered as it was before.
    assert v1(plain, [247], include="x").status_code == 200
    assert "photo_types" not in upload(plain, [247], include="7")


def test_a_tagger_file_that_fails_its_checks_is_not_served_and_nothing_changes(
        conn, tmp_path, monkeypatch):
    bad = tmp_path / "photo-tagger.npz"
    np.savez(bad, W=np.zeros((2, 12)), b=np.zeros(12), classes=np.array(CLASSES))
    phototypes.sidecar(bad).write_text('{"sha256": "0", "bands": {}, "floor": 0.35}')
    monkeypatch.setenv(phototypes.SETTING, str(bad))
    notes = []
    store = seed_two_species(conn, tmp_path)
    embed_photos(conn, store, Const("m1"),
                 photos_to_embed(conn, "m1", "large", store.location, "local"),
                 tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()
    client = TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                   backbone_loader=lambda n: Const(n), limits=open_limits(),
                                   background=False, server_key=SERVER_KEY, fungus_gate=None,
                                   note=notes.append))
    assert "photo_types" not in upload(client, [247, HABITAT])
    assert any("[photo-tagger] not served" in n and "sha256" in n for n in notes)
    assert "sha256" in client.get("/api/health", headers=SERVER).json()["photo_tagger"]["error"]
    assert client.get("/api/health").json()["photo_tagger"] == {
        "error": "not served; see the server log"}


def test_the_tagger_reads_the_gates_own_vector_and_both_leave_their_photos_out(conn, tmp_path):
    gate, tagger = FakeGate(), StubTagger()
    client = make_app(conn, tmp_path, tagger, seeded=False, gate=gate)
    r = upload(client, [247, HABITAT, 12])
    assert gate.judged == 3 and tagger.tagged == 3             # one base vector, both read it
    assert r["gate"]["left_out"] == [] and r["photo_types"]["left_out"] == [1]
    assert r["per_photo"][1]["gate"]["verdict"] == "fungus"


def test_a_tagger_for_other_vectors_than_the_gates_is_not_served(conn, tmp_path):
    gate = FakeGate()
    gate.base = "bioclip-2-336"
    notes = []
    store = seed_two_species(conn, tmp_path)
    embed_photos(conn, store, Const("m1"),
                 photos_to_embed(conn, "m1", "large", store.location, "local"),
                 tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()
    client = TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                   backbone_loader=lambda n: Const(n), limits=open_limits(),
                                   background=False, fungus_gate=gate,
                                   photo_tagger=StubTagger(), note=notes.append))
    assert "photo_types" not in upload(client, [247, HABITAT])
    assert any("the gate bioclip-2-336" in n for n in notes)


def test_a_bad_include_is_refused(apps):
    tagged, _ = apps
    files = [("photos", ("p.png", png(247), "image/png"))]
    res = tagged.post("/api/identify", files=files, data={"include": "1"})
    assert res.status_code == 400 and "photo indices 0 to 0" in res.text
    res = v1(tagged, [247], include="x")
    assert res.status_code == 400 and res.json()["code"] == "bad_request"


# --- API v1 ---------------------------------------------------------------------------

def test_the_v1_answer_carries_each_photos_type_and_what_was_left_out(apps):
    tagged, plain = apps
    body = v1(tagged, [247, HABITAT, UV]).json()
    assert [p["photo_type"]["type"] for p in body["per_photo"]] == ["specimen", "habitat", "uv"]
    assert body["per_photo"][1]["photo_type"]["left_out"] is True
    assert body["photo_types"]["left_out"] == [1]
    assert body["ranks"] == v1(plain, [247, UV]).json()["ranks"]
    anyway = v1(tagged, [247, HABITAT, UV], include="1").json()
    assert anyway["photo_types"]["included"] == [1]
    assert anyway["ranks"] == v1(plain, [247, HABITAT, UV]).json()["ranks"]


def test_an_app_session_leaves_a_habitat_photo_out_until_it_is_included(apps):
    tagged, _ = apps
    app = bearer(app_token())
    sid = tagged.post("/api/v1/sessions", json={}, headers=app).json()["session_id"]
    for red, ref in ((247, "cap"), (HABITAT, "woods")):
        res = tagged.post(f"/api/v1/sessions/{sid}/photos", headers=app,
                          files={"photo": ("p.png", png(red), "image/png")},
                          data={"photo_ref": ref})
        assert res.status_code == 200, res.text
    body = res.json()
    assert body["photo"]["photo_type"]["type"] == "habitat"
    assert body["combined"]["photo_types"]["left_out"] == [1]
    together = v1(tagged, [247, HABITAT]).json()
    assert body["combined"]["ranks"] == together["ranks"]
    res = tagged.patch(f"/api/v1/sessions/{sid}", json={"include_photos": ["woods"]},
                       headers=app)
    assert res.status_code == 200, res.text
    assert res.json()["combined"]["photo_types"]["included"] == [1]
    assert res.json()["combined"]["ranks"] == v1(tagged, [247, HABITAT], include="1").json()["ranks"]
    bad = tagged.patch(f"/api/v1/sessions/{sid}", json={"include_photos": "woods"}, headers=app)
    assert bad.status_code == 400


# --- kept answers, run links, the log ---------------------------------------------------

def test_a_kept_observation_answer_is_never_served_for_another_set_of_photos(conn, tmp_path):
    # iNat 555's photos are red 247 and, here, a habitat photo.
    client = make_app(conn, tmp_path, StubTagger(), seeded=False, site={9002: png(HABITAT)})
    first = client.post("/api/v1/identify/inat/555", json={}, headers=SERVER).json()
    assert first["cached"] is False and first["photo_types"]["left_out"] == [1]
    anyway = client.post("/api/v1/identify/inat/555", json={"include": [1]},
                         headers=SERVER).json()
    assert anyway["cached"] is False, "the answer kept without photo 2 is not this one"
    assert anyway["photo_types"]["included"] == [1] and anyway["ranks"] != first["ranks"]
    again = client.post("/api/v1/identify/inat/555", json={"include": [1]}, headers=SERVER).json()
    assert again["cached"] is True and again["ranks"] == anyway["ranks"]
    plain = client.post("/api/v1/identify/inat/555", json={}, headers=SERVER).json()
    assert plain["cached"] is True and plain["photo_types"]["left_out"] == [1]
    kept = client.get("/api/v1/answers/inat/555", params={"include": "1"}, headers=SERVER).json()
    assert kept["photo_types"]["included"] == [1]
    # The site's "from a link" answers keep apart the same way.
    site = client.post("/api/identify/observation", json={"source": "inat", "id": 555,
                                                          "models": "m1/nearest+mean"}).json()
    assert site["cached"] is True
    assert site["results"][0]["photo_types"]["left_out"] == [1]
    site_in = client.post("/api/identify/observation",
                          json={"source": "inat", "id": 555, "models": "m1/nearest+mean",
                                "include": [1]}).json()
    assert site_in["results"][0]["photo_types"]["included"] == [1]
    keys = {r["answer_key"] for r in rows(client.app) if r["source"] == "inat"}
    types = "types=" + StubTagger().fingerprint()[:8]
    assert any("include=1," in k and k.endswith(types) for k in keys)
    assert any("include" not in k and k.endswith(types) for k in keys)


def test_without_the_tagger_kept_answers_keep_their_old_keys(conn, tmp_path, monkeypatch):
    monkeypatch.delenv(phototypes.SETTING, raising=False)
    client = make_app(conn, tmp_path, True, seeded=False)
    client.post("/api/v1/identify/inat/555", json={"include": [1]}, headers=SERVER)
    [row] = [r for r in rows(client.app) if r["source"] == "inat"]
    assert row["answer_key"].endswith(":date=True,place=True,top_k=5")


def test_an_observation_counted_anyway_gets_its_own_run_link(conn, tmp_path):
    store = runs.RunStore(runs.DirObjects(tmp_path / "bucket"), write=lambda job: job(),
                          note=lambda s: None, index=runs.OwnerIndex(tmp_path / "runs.sqlite"))
    client = make_app(conn, tmp_path, StubTagger(), seeded=False, run_store=store,
                      site={9002: png(HABITAT)})
    ask = {"source": "inat", "id": 555, "models": "m1/nearest+mean"}
    plain = client.post("/api/identify/observation", json=ask).json()
    anyway = client.post("/api/identify/observation", json={**ask, "include": [1]}).json()
    assert plain["run"]["id"] != anyway["run"]["id"]
    kept = client.get(f"/api/runs/{anyway['run']['id']}").json()
    assert kept["results"][0]["photo_types"]["included"] == [1]
    assert client.post("/api/identify/observation", json=ask).json()["run"] == plain["run"]
