"""GET /api/v1/taxon-counts and the v1 credential guard (docs/api.md §1, §5, §7)."""

from fastapi.testclient import TestClient
from test_api import open_limits, post_photos
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import ISSUER, PUBLIC_KEY, SECRET, HERE, sign_in

from mycomap_vision import apiv1
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig

KEY = "k" * 40
AUTH = {"Authorization": f"Bearer {KEY}"}


def v1_app(conn, tmp_path, taxon_counts=True, signin=None, server_key=KEY):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), background=False, signin=signin,
                     server_key=server_key, taxon_counts=taxon_counts)
    return TestClient(app, follow_redirects=False)


def test_taxon_counts_are_the_served_index_reference_records(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    res = client.get("/api/v1/taxon-counts", headers=AUTH)
    assert res.status_code == 200
    doc = res.json()
    assert doc["counts_are"] == "reference_records"
    assert doc["species"] == {"A x": 4, "B y": 4}
    assert doc["genera"] == {"A": 4, "B": 4}
    assert doc["families"] == {"F": 8}
    assert doc["totals"] == {"records": 8, "species": 2, "genera": 2, "families": 1}
    assert doc["model"]["name"] == "MycoMap Vision" and "backbone" not in doc["model"]
    # The same number an identification shows on each candidate.
    [result] = post_photos(client, [247], "m1/nearest+mean").json()["results"]
    for cand in result["ranks"]["species"]:
        assert doc["species"][cand["name"]] == cand["reference_records"]


def test_taxon_counts_need_mycomap_orgs_key(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    for headers in ({}, {"Authorization": "Bearer " + "x" * 40}, {"Authorization": KEY}):
        res = client.get("/api/v1/taxon-counts", headers=headers)
        assert res.status_code == 401, headers
        assert res.json()["code"] == "invalid_token"
        assert res.headers["cache-control"] == "no-store"
    assert client.get("/api/v1/taxon-counts", headers={"X-API-Key": KEY}).status_code == 200


def test_no_v1_path_is_open_without_a_credential(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    for path in ("/api/v1/", "/api/v1/no-such-route", "/api/v1/sessions"):
        assert client.get(path).status_code == 401, path
    assert client.get("/api/v1/no-such-route", headers=AUTH).status_code == 404


def test_a_short_or_missing_server_key_accepts_nothing(conn, tmp_path):
    client = v1_app(conn, tmp_path, server_key="short")
    assert client.get("/api/v1/taxon-counts", headers={"Authorization": "Bearer short"}
                      ).status_code == 401
    assert apiv1.Authenticator(None).server_key is None


def test_v1_answers_its_credential_even_when_the_site_needs_sign_in(conn, tmp_path):
    signin = SigninConfig(mode="all", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    client = v1_app(conn, tmp_path, signin=signin)
    assert client.get("/api/v1/taxon-counts", headers=AUTH).status_code == 200
    # No redirect to the sign-in page: a JSON refusal a program can read.
    res = client.get("/api/v1/taxon-counts")
    assert res.status_code == 401 and res.json()["code"] == "invalid_token"
    # A site session is not a v1 credential.
    sign_in(client)
    assert client.get("/api/v1/taxon-counts").status_code == 401
    # The rest of the site still needs the sign-in.
    assert client.get("/api/stats").status_code == 200


def test_taxon_counts_answer_304_when_unchanged(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    first = client.get("/api/v1/taxon-counts", headers=AUTH)
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "private, max-age=3600"
    again = client.get("/api/v1/taxon-counts", headers={**AUTH, "If-None-Match": etag})
    assert again.status_code == 304 and again.headers["etag"] == etag


def test_taxon_counts_are_off_unless_turned_on(conn, tmp_path):
    client = v1_app(conn, tmp_path, taxon_counts=False)
    assert client.get("/api/v1/taxon-counts", headers=AUTH).status_code == 404


def test_a_record_without_a_name_at_a_rank_is_not_counted_there():
    doc = apiv1.taxon_counts({"species": {"A x": 2, None: 3, "": 1},
                              "genus": {"A": 2, None: 4}, "family": {}},
                             records=6, model={}, nightly_version=None)
    assert doc["species"] == {"A x": 2}
    assert doc["genera"] == {"A": 2}
    assert doc["families"] == {}
    assert doc["totals"] == {"records": 6, "species": 1, "genera": 1, "families": 0}
