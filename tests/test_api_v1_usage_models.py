"""GET /api/v1/models and GET /api/v1/usage (docs/api.md §2): the model answering, by its
public name only, and a credential's own use this month, which reading never spends."""

import json

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_api_v1_identify import DEV_KEY, FakeOrg, SERVER_KEY, app_token, bearer, identify
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, quotas as q
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

def member() -> dict:
    """A member's app token, minted when used: tokens live 60 s, and one minted at import
    expired before these tests ran in a long, busy full-suite run."""
    return bearer(app_token(sub="u_member0007", limit={"monthly": 300}))


def app(conn, tmp_path, mode="count", org=None, api_v1=True):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(), background=False, server_key=SERVER_KEY, api_v1=api_v1,
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=org or FakeOrg(),
        fungus_gate=None, note=lambda s: None,
        quotas=q.Quotas(tmp_path / "usage.sqlite", mode=mode)))


# --- /models --------------------------------------------------------------------------

def test_models_names_the_model_by_its_public_name_with_its_reference_and_limits(conn,
                                                                                   tmp_path):
    res = app(conn, tmp_path).get("/api/v1/models", headers=member())
    assert res.status_code == 200 and res.headers["cache-control"] == "private, max-age=300"
    body = res.json()
    [model] = body["models"]
    assert model["name"] == "MycoMap Vision" and model["default"] is True
    assert model["reference"] == {"records": 8, "species": 2, "photos": 8}
    assert model["ranks"] == ["species", "genus", "family", "order", "class", "phylum"]
    assert model["confident_at"] == 0.70 and set(model) >= {"version", "data", "calibrated_on"}
    assert body["limits"] == {"max_photos": 10, "max_file_mb": 25, "max_request_mb": 80,
                              "top_k_max": 10, "photos_per_minute": 20}
    assert body["features"]["identify"] is True and body["features"]["sessions"] is False


def test_an_uncalibrated_model_claims_no_likely_coverage(conn, tmp_path):
    [model] = app(conn, tmp_path).get("/api/v1/models", headers=member()).json()["models"]
    assert model["likely_coverage"] == {"species": None, "genus": None, "family": None}


def test_models_never_names_a_backbone_or_method(conn, tmp_path):
    text = json.dumps(app(conn, tmp_path).get("/api/v1/models", headers=member()).json())
    for internal in ('"m1', "nearest", "backbone", "method"):
        assert internal not in text, internal


# --- /usage ---------------------------------------------------------------------------

def test_a_member_reads_their_months_use_and_reading_costs_nothing(conn, tmp_path):
    client = app(conn, tmp_path)
    first = client.get("/api/v1/usage", headers=member())
    assert first.status_code == 200 and first.headers["cache-control"] == "no-store"
    body = first.json()
    assert body["quotas"] == "count" and body["pool"] == "member"
    assert body["usage"]["used"] == 0 and body["usage"]["limit"] == 300
    assert body["usage"]["resets_at"].endswith("Z")
    assert identify(client, member()).status_code == 200
    for _ in range(3):
        assert client.get("/api/v1/usage", headers=member()).json()["usage"]["used"] == 1


def test_mycomap_orgs_server_key_has_no_pool(conn, tmp_path):
    body = app(conn, tmp_path).get("/api/v1/usage", headers=bearer(SERVER_KEY)).json()
    assert body == {"usage": None, "quotas": "count", "pool": "none", "limit_source": None}


def test_a_developer_key_counts_in_its_owners_pool_unless_it_has_an_override(conn, tmp_path):
    org = FakeOrg()
    client = app(conn, tmp_path, org=org)
    body = client.get("/api/v1/usage", headers=bearer(DEV_KEY)).json()
    assert (body["pool"], body["limit_source"], body["usage"]["limit"]) == ("member", "plan", 2000)
    override = "vision_" + "o" * 43
    org.answers[override] = {**org.answers[DEV_KEY], "keyId": "43", "limit": {"monthly": 5},
                             "limit_source": "override"}
    body = client.get("/api/v1/usage", headers=bearer(override)).json()
    assert (body["pool"], body["limit_source"], body["usage"]["limit"]) == ("key", "override", 5)


def test_with_quotas_off_usage_is_null(conn, tmp_path):
    body = app(conn, tmp_path, mode="off").get("/api/v1/usage", headers=member()).json()
    assert body["usage"] is None and body["quotas"] == "off"


# --- both -----------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/api/v1/models", "/api/v1/usage"])
def test_both_need_a_credential(conn, tmp_path, path):
    res = app(conn, tmp_path).get(path)
    assert (res.status_code, res.json()["code"]) == (401, "invalid_token")


@pytest.mark.parametrize("path", ["/api/v1/models", "/api/v1/usage"])
def test_both_are_off_with_v1(conn, tmp_path, path):
    res = app(conn, tmp_path, api_v1=False).get(path, headers=bearer(SERVER_KEY))
    assert res.status_code == 404


def test_the_outline_marks_both_available_when_v1_is_on(conn, tmp_path):
    outline = app(conn, tmp_path).get("/api/docs/v1.json").json()
    status = {b["path"]: b["status"] for s in outline["sections"] for b in s.get("blocks") or []
              if b.get("type") == "endpoint"}
    assert status["/models"] == status["/usage"] == "available"


def test_models_state_the_likely_sets_achieved_coverage_not_their_target(conn, tmp_path):
    from mycomap_vision import evaluate
    client = app(conn, tmp_path)
    evaluate.compare(conn, ["m1"], ["nearest+mean"], embeddings_root=tmp_path / "emb",
                     log=lambda s: None)
    row_id, report = conn.execute("select id, report_json from eval_runs").fetchone()
    report = json.loads(report)
    sp = report["all_photos"]["calibration"]["species"]["sets"]
    sp.pop("target", None)                      # a fit stored before targets were kept apart
    sp["coverage"], sp["crosscheck"]["coverage"] = 0.9, 0.5513
    conn.execute("update eval_runs set report_json = ? where id = ?",
                 (json.dumps(report), row_id))
    conn.commit()
    [model] = client.get("/api/v1/models", headers=member()).json()["models"]
    assert model["likely_coverage"]["species"] == 0.55
    assert set(model["likely_coverage"]) == {"species", "genus", "family"}


def test_likely_coverage_reads_the_crosscheck_of_a_fit_stored_with_its_target():
    old = {"sets": {"species": {"coverage": 0.9, "crosscheck": {"coverage": 0.6}}}}
    assert apiv1.likely_coverage(old) == {"species": 0.6, "genus": None, "family": None}
    assert apiv1.likely_coverage(None) == {"species": None, "genus": None, "family": None}
