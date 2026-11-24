"""Stage 1 of the default switch (Steve, 2026-11-24): site visitors get v2 by default ("lets
go with v2") and "mobile app should move with the site", while mycomap.org (its server key:
auto-validation, record pages, kept answers, taxon counts), developer keys, the nightly
advance predictions, How it works and Human Feedback stay on v1 (MV_DEFAULT_BACKBONE) until
a later, separate switch. MV_SITE_DEFAULT_BACKBONE sets the site's and the app's default;
unset, everything is as before. v2's label follows the setting: "(beta)" only while it is
opt-in (Steve, 2026-11-24: "once we move to v2 as default it will no longer be beta").
Synthetic models only: v2 is a stand-in under v2's id (it sorts BEFORE v1's stand-in)."""

import json

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, post_photos
from test_api_v1_by_observation import fetcher
from test_api_v1_identify import DEV_KEY, FakeOrg, SERVER_KEY, bearer, identify
from test_api_v1_sessions import APP, add
from test_models_and_scoreboard import Const
from test_signin import HERE, PUBLIC_KEY
from test_v2_beta_prefixes import (MISSING, V1, V2, Log, embedded, scored, server as beta_server,
                                   without_measured)

from mycomap_vision import apiv1, identifylog as il, manifest
from mycomap_vision.api import SITE_DEFAULT_META, create_app, with_site_default
from mycomap_vision.guards import Limits

SERVER = bearer(SERVER_KEY)
DEV = bearer(DEV_KEY)
APP_IDENTIFY = APP                      # an app token, minted fresh per call


def server(conn, tmp_path, site=None, default=V1, api_v1=None, allowed=(V1, V2), names=(V1, V2),
           preload=None, notes=None, log=None, web_dist=None):
    """A two-model server with the API, the app's sessions, by-observation and taxon counts
    on; `site` is MV_SITE_DEFAULT_BACKBONE."""
    import time
    root = embedded(conn, tmp_path, names)
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", root,
        backbone_loader=lambda n: Const(n, 0.4 if n == V2 else 0.0),
        limits=open_limits(allowed_backbones=set(allowed), default_backbone=default,
                           api_v1_backbones=api_v1, site_default_backbone=site),
        background=False, server_key=SERVER_KEY, api_v1=True, sessions=True,
        by_observation=True, taxon_counts=True, observation_fetcher=fetcher(),
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(), fungus_gate=None,
        preload=preload, identify_log=log if log is not None else Log(),
        note=(notes.append if notes is not None else lambda s: None), web_dist=web_dist))
    for _ in range(400):
        if client.get("/api/health").json()["preload"]["state"] != "loading":
            break
        time.sleep(0.05)
    return client


def backbones(log: Log, path: str) -> set[str]:
    return {m.split("/")[0] for m in log.models(path)}


def app_session(client):
    s = client.post("/api/v1/sessions", json={}, headers=APP)
    assert s.status_code == 201, s.text
    return s


STAGE_1 = dict(site=V2, default=V1, api_v1={V1})


# --- who gets which model ---------------------------------------------------------------

def test_with_the_site_default_v2_the_site_and_the_app_get_v2_and_mycomap_org_keeps_v1(
        conn, tmp_path):
    log = Log()
    client = server(conn, tmp_path, **STAGE_1, log=log)
    # The site: the Identify page's no-model upload and "from a link".
    assert post_photos(client, [247]).status_code == 200
    site_link = client.post("/api/identify/observation", json={"source": "inat", "id": 555})
    assert site_link.status_code == 200, site_link.text
    # The app: its sessions and its identify, with an app token, naming no model.
    s = app_session(client)
    assert add(client, s.json()["session_id"], 247, "1").status_code == 200
    assert identify(client, APP_IDENTIFY).status_code == 200
    # mycomap.org's server key: identify, by-observation, kept answers, taxon counts.
    assert identify(client, SERVER).status_code == 200
    assert client.post("/api/v1/identify/inat/556", headers=SERVER).status_code == 200
    assert client.get("/api/v1/answers/inat/556", headers=SERVER).status_code == 200
    counts = client.get("/api/v1/taxon-counts", headers=SERVER).json()
    # A developer key (the validation tool) stays with mycomap.org's model too.
    assert identify(client, DEV).status_code == 200
    assert backbones(log, il.SITE_UPLOAD) == {V2}
    assert backbones(log, il.SITE_OBSERVATION) == {V2}
    assert backbones(log, il.APP_SESSION) == {V2}
    assert [m.split("/")[0] for p, m in log.rows if p == il.V1_IDENTIFY] == [V2, V1, V1]
    assert backbones(log, il.V1_OBSERVATION) == backbones(log, il.V1_ANSWERS) == {V1}
    assert counts["totals"]["records"] == 8                       # v1's reference (v2: 7)
    # /api/v1/models answers for the credential asking.
    assert (client.get("/api/v1/models", headers=SERVER).json()["models"][0]["name"]
            == "MycoMap Vision")
    assert (client.get("/api/v1/models", headers=APP_IDENTIFY).json()["models"][0]["name"]
            == "MycoMap Vision v2")


def test_the_identify_page_is_told_to_preselect_the_site_default(conn, tmp_path):
    client = server(conn, tmp_path, **STAGE_1)
    models = client.get("/api/models").json()
    assert models["site_default_backbone"] == V2 and models["site_default_ready"] is True
    # How it works reads default_backbone: still v1.
    assert models["default_backbone"] == V1 and models["default_ready"] is True


def test_the_app_may_use_the_site_default_while_mycomap_orgs_key_may_not(conn, tmp_path):
    client = server(conn, tmp_path, **STAGE_1)
    for res in (identify(client, SERVER, model=V2), identify(client, DEV, model=V2),
                client.post("/api/v1/identify/inat/555", json={"model": V2}, headers=SERVER)):
        assert res.status_code == 400 and res.json()["code"] == "model_not_offered", res.text
    assert identify(client, APP_IDENTIFY, model=V2).status_code == 200
    assert identify(client, APP_IDENTIFY, model=V1).status_code == 200     # v1 still listed
    full = client.get("/api/health", headers=SERVER).json()["default_backbone"]
    assert full["api_v1"] == [V1]
    assert full["site"] == {"backbone": V2, "from": "MV_SITE_DEFAULT_BACKBONE", "ready": True,
                            "refused": None, "api_v1_app": sorted([V1, V2])}
    assert "conflicts" not in full


def clockless(body: dict) -> dict:
    """`body` without the wall-clock second its kept answer was given at (`answered.at`,
    checked to be there), the one field besides MEASURED that differs between two runs."""
    answered = body.get("answered")
    if isinstance(answered, dict):
        assert isinstance(answered.pop("at"), str)
    return body


def test_mycomap_orgs_answers_are_byte_identical_with_the_site_default_set(conn, tmp_path):
    def asks(client):
        out = {"identify": identify(client, SERVER).json(),
               "dev": identify(client, DEV).json(),
               "by_observation": client.post("/api/v1/identify/inat/555",
                                             headers=SERVER).json(),
               "kept": client.get("/api/v1/answers/inat/555", headers=SERVER).json(),
               "models": client.get("/api/v1/models", headers=SERVER).json(),
               "counts": client.get("/api/v1/taxon-counts", headers=SERVER).json()}
        return out
    before = asks(server(conn, tmp_path, default=V1, api_v1={V1}))
    conn.close()
    other = tmp_path / "b"
    other.mkdir()
    conn_b = manifest.connect(other / "manifest.sqlite")
    after = asks(server(conn_b, other, **STAGE_1))
    conn_b.close()
    for name in ("identify", "dev", "by_observation", "kept"):
        a, b = (clockless(without_measured(x[name], "answer")) for x in (before, after))
        assert a == b, name
    for x in (before, after):                         # when the counts were made: a clock
        assert isinstance(x["counts"].pop("generated_at"), str)
    assert before["models"] == after["models"] and before["counts"] == after["counts"]
    assert before["identify"]["model"]["name"] == "MycoMap Vision"
    assert before["identify"]["model"]["version"] == after["identify"]["model"]["version"]


# --- the paths that stay on MV_DEFAULT_BACKBONE ------------------------------------------

def test_how_it_works_stays_on_mycomap_orgs_model_with_the_site_default_set(conn, tmp_path):
    client = server(conn, tmp_path, **STAGE_1)
    scored(conn, V1, "20261001-000000-aaaaaa")
    scored(conn, V2, "20261002-000000-bbbbbb")        # newer, and sorts first
    assert client.get("/api/how-good").json()["run"]["backbone"] == V1


def test_a_human_feedback_reask_stays_on_mycomap_orgs_model_with_the_site_default_set(
        conn, tmp_path):
    from test_feedback_api import FakeOrg as Roles, as_user, item_id
    from test_signin import ISSUER, SECRET

    from mycomap_vision import feedback as fb
    from mycomap_vision.signin import SigninConfig
    root = embedded(conn, tmp_path)
    fstore = fb.Store(tmp_path / "state" / "human-feedback.sqlite")
    fstore.seed([{"item_type": "photos_unlike_label", "subject": "100", "rule": "M1",
                  "record_keys": ["100"], "current": "A x", "projects": ["p1"]}], "test")
    signin = SigninConfig(mode="identify", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(allowed_backbones={V1, V2}, default_backbone=V1,
                           site_default_backbone=V2),
        signin=signin, feedback_store=fstore, feedback_org=Roles(), background=False,
        fungus_gate=None, note=lambda s: None), follow_redirects=False)
    iid = item_id(fstore, "photos_unlike_label")
    as_user(client, "admin-1")
    client.post(f"/api/feedback/items/{iid}/decision", json={"decision": "rename", "value": "A x"})
    vision = client.post(f"/api/feedback/items/{iid}/refresh").json()["refreshed"][0]["vision"]
    assert vision["available"] and vision["model"].split("/")[0] == V1


def test_advance_predictions_stay_on_mycomap_orgs_model_with_the_site_default_set(tmp_path):
    import time

    from test_nightly import QUIET, RID, FakeOutside, seed_rows
    from test_nightly_predictions import pending
    from test_prospective import FakeFetcher
    from test_v2_beta_default_nightly import V1 as N1, V2 as N2, two_model_box

    from conftest import inat_obs
    from mycomap_vision import nightly
    box = two_model_box(tmp_path)
    layer = nightly.prepare(box, **QUIET)
    photos = FakeFetcher({"400": inat_obs(400, photos=[(0, 4000, "cc0", "x")])})
    client = TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name),
        limits=open_limits(allowed_backbones={N1, N2}, default_backbone=N1,
                           site_default_backbone=N2),
        background=False, layer=layer, nightly_fetch=lambda: (seed_rows(), "g"),
        nightly_fetch_pending=lambda: ([pending(400)], "g"),
        nightly_steps=FakeOutside({300: 130}).steps(), photo_fetcher=lambda: photos,
        nightly_settings=nightly.Settings(predict=300),
        preload=[f"{N2}/nearest", f"{N1}/nearest"], note=lambda s: None))
    for _ in range(400):
        if client.get("/api/health").json()["preload"]["state"] != "loading":
            break
        time.sleep(0.05)
    report = client.app.state.run_nightly()
    assert report["predictions"]["model"] == f"{N1}/nearest"


# --- refused, never another model ---------------------------------------------------------

def test_a_site_default_that_isnt_ready_is_refused_never_replaced(conn, tmp_path):
    log, notes = Log(), []
    client = server(conn, tmp_path, site=MISSING, default=V1, api_v1={V1},
                    allowed=(V1, V2, MISSING), log=log, notes=notes)
    assert any(f"[default] MV_SITE_DEFAULT_BACKBONE={MISSING} is not served here (it has no "
               "embeddings here)" in n for n in notes)
    models = client.get("/api/models").json()
    assert models["site_default_backbone"] == MISSING and models["site_default_ready"] is False
    site = post_photos(client, [247])
    assert site.status_code == 503 and "MV_SITE_DEFAULT_BACKBONE" in site.json()["detail"]
    for res in (client.post("/api/v1/sessions", json={}, headers=APP),
                identify(client, APP_IDENTIFY)):
        assert res.status_code == 503 and res.json()["code"] == "not_served", res.text
        assert MISSING not in res.text
    assert log.rows == []
    health = client.get("/api/health", headers=SERVER).json()["default_backbone"]
    assert health["site"]["ready"] is False
    assert health["site"]["refused"] == "it has no embeddings here"
    # mycomap.org's path is untouched; a person may still choose a model on the site.
    assert identify(client, SERVER).status_code == 200
    assert backbones(log, il.V1_IDENTIFY) == {V1}
    assert post_photos(client, [247], f"{V2}/nearest").status_code == 200


def test_the_trap_1_combination_is_refused_at_startup(conn, tmp_path):
    notes = []
    client = server(conn, tmp_path, default=V2, api_v1={V1}, notes=notes)
    assert any(n.startswith(f"[default] REFUSED: MV_DEFAULT_BACKBONE={V2} is not in "
                            "MV_API_V1_BACKBONES") for n in notes), notes
    health = client.get("/api/health", headers=SERVER).json()["default_backbone"]
    assert len(health["conflicts"]) == 1 and "MV_API_V1_BACKBONES" in health["conflicts"][0]
    # The public health never says it.
    assert "conflicts" not in json.dumps(client.get("/api/health").json())
    res = identify(client, SERVER)
    assert res.status_code == 503 and res.json()["code"] == "not_served"


@pytest.mark.parametrize("settings,expected", [
    (dict(default_backbone="v2", api_v1_backbones={"v1"}), ["MV_DEFAULT_BACKBONE=v2 is not in "
                                                             "MV_API_V1_BACKBONES"]),
    (dict(default_backbone="v0"), ["MV_DEFAULT_BACKBONE=v0 is not in MV_ALLOWED_BACKBONES"]),
    (dict(default_backbone="v1", site_default_backbone="v0"),
     ["MV_SITE_DEFAULT_BACKBONE=v0 is not in MV_ALLOWED_BACKBONES"]),
    (dict(site_default_backbone="v2"), ["MV_SITE_DEFAULT_BACKBONE is set but "
                                        "MV_DEFAULT_BACKBONE is not"]),
    (dict(default_backbone="v1", api_v1_backbones={"v1"}, site_default_backbone="v2"), []),
    (dict(default_backbone="v1", api_v1_backbones={"v1"}), []),
    (dict(), []),
])
def test_every_callers_default_must_be_one_it_may_use(settings, expected):
    limits = open_limits(allowed_backbones={"v1", "v2"}, **settings)
    got = limits.default_conflicts()
    assert len(got) == len(expected) and all(g.startswith(e) for g, e in zip(got, expected)), got


def test_the_site_default_goes_to_the_site_and_app_tokens_only():
    limits = open_limits(allowed_backbones={"v1", "v2"}, default_backbone="v1",
                         site_default_backbone="v2")
    ready = ["v2", "v1"]
    assert [limits.default_for(k, ready) for k in ("site", "app", "server", "key")] == [
        "v2", "v2", "v1", "v1"]
    assert limits.default_for("app", ["v1"]) is None             # refused, not replaced
    unset = open_limits(allowed_backbones={"v1", "v2"}, default_backbone="v1")
    assert {unset.default_for(k, ready) for k in ("site", "app", "server", "key")} == {"v1"}


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (" v2 ", "v2")])
def test_the_site_default_setting_reads_from_the_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MV_SITE_DEFAULT_BACKBONE", raising=False)
    else:
        monkeypatch.setenv("MV_SITE_DEFAULT_BACKBONE", value)
    assert Limits.from_settings().site_default_backbone == expected


# --- unset: as before ---------------------------------------------------------------------

def test_unset_the_site_default_changes_nothing(conn, tmp_path):
    log, notes = Log(), []
    client = beta_server(conn, tmp_path, default=V1, log=log, notes=notes)
    models = client.get("/api/models").json()
    assert "site_default_backbone" not in models and "site_default_ready" not in models
    full = client.get("/api/health", headers=SERVER).json()["default_backbone"]
    assert set(full) == {"backbone", "from", "ready", "refused", "api_v1"}
    assert not any("MV_SITE_DEFAULT_BACKBONE" in n or "REFUSED" in n for n in notes)
    assert post_photos(client, [247]).status_code == 200
    s = app_session(client)
    assert add(client, s.json()["session_id"], 247, "1").status_code == 200
    assert backbones(log, il.SITE_UPLOAD) == backbones(log, il.APP_SESSION) == {V1}
    assert identify(client, APP_IDENTIFY, model=V2).json()["code"] == "model_not_offered"
    shell = "<html><head><title>x</title></head></html>"
    assert with_site_default(shell, None) == shell


# --- the label follows the setting --------------------------------------------------------

def test_v2_is_the_beta_while_opt_in_and_plain_v2_once_it_is_the_site_default():
    assert apiv1.named_model(V2) == "MycoMap Vision v2 (beta)"
    assert apiv1.named_model(V2, site_default=V1) == "MycoMap Vision v2 (beta)"
    assert apiv1.named_model(V2, site_default=V2) == "MycoMap Vision v2"
    assert apiv1.public_name(V2, V1, site_default=V2) == "MycoMap Vision v2"
    assert apiv1.public_name(V1, V1, site_default=V2) == "MycoMap Vision"
    full = {"preload": {"models": [f"{V1}/nearest", f"{V2}/nearest"]}}
    assert apiv1.public_health(full, V1, [V1, V2], V2)["preload"]["models"] == [
        "MycoMap Vision", "MycoMap Vision v2"]
    assert apiv1.public_health(full, V1, [V1, V2])["preload"]["models"] == [
        "MycoMap Vision", "MycoMap Vision v2 (beta)"]


@pytest.mark.parametrize("site,name", [(V2, "MycoMap Vision v2"),
                                       (None, "MycoMap Vision v2 (beta)")])
def test_the_app_and_the_public_health_name_v2_as_the_setting_says(conn, tmp_path, site, name):
    client = server(conn, tmp_path, site=site, default=V1, api_v1={V1, V2},
                    preload=[f"{V2}/nearest", f"{V1}/nearest"])
    s = client.post("/api/v1/sessions", json={"model": V2}, headers=APP)
    assert s.status_code == 201, s.text
    # No card for v2 until a release includes it, default or not.
    assert s.json()["model"]["name"] == name and s.json()["model"]["version"] is None
    v2 = identify(client, APP_IDENTIFY, model=V2).json()["model"]
    assert v2["name"] == name and v2["version"] is None and v2["data"] is None
    assert client.get("/api/health").json()["preload"]["models"] == [name, "MycoMap Vision"]


def test_the_page_shell_carries_the_site_default_only_when_set(conn, tmp_path):
    def page(conn_, where, site):
        dist = where / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><html><head><title>site</title>"
                                         "</head><body></body></html>", encoding="utf-8")
        return server(conn_, where, site=site, web_dist=dist).get("/about").text
    unset = page(conn, tmp_path, None)
    conn.close()
    other = tmp_path / "b"
    other.mkdir()
    conn_b = manifest.connect(other / "manifest.sqlite")
    stage_1 = page(conn_b, other, V2)
    conn_b.close()
    tag = f'<meta name="{SITE_DEFAULT_META}" content="{V2}" />\n  '
    assert SITE_DEFAULT_META not in unset
    assert stage_1.count(tag) == 1 and stage_1.replace(tag, "") == unset
