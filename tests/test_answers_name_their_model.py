"""Every answer is associated with a model (Steve, 2026-11-24): whatever a person or a caller
sees of an answer says which model gave it, so v1's and v2's answers are never confused. API
v1's `model` block carries `label`, the model's own public name ("MycoMap Vision v1",
"MycoMap Vision v2 (beta)"), the same whichever model is the default, card or no card; a named
model keeps its own `name` even as the default; the site's answers carry the model they came
from (the page names it), and so do kept runs, their link cards and the identification log.
v1's `name` stays "MycoMap Vision" (unchanged). Synthetic embeddings under the real v1 and v2
ids, so the public names are the real ones."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, post_photos
from test_api_v1_by_observation import fetcher
from test_api_v1_identify import FakeOrg, SERVER_KEY, bearer, identify
from test_api_v1_sessions import APP, add
from test_models_and_scoreboard import Const
from test_signin import HERE, PUBLIC_KEY
from test_v2_beta_prefixes import Log, embedded

from mycomap_vision import apiv1, identifylog as il, linkcards, runs
from mycomap_vision.api import create_app

REPO = Path(__file__).resolve().parents[1]
V1 = "bioclip-2-336-ft-20261018-171420"
V2 = apiv1.V2_BETA
SERVER = bearer(SERVER_KEY)
V1_LABEL, V2_LABEL = "MycoMap Vision v1", "MycoMap Vision v2 (beta)"


def server(conn, tmp_path, log=None, site=None):
    root = embedded(conn, tmp_path, (V1, V2))
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><head><title>site</title></head>"
                                     "<body></body></html>", encoding="utf-8")
    store = runs.RunStore(runs.DirObjects(tmp_path / "bucket"), write=lambda job: job(),
                          note=lambda s: None, index=runs.OwnerIndex(tmp_path / "runs.sqlite"))
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", root,
        backbone_loader=lambda n: Const(n, 0.4 if n == V2 else 0.0),
        limits=open_limits(allowed_backbones={V1, V2}, default_backbone=V1,
                           api_v1_backbones={V1, V2}, site_default_backbone=site, max_models=2),
        background=False, server_key=SERVER_KEY, api_v1=True, sessions=True,
        by_observation=True, taxon_counts=True, observation_fetcher=fetcher(),
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(), fungus_gate=None,
        identify_log=log if log is not None else Log(), run_store=store, web_dist=dist,
        note=lambda s: None))
    return client, store


def labels(*blocks) -> list:
    return [b["model"].get("label") for b in blocks]


# --- API v1: the app and mycomap.org --------------------------------------------------------

def test_every_api_v1_answer_names_its_model_for_v1_and_for_v2(conn, tmp_path):
    log = Log()
    client, _ = server(conn, tmp_path, log=log)
    for model, label, name in ((None, V1_LABEL, "MycoMap Vision"), (V2, V2_LABEL, V2_LABEL)):
        ask = {"model": model} if model else {}
        got = identify(client, SERVER, **ask).json()
        app = identify(client, APP, **ask).json()
        by_obs = client.post("/api/v1/identify/inat/555", json=ask, headers=SERVER).json()
        kept = client.get("/api/v1/answers/inat/555", params=ask, headers=SERVER).json()
        opened = client.post("/api/v1/sessions", json=ask, headers=APP)
        assert opened.status_code == 201, opened.text
        sid = opened.json()["session_id"]
        photo = add(client, sid, 247, "1").json()
        state = client.get(f"/api/v1/sessions/{sid}", headers=APP).json()
        blocks = [got, app, by_obs, kept, opened.json(), photo["combined"], state["combined"]]
        assert labels(*blocks) == [label] * len(blocks), (model, labels(*blocks))
        # `name` as before for v1; v2 by its own name.
        assert {b["model"]["name"] for b in blocks} == {name}
        # No card for v2 until a release includes it; the label says which model all the same.
        if model == V2:
            assert {b["model"]["version"] for b in blocks} == {None}
        for b in blocks:
            assert "bioclip" not in json.dumps(b["model"])          # no ids in a v1 answer
    # The default's own description and the taxon counts name it too.
    assert client.get("/api/v1/models", headers=SERVER).json()["models"][0]["label"] == V1_LABEL
    assert client.get("/api/v1/taxon-counts", headers=SERVER).json()["model"]["label"] == V1_LABEL
    # The identification log keeps the model of every answer on every path.
    for path in (il.V1_IDENTIFY, il.V1_OBSERVATION, il.V1_ANSWERS, il.APP_SESSION):
        assert {m.split("/")[0] for m in log.models(path)} == {V1, V2}, path


def test_a_named_model_keeps_its_own_name_as_the_default():
    # Stage 2 makes v2 the default: its answers must not read like v1's "MycoMap Vision".
    assert apiv1.public_name(V2, default=V2) == V2_LABEL
    assert apiv1.public_name(V2, default=V2, site_default=V2) == "MycoMap Vision v2"
    assert apiv1.public_name(V1, default=V1) == "MycoMap Vision"         # v1, as always
    full = {"preload": {"models": [f"{V1}/nearest", f"{V2}/nearest"]}}
    assert apiv1.public_health(full, V2, [V1, V2])["preload"]["models"] == ["model 2", V2_LABEL]
    assert apiv1.public_health(full, V1, [V1, V2])["preload"]["models"] == [
        "MycoMap Vision", V2_LABEL]


def test_the_server_and_the_website_label_each_model_alike():
    table = json.loads((REPO / "web" / "src" / "lib" / "modelLabels.fixture.json")
                       .read_text(encoding="utf-8"))
    assert len(table) >= 8
    for row in table:
        assert apiv1.model_label(row["backbone"], row["site_default"]) == row["label"], row
    assert apiv1.model_label("m1") is None and apiv1.model_label(None) is None


# --- the site, kept runs and their cards ---------------------------------------------------------

def test_every_site_answer_carries_the_model_that_gave_it(conn, tmp_path):
    log = Log()
    client, store = server(conn, tmp_path, log=log)
    one = post_photos(client, [247]).json()
    both = post_photos(client, [247], f"{V1}/nearest,{V2}/nearest").json()
    link = client.post("/api/identify/observation",
                       json={"source": "inat", "id": 555, "models": f"{V1}/nearest,{V2}/nearest"})
    assert link.status_code == 200, link.text
    again = client.post("/api/identify/observation",                     # the kept answer
                        json={"source": "inat", "id": 555, "models": f"{V1}/nearest,{V2}/nearest"})
    assert again.json()["cached"] is True
    assert [r["model"]["backbone"] for r in one["results"]] == [V1]
    for body in (both, link.json(), again.json()):
        assert [r["model"]["backbone"] for r in body["results"]] == [V1, V2]
    # A kept run's page shows each answer with its model.
    for body, want in ((one, [V1]), (both, [V1, V2]), (link.json(), [V1, V2])):
        run = client.get(f"/api/runs/{body['run']['id']}").json()
        assert [r["model"]["backbone"] for r in run["results"]] == want
    assert {m.split("/")[0] for m in log.models(il.SITE_UPLOAD)} == {V1, V2}
    assert {m.split("/")[0] for m in log.models(il.SITE_OBSERVATION)} == {V1, V2}


@pytest.mark.parametrize("models,label", [(f"{V1}/nearest", V1_LABEL),
                                          (f"{V2}/nearest", V2_LABEL),
                                          (f"{V2}/nearest,{V1}/nearest", V2_LABEL)])
def test_a_shared_answers_card_names_the_model_it_shows(conn, tmp_path, models, label):
    client, store = server(conn, tmp_path)
    rid = post_photos(client, [247], models).json()["run"]["id"]
    run = store.get(rid)
    assert linkcards.run_model_name(run) == label
    page = client.get(f"/r/{rid}", headers={"user-agent": "Twitterbot/1.0"}).text
    title = linkcards.run_title(linkcards.headline(linkcards.run_answer(run)), label)
    assert title.startswith(label) and f'content="{title}"' in page.replace("&#x27;", "'")


def test_the_card_image_line_names_the_model_and_is_unchanged_without_one():
    h = linkcards.Headline("settled", "species", "Pluteus thomsonii", 0.9)
    assert linkcards.card_lines(h, 3, V1_LABEL)[0] == "MycoMap Vision v1 says, from 3 photos"
    assert linkcards.card_lines(h, 3)[0] == "MycoMap Vision says, from 3 photos"
    for kind in ("look_alike", "unsettled", "not_fungus", "none"):
        k = linkcards.Headline(kind, "genus", "Pluteus", 0.4)
        assert linkcards.card_lines(k, 2, V2_LABEL)[0].startswith(("MycoMap Vision v2 (beta)",
                                                                   "Identified by MycoMap "
                                                                   "Vision v2 (beta)"))
        assert linkcards.run_title(k, V2_LABEL).count(V2_LABEL) == 1
        assert linkcards.run_title(k) == linkcards.run_title(k, None)
    assert linkcards.run_model_name({"models": []}) is None
    assert linkcards.run_model_name({"models": [{"result": {"model": {"backbone": "m1"}}}]}) is None
