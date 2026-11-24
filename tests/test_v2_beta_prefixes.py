"""Before the v2 opt-in beta opens (Steve, 2026-11-24): v2 is named "MycoMap Vision v2 (beta)";
the default model is a setting (MV_DEFAULT_BACKBONE) that every path naming no model uses,
refused rather than replaced while it isn't served; API v1 answers with an allow-list
(MV_API_V1_BACKBONES, default: the default model only); and no answer carries another
model's card. Synthetic models only: v2 here is a stand-in under v2's id, which sorts BEFORE
v1's stand-in, so name order alone would pick it."""

import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png, post_photos
from test_api_v1_by_observation import fetcher
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer, identify
from test_api_v1_sessions import APP, add
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, identifylog as il, modelcard
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

REPO = Path(__file__).resolve().parents[1]
V2 = apiv1.V2_BETA                    # "bioclip-2.5-336-ft-...": sorts first
V1 = "v1-ft"                          # the stand-in for v1: sorts after v2
MISSING = "v0-missing-ft"             # allowed, never embedded
SERVER = bearer(SERVER_KEY)


class Log:
    """The identification log, kept in memory: (path, model) per answer."""

    def __init__(self):
        self.rows = []

    def record(self, path, **fields):
        self.rows.append((path, fields.get("model")))

    def state(self):
        return {}

    def models(self, path):
        return [m for p, m in self.rows if p == path]


def embedded(conn, tmp_path, names=(V1, V2)):
    """The two-species set embedded for each name; v2 misses one record's only photo, so its
    reference holds 7 records against v1's 8 (taxon counts tell them apart)."""
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    for name in names:
        todo = photos_to_embed(conn, name, "large", store.location, "local")
        if name == V2:
            todo = todo[:-1]
        embed_photos(conn, store, Const(name, 0.4 if name == V2 else 0.0), todo, root / name,
                     log=lambda s: None)
    conn.commit()
    return root


def server(conn, tmp_path, default=None, api_v1=None, allowed=(V1, V2), preload=None,
           names=(V1, V2), notes=None, log=None):
    root = embedded(conn, tmp_path, names)
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", root,
        backbone_loader=lambda n: Const(n, 0.4 if n == V2 else 0.0),
        limits=open_limits(allowed_backbones=set(allowed), default_backbone=default,
                           api_v1_backbones=api_v1),
        background=False, server_key=SERVER_KEY, api_v1=True, sessions=True,
        by_observation=True, taxon_counts=True, observation_fetcher=fetcher(),
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(), fungus_gate=None,
        preload=preload, identify_log=log if log is not None else Log(),
        note=(notes.append if notes is not None else lambda s: None)))
    for _ in range(400):
        if client.get("/api/health").json()["preload"]["state"] != "loading":
            break
        time.sleep(0.05)
    return client


def open_session(client, **body):
    return client.post("/api/v1/sessions", json=body, headers=APP)


def every_default_path(client, log: Log) -> dict:
    """Asks each path that names no model once; which backbone answered, per path."""
    assert post_photos(client, [247]).status_code == 200                       # the site
    assert identify(client, SERVER).status_code == 200                         # API v1
    s = open_session(client)
    assert s.status_code == 201, s.text
    assert add(client, s.json()["session_id"], 247, "1").status_code == 200    # the app
    assert client.post("/api/v1/identify/inat/555", headers=SERVER).status_code == 200
    assert client.get("/api/v1/answers/inat/555", headers=SERVER).status_code == 200
    counts = client.get("/api/v1/taxon-counts", headers=SERVER).json()
    out = {path: {m.split("/")[0] for m in log.models(path)}
           for path in (il.SITE_UPLOAD, il.V1_IDENTIFY, il.APP_SESSION, il.V1_OBSERVATION,
                        il.V1_ANSWERS)}
    out["taxon_counts_records"] = counts["totals"]["records"]
    return out


# --- the beta's name -------------------------------------------------------------------

def test_v2_is_labelled_the_beta_and_v1_keeps_its_name():
    assert apiv1.public_name(V2, default=V1) == "MycoMap Vision v2 (beta)"
    assert apiv1.public_name(V1, default=V1) == "MycoMap Vision"
    assert apiv1.public_name("x-ft", default=V1) == apiv1.OTHER_MODEL_NAME
    # The public /api/health: the default "MycoMap Vision", v2 by its name, any other "model N".
    full = {"preload": {"models": [f"{V1}/nearest", f"{V2}/nearest", "x-ft/nearest"]}}
    body = apiv1.public_health(full, V1, [V1, V2, "x-ft"])
    assert body["preload"]["models"] == ["MycoMap Vision", "MycoMap Vision v2 (beta)", "model 3"]


def test_the_server_and_the_website_name_the_beta_alike():
    ts = (REPO / "web" / "src" / "lib" / "publicNames.ts").read_text(encoding="utf-8")
    assert re.search(r'export const V2_BETA = "([^"]+)";', ts)[1] == apiv1.V2_BETA
    named = re.search(r"NAMED_VERSIONS[^=]*= \{\s*\[V2_BETA\]: \{ name: \"([^\"]+)\"", ts)
    assert named and named[1] == apiv1.NAMED_MODELS[apiv1.V2_BETA]


# --- the default model ------------------------------------------------------------------

def test_the_default_setting_picks_v1_on_every_path_even_when_v2_sorts_first(conn, tmp_path):
    assert sorted([V1, V2])[0] == V2
    log = Log()
    # v2 preloaded FIRST: what was warm.models[0] (advance predictions, taxon counts).
    client = server(conn, tmp_path, default=V1, preload=[f"{V2}/nearest", "default"], log=log)
    assert client.get("/api/health").json()["preload"]["state"] == "ready"
    models = client.get("/api/models").json()
    assert models["default_backbone"] == V1 and models["default_ready"] is True
    got = every_default_path(client, log)
    assert got == {il.SITE_UPLOAD: {V1}, il.V1_IDENTIFY: {V1}, il.APP_SESSION: {V1},
                   il.V1_OBSERVATION: {V1}, il.V1_ANSWERS: {V1}, "taxon_counts_records": 8}
    # The public health calls the default "MycoMap Vision"; v2 by its own name.
    public = client.get("/api/health").json()
    assert public["preload"]["models"] == ["MycoMap Vision v2 (beta)", "MycoMap Vision"]
    full = client.get("/api/health", headers=SERVER).json()
    assert full["default_backbone"] == {"backbone": V1, "from": "MV_DEFAULT_BACKBONE",
                                        "ready": True, "refused": None, "api_v1": [V1]}


def test_unset_the_default_is_the_first_ready_name_as_before(conn, tmp_path):
    log, notes = Log(), []
    client = server(conn, tmp_path, preload=[f"{V2}/nearest"], log=log, notes=notes)
    assert "default_backbone" not in client.get("/api/models").json()
    got = every_default_path(client, log)
    assert got[il.SITE_UPLOAD] == got[il.V1_IDENTIFY] == {V2}
    assert got["taxon_counts_records"] == 7
    assert any("[default] MV_DEFAULT_BACKBONE is unset" in n for n in notes)


def test_a_named_default_that_isnt_ready_is_refused_not_replaced(conn, tmp_path):
    log, notes = Log(), []
    client = server(conn, tmp_path, default=MISSING, allowed=(V1, V2, MISSING),
                    preload=["default"], log=log, notes=notes)
    assert any(f"[default] MV_DEFAULT_BACKBONE={MISSING} is not served here (it has no "
               "embeddings here)" in n for n in notes)
    health = client.get("/api/health", headers=SERVER).json()
    assert health["preload"]["state"] == "failed"
    assert MISSING in health["preload"]["error"]
    assert health["default_backbone"]["ready"] is False
    assert health["default_backbone"]["refused"] == "it has no embeddings here"
    models = client.get("/api/models").json()
    assert models["default_backbone"] == MISSING and models["default_ready"] is False
    # Every path that names no model is refused; nothing answers in its place.
    assert post_photos(client, [247]).status_code == 503
    for res in (identify(client, SERVER), open_session(client),
                client.post("/api/v1/identify/inat/555", headers=SERVER),
                client.get("/api/v1/taxon-counts", headers=SERVER)):
        assert res.status_code == 503 and res.json()["code"] == "not_served", res.text
        assert MISSING not in res.text
    assert log.rows == []
    # A person may still choose a model on the site (the beta's opt-in).
    assert post_photos(client, [247], f"{V1}/nearest").status_code == 200


def test_a_named_default_that_isnt_allowed_says_so(conn, tmp_path):
    notes = []
    client = server(conn, tmp_path, default=MISSING, notes=notes)
    assert any("(it is not in MV_ALLOWED_BACKBONES)" in n for n in notes)
    assert identify(client, SERVER).json()["code"] == "not_served"


# --- API v1 during the beta ---------------------------------------------------------------

def test_api_v1_refuses_v2_by_default_and_serves_v1_to_a_request_naming_no_model(conn,
                                                                                   tmp_path):
    log = Log()
    client = server(conn, tmp_path, default=V1, log=log)
    for res in (identify(client, SERVER, model=V2), identify(client, bearer(app_token()),
                                                             model=f"{V2}/nearest"),
                open_session(client, model=V2),
                client.post("/api/v1/identify/inat/555", json={"model": V2}, headers=SERVER)):
        assert res.status_code == 400 and res.json()["code"] == "model_not_offered", res.text
        assert V2 not in res.json()["detail"]
    assert identify(client, SERVER, model=V1).status_code == 200
    plain = identify(client, SERVER)
    assert plain.status_code == 200 and plain.json()["model"]["name"] == "MycoMap Vision"
    assert {m.split("/")[0] for m in log.models(il.V1_IDENTIFY)} == {V1}
    # The site still offers v2 to anyone who chooses it.
    assert post_photos(client, [247], f"{V2}/nearest").status_code == 200


def test_an_allow_listed_v2_answers_through_api_v1_under_its_own_name(conn, tmp_path):
    client = server(conn, tmp_path, default=V1, api_v1={V1, V2})
    res = identify(client, SERVER, model=V2)
    assert res.status_code == 200, res.text
    assert res.json()["model"]["name"] == "MycoMap Vision v2 (beta)"


# --- no answer carries another model's card -----------------------------------------------

def test_no_answer_carries_another_models_card(conn, tmp_path):
    client = server(conn, tmp_path, default=V1, api_v1={V1, V2})
    v1 = identify(client, SERVER).json()["model"]
    assert v1["name"] == "MycoMap Vision" and set(v1["data"]) == {"snapshot", "records",
                                                                  "species", "genera"}
    v2 = identify(client, SERVER, model=V2).json()["model"]
    assert v2["version"] is None and v2["data"] is None
    s = open_session(client, model=V2).json()
    assert s["model"]["version"] is None and s["model"]["data"] is None
    by_obs = client.post("/api/v1/identify/inat/555", json={"model": V2}, headers=SERVER)
    assert by_obs.status_code == 200 and by_obs.json()["model"]["data"] is None
    kept = client.get("/api/v1/answers/inat/555", params={"model": V2}, headers=SERVER)
    assert kept.status_code == 200 and kept.json()["model"]["version"] is None
    assert client.get("/api/v1/models", headers=SERVER).json()["models"][0]["data"] == v1["data"]


def test_the_card_belongs_to_the_releases_own_models(tmp_path):
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "release.json").write_text(json.dumps({"version": "2026-10-13.4",
                                                  "backbones": [V1]}), encoding="utf-8")
    card = modelcard.card(rel, None)
    assert card["version"] == "2026-10-13.4" and card["backbones"] == [V1]
    assert modelcard.belongs_to(card, V1, default=V2)            # the release's, default or not
    assert not modelcard.belongs_to(card, V2, default=V2)        # added later: no card
    loose = modelcard.card(None, None)                           # no release: the default's
    assert modelcard.belongs_to(loose, V1, default=V1)
    assert not modelcard.belongs_to(loose, V2, default=V1)
    assert apiv1.public_model(card, None, name="x", own_card=False) == {
        "name": "x", "version": None, "data": None, "calibrated_on": None}


# --- a v1-only server answers as before ---------------------------------------------------

# What differs between two runs of the same answer because it is MEASURED (time taken) or
# drawn at random, by exact path: each must be present on both sides and of its kind, and is
# then left out of the byte-for-byte comparison. Nothing else is left out.
TIMING_MS = ("queue_ms", "decode_ms", "embed_ms", "score_ms", "model_ms", "total_ms")
MEASURED = {
    "answer": [*(("timing", k) for k in TIMING_MS), ("timing", "fetch_ms")],
    "health": [("started",), ("preload", "seconds"), ("queue", "embed_seconds_per_photo"),
               ("queue", "search_seconds"), ("excluded_preload", "last_map_seconds"),
               ("excluded_preload", "last_pass_seconds"), ("excluded_preload", "last_map_at")],
}
MAY_BE_NULL = {("timing", "fetch_ms"), ("preload", "seconds"),
               ("excluded_preload", "last_map_seconds"), ("excluded_preload", "last_pass_seconds"),
               ("excluded_preload", "last_map_at")}


def without_measured(body: dict, kind: str) -> dict:
    """`body` (a deep copy) without its measured fields (MEASURED[kind]), each checked to be
    there and a number (or null where MAY_BE_NULL), and without a v1 request id (checked to
    be one)."""
    out = json.loads(json.dumps(body))
    for path in MEASURED[kind]:
        holder = out
        for key in path[:-1]:
            assert isinstance(holder, dict) and key in holder, (kind, path)
            holder = holder[key]
        assert path[-1] in holder, (kind, path)
        value = holder.pop(path[-1])
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        assert ok or (value is None and path in MAY_BE_NULL), (kind, path, value)
    if "request_id" in out:
        assert out.pop("request_id").startswith("r_")
    return out


def test_a_v1_only_server_answers_the_same_with_or_without_the_new_settings(conn, tmp_path):
    a = server(conn, tmp_path, names=(V1,), allowed=(V1,))
    before = {"identify": identify(a, SERVER).json(), "site": post_photos(a, [247]).json(),
              "models": a.get("/api/models").json(), "health": a.get("/api/health").json()}
    conn.close()
    other = tmp_path / "b"
    other.mkdir()
    from mycomap_vision import manifest
    conn_b = manifest.connect(other / "manifest.sqlite")
    b = server(conn_b, other, names=(V1,), allowed=(V1,), default=V1, api_v1={V1})
    after = {"identify": identify(b, SERVER).json(), "site": post_photos(b, [247]).json(),
             "models": b.get("/api/models").json(), "health": b.get("/api/health").json()}
    conn_b.close()
    for name, kind in (("identify", "answer"), ("site", "answer"), ("health", "health")):
        assert without_measured(before[name], kind) == without_measured(after[name], kind), name
    # Unset, /api/models has no new field at all; set, only the default's two.
    assert {k: v for k, v in after["models"].items()
            if k not in ("default_backbone", "default_ready")} == before["models"]
    assert (after["models"]["default_backbone"], after["models"]["default_ready"]) == (V1, True)
    assert "default_backbone" not in before["models"]
    assert before["identify"]["model"]["name"] == "MycoMap Vision"
    assert set(before["identify"]["model"]["data"]) == {"snapshot", "records", "species",
                                                        "genera"}


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (" a , b ,", {"a", "b"})])
def test_the_api_v1_list_reads_as_a_set(monkeypatch, value, expected):
    from mycomap_vision.guards import Limits
    if value is None:
        monkeypatch.delenv("MV_API_V1_BACKBONES", raising=False)
    else:
        monkeypatch.setenv("MV_API_V1_BACKBONES", value)
    monkeypatch.setenv("MV_DEFAULT_BACKBONE", " v1 ")
    limits = Limits.from_settings()
    assert limits.api_v1_backbones == expected and limits.default_backbone == "v1"
    assert limits.default_of(["a", "v1"]) == "v1" and limits.default_of(["a"]) is None


def scored(conn, backbone, comparison, method="nearest+mean"):
    """An open (not sealed) scoreboard run of `backbone`, as mv compare saves one."""
    from mycomap_vision import evaluate
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    report = {"dataset_release": {"dataset_release": "v1", "draft": False}}
    with conn:
        conn.execute("insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, "
                     "n_reference, n_test, record_set, species_top1, report_json, created_at) "
                     "values (?, ?, ?, '2026-08-17', 28, 9, 9, 'r', 0.5, ?, 't')",
                     (comparison, backbone, method, json.dumps(report)))


@pytest.mark.parametrize("default,expected", [(V1, V1), (None, V2)])
def test_how_it_works_shows_the_default_models_numbers(conn, tmp_path, default, expected):
    client = server(conn, tmp_path, default=default)
    scored(conn, V1, "20261001-000000-aaaaaa")
    scored(conn, V2, "20261002-000000-bbbbbb")        # newer, and sorts first
    assert client.get("/api/how-good").json()["run"]["backbone"] == expected


@pytest.mark.parametrize("default,expected", [(V1, V1), (None, V2)])
def test_a_human_feedback_reask_uses_the_default_model(conn, tmp_path, default, expected):
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
        limits=open_limits(allowed_backbones={V1, V2}, default_backbone=default),
        signin=signin, feedback_store=fstore, feedback_org=Roles(), background=False,
        fungus_gate=None, note=lambda s: None), follow_redirects=False)
    iid = item_id(fstore, "photos_unlike_label")
    as_user(client, "admin-1")
    client.post(f"/api/feedback/items/{iid}/decision", json={"decision": "rename", "value": "A x"})
    vision = client.post(f"/api/feedback/items/{iid}/refresh").json()["refreshed"][0]["vision"]
    assert vision["available"] and vision["model"].split("/")[0] == expected
