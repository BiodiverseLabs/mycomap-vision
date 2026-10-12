"""/api/health is public: it never names a backbone or method (Steve's naming rule: the
model is "MycoMap Vision"). mycomap.org's server key still reads the full detail, and the
fields the deploy scripts read keep working."""

import json

from fastapi.testclient import TestClient
from test_api import open_limits
from test_api_v1_identify import FakeOrg, SERVER_KEY, bearer, identify
from test_api_v1_sessions import FakeGate
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, nightly_notes
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

SERVED = "bioclip-2-ft-20261007-165400"
OTHER = "bioclip-2-ft-20261012-030000"


# Tests of what operators read (preload, nightly, search, embed, vectors) read the full
# body with mycomap.org's server key, as the box's own checks can.
HEALTH_KEY = "h" * 40


def full_health(client) -> dict:
    return client.get("/api/health",
                      headers={"Authorization": f"Bearer {HEALTH_KEY}"}).json()


def box_health() -> dict:
    """/api/health as the box gave it on 2026-10-12, ids included."""
    return {
        "ok": True, "code_version": "85b47a2",
        "preload": {"state": "failed", "models": [f"{SERVED}/nearest+mean+place"],
                    "seconds": 41.2, "error": f"FileNotFoundError: models/{SERVED}.pt"},
        "nightly": {"enabled": True, "last_ok_at": "2026-10-12T07:40:00Z", "state": "ok",
                    "last_run_at": "2026-10-12T07:00:00Z", "next_run": "2026-10-13T07:00:00Z",
                    "refused": False, "new_release_due": True,
                    "changed": {"new": 3, "removed": 0, "renamed": 1, "embedded": 9,
                                "predicted": 2},
                    "layer": {SERVED: {"release_photos": 590000, "nightly_photos": 70000,
                                       "share": 0.12, "new_release_due": True},
                              OTHER: {"release_photos": 600000, "nightly_photos": 10,
                                      "share": 0.0, "new_release_due": False}}},
        "queue": {"queued_jobs": 0, "queued_photos": 0, "embed_seconds_per_photo": 1.6,
                  "search_seconds": 0.06, "batches": 12},
        "search": {"mode": "fbgemm", "packed_rows": 593224, "packed_bytes": 465743740,
                   "last_error": f"MemoryError packing {SERVED}"},
        "embed": {SERVED: {"mode": "torch", "reason": f"models/{SERVED}.onnx is missing"}},
        "quotas": {"mode": "off"},
        "sessions": None,
        "fungus_gate": {"mode": "probe", "base": "bioclip-2", "fingerprint": "ab12cd34",
                        "thresholds": {"not_fungus": 0.5, "unsure": 0.8}},
        "vectors": {f"{SERVED}/nearest+mean+place": {"int8_rows": 593224, "float_rows": 0},
                    f"{SERVED}/nearest": {"int8_rows": 593224, "float_rows": 0}},
    }


def strings(v):
    """Every key and string value in a JSON body."""
    if isinstance(v, dict):
        for k, x in v.items():
            yield str(k)
            yield from strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)
    elif isinstance(v, str):
        yield v


def internal(body) -> list[str]:
    return [s for s in strings(body) if any(m in s.lower() for m in ("bioclip", "-ft-"))]


def test_no_public_health_string_names_a_backbone_or_method():
    public = apiv1.public_health(box_health(), SERVED, {SERVED, OTHER})
    assert internal(public) == []
    assert not any("nearest" in s for s in strings(public))
    assert public["preload"]["models"] == ["MycoMap Vision"]
    assert list(public["vectors"]) == ["MycoMap Vision", "MycoMap Vision (2)"]
    assert list(public["nightly"]["layer"]) == ["MycoMap Vision", "model 2"]
    assert list(public["embed"]) == ["MycoMap Vision"] and "reason" not in public["embed"]["MycoMap Vision"]
    assert "base" not in public["fungus_gate"]


def test_error_text_is_not_public_but_the_figures_are():
    public = apiv1.public_health(box_health(), SERVED, {SERVED, OTHER})
    assert public["preload"]["error"] == "failed to load; see the server log"
    assert public["search"]["last_error"] == "see the server log"
    assert public["search"]["packed_bytes"] == 465743740
    assert public["queue"] == box_health()["queue"]
    assert public["nightly"]["changed"] == box_health()["nightly"]["changed"]


def test_an_id_it_was_not_told_of_still_never_shows():
    public = apiv1.public_health(box_health(), None, set())
    assert internal(public) == []


def test_the_deploy_scripts_still_read_what_they_need():
    public = apiv1.public_health(box_health(), SERVED, {SERVED, OTHER})
    # deploy.sh greps the compact JSON for the preload state and for "ok":true.
    text = json.dumps(public, separators=(",", ":"))
    assert text.startswith('{"ok":true') and '"preload":{"state":"failed"' in text
    # verify.sh pipes it through nightly_notes: the same lines, the model by its public name.
    lines = nightly_notes.notes(public["nightly"])
    assert [lvl for lvl, _ in lines] == [lvl for lvl, _ in nightly_notes.notes(box_health()["nightly"])]
    assert any(t.startswith("MycoMap Vision: the nights added 70,000 photos") for _, t in lines)


# --- through the API --------------------------------------------------------------------

def app(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, SERVED, "large", store.location, "local")
    embed_photos(conn, store, Const(SERVED), todo, root / SERVED, log=lambda s: None)
    conn.commit()
    gate = FakeGate()
    gate.base = "bioclip-2"
    gate.summary = lambda: {"fingerprint": gate.fp, "base": gate.base}
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(allowed_backbones={SERVED}), background=False,
        server_key=SERVER_KEY, api_v1=True, app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE),
        introspect=FakeOrg(), fungus_gate=gate, note=lambda s: None))


def test_anyone_gets_public_health_and_the_server_key_gets_the_ids(conn, tmp_path):
    client = app(conn, tmp_path)
    assert identify(client, bearer(SERVER_KEY)).status_code == 200    # an index is served
    public = client.get("/api/health")
    assert public.status_code == 200 and public.json()["ok"] is True
    assert internal(public.json()) == [] and "MycoMap Vision" in public.json()["vectors"]
    for wrong in ("Bearer nope", f"Bearer {SERVER_KEY}x", SERVER_KEY):
        assert internal(client.get("/api/health", headers={"Authorization": wrong}).json()) == []
    full = client.get("/api/health", headers=bearer(SERVER_KEY)).json()
    assert any(k.startswith(SERVED) for k in full["vectors"])
    assert full["fungus_gate"]["base"] == "bioclip-2"
