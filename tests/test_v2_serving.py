"""MycoMap Vision v2 served beside v1 (servedbases.py; Steve, 2026-11-23): each backbone reads
the not-a-fungus gate and the photo-type tagger of ITS OWN base model, forked from its own
forward pass; a backbone with no gate (or tagger) for its base is refused, never answered
ungated; and a v1-only server answers as before. Synthetic heads and test-only pins only."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png, post_photos
from test_api_photo_types import BANDS, CLASSES
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import config, fungusgate, phototypes, servedbases
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.fungusgate import FungusGate, Thresholds
from mycomap_vision.phototypes import PhotoTagger

V1, V2 = "v1-ft", "v2-ft"                      # stand-ins for the two fine-tunes
BASE1, BASE2 = "bioclip-2-336", "bioclip-2.5-336"
TWIST = {V1: 0.0, V2: 0.4}                     # v2's own vectors differ from v1's


class FineTune(Const):
    """A fine-tune stand-in on `base`: its own vectors as Const's (twisted for v2); the
    base vector it forks from the same pass is tagged with its base (+1 for BASE1, -1 for
    BASE2, times the photo's own vector), so a gate can tell whose vector it was handed."""

    def __init__(self, name, base):
        super().__init__(name, TWIST[name])
        self.base_name = base
        self.forks = 0

    def encode_with_base(self, images):
        self.forks += 1
        v = self.encode(images)
        return v, v * (1.0 if self.base_name == BASE1 else -1.0)


class SpyGate:
    """A gate on `base` that gives every photo `verdict` and keeps the vectors it read."""

    def __init__(self, base, verdict="fungus"):
        self.base, self.verdict, self.read = base, verdict, []

    def fingerprint(self):
        return f"gate-{self.base}"

    def judge(self, base_vectors):
        self.read.extend(np.asarray(base_vectors).tolist())
        return [{"verdict": self.verdict, "p_fungus": 0.9 if self.verdict == "fungus" else 0.3,
                 "message": None} for _ in range(len(base_vectors))]

    def summary(self):
        return {"base": self.base, "fingerprint": self.fingerprint()}


class SpyTagger(PhotoTagger):
    """A tagger on `base` that calls every photo `kind` (tag-only kinds keep the answer)."""

    def __init__(self, base, kind="specimen"):
        super().__init__(np.zeros((2, len(CLASSES)), np.float32), np.zeros(len(CLASSES)),
                         CLASSES, BANDS, 0.35, base, f"test-{base}")
        self.kind, self.read = kind, []

    def probs(self, base_vectors):
        self.read.extend(np.asarray(base_vectors).tolist())
        row = np.full(len(CLASSES), 0.1 / (len(CLASSES) - 1))
        row[CLASSES.index(self.kind)] = 0.9
        return np.tile(row, (len(base_vectors), 1))


def record_finetune(name, base):
    """The fine-tune's record as a release ships it: data/models/<name>.json with its base."""
    root = config.DATA_DIR / "models"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps({"name": name, "base": base}),
                                       encoding="utf-8")


@pytest.fixture
def emb(conn, tmp_path):
    """Both backbones embedded over the same records, their fine-tunes recorded."""
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    for name, base in ((V1, BASE1), (V2, BASE2)):
        todo = photos_to_embed(conn, name, "large", store.location, "local")
        embed_photos(conn, store, Const(name, TWIST[name]), todo, root / name,
                     log=lambda s: None)
        record_finetune(name, base)
    conn.commit()
    return root


def server(tmp_path, root, gates, taggers=None, allowed=(V1, V2), loaded=None):
    bases = {V1: BASE1, V2: BASE2}
    models = loaded if loaded is not None else {}

    def loader(name):
        models[name] = FineTune(name, bases[name])
        return models[name]
    return TestClient(create_app(tmp_path / "manifest.sqlite", root, backbone_loader=loader,
                                 limits=open_limits(allowed_backbones=set(allowed)),
                                 background=False, fungus_gate=gates,
                                 photo_tagger=taggers if taggers is not None else None,
                                 note=lambda s: None))


def answer(client, backbone, reds=(247, 12)):
    res = post_photos(client, list(reds), f"{backbone}/nearest")
    assert res.status_code == 200, res.text
    [r] = res.json()["results"]
    return r


def stable(r):
    """An answer without what differs between two runs of the same answer."""
    return json.loads(json.dumps({k: v for k, v in r.items()
                                  if k not in ("timing", "run", "request_id")}))


# --- each backbone reads its own base's gate and tagger --------------------------------

def test_a_v2_backbone_on_another_base_gets_its_own_gate_and_tagger_vectors(emb, tmp_path):
    g1, g2 = SpyGate(BASE1), SpyGate(BASE2, verdict="unsure")
    t1, t2 = SpyTagger(BASE1), SpyTagger(BASE2, kind="uv")
    loaded = {}
    client = server(tmp_path, emb, {BASE1: g1, BASE2: g2}, [t1, t2], loaded=loaded)
    r2 = answer(client, V2)
    # v2's photos went to v2's gate and tagger only, as the base vectors v2's own pass forked.
    assert g1.read == [] and t1.read == []
    v2 = FineTune(V2, BASE2)
    forked = v2.encode_with_base([png_image(247)])[1].tolist() + \
        v2.encode_with_base([png_image(12)])[1].tolist()
    assert np.allclose(g2.read, forked) and np.allclose(t2.read, forked)
    assert loaded[V2].forks == 2                       # one pass per photo, nothing else run
    assert r2["gate"]["verdict"] == "unsure"           # v2's gate decided
    assert [p["photo_type"]["type"] for p in r2["per_photo"]] == ["uv", "uv"]
    # v1 still reads v1's.
    r1 = answer(client, V1)
    assert r1["gate"]["verdict"] == "fungus" and len(g1.read) == 2 and len(g2.read) == 2
    assert [p["photo_type"]["type"] for p in r1["per_photo"]] == ["specimen", "specimen"]
    assert len(t1.read) == 2 and len(t2.read) == 2


def png_image(red):
    import io

    from PIL import Image
    return Image.open(io.BytesIO(png(red))).convert("RGB")


def test_health_names_every_gate_and_tagger_by_base_when_two_are_served(emb, tmp_path):
    client = server(tmp_path, emb, {BASE1: SpyGate(BASE1), BASE2: SpyGate(BASE2)},
                    [SpyTagger(BASE1), SpyTagger(BASE2)])
    health = client.get("/api/health").json()
    assert "bioclip" not in json.dumps(health)          # public: no base named
    assert len(health["fungus_gates"]) == 2 and len(health["photo_taggers"]) == 2
    assert "refused_backbones" not in health


# --- fail closed ------------------------------------------------------------------------

def test_a_backbone_with_no_gate_for_its_base_is_refused_not_served_ungated(emb, tmp_path):
    g1, loaded = SpyGate(BASE1), {}
    client = server(tmp_path, emb, {BASE1: g1}, loaded=loaded)
    res = post_photos(client, [247], f"{V2}/nearest")
    assert res.status_code == 503 and "no not-a-fungus gate reads bioclip-2.5-336" in res.text
    assert V2 not in loaded and g1.read == []           # never loaded, never answered
    listing = client.get("/api/models").json()
    assert V2 not in listing["ready"] and V1 in listing["ready"]
    [v2] = [b for b in listing["backbones"] if b["backbone"] == V2]
    assert "no not-a-fungus gate" in v2["refused"]
    assert answer(client, V1)["gate"]["verdict"] == "fungus"   # v1 answers, gated


def test_a_refused_backbone_is_shown_in_health_with_its_reason_only_to_the_server_key(
        emb, tmp_path):
    key = "k" * 40
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", emb, backbone_loader=lambda n: FineTune(n, BASE1),
        limits=open_limits(allowed_backbones={V1, V2}), background=False,
        fungus_gate={BASE1: SpyGate(BASE1)}, server_key=key, note=lambda s: None))
    full = client.get("/api/health", headers={"Authorization": f"Bearer {key}"}).json()
    assert "no not-a-fungus gate reads bioclip-2.5-336" in full["refused_backbones"][V2]
    public = client.get("/api/health").json()
    assert list(public["refused_backbones"].values()) == ["not served; see the server log"]
    assert V2 not in json.dumps(public)


def test_a_backbone_whose_loaded_model_cant_fork_its_gates_base_is_refused(emb, tmp_path):
    # The record says bioclip-2-336, but the model loaded gives no base vector (an ONNX
    # tower, say): refused at the photo, never answered without the gate.
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", emb, backbone_loader=lambda n: Const(n, TWIST[n]),
        limits=open_limits(allowed_backbones={V1}), background=False,
        fungus_gate={BASE1: SpyGate(BASE1)}, note=lambda s: None))
    res = post_photos(client, [247], f"{V1}/nearest")
    assert res.status_code == 503 and "not served ungated" in res.text


def test_a_backbone_with_no_tagger_for_its_base_is_refused_unless_named_untagged(
        emb, tmp_path, monkeypatch):
    gates = {BASE1: SpyGate(BASE1), BASE2: SpyGate(BASE2)}
    client = server(tmp_path, emb, gates, [SpyTagger(BASE1)])
    res = post_photos(client, [247], f"{V2}/nearest")
    assert res.status_code == 503 and "no photo-type tagger reads bioclip-2.5-336" in res.text
    monkeypatch.setenv(servedbases.UNTAGGED_SETTING, V2)
    client = server(tmp_path, emb, gates, [SpyTagger(BASE1)])
    r = answer(client, V2)
    assert r["gate"]["verdict"] == "fungus" and "photo_types" not in r
    [shown] = client.get("/api/health").json()["untagged_backbones"]   # public: a label
    assert shown.startswith("model ")


def test_a_tagger_whose_base_no_gate_reads_is_not_served(emb, tmp_path):
    # A 2.5 head with only the 336 gate: the head isn't served (it would need a second
    # tower), so it refuses nothing, and v2 stays refused for want of its gate.
    client = server(tmp_path, emb, {BASE1: SpyGate(BASE1)}, [SpyTagger(BASE1), SpyTagger(BASE2)])
    assert [p["photo_type"]["type"] for p in answer(client, V1)["per_photo"]] == [
        "specimen", "specimen"]
    health = client.get("/api/health").json()
    assert "photo_taggers" not in health                  # one head served, not two
    assert health["photo_tagger"]["version"] is not None
    assert post_photos(client, [247], f"{V2}/nearest").status_code == 503


# --- v1 alone: as before -----------------------------------------------------------------

def test_serving_v2_beside_v1_leaves_v1s_answers_unchanged(emb, tmp_path):
    alone = server(tmp_path, emb, {BASE1: SpyGate(BASE1)}, [SpyTagger(BASE1)], allowed=(V1,))
    both = server(tmp_path, emb, {BASE1: SpyGate(BASE1), BASE2: SpyGate(BASE2, "unsure")},
                  [SpyTagger(BASE1), SpyTagger(BASE2, "uv")])
    for reds in ((247,), (247, 12), (12, 240, 15)):
        assert stable(answer(alone, V1, reds)) == stable(answer(both, V1, reds))


def test_a_one_base_server_reads_the_same_gate_file_and_reports_health_as_before(
        emb, tmp_path, monkeypatch):
    # The release's fungus-gate.npz alone, one tagger: the same gate, no new health keys.
    models = config.DATA_DIR / "models"
    gate = FungusGate(np.array(["mushroom", "bird"]), np.array(["fungus", "other"]),
                      np.array([[1.0, 0.0], [0.0, 1.0]]), 10.0, Thresholds(0.2, 0.5), BASE1)
    gate.save(models)
    gates, errors = fungusgate.load_all(models)
    assert list(gates) == [BASE1] and errors == []
    assert gates[BASE1].fingerprint() == FungusGate.load(models).fingerprint()
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", emb, backbone_loader=lambda n: FineTune(n, BASE1),
        limits=open_limits(allowed_backbones={V1}), background=False,
        photo_tagger=SpyTagger(BASE1), note=lambda s: None))
    health = client.get("/api/health").json()
    for new in ("fungus_gates", "photo_taggers", "refused_backbones", "untagged_backbones",
                "fungus_gate_errors"):
        assert new not in health
    assert health["fungus_gate"]["fingerprint"] == gate.fingerprint()
    assert answer(client, V1)["gate"]["photos"][0]["verdict"] in ("fungus", "unsure",
                                                                   "not_fungus")


# --- the files ---------------------------------------------------------------------------

def gate_on(base, unsure=0.5):
    return FungusGate(np.array(["mushroom", "bird"]), np.array(["fungus", "other"]),
                      np.array([[1.0, 0.0], [0.0, 1.0]]), 10.0, Thresholds(0.2, unsure), base)


def test_a_gate_for_another_base_is_read_from_its_own_file_beside_the_default(tmp_path):
    gate_on(BASE1).save(tmp_path)
    path = gate_on(BASE2, unsure=0.6).save(tmp_path, by_base=True)
    assert path.name == f"fungus-gate.{BASE2}.json"
    gates, errors = fungusgate.load_all(tmp_path)
    assert sorted(gates) == [BASE1, BASE2] and errors == []
    assert gates[BASE2].thresholds.unsure == 0.6 and gates[BASE1].thresholds.unsure == 0.5


def test_a_gate_file_naming_another_base_than_its_own_is_refused(tmp_path):
    gate_on(BASE1).save(tmp_path)
    gate_on(BASE2).save(tmp_path, by_base=True)
    (tmp_path / f"fungus-gate.{BASE2}.json").rename(tmp_path / "fungus-gate.other-base.json")
    (tmp_path / f"fungus-gate.{BASE2}.npz").rename(tmp_path / "fungus-gate.other-base.npz")
    gates, errors = fungusgate.load_all(tmp_path)
    assert list(gates) == [BASE1] and "says it reads bioclip-2.5-336 vectors" in errors[0]
    # A second gate for the default's base: the default stands.
    gate_on(BASE1, unsure=0.7).save(tmp_path, by_base=True)
    gates, errors = fungusgate.load_all(tmp_path)
    assert gates[BASE1].thresholds.unsure == 0.5 and any("already served" in e for e in errors)


FAKE_25_SHA = "0" * 63 + "f"                   # test-only pin, never a real head


def tagger_file(tmp_path, base, sha, name):
    t = PhotoTagger(np.zeros((2, len(CLASSES)), np.float32), np.zeros(len(CLASSES)), CLASSES,
                    BANDS, 0.35, base, None, {"source_sha256": sha})
    return t.save(tmp_path / name)


def test_mv_photo_tagger_names_one_head_per_base(tmp_path, monkeypatch):
    sha336 = next(s for s, pin in phototypes.PINNED.items() if pin["base"] == BASE1)
    monkeypatch.setitem(phototypes.PINNED, FAKE_25_SHA,
                        {"version": "test-2.5", "base": BASE2})
    a = tagger_file(tmp_path, BASE1, sha336, "a.npz")
    b = tagger_file(tmp_path, BASE2, FAKE_25_SHA, "b.npz")
    monkeypatch.setenv(phototypes.SETTING, f"{a}, {b}")
    taggers, errors = PhotoTagger.all_from_setting()
    assert [(t.base, t.version) for t in taggers] == [(BASE1, "v2.1-336"), (BASE2, "test-2.5")]
    assert errors == []
    assert PhotoTagger.from_setting().base == BASE1            # the first, as before
    # A second head for one base is refused; the first named stands.
    c = tagger_file(tmp_path, BASE1, sha336, "c.npz")
    monkeypatch.setenv(phototypes.SETTING, f"{a},{c}")
    taggers, errors = PhotoTagger.all_from_setting()
    assert len(taggers) == 1 and "already served" in errors[0]


def test_a_25_head_without_its_pin_is_refused(tmp_path, monkeypatch):
    b = tagger_file(tmp_path, BASE2, FAKE_25_SHA, "b.npz")
    monkeypatch.setenv(phototypes.SETTING, str(b))
    taggers, errors = PhotoTagger.all_from_setting()
    assert taggers == [] and "not a pinned one" in errors[0]
