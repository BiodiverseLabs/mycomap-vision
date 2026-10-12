"""The end-to-end contract harness (scripts/e2e/vision_e2e.py) passes against this server, in
the server-key mode the box runs it in: the harness and the API keep agreeing."""

import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_api import open_limits
from test_api_v1_sessions import FakeGate
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import modelcard
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

uvicorn = pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "e2e"))
import vision_e2e  # noqa: E402

KEY = "k" * 40


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    monkeypatch.setattr(modelcard, "card", lambda *a: {
        "version": "2026-10-06", "data": {"snapshot": "2026-10-06", "records": 8,
                                          "species": 2, "genera": 2}})
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), background=False, server_key=KEY, api_v1=True,
                     sessions=True, fungus_gate=FakeGate(), app_tokens=False, introspect=False,
                     note=lambda s: None)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(10)


def photo(path: Path, red: int) -> str:
    rng = np.random.default_rng(red)
    a = np.clip(rng.normal(0, 8, (300, 400, 3)) + [red, 30, 20], 0, 255).astype("uint8")
    Image.fromarray(a).save(path, quality=90)
    return str(path)


def test_the_contract_harness_passes_with_the_server_key(server, tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text(KEY)
    photos = [photo(tmp_path / f"p{i}.jpg", red) for i, red in enumerate((247, 240, 12, 245))]
    out = tmp_path / "e2e.json"
    monkeypatch.setattr(sys, "argv", ["vision_e2e.py", server, "--key-file", str(key_file),
                                      "--photos", *photos, "--json", str(out)])
    vision_e2e.RESULTS.clear()
    code = vision_e2e.main()
    failed = [r for r in vision_e2e.RESULTS if r["result"] == "FAIL"]
    assert code == 0 and not failed, failed
    passed = sum(r["result"] == "PASS" for r in vision_e2e.RESULTS)
    assert passed >= 25 and out.is_file()


def test_minting_tokens_needs_the_org_folder(monkeypatch):
    monkeypatch.delenv("VISION_E2E_MINT_CWD", raising=False)
    with pytest.raises(SystemExit, match="VISION_E2E_MINT_CWD"):
        vision_e2e.api_server()


def test_app_copies_are_jpegs_of_at_most_1024_px_without_exif(tmp_path):
    big = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 2000), (120, 80, 40)).save(big)
    from io import BytesIO
    copy = Image.open(BytesIO(vision_e2e.app_copy(big)))
    assert copy.format == "JPEG" and max(copy.size) == 1024 and not copy.getexif()
