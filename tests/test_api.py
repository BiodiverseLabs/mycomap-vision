import io
from collections import Counter

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.identify import improvement_hints, softmax_confidence


def png(red):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def app_with_model(conn, tmp_path, embed_all=True):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    if embed_all:
        todo = photos_to_embed(conn, "m1", "large", store.location, "local")
        embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    manifest = tmp_path / "manifest.sqlite"
    return TestClient(create_app(manifest, root, backbone_loader=lambda name: Const(name)))


def post_photos(client, reds, models=""):
    files = [("photos", (f"p{i}.png", png(r), "image/png")) for i, r in enumerate(reds)]
    return client.post("/api/identify", files=files, data={"models": models})


def test_identify_returns_every_rank_nearest_specimens_and_links(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    res = post_photos(client, [247], "m1/nearest")
    assert res.status_code == 200
    [r] = res.json()["results"]
    assert [c["name"] for c in r["ranks"]["species"]][:1] == ["A x"]
    assert r["ranks"]["genus"][0]["name"] == "A"
    assert r["ranks"]["family"][0]["name"] == "F"
    assert 0 < r["ranks"]["species"][0]["confidence"] <= 1
    spec = r["specimens"][0]
    assert spec["species"] == "A x"
    assert spec["inat_url"].startswith("https://www.inaturalist.org/observations/")
    assert spec["species_url"] == "https://mycomap.org/species/A%20x"
    assert spec["photo_url"].endswith("/medium.jpg")
    assert any("Add more photos" in h for h in r["hints"])


def test_two_models_run_side_by_side_on_the_same_photos(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    res = post_photos(client, [12, 14], "m1/nearest,m1/species-mean")
    methods = [r["model"]["method"] for r in res.json()["results"]]
    assert methods == ["nearest", "species-mean"]
    assert all(r["photos"] == 2 for r in res.json()["results"])


def test_identify_refuses_bad_requests_with_clear_reasons(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    assert post_photos(client, [1], "nope/nearest").status_code == 400
    assert post_photos(client, [1], "m1/magic").status_code == 400
    assert post_photos(client, [1] * 11).status_code == 400
    bad = client.post("/api/identify", files=[("photos", ("x.png", b"not an image", "image/png"))])
    assert bad.status_code == 400 and "not an image" in bad.json()["detail"]


def test_identify_without_any_embedded_model_says_so(conn, tmp_path):
    client = app_with_model(conn, tmp_path, embed_all=False)
    assert post_photos(client, [1]).status_code == 503


def test_stats_models_and_scoreboard_are_served(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    stats = client.get("/api/stats").json()
    assert stats["records"] == 8 and stats["embedded"] == {"m1": 8}
    assert stats["names_by_records"]["3-5"] == 2
    assert "m1" in client.get("/api/models").json()["ready"]
    evaluate.compare(conn, ["m1"], embeddings_root=tmp_path / "emb", log=lambda s: None)
    runs = client.get("/api/scoreboard").json()["runs"]
    assert runs and runs[0]["backbone"] == "m1"
    assert client.get(f"/api/scoreboard/{runs[0]['id']}").json()["method"] == "nearest"
    assert client.get("/api/scoreboard/9999").status_code == 404


def test_confidence_sums_to_one_and_favours_the_higher_score():
    c = softmax_confidence(np.array([0.90, 0.88, 0.5]))
    assert abs(c.sum() - 1) < 1e-9 and c[0] > c[1] > c[2]


def test_hints_name_close_species_and_single_specimen_matches():
    ranks = {"species": [{"name": "Russula a", "confidence": 0.4},
                         {"name": "Russula b", "confidence": 0.35}],
             "genus": [{"name": "Russula", "confidence": 0.9}]}
    hints = improvement_hints(3, ranks, top_species_refs=1)
    assert any("Russula a and Russula b are close" in h for h in hints)
    assert any("single DNA-verified specimen" in h for h in hints)
    assert not any("Add more photos" in h for h in hints)
    weak = improvement_hints(2, {"species": [{"name": "X", "confidence": 0.1}],
                                 "genus": [{"name": "X", "confidence": 0.2}]}, 5)
    assert any("Nothing in the DNA-verified set" in h for h in weak)


def test_the_built_site_is_served_with_client_routes_falling_back_to_index(conn, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "a.js").write_text("js")
    (tmp_path / "secret.txt").write_text("outside")
    seed_two_species(conn, tmp_path)
    conn.commit()
    client = TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                   backbone_loader=lambda n: Const(n), web_dist=dist))
    assert client.get("/assets/a.js").text == "js"
    assert client.get("/models").text == "<html>app</html>"
    assert client.get("/../secret.txt").text != "outside"
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api/health").json()["ok"] is True


def test_confidence_uses_the_models_latest_calibration_once_compared(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    before = post_photos(client, [247], "m1/nearest").json()["results"][0]
    assert before["calibration"] is None and "Not calibrated" in before["confidence_note"]
    evaluate.compare(conn, ["m1"], embeddings_root=tmp_path / "emb", log=lambda s: None)
    after = post_photos(client, [247], "m1/nearest").json()["results"][0]
    assert after["calibration"]["comparison_id"]
    assert "Calibrated on" in after["confidence_note"]


def test_specimens_are_ranked_by_all_your_photos_not_one_lucky_match():
    from mycomap_vision.evaluate import Record, build_index, NearestSpecimen
    from mycomap_vision.identify import Identifier

    def unit(*v):
        a = np.asarray(v, dtype=np.float32)
        return a / np.linalg.norm(a)

    # Query photos: a cap view and an underside view.
    q = np.stack([unit(1, 0, 0), unit(0, 1, 0)]).astype(np.float16)
    vecs = np.stack([
        unit(1, 0, 0), unit(0, 0, 1),        # record X: perfect cap match, nothing else
        unit(1, .8, 0), unit(.8, 1, 0),      # record Y: good match to both views
    ]).astype(np.float16)
    recs = [Record("X", "Xus a", "Xus", "F", "2026-01-01", "u", [0, 1]),
            Record("Y", "Yus b", "Yus", "F", "2026-01-01", "u", [2, 3])]
    ident = Identifier.__new__(Identifier)
    ident.index = build_index(recs)
    ident.model = ident.nearest = NearestSpecimen()
    ident.nearest.fit(vecs, ident.index)
    ident.col_photo = np.arange(4)[ident.index.cols]
    rec_of = {row: r for r in recs for row in r.photo_rows}
    ident.col_record = [rec_of[int(c)] for c in ident.index.cols.tolist()]
    ident.rec_starts = np.asarray([i for i, r in enumerate(ident.col_record)
                                   if i == 0 or r is not ident.col_record[i - 1]])
    ident.photos, ident.records, ident.embedded, ident.calibration = {}, 2, 4, None
    ident.backbone, ident.method = "t", "nearest"
    ident.rank_counts = {k: Counter() for k in ("family", "genus", "species")}
    out = ident.identify_vectors(q)
    assert [s["observation_id"] for s in out["specimens"]] == ["Y", "X"]
    assert out["ranks"]["species"][0]["name"] == "Yus b"
