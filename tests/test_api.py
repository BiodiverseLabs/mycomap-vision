import io
from collections import Counter

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.guards import Limits
from mycomap_vision.identify import improvement_hints, softmax_confidence


def png(red):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def open_limits(**kw):
    base = dict(rate=(1000, 60.0), max_in_flight=4, allowed_backbones={"m1"})
    base.update(kw)
    return Limits(**base)


def app_with_model(conn, tmp_path, embed_all=True, limits=None, arr_photos=()):
    store = seed_two_species(conn, tmp_path)
    conn.executemany("update photos set license_class = 'arr' where photo_id = ?",
                     [(p,) for p in arr_photos])
    root = tmp_path / "emb"
    if embed_all:
        todo = photos_to_embed(conn, "m1", "large", store.location, "local")
        embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    manifest = tmp_path / "manifest.sqlite"
    return TestClient(create_app(manifest, root, backbone_loader=lambda name: Const(name),
                                 limits=limits or open_limits(), background=False))


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
                                   backbone_loader=lambda n: Const(n), web_dist=dist,
                                   limits=open_limits()))
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


def unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Query photos: a cap view and an underside view.
CAP_AND_UNDERSIDE = np.stack([unit(1, 0, 0), unit(0, 1, 0)]).astype(np.float16)


def cap_and_underside_identifier(photos=None):
    """Record X matches the cap perfectly and nothing else; record Y matches both views
    well (its photo 2 is closer to the cap, photo 3 to the underside)."""
    from mycomap_vision.evaluate import Record, build_index, NearestSpecimen
    from mycomap_vision.identify import Identifier

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
    ident.photos, ident.records, ident.embedded, ident.calibration = photos or {}, 2, 4, None
    ident.backbone, ident.method, ident.uses_context = "t", "nearest", False
    ident.rank_counts = {k: Counter() for k in ("family", "genus", "species")}
    return ident


def test_specimens_are_ranked_by_all_your_photos_not_one_lucky_match():
    out = cap_and_underside_identifier().identify_vectors(CAP_AND_UNDERSIDE)
    assert [s["observation_id"] for s in out["specimens"]] == ["Y", "X"]
    assert out["ranks"]["species"][0]["name"] == "Yus b"


def test_each_photo_is_also_scored_on_its_own_and_says_where_it_puts_the_overall_answer():
    out = cap_and_underside_identifier().identify_vectors(CAP_AND_UNDERSIDE)
    cap, underside = out["per_photo"]
    assert (cap["photo"], underside["photo"]) == (0, 1)
    # The cap alone prefers X (a perfect match); the underside alone prefers Y.
    assert cap["ranks"]["species"][0]["name"] == "Xus a"
    assert underside["ranks"]["species"][0]["name"] == "Yus b"
    # Both say where they put the answer from all photos together (Yus b).
    assert cap["overall_top"]["species"]["name"] == "Yus b"
    assert cap["overall_top"]["species"]["position"] == 2
    assert underside["overall_top"]["species"]["position"] == 1
    genus = cap["overall_top"]["genus"]
    assert (genus["name"], genus["position"]) == ("Yus", 2)
    assert cap["overall_top"]["species"]["confidence"] < 0.5
    # The main answer is the average of the photos' own scores.
    overall = {c["name"]: c["score"] for c in out["ranks"]["species"]}
    own = [{c["name"]: c["score"] for c in p["ranks"]["species"]} for p in out["per_photo"]]
    for name, score in overall.items():
        assert abs(score - (own[0][name] + own[1][name]) / 2) < 1e-3


def test_a_photos_own_specimens_show_the_reference_photo_that_matched_that_photo():
    from mycomap_vision.identify import PhotoInfo
    photos = {p: PhotoInfo(f"https://static.inaturalist.org/photos/{p}/large.jpg", "open", "o")
              for p in range(4)}
    out = cap_and_underside_identifier(photos).identify_vectors(CAP_AND_UNDERSIDE)
    cap, underside = out["per_photo"]
    assert [s["observation_id"] for s in cap["specimens"]] == ["X", "Y"]
    assert [s["observation_id"] for s in underside["specimens"]] == ["Y", "X"]
    assert all(s["matched_query_photo"] == 0 for s in cap["specimens"])
    assert all(s["matched_query_photo"] == 1 for s in underside["specimens"])
    y_for_cap = next(s for s in cap["specimens"] if s["observation_id"] == "Y")
    assert "/photos/2/" in y_for_cap["photo_url"], "Y's photo closest to the cap"
    assert "/photos/3/" in underside["specimens"][0]["photo_url"], "Y's photo closest to the underside"
    assert underside["specimens"][0]["similarity"] < 1 and cap["specimens"][0]["similarity"] > 0.99


def test_every_offered_method_answers_and_gives_each_photo_its_own_answer(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    offered = client.get("/api/models").json()["methods"]
    assert "nearest+prior" in offered and "species-mean+prior" in offered
    for method in offered:
        res = post_photos(client, [247, 12], f"m1/{method}")
        assert res.status_code == 200, f"{method}: {res.text}"
        [r] = res.json()["results"]
        red, blue = r["per_photo"]
        assert red["specimens"] and blue["specimens"], method
        if not method.startswith(("linear", "hybrid")):     # trained ones: too few records here
            assert red["ranks"]["species"][0]["name"] == "A x", method
            assert blue["ranks"]["species"][0]["name"] == "B y", method
        [single] = post_photos(client, [247], f"m1/{method}").json()["results"]
        assert single["per_photo"][0]["ranks"] == single["ranks"], \
            f"{method}: one photo on its own is the whole answer"



def test_an_image_too_large_in_pixels_is_refused_before_decoding(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    buf = io.BytesIO()
    Image.new("L", (8000, 6000)).save(buf, "PNG")           # 48 MP, a small file
    res = client.post("/api/identify", files=[("photos", ("big.png", buf.getvalue(), "image/png"))])
    assert res.status_code == 413 and "MP limit" in res.json()["detail"]


def test_a_request_declaring_too_many_bytes_is_refused(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    res = client.post("/api/identify", content=b"x", headers={"content-length": str(200 * 2**20),
                                                               "content-type": "multipart/form-data; boundary=x"})
    assert res.status_code == 413


def test_identifications_are_rate_limited_per_address(conn, tmp_path):
    client = app_with_model(conn, tmp_path, limits=open_limits(rate=(2, 600.0)))
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    third = post_photos(client, [247], "m1/nearest")
    assert third.status_code == 429 and int(third.headers["retry-after"]) > 0


def test_a_full_gpu_queue_answers_busy_instead_of_piling_up(conn, tmp_path):
    client = app_with_model(conn, tmp_path, limits=open_limits(max_in_flight=0))
    res = post_photos(client, [247], "m1/nearest")
    assert res.status_code == 503 and res.headers["retry-after"]


def test_only_allowed_backbones_are_offered_or_loaded(conn, tmp_path):
    loaded = []
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()
    client = TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                   backbone_loader=lambda n: loaded.append(n) or Const(n),
                                   limits=open_limits(allowed_backbones={"other"})))
    assert client.get("/api/models").json()["ready"] == []
    res = post_photos(client, [247], "m1/nearest")
    assert res.status_code == 400 and "not offered" in res.json()["detail"]
    assert loaded == []


def test_rate_spec_and_limiter_window():
    from mycomap_vision.guards import RateLimiter, parse_rate
    assert parse_rate("30/600") == (30, 600.0) and parse_rate(None) == (30, 600.0)
    t = [0.0]
    lim = RateLimiter(1, 10, clock=lambda: t[0])
    assert lim.check("a") == 0 and lim.check("a") == 10 and lim.check("b") == 0
    t[0] = 10.5
    assert lim.check("a") == 0


def jpeg_with_exif(lat, lon, when):
    from fractions import Fraction
    img = Image.new("RGB", (8, 8), (247, 0, 0))
    exif = img.getexif()
    gps = exif.get_ifd(0x8825)

    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        s = Fraction(round(((v - d) * 60 - m) * 60 * 100), 100)
        return (Fraction(d), Fraction(m), s)
    gps[1], gps[2] = ("N" if lat >= 0 else "S"), dms(lat)
    gps[3], gps[4] = ("E" if lon >= 0 else "W"), dms(lon)
    exif.get_ifd(0x8769)[36867] = when
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_place_and_date_are_read_from_photo_exif():
    from mycomap_vision.exif import place_and_date
    lat, lon, when = place_and_date(Image.open(io.BytesIO(jpeg_with_exif(39.1653, -86.5264,
                                                                          "2026:09:12 10:03:00"))))
    assert round(lat, 3) == 39.165 and round(lon, 3) == -86.526 and when == "2026-09-12"
    assert place_and_date(Image.new("RGB", (4, 4))) == (None, None, None)


def test_entered_place_wins_and_photo_fills_the_gaps_rounded_in_the_answer():
    from mycomap_vision.api import fill_context
    from mycomap_vision.prior import Context
    ctx, used = fill_context(Context(None, None, "2026-01-02"),
                             [(None, None, None), (39.1653, -86.5264, "2026-09-12")])
    assert (ctx.latitude, ctx.longitude, ctx.observed_on) == (39.1653, -86.5264, "2026-01-02")
    assert used == {"latitude": 39.2, "longitude": -86.5, "observed_on": "2026-01-02",
                    "place_from": "photo", "date_from": "entered"}


def test_a_1970_01_01_date_typed_or_from_a_camera_clock_is_no_date():
    from mycomap_vision.api import fill_context
    from mycomap_vision.exif import place_and_date
    from mycomap_vision.prior import Context
    # A camera whose clock was never set: the place is read, the date is not.
    lat, lon, when = place_and_date(Image.open(io.BytesIO(jpeg_with_exif(39.1653, -86.5264,
                                                                          "1970:01:01 00:00:00"))))
    assert round(lat, 3) == 39.165 and when is None
    # Typed (or sent by an API caller), it doesn't stop a photo's real date filling in,
    # and a photo's placeholder is passed over for the next photo's date.
    ctx, used = fill_context(Context(None, None, "1970-01-01"),
                             [(None, None, "1970-01-01"), (None, None, "2026-09-12")])
    assert ctx.observed_on == "2026-09-12"
    assert (used["observed_on"], used["date_from"]) == ("2026-09-12", "photo")
    ctx, used = fill_context(Context(None, None, "1/1/1970"), [(None, None, "1970-01-01")])
    assert ctx.observed_on is None and ctx.day_of_year is None
    assert (used["observed_on"], used["date_from"]) == (None, None)


def test_identify_ignores_a_1970_01_01_date(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    files = [("photos", ("p.jpg", jpeg_with_exif(39.1653, -86.5264, "1970:01:01 00:00:00"),
                         "image/jpeg"))]
    res = client.post("/api/identify", files=files,
                      data={"models": "m1/nearest", "observed_on": "1970-01-01"}).json()
    assert (res["context_used"]["observed_on"], res["context_used"]["date_from"]) == (None, None)
    assert res["context_used"]["place_from"] == "photo"


def test_identify_reports_the_context_it_used(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    files = [("photos", ("p.jpg", jpeg_with_exif(39.1653, -86.5264, "2026:09:12 10:03:00"),
                         "image/jpeg"))]
    res = client.post("/api/identify", files=files, data={"models": "m1/nearest"}).json()
    assert res["context_used"]["place_from"] == "photo"
    assert res["context_used"]["latitude"] == 39.2
    bad = client.post("/api/identify", files=files, data={"models": "m1/nearest", "lat": "95"})
    assert bad.status_code == 400


def test_results_never_show_an_all_rights_reserved_photo(conn, tmp_path):
    # Record 100 (the reddest A x) has only an all-rights-reserved photo.
    client = app_with_model(conn, tmp_path, arr_photos=[1000])
    [r] = post_photos(client, [250], "m1/nearest").json()["results"]
    by_obs = {s["observation_id"]: s for s in r["specimens"]}
    hidden = by_obs["100"]
    assert hidden["photo_url"] is None and hidden["photo_owner"] is None
    assert hidden["photo_withheld"] is True
    assert hidden["inat_url"].endswith("/100")          # the record itself is still linked
    shown = [s for o, s in by_obs.items() if o != "100"]
    assert shown and all(s["photo_url"] and s["photo_withheld"] is False for s in shown)


def test_a_record_shows_its_best_matching_openly_licensed_photo():
    from mycomap_vision.identify import pick_shown
    scores = [0.9, 0.8, 0.7, 0.95]
    assert pick_shown(scores, ["arr", "nc", "open", "arr"]) == 1
    assert pick_shown(scores, ["arr", None, "arr", "arr"]) is None
    assert pick_shown(scores, ["open", "open", "open", "open"]) == 3


def _row(oid, name, when, project="Macrofungi of Indiana", continent="North America"):
    return {"source": "iNaturalist", "observation_id": str(oid), "scientific_name": name, "genus": name.split()[0],
            "family": "F", "continent": continent, "validation_status_1": "yes",
            "validation_project_1": project, "validation_date_1": when}


def test_stats_credit_who_built_the_reference_set_and_count_provisional_names(conn):
    from conftest import inat_obs
    from test_models_and_scoreboard import OPEN

    from mycomap_vision.api import reference_stats
    from mycomap_vision.inat import save_batch
    from mycomap_vision.records import build_records, save_records
    rows = [_row(1, "Russula emetica", "9/1/2026"),
            _row(2, "Russula sp. 'IN01'", "9/30/2026", project="MycoMap BC"),
            _row(3, "Cortinarius sp. 'CA02'", "10/6/2026"),
            _row(4, "Cortinarius sp. 'CA02'", "10/5/2026"),
            # Outside North America: not in the reference set, so not counted.
            _row(5, "Amanita sp. 'EU01'", "10/6/2026", project="Elsewhere", continent="Europe")]
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, [r["observation_id"] for r in rows],
               [inat_obs(1, "ann", photos=[(0, 11, "cc0", OPEN)]),
                inat_obs(2, "bob", photos=[(0, 12, "cc0", OPEN)]),
                inat_obs(3, "ann", photos=[(0, 13, "cc0", OPEN)]),
                inat_obs(4, "cy", photos=[(0, 14, "cc0", OPEN)]),
                inat_obs(5, "dee", photos=[(0, 15, "cc0", OPEN)])], "t")
    s = reference_stats(conn, set())
    assert (s["names"], s["names_provisional"]) == (3, 2)
    assert s["projects"] == 2                   # Indiana and BC; Europe's project not counted
    assert s["photographers"] == 3              # ann, bob, cy: dee's record is in Europe
    # The newest week runs to the newest validation (6 Oct): 30 Sep is in, 1 Sep is not.
    assert s["recent_week"] == {"records": 3, "from": "2026-09-30", "through": "2026-10-06"}


def test_stats_have_no_recent_week_before_any_record_is_validated(conn):
    from mycomap_vision.api import reference_stats
    s = reference_stats(conn, set())
    assert s["recent_week"] is None and s["projects"] == 0 and s["photographers"] == 0


def test_models_say_when_a_fine_tuned_model_was_trained_through(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    assert client.get("/api/models").json()["backbones"][-1]["trained_through"] is None
    conn.executescript(evaluate.FINETUNE_SCHEMA)
    conn.execute("insert into finetunes values ('m1', 'bioclip-2', '2026-09-07', 28, '{}', 'now')")
    conn.commit()
    m1 = [b for b in client.get("/api/models").json()["backbones"] if b["backbone"] == "m1"]
    assert m1[0]["trained_through"] == "2026-09-07"
