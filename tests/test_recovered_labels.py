"""Human Feedback "Recovered labels" (recovered_labels.py; Steve, 2026-10-26): a specimen crop for
a field-slip photo a person left out. The slip exclusion never changes; each crop goes back into
Vision unless a person rejects it, and only while the photo stays left out as a slip. A crop is
coordinates on the original photo, applied in memory when embedding: no new image file. The
paper's exclusion export never changes because of a crop."""

import io
import json
import sqlite3

import numpy as np
import pytest
from PIL import Image
from test_feedback_api import as_user
from test_models_and_scoreboard import Const, seed_two_species
from test_photo_review import site_with_photos

from mycomap_vision import embed, evaluate
from mycomap_vision import feedback as fb
from mycomap_vision import feedback_export as fx
from mycomap_vision import photo_review as pr
from mycomap_vision import recovered_labels as rl
from mycomap_vision import slipcrop

INAT = "https://www.inaturalist.org/observations/"
ME = {"sub": "admin-1", "name": "A"}
COLS = "\t".join(rl.SEED_COLUMNS)


def slip_item(key, record, rule="R10 v2 slip p=0.83 (high)", projects=("p1",)):
    return {"item_type": pr.ITEM_TYPE, "subject": key, "rule": rule, "granularity": "photo",
            "links": [INAT + record], "current": "A x", "projects": list(projects)}


def open_store(path):
    """A store whose seal lets every record through (these records have no manifest dates)."""
    return fb.Store(path, gate=fb.SealGate(lambda key: None, seal=None))


def decide_photo(s, key, decision, reason=None):
    it = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == key)
    s.decide(it["id"], decision, ME, value=reason)


def seed_line(key, record, p, crops, w=1000, h=800, area=0.05):
    return "\t".join(map(str, (key, record, "A x", "" if p is None else f"{p:.3f}",
                               rl.tranche_of(p), w, h, 0.3, area, 0.0,
                               json.dumps(crops), slipcrop.VERSION)))


def write_seed(path, lines):
    path.write_text(COLS + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path, conn):
    """Photos 1 and 2 left out as field slips, 3 left out as a micrograph, 4 still open; the
    detector found crops on 1 (two), 2 (one) and 3, nothing on 5."""
    s = open_store(tmp_path / "f.sqlite")
    s.seed([slip_item("inat:1", "10"), slip_item("inat:2", "11", "R10 v2 slip p=0.55 (medium)"),
            slip_item("inat:3", "12"), slip_item("inat:4", "13")], "slips")
    decide_photo(s, "inat:1", "exclude_photo", "slip")
    decide_photo(s, "inat:2", "exclude_photo", "slip")
    decide_photo(s, "inat:3", "exclude_photo", "micrograph")
    seed = write_seed(tmp_path / "crops.tsv", [
        seed_line("inat:1", "10", 0.83, [[100, 100, 300, 300], [500, 400, 900, 800]], area=0.02),
        seed_line("inat:2", "11", 0.55, [[0, 0, 400, 400]], area=0.2),
        seed_line("inat:3", "12", 0.81, [[10, 10, 50, 50]]),
        seed_line("inat:4", "13", 0.75, [[10, 10, 50, 50]]),
        seed_line("inat:5", "14", 0.90, [])])
    out = fb.seed_files(s, conn, slip_crops=seed)
    return s, out


def item_of(s, key):
    return next(i for i in s.items_of(rl.ITEM_TYPE) if i["subject"] == key)


# --- the rules -----------------------------------------------------------------------------------

def test_tranches_come_from_the_tagger_r10_slip_p_and_an_earlier_rule_is_its_own():
    assert [rl.tranche_of(p) for p in (0.95, 0.70, 0.69, 0.6, 0.55, 0.45, 0.35, 0.2, None)] == [
        "high", "high", "high", "high", "medium", "medium", "low", "low", "rule"]
    assert rl.tranche_of(0.19) == "rule"                     # below the lowest band
    assert rl.slip_p("R9 v2 slip p=0.91 (high); R10 v2 slip p=0.64 (low)") == 0.64
    assert rl.slip_p("R6b view slip p=0.81 (label cosine 0.44)") is None
    assert rl.slip_p("R10 v2 micrograph p=0.99 (high)") is None


def test_only_photos_a_person_left_out_as_a_slip_are_queued_with_the_slips_projects(store):
    s, out = store
    assert out[0]["added"] == 2 and out[0]["no_specimen"] == 1
    assert out[0]["not_slip_excluded"] == 2                  # a micrograph, and one still open
    one = item_of(s, "inat:1")
    assert one["status"] == "open" and one["projects"] == ["p1"]
    assert one["evidence"]["tranche"] == "high" and [c["n"] for c in one["evidence"]["crops"]] == [1, 2]
    assert pr.record_of_item(one) == "10"
    assert s.summary()[[t["type"] for t in s.summary()].index(rl.ITEM_TYPE)]["section"] == "cleanup"
    # the slip exclusions themselves are untouched: still excluded, nothing reopened
    assert {"inat:1", "inat:2", "inat:3"} <= s.excluded_photos()


def test_a_crop_outside_its_photo_refuses_the_file(tmp_path, conn):
    s = open_store(tmp_path / "f.sqlite")
    bad = write_seed(tmp_path / "bad.tsv", [seed_line("inat:1", "10", 0.8, [[0, 0, 1200, 300]])])
    with pytest.raises(fb.FeedbackError, match="not inside"):
        fb.seed_files(s, conn, slip_crops=bad)


def test_every_crop_counts_until_rejected_and_only_while_the_photo_stays_a_slip(store):
    s, _ = store
    got = rl.accepted_crops(s)
    assert set(got) == {"inat:1", "inat:2"}
    assert got["inat:1"]["boxes"] == [(1, [0.1, 0.125, 0.3, 0.375]), (2, [0.5, 0.5, 0.9, 1.0])]
    rl.batch_decide(s, [{"item_id": item_of(s, "inat:1")["id"], "include": [2]},
                        {"item_id": item_of(s, "inat:2")["id"], "include": []}], ME)
    got = rl.accepted_crops(s)
    assert list(got) == ["inat:1"] and [n for n, _ in got["inat:1"]["boxes"]] == [2]
    decide_photo(s, "inat:1", "keep_photo")             # the photo itself is back: no crop
    assert rl.accepted_crops(s) == {}
    decide_photo(s, "inat:1", "exclude_photo", "blurry")   # left out for another reason: none
    assert rl.accepted_crops(s) == {}
    decide_photo(s, "inat:1", "exclude_photo", "slip")
    assert list(rl.accepted_crops(s)) == ["inat:1"]


def test_a_page_save_is_whole_or_nothing_and_idempotent(store):
    s, _ = store
    one, two = item_of(s, "inat:1")["id"], item_of(s, "inat:2")["id"]
    other = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == "inat:4")["id"]
    for bad in ({"item_id": two, "include": [2]},          # item 2 has one crop
                {"item_id": two, "include": "1"},           # not a list
                {"item_id": two, "include": [True]},        # not a crop number
                {"item_id": other, "include": []},          # another queue's item
                {"item_id": one, "include": [1]},           # twice in the batch
                {"item_id": 99999, "include": []}):         # no such item
        with pytest.raises(fb.FeedbackError):
            rl.batch_decide(s, [{"item_id": one, "include": [1, 2]}, bad], ME)
    assert s.conn.execute("select count(*) from decisions d join items i on i.id = d.item_id "
                          "where i.item_type = ?", (rl.ITEM_TYPE,)).fetchone()[0] == 0
    out = rl.batch_decide(s, [{"item_id": one, "include": [2, 1, 1]},
                              {"item_id": two, "include": []}], ME)
    assert out == {"saved": 2, "unchanged": 0, "reasons_changed": 0, "included": 1, "rejected": 1}
    last = s.get(one)["decisions"][-1]
    assert (last["decision"], last["value"], last["applies"]) == ("include_crops", "[1, 2]", "vision")
    assert s.get(two)["decisions"][-1]["decision"] == "reject_crops"
    again = rl.batch_decide(s, [{"item_id": one, "include": [1, 2]}], ME)
    assert (again["saved"], again["unchanged"]) == (0, 1)


def test_tiles_sort_riskiest_first_filter_by_tranche_and_say_when_a_photo_is_back(store):
    s, _ = store
    items = s.items_of(rl.ITEM_TYPE)
    slips = set(rl.slip_exclusions(s))
    assert [t["photo_key"] for t in rl.tiles(items, slips=slips)] == ["inat:1", "inat:2"]
    assert [t["photo_key"] for t in rl.tiles(items, sort="largest")] == ["inat:2", "inat:1"]
    assert [t["photo_key"] for t in rl.tiles(items, "medium")] == ["inat:2"]
    t = rl.tiles(items, slips=slips)[0]
    assert t["crops"][0] == {"n": 1, "box": [100, 100, 300, 300], "proposed": [100, 100, 300, 300],
                             "area": 0.05, "decision": None}
    assert t["still_excluded"] is True
    assert {c["code"]: c["open"] for c in rl.tranche_counts(items)} == {
        "high": 1, "medium": 1, "low": 0, "rule": 0}
    rl.batch_decide(s, [{"item_id": t["item_id"], "include": [2]}], ME)
    decide_photo(s, "inat:1", "keep_photo")
    done = rl.tiles(s.items_of(rl.ITEM_TYPE), status="decided", slips=set(rl.slip_exclusions(s)))
    assert [c["decision"] for c in done[0]["crops"]] == ["reject", "include"]
    assert done[0]["still_excluded"] is False
    for bad in ({"tranche": "nope"}, {"status": "any"}, {"sort": "p"}):
        with pytest.raises(fb.FeedbackError):
            rl.tiles(items, **bad)


# --- into Vision: the manifest, embedding, the reference --------------------------------------

def test_the_manifest_holds_the_kept_crops_as_fractions_and_its_fingerprint_follows(store, conn):
    s, _ = store
    fp0 = fb.crops_fingerprint(conn)
    out = fb.sync_crops(s, conn)
    assert out["crops"] == 3 and out["photos"] == 2 and out["fingerprint"] != fp0
    rows = conn.execute("select photo_id, n, l, t, r, b from feedback_photo_crops "
                        "order by photo_id, n").fetchall()
    assert [tuple(r) for r in rows] == [(1, 1, 0.1, 0.125, 0.3, 0.375), (1, 2, 0.5, 0.5, 0.9, 1.0),
                                        (2, 1, 0.0, 0.0, 0.4, 0.5)]
    rl.batch_decide(s, [{"item_id": item_of(s, "inat:2")["id"], "include": []}], ME)
    assert fb.sync_crops(s, conn)["crops"] == 2 and fb.crops_fingerprint(conn) != out["fingerprint"]


def test_a_crop_vector_id_is_negative_and_the_box_is_applied_in_memory():
    assert embed.crop_vector_id(1000, 1) == -16001 and embed.crop_vector_id(1000, 2) == -16002
    with pytest.raises(ValueError):
        embed.crop_vector_id(1, 16)
    img = Image.new("RGB", (100, 50))
    assert embed.apply_box(img, (0.5, 0.0, 1.0, 0.5)).size == (50, 25)
    assert embed.apply_box(img, (0.0, 0.0, 1.5, 1.0)).size == (100, 50)       # kept inside
    with pytest.raises(ValueError):
        embed.apply_box(img, (0.5, 0.5, 0.5, 0.9))


def slip_with_crop(conn, tmp_path, include=True):
    """seed_two_species, with record 100's only photo (1000) red on its left half and black on
    its right, left out as a field slip; its crop is the red half."""
    store = seed_two_species(conn, tmp_path)
    im = Image.new("RGB", (16, 8), (0, 0, 0))
    im.paste((250, 0, 0), (0, 0, 8, 8))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    rel = conn.execute("select path from photo_copies where photo_id = 1000").fetchone()[0]
    store.put(rel, buf.getvalue())
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([slip_item("inat:1000", "100")], "slips")
    decide_photo(s, "inat:1000", "exclude_photo", "slip")
    seed = write_seed(tmp_path / "crops.tsv", [seed_line("inat:1000", "100", 0.8, [[0, 0, 8, 8]],
                                                         w=16, h=8)])
    fb.seed_files(s, conn, slip_crops=seed)
    if not include:
        rl.batch_decide(s, [{"item_id": item_of(s, "inat:1000")["id"], "include": []}], ME)
    fb.sync_exclusions(s, conn)
    fb.sync_crops(s, conn)
    conn.commit()
    return store, s


def test_a_kept_crop_is_embedded_from_its_box_and_joins_its_record_in_the_reference(conn, tmp_path):
    store, s = slip_with_crop(conn, tmp_path)
    m1 = Const("m1")
    root = tmp_path / "emb"
    embed.embed_photos(conn, store, m1, embed.photos_to_embed(conn, "m1", "large", store.location,
                                                             "local"), root / "m1", log=lambda x: None)
    todo = embed.crops_to_embed(conn, "m1", "large", store.location)
    assert todo == [(-16001, todo[0][1], (0.0, 0.0, 0.5, 1.0))]
    embed.embed_photos(conn, store, m1, todo, root / "m1", log=lambda x: None)
    assert embed.crops_to_embed(conn, "m1", "large", store.location) == []      # once
    ids, vecs = embed.load_embeddings(conn, "m1", root / "m1")
    row = {int(p): i for i, p in enumerate(ids.tolist())}
    want = embed.normalise(m1.encode([Image.new("RGB", (8, 8), (250, 0, 0))]))[0]
    whole = embed.normalise(m1.encode([Image.open(io.BytesIO(store.get(
        conn.execute("select path from photo_copies where photo_id = 1000").fetchone()[0])))
        .convert("RGB")]))[0]
    assert np.allclose(vecs[row[-16001]], want) and not np.allclose(want, whole)
    recs = {r.observation_id: r for r in evaluate.load_records(conn, row)}
    assert recs["100"].photo_rows == [row[-16001]]          # the crop, never the slip photo
    # another backbone has no crop vector: its read of the record is as before (none)
    assert "100" not in {r.observation_id for r in evaluate.load_records(
        conn, {p: i for p, i in row.items() if p > 0})}


def test_a_rejected_crop_is_never_embedded_and_the_record_keeps_no_photo(conn, tmp_path):
    store, _ = slip_with_crop(conn, tmp_path, include=False)
    assert embed.crops_to_embed(conn, "m1", "large", store.location) == []
    ids = {int(p[0]): int(p[0]) for p in conn.execute("select photo_id from observation_photos")}
    assert "100" not in {r.observation_id for r in evaluate.load_records(conn, ids)}


def test_the_night_fetches_a_crops_photo_and_keeps_it_until_the_crop_is_embedded(conn, tmp_path):
    from mycomap_vision import nightly
    store, _ = slip_with_crop(conn, tmp_path)
    m1 = Const("m1")
    root = tmp_path / "emb"
    embed.embed_photos(conn, store, m1, embed.photos_to_embed(conn, "m1", "large", store.location,
                                                             "local"), root / "m1", log=lambda x: None)
    held = conn.execute("select count(*) from photo_copies where photo_id = 1000").fetchone()[0]
    assert nightly.drop_embedded_photos(conn, store, ["m1"]) == 7     # all but the crop's photo
    assert conn.execute("select count(*) from photo_copies where photo_id = 1000").fetchone()[0] == held
    embed.embed_photos(conn, store, m1, embed.crops_to_embed(conn, "m1", "large", store.location),
                       root / "m1", log=lambda x: None)
    assert nightly.drop_embedded_photos(conn, store, ["m1"]) == 1
    # with its copy gone and the crop embedded, there is nothing to fetch; a new backbone would
    assert embed.crop_photos_to_fetch(conn, "m1", "large", store.location) == []
    assert [r[0] for r in embed.crop_photos_to_fetch(conn, "m2", "large", store.location)] == [1000]
    assert nightly.Settings().max_crops == 3000


# --- the API --------------------------------------------------------------------------------------

def test_the_api_pages_tiles_and_saves_for_reviewers_and_read_only_viewers_only_read(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    for key in ("inat:1000", "inat:1100"):
        decide_photo(fstore, key, "exclude_photo", "slip")
    fstore.seed([{**r, "projects": ["p1"]} for r in rl.rows_from_slip_crops(write_seed(
        tmp_path / "crops.tsv", [seed_line("inat:1000", "100", 0.9, [[10, 10, 90, 90]], area=0.01),
                                 seed_line("inat:1100", "100", 0.45, [[0, 0, 50, 50]], area=0.3)]))[0]],
                "crops")
    as_user(c, "admin-1")
    r = c.get("/api/feedback/recovered-labels")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and [t["photo_key"] for t in body["tiles"]] == ["inat:1000", "inat:1100"]
    cc, arr = body["tiles"]
    assert cc["shown"] == "public" and arr["shown"] == "private"
    assert arr["full"] == "/api/feedback/photo/inat:1100?size=full" and arr["still_excluded"] is True
    assert {t["code"]: t["open"] for t in body["tranches"]}["medium"] == 1
    assert c.get("/api/feedback/recovered-labels", params={"tranche": "high"}).json()["total"] == 1
    assert c.get("/api/feedback/recovered-labels", params={"tranche": "nope"}).status_code == 400
    saved = c.post("/api/feedback/recovered-labels/batch", json={"decisions": [
        {"item_id": cc["item_id"], "include": [1]}, {"item_id": arr["item_id"], "include": []}]})
    assert saved.status_code == 200 and saved.json()["saved"] == 2
    assert c.post("/api/feedback/recovered-labels/batch", json={"decisions": [
        {"item_id": cc["item_id"], "include": [7]}]}).status_code == 400
    summary = {t["type"]: t for t in c.get("/api/feedback/summary").json()}
    assert summary[rl.ITEM_TYPE]["decided"] == 2 and summary[rl.ITEM_TYPE]["records_decided"] == 1
    as_user(c, "member-1")                                     # another project's contributor
    assert c.get("/api/feedback/recovered-labels").json()["total"] == 0
    assert c.post("/api/feedback/recovered-labels/batch", json={"decisions": [
        {"item_id": cc["item_id"], "include": []}]}).status_code == 400
    c.cookies.clear()                                          # read-only: every photo item
    ro = c.get("/api/feedback/recovered-labels", params={"status": "decided"}).json()
    assert [t["shown"] for t in ro["tiles"]] == ["public", "hidden"]
    assert "owner_login" not in ro["tiles"][1] and "full" not in ro["tiles"][1]
    assert c.post("/api/feedback/recovered-labels/batch", json={"decisions": [
        {"item_id": cc["item_id"], "include": []}]}).status_code == 401


# --- the paper's files ----------------------------------------------------------------------------

T0 = 1_790_000_000.0
CUT = "2026-09-21T14:30:00Z"                               # T0 + 1000
KEY = b"test-key"


def at(s, when):
    s.conn.execute("update decisions set at = ? where id = (select max(id) from decisions)", (when,))
    s.conn.commit()


def backup(live, dest):
    src, dst = sqlite3.connect(live), sqlite3.connect(dest)
    src.backup(dst)
    src.close()
    dst.close()
    return dest


def paper_site(tmp_path, conn):
    live = tmp_path / "state" / "human-feedback.sqlite"
    s = open_store(live)
    s.seed([slip_item("inat:1", "10"), slip_item("inat:2", "11"), slip_item("inat:3", "12")], "slips")
    for key in ("inat:1", "inat:2", "inat:3"):
        decide_photo(s, key, "exclude_photo", "slip")
        at(s, T0 + 1)
    s.conn.execute("update items set created_at = ?", (T0,))
    s.conn.commit()
    rel = tmp_path / "release.sqlite"
    r = sqlite3.connect(rel)
    r.executescript("create table records (record_key text, split text);"
                    "create table photos (photo_key text, record_key text);")
    r.executemany("insert into records values (?,?)", [("10", "train"), ("11", "val"), ("12", "val")])
    r.executemany("insert into photos values (?,?)", [("inat:1", "10"), ("inat:2", "11"),
                                                      ("inat:3", "12")])
    r.commit()
    r.close()
    return s, live, rel


def test_the_exclusion_export_is_byte_identical_with_crops_and_the_crop_export_is_its_own(tmp_path, conn):
    s, live, rel = paper_site(tmp_path, conn)
    before, _ = fx.export(backup(live, tmp_path / "a.sqlite"), rel, CUT, KEY, live)
    seed = write_seed(tmp_path / "crops.tsv", [
        seed_line("inat:1", "10", 0.8, [[100, 100, 300, 300], [500, 400, 900, 800]]),
        seed_line("inat:2", "11", 0.8, [[0, 0, 400, 400]]),
        seed_line("inat:3", "12", 0.8, [[0, 0, 100, 100]])])
    fb.seed_files(s, conn, slip_crops=seed)
    s.conn.execute("update items set created_at = ? where item_type = ?", (T0 + 2, rl.ITEM_TYPE))
    s.conn.commit()
    rl.batch_decide(s, [{"item_id": item_of(s, "inat:1")["id"], "include": [2],
                         "boxes": {"2": [520, 420, 880, 780]}}], ME)
    at(s, T0 + 3)
    rl.batch_decide(s, [{"item_id": item_of(s, "inat:2")["id"], "include": []}], ME)
    at(s, T0 + 5000)                                   # after the cut: not in the file
    after, _ = fx.export(backup(live, tmp_path / "b.sqlite"), rel, CUT, KEY, live)
    assert after == before                             # crops never change E34/E35's list
    data, summary = fx.export_crops(backup(live, tmp_path / "c.sqlite"), rel, CUT, KEY, live)
    lines = data.decode().splitlines()
    assert lines[0].split("\t") == list(fx.CROP_COLUMNS)
    rows = [dict(zip(fx.CROP_COLUMNS, x.split("\t"))) for x in lines[1:]]
    assert [(r["photo_key"], r["n"], r["decision"]) for r in rows] == [
        ("inat:1", "2", "include_crops"), ("inat:2", "1", "default"), ("inat:3", "1", "default")]
    assert (rows[0]["l"], rows[0]["b"], rows[0]["v1_split"]) == ("0.520000", "0.975000", "train")
    assert [r["box_source"] for r in rows] == ["reviewer", "detector", "detector"]
    assert rows[1]["reviewer_pseudonym"] == "" and len(rows[0]["reviewer_pseudonym"]) == 16
    assert summary["rows"] == 3 and summary["photos"] == 3
    again, _ = fx.export_crops(backup(live, tmp_path / "d.sqlite"), rel, CUT, KEY, live)
    assert again == data
    with pytest.raises(fx.ExportError, match="live store"):
        fx.export_crops(live, rel, CUT, KEY, live)


def test_the_crop_export_refuses_a_v1_test_photo(tmp_path, conn):
    s, live, rel = paper_site(tmp_path, conn)
    r = sqlite3.connect(rel)
    r.execute("update records set split = 'test' where record_key = '12'")
    r.commit()
    r.close()
    fb.seed_files(s, conn, slip_crops=write_seed(tmp_path / "crops.tsv", [
        seed_line("inat:3", "12", 0.8, [[0, 0, 100, 100]])]))
    s.conn.execute("update items set created_at = ?", (T0,))
    s.conn.commit()
    with pytest.raises(fx.ExportError, match="v1-test"):
        fx.export_crops(backup(live, tmp_path / "c.sqlite"), rel, CUT, KEY, live)


# --- the batch job --------------------------------------------------------------------------------

def test_the_job_reads_slips_from_a_copy_writes_seed_lines_and_resumes(tmp_path, conn):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([slip_item("inat:1", "10"), slip_item("inat:2", "11", "R9 v2 slip p=0.95 (high)"),
            slip_item("inat:3", "12")], "slips")
    decide_photo(s, "inat:1", "exclude_photo", "slip")
    decide_photo(s, "inat:2", "exclude_photo", "slip")
    decide_photo(s, "inat:3", "keep_photo")
    copy = sqlite3.connect(backup(tmp_path / "f.sqlite", tmp_path / "copy.sqlite"))
    picks = rl.slip_photos_in_copy(copy)
    assert [(p["photo_key"], p["record_key"], p["tranche"]) for p in picks] == [
        ("inat:2", "11", "rule"), ("inat:1", "10", "high")]          # R9 is not the R10 tagger
    seen = []

    def detect(img):
        seen.append(img.size)
        return {"version": slipcrop.VERSION, "width": img.size[0], "height": img.size[1],
                "slip_area": 0.4, "specimen_area": 0.1, "on_slip_area": 0.0,
                "crops": [{"box": [0, 0, 5, 5], "area": 0.1}] if len(seen) == 1 else []}
    out = tmp_path / "seed.tsv"
    check = rl.v1_check({"10": "train", "11": "val"}, fb.SealGate(lambda key: None, seal=None))
    first = rl.run_slip_crops(picks, lambda p: Image.new("RGB", (20, 10)), detect, out, check,
                              limit=1)
    assert (first["photos"], first["with_crops"], first["already_done"]) == (1, 1, 0)
    second = rl.run_slip_crops(picks, lambda p: Image.new("RGB", (20, 10)), detect, out, check)
    assert (second["photos"], second["no_specimen"], second["already_done"]) == (1, 1, 1)
    rows, counts = rl.rows_from_slip_crops(out)
    assert [r["subject"] for r in rows] == ["inat:2"] and counts == {"no_specimen": 1}
    assert rows[0]["evidence"]["crops"] == [{"n": 1, "box": [0, 0, 5, 5]}]
    assert rows[0]["evidence"]["width"] == 20


# --- the detector's geometry (numpy only) ---------------------------------------------------------

def test_a_crop_is_the_specimen_box_with_a_margin_square_where_the_photo_allows():
    assert slipcrop.square_box(100, 100, 200, 150, 1000, 800) == (85, 60, 215, 190)
    l, t, r, b = slipcrop.square_box(0, 0, 900, 100, 1000, 800)
    assert (l, r) == (0, 1000) and b - t == 800                    # wider than the photo allows
    assert slipcrop.square_box(950, 750, 1000, 800, 1000, 800)[2:] == (1000, 800)


def test_parts_close_together_are_one_crop_and_far_apart_are_two_smallest_first():
    spec = np.zeros((60, 80), bool)
    spec[5:10, 5:10] = True                        # small, top left
    spec[40:58, 50:78] = True                      # large, bottom right
    two = slipcrop.crop_boxes(spec, 800, 600, min_cluster=0.001)
    assert len(two) == 2 and two[0]["area"] < two[1]["area"]
    spec[10:40, 8:52] = True                       # joins them
    assert len(slipcrop.crop_boxes(spec, 800, 600, min_cluster=0.001)) == 1
    tiny = np.zeros((60, 80), bool)
    tiny[0, 0] = True
    assert slipcrop.crop_boxes(tiny, 800, 600) == []


def test_something_lying_on_the_slip_counts_as_specimen():
    slip = np.zeros((40, 40), bool)
    slip[5:35, 5:35] = True
    slip[15:25, 15:25] = False                     # a hole: neither slip nor specimen
    _, spec, on_slip = slipcrop.specimen_region(slip, np.zeros_like(slip))
    assert spec[20, 20] and on_slip.sum() == 100
    _, spec, _ = slipcrop.specimen_region(slip, np.zeros_like(slip), min_hole=0.2)
    assert not spec.any()                          # a hole too small is only a gap


def test_a_night_fetches_embeds_and_counts_a_kept_crop_once_and_a_zero_cap_skips_it(tmp_path):
    from test_nightly import RID, FakeOutside, night, seed_rows
    from mycomap_vision import manifest, nightly
    from mycomap_vision.photos import Result, save_result
    root = tmp_path / "srv"
    rel = root / "releases" / RID
    rel.mkdir(parents=True)
    conn = manifest.connect(rel / "manifest.sqlite")
    store = seed_two_species(conn, tmp_path)
    embed.embed_photos(conn, store, Const("m1"), embed.photos_to_embed(
        conn, "m1", "large", store.location, "local"), rel / "embeddings" / "m1", log=lambda s: None)
    conn.commit()
    conn.close()
    (root / "current.txt").write_text(RID + "\n", encoding="utf-8")
    layer = nightly.prepare(root, log=lambda s: None)
    c = manifest.connect(layer.manifest)
    fb.ensure_exclusions_schema(c)
    fb.ensure_crops_schema(c)
    c.execute("insert into feedback_photo_exclusions values (1000, 'inat:1000')")
    c.execute("insert into feedback_photo_crops values (1000, 1, 'inat:1000', 0, 0, 0.5, 1, 'v')")
    c.commit()
    c.close()
    fetched = []

    def crop_photos(conn, store, backbone, size, limit=None, log=print):
        rows = embed.crop_photos_to_fetch(conn, backbone, size, store.location, limit=limit)
        with conn:
            for r in rows:
                im = Image.new("RGB", (16, 8))
                im.paste((250, 0, 0), (0, 0, 8, 8))
                buf = io.BytesIO()
                im.save(buf, "PNG")
                store.put(f"c/{r['photo_id']}.png", buf.getvalue())
                save_result(conn, Result(r["photo_id"], "done", f"c/{r['photo_id']}.png", 1, "h"),
                            size, "now", store.location)
                fetched.append(r["photo_id"])
        return {"done": len(rows)}

    class Outside(FakeOutside):
        def steps(self):
            return nightly.Steps(fetch_inat=self.fetch_inat, taxonomy=self.taxonomy,
                                 download=self.download, crop_photos=crop_photos,
                                 misidentifications=self.misidentifications)
    off = night(layer, seed_rows(), Outside({}), settings=nightly.Settings(max_crops=0))
    assert "crops_embedded" not in off and fetched == []
    first = night(layer, seed_rows(), Outside({}))
    assert first["ok"] and first["crops_embedded"] == {"m1": 1} and first["changed"]
    assert fetched == [1000] and first["photos_deleted"] == 1        # its copy goes once embedded
    again = night(layer, seed_rows(), Outside({}))
    assert again["crops_embedded"] == {"m1": 0} and fetched == [1000]
    c = manifest.connect(layer.manifest)
    assert c.execute("select count(*) from embeddings where backbone = 'm1' and photo_id = -16001"
                     ).fetchone()[0] == 1
    c.close()
    assert -16001 in np.load(next((layer.embeddings / "m1").glob("*.ids.npy"))).tolist()


def test_the_job_reads_only_v1_train_and_val_photos_and_never_a_v1_test_record(tmp_path):
    """The Paper Coordinator, 2026-10-26: never a v1-test record's photo before the unseal, and
    the paper's seal asserted on each photo besides (records first green after the seal)."""
    seal = "2026-10-12T13:17:51+00:00"
    dates = {"10": "2026-09-01", "11": "2026-09-02", "12": "2026-10-20", "14": "2026-10-20"}
    gate = fb.SealGate(dates.get, seal=seal)
    splits = {"10": "train", "11": "val", "12": "test", "14": "train"}   # 14: a stale split
    check = rl.v1_check(splits, gate)
    picks = [{"photo_key": f"inat:{r}", "record_key": r, "label": "A x", "slip_p": 0.8}
             for r in ("10", "11", "12", "13", "14")]
    ok, refused = rl.job_photos(picks, check)
    assert [p["record_key"] for p in ok] == ["10", "11"]
    assert refused == {"v1 test": 1, "not a v1 train or val record": 1, "seal: sealed": 1}
    read = []

    def image(p):
        read.append(p["photo_key"])
        return Image.new("RGB", (20, 10))

    def detect(img):
        return {"version": "v", "width": 20, "height": 10, "slip_area": 0.4,
                "specimen_area": 0.1, "on_slip_area": 0.0, "crops": []}
    with pytest.raises(fb.FeedbackError, match="v1 test"):
        rl.run_slip_crops([picks[0], picks[2]], image, detect, tmp_path / "seed.tsv", check)
    assert read == ["inat:10"]                 # stopped before the v1-test photo was read


# --- a reviewer's own box, and the exclusion reason from the tile (Steve, 2026-10-27) ------------

def test_a_reviewers_box_replaces_the_detectors_wherever_crops_are_used_and_carries_over(store, conn):
    s, _ = store
    one = item_of(s, "inat:1")["id"]
    rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "boxes": {"2": [400, 300, 900, 800]}}], ME)
    last = s.get(one)["decisions"][-1]
    assert json.loads(last["value"]) == {"include": [1, 2], "boxes": {"2": [400, 300, 900, 800]}}
    got = rl.accepted_crops(s)["inat:1"]["boxes"]
    assert got == [(1, [0.1, 0.125, 0.3, 0.375]), (2, [0.4, 0.375, 0.9, 1.0])]
    t = next(x for x in rl.tiles(s.items_of(rl.ITEM_TYPE), status="decided") if x["item_id"] == one)
    assert t["crops"][1]["box"] == [400, 300, 900, 800] and t["crops"][1]["proposed"] == [500, 400, 900, 800]
    fb.sync_crops(s, conn)
    assert tuple(conn.execute("select l, t from feedback_photo_crops where n = 2").fetchone()) == (0.4, 0.375)
    # a later save without boxes keeps the reviewer's box; back on the detector's, it is no edit
    rl.batch_decide(s, [{"item_id": one, "include": [2]}], ME)
    assert rl.boxes_now(s.get(one))[2] == [400, 300, 900, 800]
    rl.batch_decide(s, [{"item_id": one, "include": [2], "boxes": {"2": [500, 400, 900, 800]}}], ME)
    assert s.get(one)["decisions"][-1]["value"] == "[2]"
    again = rl.batch_decide(s, [{"item_id": one, "include": [2]}], ME)
    assert (again["saved"], again["unchanged"]) == (0, 1)


def test_a_box_outside_its_photo_too_small_or_on_a_rejected_crop_writes_nothing(store):
    s, _ = store
    one, two = item_of(s, "inat:1")["id"], item_of(s, "inat:2")["id"]
    ok = {"item_id": two, "include": [1], "boxes": {"1": [0, 0, 300, 300]}}
    for bad in ({"item_id": one, "include": [1], "boxes": {"1": [0, 0, 1200, 300]}},   # past the edge
                {"item_id": one, "include": [1], "boxes": {"1": [10, 10, 30, 30]}},    # under 32 px
                {"item_id": one, "include": [1], "boxes": {"2": [0, 0, 300, 300]}},    # crop 2 rejected
                {"item_id": one, "include": [1], "boxes": {"1": [0, 0, 300]}},         # not a box
                {"item_id": one, "include": [1], "boxes": [[0, 0, 300, 300]]}):        # not by number
        with pytest.raises(fb.FeedbackError):
            rl.batch_decide(s, [ok, bad], ME)
    assert s.conn.execute("select count(*) from decisions d join items i on i.id = d.item_id "
                          "where i.item_type = ?", (rl.ITEM_TYPE,)).fetchone()[0] == 0
    assert rl.min_side(1000, 800) == 32


def test_changing_the_reason_on_the_tile_changes_the_photos_exclusion_and_only_a_slip_gives_crops(store):
    s, _ = store
    one = item_of(s, "inat:1")["id"]
    photo = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == "inat:1")["id"]
    out = rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "reason": "dried"}], ME)
    assert out["reasons_changed"] == 1
    last = s.get(photo)["decisions"][-1]
    assert (last["decision"], last["value"], last["note"]) == ("exclude_photo", "dried", "recovered labels")
    assert "inat:1" not in rl.accepted_crops(s) and "inat:1" in s.excluded_photos()
    reasons = rl.photo_reasons(s.items_of(pr.ITEM_TYPE))
    t = next(x for x in rl.tiles(s.items_of(rl.ITEM_TYPE), status="decided", slips=reasons)
             if x["item_id"] == one)
    assert (t["reason"], t["still_excluded"]) == ("dried", False)
    same = rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "reason": "dried"}], ME)
    assert same["reasons_changed"] == 0                                 # nothing written twice
    rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "reason": "slip"}], ME)
    assert "inat:1" in rl.accepted_crops(s)
    rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "reason": "keep"}], ME)
    assert s.get(photo)["decisions"][-1]["decision"] == "keep_photo"
    assert "inat:1" not in s.excluded_photos() and "inat:1" not in rl.accepted_crops(s)
    with pytest.raises(fb.FeedbackError, match="reason"):
        rl.batch_decide(s, [{"item_id": one, "include": [1], "reason": "fungus"}], ME)


def test_the_api_saves_a_moved_box_and_a_new_reason(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    decide_photo(fstore, "inat:1000", "exclude_photo", "slip")
    fstore.seed([{**r, "projects": ["p1"]} for r in rl.rows_from_slip_crops(write_seed(
        tmp_path / "crops.tsv", [seed_line("inat:1000", "100", 0.9, [[10, 10, 90, 90]], w=200, h=200)]))[0]],
                "crops")
    as_user(c, "admin-1")
    body = c.get("/api/feedback/recovered-labels").json()
    assert {"code": "slip", "label": "Field slip or label"} in body["reasons"]
    t = body["tiles"][0]
    assert t["reason"] == "slip" and t["crops"][0]["proposed"] == [10, 10, 90, 90]
    r = c.post("/api/feedback/recovered-labels/batch", json={"decisions": [
        {"item_id": t["item_id"], "include": [1], "boxes": {"1": [20, 20, 120, 120]}, "reason": "dried"}]})
    assert r.status_code == 200 and r.json()["reasons_changed"] == 1
    done = c.get("/api/feedback/recovered-labels", params={"status": "decided"}).json()["tiles"][0]
    assert done["crops"][0]["box"] == [20, 20, 120, 120] and done["reason"] == "dried"
    assert done["still_excluded"] is False
    bad = c.post("/api/feedback/recovered-labels/batch", json={"decisions": [
        {"item_id": t["item_id"], "include": [1], "boxes": {"1": [0, 0, 500, 500]}}]})
    assert bad.status_code == 400


def test_the_touching_box_rule_takes_a_weak_box_beside_a_sure_one_but_never_chains_or_spreads():
    stem = ([354, 260, 454, 375], 0.18, "specimen")
    cap = ([297, 163, 402, 267], 0.04, "specimen")                 # overlaps the stem's box
    logo = ([177, 522, 202, 548], 0.04, "specimen")                # touches nothing kept
    beyond = ([200, 100, 300, 170], 0.04, "specimen")              # touches only the cap
    huge = ([300, 200, 560, 420], 0.05, "specimen")                # overlaps, 5x the stem's area
    area = 768 * 1024
    off = slipcrop.keep_detections([stem, cap, logo, beyond, huge], area)
    assert [b for b, _ in off["specimen"]] == [stem[0]]
    on = slipcrop.keep_detections([stem, cap, logo, beyond, huge], area, grow=0.03)
    assert sorted(b for b, _ in on["specimen"]) == sorted([stem[0], cap[0]])
    assert slipcrop.version() == "slipcrop-1" and slipcrop.version(0.03) == "slipcrop-1+grow0.03"


# --- every crop is the square the model reads (Steve, 2026-10-27) --------------------------------

def test_a_crop_is_the_central_square_of_its_box_wherever_it_is_shown_stored_or_saved(tmp_path, conn):
    assert rl.effective_box([0, 0, 300, 200]) == [50, 0, 250, 200]
    assert rl.effective_box([10, 20, 110, 320]) == [10, 120, 110, 220]
    assert rl.effective_box([5, 5, 55, 55]) == [5, 5, 55, 55]
    s = open_store(tmp_path / "f.sqlite")
    s.seed([slip_item("inat:1", "10")], "slips")
    decide_photo(s, "inat:1", "exclude_photo", "slip")
    fb.seed_files(s, conn, slip_crops=write_seed(tmp_path / "crops.tsv", [
        seed_line("inat:1", "10", 0.8, [[0, 0, 1000, 600], [100, 100, 300, 300]])]))
    t = rl.tiles(s.items_of(rl.ITEM_TYPE))[0]
    assert [c["box"] for c in t["crops"]] == [[200, 0, 800, 600], [100, 100, 300, 300]]
    assert t["crops"][0]["proposed"] == [200, 0, 800, 600]
    assert rl.accepted_crops(s)["inat:1"]["boxes"][0] == (1, [0.2, 0.0, 0.8, 0.75])
    one = item_of(s, "inat:1")["id"]
    rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "boxes": {"2": [100, 100, 400, 300]}}], ME)
    assert rl.boxes_now(s.get(one))[2] == [150, 100, 350, 300]          # saved as its square
    rl.batch_decide(s, [{"item_id": one, "include": [1, 2], "boxes": {"1": [200, 0, 800, 600]}}], ME)
    assert 1 not in rl.read_value(s.get(one)["decisions"][-1])[1]      # the detector's square: no edit


def test_a_moved_box_is_embedded_again_at_the_next_night(conn, tmp_path):
    store, s = slip_with_crop(conn, tmp_path)
    m1 = Const("m1")
    root = tmp_path / "emb"
    embed.embed_photos(conn, store, m1, embed.crops_to_embed(conn, "m1", "large", store.location),
                       root / "m1", log=lambda x: None)
    assert embed.crops_to_embed(conn, "m1", "large", store.location) == []
    rl.batch_decide(s, [{"item_id": item_of(s, "inat:1000")["id"], "include": [1],
                         "boxes": {"1": [8, 0, 16, 8]}}], ME)
    out = fb.sync_crops(s, conn)
    assert out["moved"] == 1
    todo = embed.crops_to_embed(conn, "m1", "large", store.location)
    assert todo == [(-16001, todo[0][1], (0.5, 0.0, 1.0, 1.0))]          # the black half now
    embed.embed_photos(conn, store, m1, todo, root / "m1", log=lambda x: None)
    ids, vecs = embed.load_embeddings(conn, "m1", root / "m1")
    row = {int(p): i for i, p in enumerate(ids.tolist())}                 # the newest vector wins
    want = embed.normalise(m1.encode([Image.new("RGB", (8, 8), (0, 0, 0))]))[0]
    assert np.allclose(vecs[row[-16001]], want)
    assert fb.sync_crops(s, conn)["moved"] == 0                           # nothing moved since


def test_the_square_shown_is_exactly_what_the_served_transform_reads():
    """The served models' eval transform (open_clip, resize_mode shortest: Resize then
    CenterCrop) reads only the central square of a crop box: the box and its effective square
    give the same model input."""
    open_clip = pytest.importorskip("open_clip")
    torch = pytest.importorskip("torch")
    t = open_clip.image_transform(336, is_train=False)
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
    raw = [40, 60, 440, 300]                                  # 400 x 240: wider than tall
    sq = rl.effective_box(raw)
    assert sq == [120, 60, 360, 300]
    w, h = img.size
    frac = lambda b: [b[0] / w, b[1] / h, b[2] / w, b[3] / h]  # noqa: E731
    a = t(embed.apply_box(img, frac(raw)))
    b = t(embed.apply_box(img, frac(sq)))
    assert a.shape == b.shape == (3, 336, 336)
    # the same pixels; only the resampling at the square's edges can differ a little
    assert float(torch.mean(torch.abs(a - b))) < 0.05
    off = t(embed.apply_box(img, frac([40, 60, 280, 300])))   # the left square: other pixels
    assert float(torch.mean(torch.abs(a - off))) > 0.5


def test_box_source_says_reviewer_only_for_a_box_whose_square_differs_from_the_detectors():
    item = {"status": "decided", "evidence": {"crops": [{"n": 1, "box": [0, 0, 300, 200]},
                                                        {"n": 2, "box": [10, 10, 60, 60]}]},
            "last_decision": {"decision": "include_crops",
                              "value": json.dumps({"include": [1, 2],
                                                   "boxes": {"1": [50, 0, 250, 200],     # = its square
                                                             "2": [20, 10, 80, 60]}})}}  # an old rectangle
    assert rl.box_source(item, 1) == "detector"
    assert rl.box_source(item, 2) == "reviewer"
    assert rl.boxes_now(item)[2] == [25, 10, 75, 60]
    open_item = {"status": "open", "evidence": item["evidence"], "last_decision": None}
    assert [rl.box_source(open_item, n) for n in (1, 2)] == ["detector", "detector"]
