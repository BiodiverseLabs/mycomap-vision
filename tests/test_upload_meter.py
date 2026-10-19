"""How often an uploaded photo is a reference photo (uploadmeter.py; Steve, 2026-10-19): a
count-only meter. Counters per day and path, nothing else kept; shown only to mycomap.org's
server key; answers unchanged."""

import sqlite3
from datetime import datetime, timezone

from test_api import png
from test_api_v1_identify import SERVER_KEY, bearer, v1_app
from test_api_v1_sessions import add, app as sessions_app, open_session
from test_link_twins import write_table

from mycomap_vision import dhash, twins, uploadmeter

def patterned_png(seed: int, red: int = 20) -> bytes:
    """A photo with a distinctive dHash (structure in green and blue) whose red, all the
    stand-in backbone reads, is flat at `red`."""
    import io

    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    px = np.zeros((64, 72, 3), dtype=np.uint8)
    px[..., 0] = red
    px[..., 1:] = rng.integers(0, 256, size=(8, 9, 2)).repeat(8, 0).repeat(8, 1)
    buf = io.BytesIO()
    Image.fromarray(px, "RGB").save(buf, "PNG")
    return buf.getvalue()


DAY = datetime(2026, 10, 19, 12, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, at=DAY):
        self.at = at

    def __call__(self):
        return self.at


def meter(tmp_path, table=True, clock=None, **table_kw):
    path = tmp_path / twins.FILE
    if table:
        write_table(path, **table_kw)
    return uploadmeter.UploadMeter(tmp_path / "vision-usage.sqlite",
                                   lambda: path if path.is_file() else None,
                                   clock=clock or Clock(), note=lambda s: None)


# --- the rule ---------------------------------------------------------------------------

def test_a_photo_within_three_bits_of_a_reference_photo_is_a_match(tmp_path):
    h = 0x0123456789ABCDEF
    m = meter(tmp_path, hashes={1: h, 2: h ^ (0xFF << 40)})
    m.record("site", 3, [h ^ 0b111, h ^ 0b1111, None])      # 3 bits | 4 bits | unhashable
    t = m.summary()["total"]
    assert t == {"uploads": 1, "metered": 1, "matched": 1, "photos": 3,
                 "photos_metered": 2, "photos_matched": 1}
    m.record("api", 1, [h ^ (0b1111 << 20)])                 # 4 bits from both: no match
    assert m.summary()["total"]["matched"] == 1


def test_a_template_photo_is_never_a_match(tmp_path):
    h = 0x0F0F0F0F0F0F0F0F
    m = meter(tmp_path, hashes={1: h, 2: h ^ 1}, templates=[h])
    m.record("site", 2, [h, h ^ 2])          # the template | 2 bits from photo 2 (h ^ 1)
    assert m.summary()["total"]["photos_matched"] == 1


def test_without_a_twin_table_uploads_are_counted_but_nothing_is_hashed(tmp_path):
    m = meter(tmp_path, table=False)
    assert m.hashing() is False
    m.record("site", 2, None)
    t = m.summary()
    assert t["table"] is False
    assert t["total"] == {"uploads": 1, "metered": 0, "matched": 0, "photos": 2,
                          "photos_metered": 0, "photos_matched": 0}


def test_counts_are_kept_per_utc_day_and_path_and_the_last_seven_days_are_shown(tmp_path):
    clock = Clock()
    m = meter(tmp_path, clock=clock, hashes={1: 5})
    for days_ago, path in [(0, "site"), (0, "api"), (0, "site"), (6, "session"), (7, "site")]:
        clock.at = DAY - days_ago * 86400
        m.record(path, 1, [5])
    clock.at = DAY
    s = m.summary()
    assert [d["day"] for d in s["days"]] == ["2026-10-19", "2026-10-13"]   # 10-12 is too old
    assert s["days"][0]["paths"]["site"]["uploads"] == 2 and s["days"][0]["uploads"] == 3
    assert s["total"]["uploads"] == 4 and s["total"]["matched"] == 4


def test_only_counters_are_stored(tmp_path):
    h = 0x0123456789ABCDEF
    m = meter(tmp_path, hashes={1: h})
    m.record("site", 1, [h])
    db = sqlite3.connect(tmp_path / "vision-usage.sqlite")
    cols = [r[1] for r in db.execute("pragma table_info(upload_matches)")]
    assert cols == ["day", "path", *uploadmeter.COUNTERS]
    assert db.execute("select * from upload_matches").fetchall() == [
        ("2026-10-19", "site", 1, 1, 1, 1, 1, 1)]


def test_a_broken_store_never_fails_the_identification(tmp_path):
    m = meter(tmp_path, hashes={1: 5})
    m.path = tmp_path / "no-such-dir" / "x" / "usage.sqlite"     # unwritable now
    m.record("site", 1, [5])                                      # no exception


# --- through the API --------------------------------------------------------------------

def health(client, key=True):
    return client.get("/api/health", headers=bearer(SERVER_KEY) if key else {}).json()


def site_upload(client, bodies):
    return client.post("/api/identify",
                       files=[("photos", (f"p{i}.png", b, "image/png"))
                              for i, b in enumerate(bodies)])


def test_a_site_upload_of_a_reference_photo_is_counted(conn, tmp_path):
    photo = patterned_png(3, red=20)
    write_table(tmp_path / twins.FILE, hashes={1004: dhash.dhash64(photo) ^ 1})
    client = v1_app(conn, tmp_path)
    assert site_upload(client, [photo, patterned_png(4, red=240)]).status_code == 200
    got = health(client)["upload_matches"]
    assert got["table"] is True
    assert got["days"][0]["paths"]["site"] == {"uploads": 1, "metered": 1, "matched": 1,
                                               "photos": 2, "photos_metered": 2,
                                               "photos_matched": 1}


def test_api_and_session_uploads_are_counted_on_their_own_paths(conn, tmp_path):
    photo = patterned_png(3, red=20)
    write_table(tmp_path / twins.FILE, hashes={1004: dhash.dhash64(photo)})
    client = sessions_app(conn, tmp_path)
    files = [("photos", ("p.png", photo, "image/png"))]
    assert client.post("/api/v1/identify", files=files,
                       headers=bearer(SERVER_KEY)).status_code == 200
    sid = open_session(client)["session_id"]
    assert add(client, sid, 30, "a").status_code == 200        # a flat red: not a reference
    paths = health(client)["upload_matches"]["days"][0]["paths"]
    assert (paths["api"]["matched"], paths["api"]["photos"]) == (1, 1)
    assert (paths["session"]["uploads"], paths["session"]["matched"]) == (1, 0)


def test_the_meter_changes_no_answer(conn, tmp_path):
    photo = patterned_png(3, red=20)
    client = v1_app(conn, tmp_path)
    before = site_upload(client, [photo]).json()["results"][0]
    write_table(tmp_path / twins.FILE, hashes={1004: dhash.dhash64(photo)})
    after = site_upload(client, [photo]).json()["results"][0]
    assert after["ranks"] == before["ranks"] and after["specimens"] == before["specimens"]
    assert health(client)["upload_matches"]["total"]["matched"] == 1


def test_only_the_server_key_sees_the_counts(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    assert site_upload(client, [png(247)]).status_code == 200
    assert "upload_matches" not in health(client, key=False)
    assert "upload_matches" not in client.get(
        "/api/health", headers={"Authorization": "Bearer nope"}).json()
    assert health(client)["upload_matches"]["total"]["uploads"] == 1
