import hashlib
import threading

from conftest import inat_obs

from mycomap_vision.inat import save_batch
from mycomap_vision.photos import (MAX_ATTEMPTS, HostGate, HostPolicy, Result, download_one,
                                   pending_photos, save_result)
from mycomap_vision.ratelimit import ByteBudget
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 100
URL = "https://static.inaturalist.org/photos/1234/square.jpeg?99"


class FakeResp:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, resp):
        self.resp = resp
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.resp


def gate():
    return HostGate(HostPolicy(concurrency=1, min_interval=0, budgets=[ByteBudget(10_000, 3600)]))


def test_a_downloaded_photo_is_stored_at_medium_size_with_its_hash(tmp_path):
    s = FakeSession(FakeResp(200, JPEG))
    g = gate()
    r = download_one(s, g, 1234, URL, "medium", LocalStore(tmp_path), threading.Event())
    assert s.urls == ["https://static.inaturalist.org/photos/1234/medium.jpeg"]
    assert r.status == "done" and r.local_path == "photos/medium/234/1234.jpeg"
    assert (tmp_path / r.local_path).read_bytes() == JPEG
    assert r.sha256 == hashlib.sha256(JPEG).hexdigest()
    assert g.policy.budgets[0].used() == len(JPEG)


def test_a_deleted_photo_is_missing_not_an_error(tmp_path):
    r = download_one(FakeSession(FakeResp(404)), gate(), 1, URL, "medium", LocalStore(tmp_path),
                     threading.Event())
    assert r.status == "missing"


def test_an_html_error_page_is_not_saved_as_a_photo(tmp_path):
    r = download_one(FakeSession(FakeResp(200, b"<html>blocked</html>")), gate(), 1, URL,
                     "medium", LocalStore(tmp_path), threading.Event())
    assert r.status == "error" and r.error == "not an image"
    assert not any(tmp_path.rglob("*.jpeg"))


def test_rate_limited_answer_pauses_the_host(tmp_path):
    g = gate()
    r = download_one(FakeSession(FakeResp(429)), g, 1, URL, "medium", LocalStore(tmp_path), threading.Event())
    assert r.status == "error" and r.error == "http 429"
    assert g.pacer._next > 0


def test_an_exhausted_byte_budget_waits_instead_of_downloading(tmp_path):
    g = gate()
    g.policy.budgets[0].add(10_000)
    stop = threading.Event()
    stop.set()                   # stop immediately rather than wait out the hour
    s = FakeSession(FakeResp(200, JPEG))
    r = download_one(s, g, 1, URL, "medium", LocalStore(tmp_path), stop)
    assert r.error == "stopped" and s.urls == []


def _seed(conn):
    rows = [{"observation_id": oid, "scientific_name": "X", "continent": cont,
             "validation_status_1": "yes"} for oid, cont in [("5", "North America"),
                                                             ("6", "Europe")]]
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, ["5", "6"], [
        inat_obs(5, photos=[(0, 11, None, "static.inaturalist.org"),
                            (1, 12, "cc0", "inaturalist-open-data.s3.amazonaws.com")]),
        inat_obs(6, photos=[(0, 21, "cc0", "inaturalist-open-data.s3.amazonaws.com")]),
    ], "t")


def test_pending_photos_cover_green_north_american_records_until_attempts_run_out(conn):
    _seed(conn)
    assert sorted(r["photo_id"] for r in pending_photos(conn, True, None)) == [11, 12]
    assert sorted(r["photo_id"] for r in pending_photos(conn, False, None)) == [11, 12, 21]
    save_result(conn, Result(11, "done", "p", 5, "h"), "medium", "now")
    for _ in range(MAX_ATTEMPTS):
        save_result(conn, Result(12, "error", error="http 500"), "medium", "now")
    assert pending_photos(conn, True, None) == []


def test_photos_held_at_another_size_are_queued_again_for_the_new_size(conn):
    _seed(conn)
    save_result(conn, Result(11, "done", "p", 5, "h"), "medium", "now")
    save_result(conn, Result(12, "done", "p", 5, "h"), "large", "now")
    assert [r["photo_id"] for r in pending_photos(conn, True, None, "large")] == [11]
    assert [r["photo_id"] for r in pending_photos(conn, True, None, "medium")] == [12]


def test_a_stopped_download_does_not_use_up_an_attempt(conn):
    _seed(conn)
    save_result(conn, Result(11, "error", error="stopped"), "medium", "now")
    row = conn.execute("select status, attempts from photos where photo_id = 11").fetchone()
    assert tuple(row) == ("pending", 0)


class FailingStore:
    location = "s3://b/"

    def put(self, relpath, body):
        raise ConnectionError("S3 unreachable")


def test_a_failed_store_write_is_an_error_to_retry_not_a_done_photo(tmp_path):
    r = download_one(FakeSession(FakeResp(200, JPEG)), gate(), 1, URL, "large", FailingStore(),
                     threading.Event())
    assert r.status == "error" and r.error == "store: ConnectionError"


def test_photos_held_locally_are_queued_again_when_the_store_moves_to_s3(conn):
    _seed(conn)
    save_result(conn, Result(11, "done", "p", 5, "h"), "large", "now", "C:/data")
    save_result(conn, Result(12, "done", "p", 5, "h"), "large", "now", "s3://b/")
    assert [r["photo_id"] for r in pending_photos(conn, True, None, "large", "s3://b/")] == [11]
    # Without a store given, only the size matters.
    assert pending_photos(conn, True, None, "large") == []


def test_random_order_still_returns_only_pending_photos(conn):
    _seed(conn)
    save_result(conn, Result(11, "done", "p", 5, "h"), "medium", "now")
    got = pending_photos(conn, True, None, random_order=True)
    assert [r["photo_id"] for r in got] == [12]


def test_a_record_sample_takes_every_photo_of_the_chosen_records(conn):
    _seed(conn)
    got = pending_photos(conn, True, None, record_sample=1)
    assert sorted(r["photo_id"] for r in got) == [11, 12]    # both photos of record 5


def test_a_laptop_sample_and_the_s3_set_are_tracked_as_separate_copies(conn):
    _seed(conn)
    save_result(conn, Result(11, "done", "p/11.jpg", 5, "h"), "medium", "now", "C:/data")
    # Held on the laptop at medium, so still wanted in S3 at large...
    assert 11 in [r["photo_id"] for r in pending_photos(conn, True, None, "large", "s3://b/")]
    save_result(conn, Result(11, "done", "p/11.jpg", 9, "h"), "large", "now", "s3://b/")
    assert 11 not in [r["photo_id"] for r in pending_photos(conn, True, None, "large", "s3://b/")]
    # ...and the laptop copy is still there for local embedding.
    assert 11 not in [r["photo_id"] for r in pending_photos(conn, True, None, "medium", "C:/data")]
    copies = {tuple(r) for r in conn.execute("select store, size from photo_copies where photo_id = 11")}
    assert copies == {("C:/data", "medium"), ("s3://b/", "large")}


def test_only_failures_count_toward_the_attempt_limit(conn):
    _seed(conn)
    for size in ("small", "medium", "large", "original", "medium"):
        save_result(conn, Result(11, "done", "p", 5, "h"), size, "now", "C:/data")
    assert conn.execute("select attempts from photos where photo_id = 11").fetchone()[0] == 0


def test_photos_gone_from_inat_are_not_queued_again(conn):
    _seed(conn)
    save_result(conn, Result(11, "missing", error="http 404"), "large", "now")
    assert 11 not in [r["photo_id"] for r in pending_photos(conn, True, None, "large", "s3://b/")]
