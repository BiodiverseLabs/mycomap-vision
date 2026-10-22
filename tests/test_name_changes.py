"""mycomap.org's name-change log read into the nightly layer (name_changes.py): records are
renamed in the layer's manifest only while they still carry the old name, the cursor is kept
across runs and releases, truncated changes are paged, a failure moves nothing, the served
index answers with the new names at once, the nightly reads the log first, and no frozen
file (the served release, v1, v1-cc, v1-test, the pinned v1 label vocabulary r4, the r4
erratum) changes by a byte."""

import hashlib
import os
import sqlite3
import stat
import threading
import time

import pytest
import requests
from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const
from test_nightly import QUIET, RID, FakeOutside, box, names_in, row, seed_rows, top_species  # noqa: F401

from mycomap_vision import feedback as fb
from mycomap_vision import api, manifest, name_changes, nightly
from mycomap_vision.api import create_app


class FakeLog:
    """GET /api/vision/name-changes and /:id/records as .org answers them
    (services/nameChangeLog.ts: oldest first, `next` the last id, records capped per change)."""

    def __init__(self, changes=(), per_change=2000):
        self.changes, self.per_change = list(changes), per_change
        self.calls, self.down, self.absent, self.records_down = [], False, False, False

    def add(self, cid, old, new, refs, **extra):
        self.changes.append({"id": cid, "at": "2026-10-18T14:20:03.000Z", "kind": "rename",
                             "old_name": old, "new_name": new, "source": "name_update",
                             "request_id": None, "decision_ref": None,
                             "records": [{"ref": r, "old_name": old, "new_name": new}
                                         for r in refs], **extra})

    def name_changes(self, after, limit=100):
        self.calls.append(("feed", after, limit))
        if self.down:
            raise requests.ConnectionError("down")
        if self.absent:
            return None                              # .org without the log yet (404)
        page = [c for c in self.changes if c["id"] > after][:limit]
        out = []
        for c in page:
            recs = c["records"]
            out.append({**c, "records": recs[:self.per_change],
                        "records_truncated": len(recs) > self.per_change})
        return {"changes": out, "next": page[-1]["id"] if page else after}

    def name_change_records(self, cid, after=0):
        self.calls.append(("records", cid, after))
        if self.records_down:
            raise requests.ConnectionError("down")
        recs = next(c["records"] for c in self.changes if c["id"] == cid)
        numbered = list(enumerate(recs, start=1))
        page = [(i, r) for i, r in numbered if i > after][:2]
        return {"records": [r for _, r in page], "next": page[-1][0] if page else after}

    # Human Feedback's calls, unused here
    def roles(self, user_id):
        return None

    def record(self, key):
        return None


def layer_db_of(path):
    return lambda: manifest.connect(path)


def reader(tmp_path, log, layer_manifest):
    return name_changes.Reader(tmp_path / "state" / "org-name-changes.sqlite", log,
                               layer_db_of(layer_manifest), note=lambda s: None)


@pytest.fixture
def layer(box):
    return nightly.prepare(box, **QUIET)


def test_log_records_map_to_the_manifests_keys():
    assert name_changes.record_key("inat:123") == "123"
    assert name_changes.record_key("mo:45") == "mo:45"
    for ref in ("seq:9", "mp:3", "local:2", "inat:", "inat:12a", "", None):
        assert name_changes.record_key(ref) is None


def test_records_are_renamed_only_while_they_still_carry_the_old_name(tmp_path, layer):
    log = FakeLog()
    log.add(1, "B y", "B org", ["inat:103", "inat:104", "inat:999", "seq:5", "mo:1"])
    log.add(2, "A x", "A org", ["inat:100"])
    log.add(3, "A x", "A later", ["inat:100"])           # 100 no longer says A x: left alone
    out = reader(tmp_path, log, layer.manifest).run()
    assert out == {"read": 3, "renamed": 3, "not_held": 3, "name_moved": 1, "after": 3}
    names = names_in(layer.manifest)
    assert names["103"] == names["104"] == "B org" and names["105"] == "B y"
    assert names["100"] == "A org" and names["101"] == "A x"
    c = sqlite3.connect(layer.manifest)
    assert c.execute("select change_id, record_key, old_name, new_name from org_renames "
                     "order by id").fetchall() == [(1, "103", "B y", "B org"),
                                                   (1, "104", "B y", "B org"),
                                                   (2, "100", "A x", "A org")]
    assert name_changes.version(c) == 3
    c.close()


def test_the_cursor_is_kept_and_a_change_is_never_applied_twice(tmp_path, layer):
    log = FakeLog()
    log.add(1, "B y", "B org", ["inat:103"])
    assert reader(tmp_path, log, layer.manifest).run()["read"] == 1
    again = reader(tmp_path, log, layer.manifest)            # a new process, the same state
    assert again.cursor() == 1 and again.run()["read"] == 0
    assert log.calls[-1] == ("feed", 1, name_changes.PAGE)
    log.add(2, "B org", "B y", ["inat:103"])                 # renamed back later
    assert again.run() == {"read": 1, "renamed": 1, "not_held": 0, "name_moved": 0, "after": 2}
    assert names_in(layer.manifest)["103"] == "B y"


def test_a_long_log_is_read_page_by_page_and_truncated_changes_record_by_record(
        tmp_path, layer, monkeypatch):
    monkeypatch.setattr(name_changes, "PAGE", 2)
    log = FakeLog(per_change=2)
    log.add(1, "B y", "B org", ["inat:103", "inat:104", "inat:105", "inat:107", "seq:1"])
    log.add(2, "A x", "A org", ["inat:100"])
    log.add(3, "A x", "A org", ["inat:101"])
    out = reader(tmp_path, log, layer.manifest).run()
    assert out["read"] == 3 and out["renamed"] == 6 and out["not_held"] == 1
    assert [c for c in log.calls if c[0] == "feed"] == [("feed", 0, 2), ("feed", 2, 2)]
    assert ("records", 1, 0) in log.calls and ("records", 1, 4) in log.calls
    names = names_in(layer.manifest)
    assert {names[k] for k in ("103", "104", "105", "107")} == {"B org"}


def test_a_failure_moves_neither_the_cursor_nor_a_name(tmp_path, layer):
    log = FakeLog(per_change=1)
    log.add(1, "B y", "B org", ["inat:103", "inat:104"])    # truncated: its records are paged
    r = reader(tmp_path, log, layer.manifest)
    log.records_down = True
    out = r.run()
    assert out["read"] == 0 and "ConnectionError" in out["error"] and r.cursor() == 0
    assert names_in(layer.manifest)["103"] == "B y"
    log.down = True
    assert "error" in r.run() and r.cursor() == 0
    log.down = log.records_down = False
    assert r.run()["renamed"] == 2 and r.cursor() == 1


def test_an_org_without_the_log_yet_is_not_an_error(tmp_path, layer):
    log = FakeLog()
    log.absent = True
    out = reader(tmp_path, log, layer.manifest).run()
    assert out["available"] is False and out["read"] == 0 and "error" not in out


def sha_of_every_file(root, skip):
    out = {}
    for dirpath, _, files in os.walk(root):
        if any(os.path.commonpath([dirpath, s]) == str(s) for s in skip):
            continue
        for f in files:
            p = os.path.join(dirpath, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def frozen_releases(root, records):
    """v1, v1-cc and v1-test as dataset_release writes them (a release.sqlite holding the
    records, MANIFEST.json, read-only files), the pinned label vocabulary r4 and the r4
    erratum, all holding the names the log renames."""
    out = root / "dataset-releases"
    for rid in ("v1", "v1-cc", "v1-test"):
        d = out / rid
        d.mkdir(parents=True)
        db = sqlite3.connect(d / "release.sqlite")
        db.execute("create table records (record_key text primary key, label text)")
        db.executemany("insert into records values (?, ?)", records)
        db.commit()
        db.close()
        (d / "MANIFEST.json").write_text(f'{{"id": "{rid}", "frozen": true}}\n', encoding="utf-8")
    (out / "v1-label-vocabulary-r4.txt").write_text(
        "".join(sorted({f"{lab}\n" for _, lab in records})), encoding="utf-8")
    (out / "v1-errata-r4.tsv").write_text(
        "kind\tkey\treason\tnote\nrecord\t103\tcrosspost_twin\tB y\n", encoding="utf-8")
    for p in out.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
    return out


def test_the_log_leaves_every_frozen_release_and_the_pinned_files_byte_identical(tmp_path, box):
    """Renames from .org reach live data and v2 only (Steve, 2026-10-18): applying the log
    touches the nightly layer and its own state, never a byte of the served release, the
    frozen dataset releases (v1, v1-cc, v1-test), the pinned v1-label-vocabulary-r4 or the r4
    erratum."""
    layer = nightly.prepare(box, **QUIET)
    released = names_in(box / "releases" / RID / "manifest.sqlite")
    frozen = frozen_releases(box, sorted(released.items()))
    state = box / "state"
    before = sha_of_every_file(box, skip=[state])
    assert {"dataset-releases/v1/release.sqlite", "dataset-releases/v1-cc/release.sqlite",
            "dataset-releases/v1-test/release.sqlite",
            "dataset-releases/v1-label-vocabulary-r4.txt", "dataset-releases/v1-errata-r4.tsv",
            f"releases/{RID}/manifest.sqlite"} <= {k.replace(os.sep, "/") for k in before}
    log = FakeLog()
    log.add(1, "B y", "B org", [f"inat:{k}" for k, v in released.items() if v == "B y"])
    log.add(2, "A x", "A org", [f"inat:{k}" for k, v in released.items() if v == "A x"])
    r = name_changes.Reader(state / "org-name-changes.sqlite", log,
                            layer_db_of(layer.manifest), note=lambda s: None)
    out = r.run()
    assert out["renamed"] == len(released)
    assert set(names_in(layer.manifest).values()) == {"A org", "B org"}
    assert sha_of_every_file(box, skip=[state]) == before
    assert names_in(box / "releases" / RID / "manifest.sqlite") == released
    assert frozen.is_dir()


# --- the server ------------------------------------------------------------------------------

def site(box, tmp_path, layer, rows, log):
    return TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        layer=layer, nightly_fetch=lambda: (rows, "2026-10-18T07:00:00.000Z"),
        nightly_steps=FakeOutside({}).steps(), note=lambda s: None,
        feedback_store=fb.Store(tmp_path / "fb.sqlite"), feedback_org=log))


def test_a_rename_from_the_log_reaches_the_served_index_without_waiting_for_the_night(
        tmp_path, box):
    layer = nightly.prepare(box, **QUIET)
    log = FakeLog()
    client = site(box, tmp_path, layer, seed_rows(), log)
    assert top_species(client, 12) == "B y"
    log.add(1, "B y", "B org", [f"inat:{100 + i}" for i in (3, 4, 5, 7)])
    out = client.app.state.read_name_changes()
    assert out["renamed"] == 4
    assert eventually(lambda: top_species(client, 12)) == "B org"
    assert (box / "state" / "org-name-changes.sqlite").is_file()


def eventually(read, want="B org", seconds=10.0):
    """What `read` gives once it gives `want` (a rebuild running behind), else its last."""
    end = time.monotonic() + seconds
    while (got := read()) != want and time.monotonic() < end:
        time.sleep(0.05)
    return got


def test_a_rename_is_answered_from_the_old_index_while_the_new_one_builds_behind_it(
        tmp_path, box, monkeypatch):
    """A rename never makes Identify refuse (Steve, 2026-10-22: six 503s an afternoon, one
    per rename): the index answering keeps answering while the renamed one builds, which
    starts as soon as the rename is read, not when the next person asks."""
    monkeypatch.setenv("MV_INDEX_WAIT_SECONDS", "0.2")
    layer = nightly.prepare(box, **QUIET)
    log = FakeLog()
    client = site(box, tmp_path, layer, seed_rows(), log)
    assert top_species(client, 12) == "B y"
    real, go, started = api.Identifier, threading.Event(), threading.Event()

    def slow(*a, **kw):
        started.set()
        assert go.wait(10)
        return real(*a, **kw)
    monkeypatch.setattr(api, "Identifier", slow)
    log.add(1, "B y", "B org", [f"inat:{100 + i}" for i in (3, 4, 5, 7)])
    assert client.app.state.read_name_changes()["renamed"] == 4
    assert started.wait(5), "the rebuild starts when the rename is read"
    assert top_species(client, 12) == "B y", "answered at once, from the old index"
    go.set()
    assert eventually(lambda: top_species(client, 12)) == "B org"


def test_the_nightly_reads_the_log_first_so_it_counts_only_the_rest(tmp_path, box):
    layer = nightly.prepare(box, **QUIET)
    log = FakeLog()
    log.add(1, "A x", "A org", [f"inat:{100 + i}" for i in (0, 1, 2, 6)])
    rows = [dict(r, scientific_name="A org") if r["scientific_name"] == "A x" else r
            for r in seed_rows()]
    rows = [dict(r, scientific_name="B night") if r["observation_id"] == "103" else r
            for r in rows]
    client = site(box, tmp_path, layer, rows, log)
    report = client.app.state.run_nightly()
    assert report["ok"] and report["records"]["renamed"] == 1      # only B night
    assert ("feed", 0, name_changes.PAGE) in log.calls
    names = names_in(layer.manifest)
    assert names["100"] == "A org" and names["103"] == "B night"


def test_without_a_layer_or_an_org_address_nothing_reads_the_log(tmp_path, box, monkeypatch):
    log = FakeLog()
    client = TestClient(create_app(
        box / "releases" / RID / "manifest.sqlite", box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        note=lambda s: None, feedback_store=fb.Store(tmp_path / "fb.sqlite"), feedback_org=log))
    assert client.app.state.name_change_reader is None          # never the release's manifest
    monkeypatch.delenv("MV_ORG_BASE_URL", raising=False)
    layer = nightly.prepare(box, **QUIET)
    client = TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        layer=layer, note=lambda s: None, feedback_store=fb.Store(tmp_path / "fb2.sqlite")))
    assert client.app.state.name_change_reader is None          # no address: tests never read .org
