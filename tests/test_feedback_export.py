"""The photo-exclusion export the paper freezes by sha256 (feedback_export.py; E34, the paper
coordinator, 2026-10-21): a decision after the cut-off leaves the bytes unchanged, two runs give
the same sha, reviewers appear only as an HMAC pseudonym, a v1-test photo is refused (paper
mode), and it reads a .backup copy, never the live store, writing nothing to it."""

import hashlib
import hmac
import json
import sqlite3
from pathlib import Path

import pytest

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_export as fx

INAT = "https://www.inaturalist.org/observations/"
KEY = b"test-key"
T0 = 1_790_000_000.0                       # 2026-09-21; the cut-off below is T0 + 1000
CUT = "2026-09-21T14:30:00Z"


V1_HASH = "a1" * 32                       # a synthetic release v1's release_hash


def pin_v1(rel, release_hash=V1_HASH):
    """The MANIFEST.json write_release leaves beside a release (here only what the exporter
    reads: the release_hash and the release file's sha256), for a synthetic release v1, written
    again on each call so it matches the file as it is now."""
    rel = Path(rel)
    (rel.parent / "MANIFEST.json").write_text(json.dumps({
        "dataset_release": "v1", "hashes": {"release": release_hash},
        "files": [{"path": rel.name, "sha256": hashlib.sha256(rel.read_bytes()).hexdigest()}]}),
        encoding="utf-8")
    return rel


def make_v1_test(path, records, photos, test_for="v1", for_hash=V1_HASH):
    """A synthetic v1-test release: its info (as write_release stores it, JSON values),
    records and photos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = sqlite3.connect(path)
    r.executescript("create table info (key text primary key, value text not null);"
                    "create table records (record_key text, split text);"
                    "create table photos (photo_key text, record_key text);")
    info = {"id": "v1-test", "test_for": test_for, "test_for_release_hash": for_hash}
    r.executemany("insert into info values (?,?)",
                  [(k, json.dumps(v)) for k, v in info.items() if v is not None])
    r.executemany("insert into records values (?,?)", records)
    r.executemany("insert into photos values (?,?)", photos)
    r.commit()
    r.close()
    return path


def hold_file(path, keys):
    """A hold list and its sha256 (an empty one holds no record)."""
    path = Path(path)
    path.write_bytes("".join(k + "\n" for k in keys).encode("utf-8"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def export_paper(copy, rel, *a, v1_test=None, **kw):
    """The exclusion export in paper mode (its mode has no default since the ruling of
    15 Nov 2026), with release v1 pinned, a v1-test release (by default one record, 99, and
    photo inat:99) and, unless given, an empty pinned hold list."""
    rel = pin_v1(rel)
    if v1_test is None:
        v1_test = rel.parent / "v1-test" / "release.sqlite"
        if not v1_test.exists():
            make_v1_test(v1_test, [("99", "test")], [("inat:99", "99")])
    if "hold_keys" not in kw and "hold_sha256" not in kw:
        kw["hold_keys"], kw["hold_sha256"] = hold_file(rel.parent / "hold-empty.txt", [])
    return fx.export(copy, rel, *a, mode="paper", v1_test_release=v1_test, **kw)


def item(key, record, rule):
    return {"item_type": "non_specimen_photo", "subject": key, "rule": rule,
            "granularity": "photo", "links": [INAT + record], "current": "A x"}


@pytest.fixture
def site(tmp_path):
    live = tmp_path / "state" / "human-feedback.sqlite"
    s = fb.Store(live)
    s.seed([item("inat:1", "10", "R6 micrograph probe p=0.91"),
            item("inat:2", "10", "R6 view habitat p=0.30; R7 uv probe p=0.99"),
            item("inat:3", "11", "R7 tree probe p=0.97"),
            item("inat:4", "11", "R6b view slip p=0.81 (label cosine 0.44)"),
            item("inat:5", "11", "reviewer: left out in photo review")], "seed")
    rel = tmp_path / "release.sqlite"
    r = sqlite3.connect(rel)
    r.executescript("create table records (record_key text, split text);"
                    "create table photos (photo_key text, record_key text);")
    r.executemany("insert into records values (?,?)", [("10", "train"), ("11", "val")])
    r.executemany("insert into photos values (?,?)", [("inat:1", "10"), ("inat:2", "10"),
                                                      ("inat:3", "11"), ("inat:4", "11"),
                                                      ("inat:5", "11")])
    r.commit()
    r.close()
    return s, live, rel, tmp_path


def decide(s, key, decision, at, value=None, note=None, user="steve-123", name="Steve R"):
    it = next(i for i in s.items_of("non_specimen_photo") if i["subject"] == key)
    s.decide(it["id"], decision, {"sub": user, "name": name}, value=value, note=note)
    s.conn.execute("update decisions set at = ? where id = (select max(id) from decisions)", (at,))
    s.conn.commit()


def backup(live, dest):
    src, dst = sqlite3.connect(live), sqlite3.connect(dest)
    src.backup(dst)
    src.close()
    dst.close()
    return dest


def decided(s):
    decide(s, "inat:1", "exclude_photo", T0 + 1, "micrograph", "summary: shown")
    decide(s, "inat:2", "exclude_photo", T0 + 2, "uv", "summary: flipped")
    decide(s, "inat:3", "keep_photo", T0 + 3, None, "summary: flipped", user="ana-9", name="Ana")
    decide(s, "inat:5", "exclude_photo", T0 + 4, "species_description", None)


def test_the_columns_say_how_each_exclusion_was_confirmed_and_what_flagged_it(site):
    s, live, rel, tmp = site
    decided(s)
    data, summary = export_paper(backup(live, tmp / "copy.sqlite"), rel, CUT, KEY, live)
    lines = data.decode().split("\n")
    assert lines[0].split("\t") == list(fx.COLUMNS) and data.endswith(b"\n")
    rows = {l.split("\t")[0]: dict(zip(fx.COLUMNS, l.split("\t"))) for l in lines[1:] if l}
    assert sorted(rows) == ["inat:1", "inat:2", "inat:3", "inat:5"]       # 4: undecided, absent
    assert [rows[k]["confirmation"] for k in sorted(rows)] == ["shown", "flipped", "flipped",
                                                                "record"]
    assert rows["inat:2"]["flag_source"] == "R6 view habitat; R7 uv probe"
    assert rows["inat:5"]["flag_source"] == "reviewer: left out in photo review"
    assert rows["inat:3"]["decision"] == "keep_photo" and rows["inat:3"]["reason"] == ""
    assert (rows["inat:1"]["record_key"], rows["inat:1"]["v1_split"],
            rows["inat:1"]["manifest_photo_id"]) == ("10", "train", "1")
    assert rows["inat:1"]["decided_at"] == "2026-09-21T14:13:21.000000Z"
    assert summary["exclusions_by_confirmation"] == {"flipped": 1, "record": 1, "shown": 1}
    assert summary["sha256"] == hashlib.sha256(data).hexdigest()


def test_a_decision_after_the_cut_off_leaves_the_bytes_unchanged(site):
    s, live, rel, tmp = site
    decided(s)
    before, _ = export_paper(backup(live, tmp / "a.sqlite"), rel, CUT, KEY, live)
    decide(s, "inat:1", "keep_photo", T0 + 5000, None, "summary: flipped")       # after the cut
    decide(s, "inat:4", "exclude_photo", T0 + 6000, "slip", "summary: shown")    # after the cut
    after, _ = export_paper(backup(live, tmp / "b.sqlite"), rel, CUT, KEY, live)
    assert after == before
    later, _ = export_paper(tmp / "b.sqlite", rel, "2026-09-22T00:00:00Z", KEY, live)
    assert later != before and b"inat:4" in later                   # the later cut sees them


def test_two_runs_give_the_same_sha(site):
    s, live, rel, tmp = site
    decided(s)
    copy = backup(live, tmp / "copy.sqlite")
    one = export_paper(copy, rel, CUT, KEY, live)[1]["sha256"]
    two = export_paper(backup(live, tmp / "copy2.sqlite"), rel, CUT, KEY, live)[1]["sha256"]
    assert one == two


def test_reviewers_appear_only_as_an_hmac_pseudonym(site):
    s, live, rel, tmp = site
    decided(s)
    copy = backup(live, tmp / "copy.sqlite")
    data, _ = export_paper(copy, rel, CUT, KEY, live)
    for leak in (b"steve-123", b"Steve R", b"ana-9", b"Ana"):
        assert leak not in data
    want = hmac.new(KEY, b"steve-123", hashlib.sha256).hexdigest()[:16]
    assert want.encode() in data
    other, _ = export_paper(copy, rel, CUT, b"another-key", live)
    assert want.encode() not in other
    with pytest.raises(fx.ExportError, match="HMAC key"):
        export_paper(copy, rel, CUT, b"", live)


def test_a_decided_photo_on_a_record_the_v1_test_release_lists_is_refused(site):
    """The real path: v1-test is its own release (release v1 has no test split), so the
    refusal reads the v1-test release's records."""
    s, live, rel, tmp = site
    decided(s)
    v1_test = make_v1_test(tmp / "t" / "release.sqlite", [("12", "test")], [("inat:9", "12")])
    assert export_paper(backup(live, tmp / "a.sqlite"), rel, CUT, KEY, live, v1_test=v1_test)
    s.seed([item("inat:9", "12", "R7 uv probe p=0.99")], "after-the-queues-reopened")
    decide(s, "inat:9", "exclude_photo", T0 + 7, "uv", "summary: shown")
    with pytest.raises(fx.ExportError, match="1 decided photo.*v1-test.*inat:9"):
        export_paper(backup(live, tmp / "b.sqlite"), rel, CUT, KEY, live, v1_test=v1_test)


def test_it_reads_a_backup_copy_and_never_the_live_store(site):
    s, live, rel, tmp = site
    decided(s)
    with pytest.raises(fx.ExportError, match="live store"):
        export_paper(live, rel, CUT, KEY, live)
    with pytest.raises(fx.ExportError, match="-wal"):               # a live database elsewhere
        export_paper(live, rel, CUT, KEY, None)
    copy = backup(live, tmp / "copy.sqlite")
    sha = hashlib.sha256(copy.read_bytes()).hexdigest()
    export_paper(copy, rel, CUT, KEY, live)
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == sha      # nothing written
    assert not (tmp / "copy.sqlite-wal").exists()


def test_the_summary_names_the_keys_fingerprint_and_a_run_refuses_another_key(site):
    s, live, rel, tmp = site
    decided(s)
    copy = backup(live, tmp / "copy.sqlite")
    fp = hmac.new(KEY, b"mycomap-e34", hashlib.sha256).hexdigest()
    data, summary = export_paper(copy, rel, CUT, KEY, live, expect_fingerprint=fp)
    assert summary["key_fingerprint"] == fp and KEY not in data
    with pytest.raises(fx.ExportError, match="fingerprint"):
        export_paper(copy, rel, CUT, b"another-key", live, expect_fingerprint=fp)


def test_a_default_saved_by_the_reviewers_save_is_its_own_confirmation():
    """Auto-save off (the review-autosave-toggle branch): the default Exclude written by the
    reviewer's Save carries "summary: saved", neither shown nor flipped."""
    assert fx.confirmation("summary: saved") == "saved"
    assert fx.confirmation("summary: shown") == "shown"
    assert fx.confirmation("summary: flipped") == "flipped"
    assert fx.confirmation(None) == "record"

def test_past_drawing_decisions_export_their_own_codes_and_a_new_one_exports_drawing(site):
    """Steve, 2026-10-27: one Drawings reason. Rows decided before keep drawing_micro and
    drawing_specimen in the reason column; only a new decision says "drawing"; and a reason
    filter for micrographs (paper E35's micro arm) never takes a drawing."""
    s, live, rel, tmp = site
    decided(s)
    decide(s, "inat:4", "exclude_photo", T0 + 5, "drawing_micro", None)
    data, summary = export_paper(backup(live, tmp / "a.sqlite"), rel, CUT, KEY, live)
    rows = {l.split("\t")[0]: dict(zip(fx.COLUMNS, l.split("\t")))
            for l in data.decode().split("\n")[1:] if l}
    assert rows["inat:4"]["reason"] == "drawing_micro"
    assert summary["exclusions_by_reason"]["drawing_micro"] == 1
    assert [k for k, r in rows.items() if r["reason"] == "micrograph"] == ["inat:1"]
    decide(s, "inat:4", "exclude_photo", T0 + 6, "drawing_specimen", None)
    again = export_paper(backup(live, tmp / "b.sqlite"), rel, CUT, KEY, live)[0]
    assert "\tdrawing_specimen\t" in again.decode() and "\tdrawing\t" not in again.decode()
