"""The exclusion export's mode (the paper coordinator's ruling of 15 Nov 2026, research
f02ad4a): an export for a v1 paper row refuses any decided photo on a v1-test record, as the
frozen E34 (c5fc1d13...) and E35 (2321d187...) exports did; an export for a served or later
model (v2.1, v2.2) includes them, counted in its summary. The choice is an explicit argument:
there is no default. A record held out of every queue (MV_FEEDBACK_HOLD_KEYS, the E17 blind
check) refuses the export in either mode, without naming it. All stores here are synthetic."""

import argparse
import hashlib
import json
import sqlite3

import pytest

from mycomap_vision import cli
from mycomap_vision import feedback as fb
from mycomap_vision import feedback_export as fx

INAT = "https://www.inaturalist.org/observations/"
MO = "https://mushroomobserver.org/obs/"
KEY = b"golden-key"
T0 = 1_790_000_000.0                       # 2026-09-21 14:13:20 UTC; the cut is T0 + 1000
CUT = "2026-09-21T14:30:00Z"


def item(key, link, rule):
    return {"item_type": "non_specimen_photo", "subject": key, "rule": rule,
            "granularity": "photo", "links": [link], "current": "A x"}


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


def release(path, records, photos):
    r = sqlite3.connect(path)
    r.executescript("create table records (record_key text, split text);"
                    "create table photos (photo_key text, record_key text);")
    r.executemany("insert into records values (?,?)", records)
    r.executemany("insert into photos values (?,?)", photos)
    r.commit()
    r.close()
    return path


def hold_file(path, keys):
    path.write_bytes("".join(k + "\n" for k in keys).encode("utf-8"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def build_site(tmp_path):
    """A store shaped like the paper-era one: v1 train and val photos decided (every
    confirmation, several reasons, a re-decision, a decision after the cut), an MO photo, a
    photo of a record release v1 left out (v1_split none) and an undecided photo; plus, as
    after the queues reopened, one decided photo of each of two v1-test records ("20" and
    MO "mo:21"), and a v1-test record ("22") whose photo is undecided."""
    live = tmp_path / "state" / "human-feedback.sqlite"
    s = fb.Store(live)
    s.seed([item("inat:1", INAT + "10", "R6 micrograph probe p=0.91"),
            item("inat:2", INAT + "10", "R6 view habitat p=0.30; R7 uv probe p=0.99"),
            item("inat:3", INAT + "11", "R7 tree probe p=0.97"),
            item("inat:4", INAT + "11", "R6b view slip p=0.81 (label cosine 0.44)"),
            item("inat:5", INAT + "11", "reviewer: left out in photo review"),
            item("mo:6", MO + "12", "R6 micrograph probe p=0.88"),
            item("inat:7", INAT + "13", "R7 drawing probe p=0.96"),
            item("inat:8", INAT + "11", "R6 view dried p=0.40")], "seed")
    decide(s, "inat:1", "exclude_photo", T0 + 1, "micrograph", "summary: shown")
    decide(s, "inat:2", "exclude_photo", T0 + 2, "uv", "summary: flipped")
    decide(s, "inat:3", "keep_photo", T0 + 3, None, "summary: flipped", user="ana-9", name="Ana")
    decide(s, "inat:4", "exclude_photo", T0 + 4, "habitat", "summary: saved")
    decide(s, "inat:4", "exclude_photo", T0 + 5, "slip", None)              # the latest wins
    decide(s, "inat:5", "exclude_photo", T0 + 6, "species_description", None)
    decide(s, "mo:6", "exclude_photo", T0 + 7, "micrograph", "summary: shown")
    decide(s, "inat:7", "exclude_photo", T0 + 8, "drawing", "summary: shown")
    decide(s, "inat:1", "keep_photo", T0 + 5000, None, "summary: flipped")  # after the cut
    rel = release(tmp_path / "v1.sqlite",
                  [("10", "train"), ("11", "val"), ("mo:12", "train"), ("13", None)],
                  [("inat:1", "10"), ("inat:2", "10"), ("inat:3", "11"), ("inat:4", "11"),
                   ("inat:5", "11"), ("mo:6", "mo:12"), ("inat:7", "13"), ("inat:8", "11")])
    test = release(tmp_path / "v1-test.sqlite",
                   [("20", "test"), ("mo:21", "test"), ("22", "test"), ("23", None)],
                   [("inat:30", "20"), ("mo:31", "mo:21"), ("inat:32", "22")])
    return s, live, rel, test, tmp_path


@pytest.fixture
def site(tmp_path):
    return build_site(tmp_path)


def reopen(s):
    """The queues after the 15 Nov ruling: photos of v1-test records queued and decided."""
    s.seed([item("inat:30", INAT + "20", "R6 micrograph probe p=0.93"),
            item("mo:31", MO + "21", "R7 uv probe p=0.99"),
            item("inat:32", INAT + "22", "R6 view habitat p=0.50")], "nightly")
    decide(s, "inat:30", "exclude_photo", T0 + 20, "micrograph", "summary: shown")
    decide(s, "mo:31", "keep_photo", T0 + 21, None, "summary: flipped")


def run(site, name, mode, **kw):
    s, live, rel, test, tmp = site
    return fx.export(backup(live, tmp / name), rel, CUT, KEY, live, mode=mode,
                     v1_test_release=test, **kw)


def rows_of(data):
    lines = data.decode().split("\n")
    return {l.split("\t")[0]: dict(zip(fx.COLUMNS, l.split("\t"))) for l in lines[1:] if l}


# --- paper mode: the frozen exports' bytes ------------------------------------------------------

# What the exporter wrote before it had a mode, on the fixture above: the same bytes at Vision
# cd3eb23 (the E34/E35 training code), 264d457 (-a4/-a5 scoring code) and main 2eb5478, its
# `export` unchanged since d7befa3 (21 Oct). Paper mode must give exactly these, so a frozen
# export is reproduced byte for byte; the old summary's every field is kept as it was.
GOLDEN = "".join("\t".join(row) + "\n" for row in [
    fx.COLUMNS,
    ("inat:1", "1", "10", "train", "exclude_photo", "micrograph", "1",
     "2026-09-21T14:13:21.000000Z", "2d22c24c81100272", "shown", "R6 micrograph probe"),
    ("inat:2", "2", "10", "train", "exclude_photo", "uv", "2",
     "2026-09-21T14:13:22.000000Z", "2d22c24c81100272", "flipped",
     "R6 view habitat; R7 uv probe"),
    ("inat:3", "3", "11", "val", "keep_photo", "", "3",
     "2026-09-21T14:13:23.000000Z", "166a1a9f73721d12", "flipped", "R7 tree probe"),
    ("inat:4", "4", "11", "val", "exclude_photo", "slip", "5",
     "2026-09-21T14:13:25.000000Z", "2d22c24c81100272", "record", "R6b view slip"),
    ("inat:5", "5", "11", "val", "exclude_photo", "species_description", "6",
     "2026-09-21T14:13:26.000000Z", "2d22c24c81100272", "record",
     "reviewer: left out in photo review"),
    ("inat:7", "7", "13", "none", "exclude_photo", "drawing", "8",
     "2026-09-21T14:13:28.000000Z", "2d22c24c81100272", "shown", "R7 drawing probe"),
    ("mo:6", "1000000000006", "mo:12", "train", "exclude_photo", "micrograph", "7",
     "2026-09-21T14:13:27.000000Z", "2d22c24c81100272", "shown", "R6 micrograph probe")])
GOLDEN_SHA256 = "6e36b2d5aaad8d3f4ba1ba832a126895380c241f7c7545808c499f8e6ee7aa0f"
GOLDEN_SUMMARY = {
    "as_of": CUT, "rows": 7, "sha256": GOLDEN_SHA256,
    "key_fingerprint": "79d99a73ad8daf82a8654984d965e465e0a55463d0b526ddeca0787dad9e7502",
    "by_decision": {"exclude_photo": 6, "keep_photo": 1},
    "exclusions_by_confirmation": {"flipped": 1, "record": 2, "shown": 3},
    "exclusions_by_reason": {"drawing": 1, "micrograph": 2, "slip": 1,
                             "species_description": 1, "uv": 1},
    "by_split": {"none": 1, "train": 3, "val": 3}}


def test_paper_mode_reproduces_the_frozen_exporters_bytes(site):
    data, summary = run(site, "copy.sqlite", "paper")
    assert hashlib.sha256(GOLDEN.encode()).hexdigest() == GOLDEN_SHA256
    assert data == GOLDEN.encode("utf-8")
    assert {k: summary[k] for k in GOLDEN_SUMMARY} == GOLDEN_SUMMARY
    assert summary["mode"] == "paper" and summary["v1_test_photos"] == 0


def test_paper_mode_bytes_do_not_depend_on_the_hold_list(site):
    s, live, rel, test, tmp = site
    path, sha = hold_file(tmp / "hold.txt", ["20", "99"])
    data, summary = run(site, "copy.sqlite", "paper", hold_keys=path, hold_sha256=sha)
    assert data == GOLDEN.encode("utf-8")
    assert summary["hold"] == {"checked": True, "held_keys": 2}


def test_paper_mode_refuses_a_decided_photo_on_a_v1_test_record(site):
    s = site[0]
    reopen(s)
    with pytest.raises(fx.ExportError, match="v1-test") as e:
        run(site, "copy.sqlite", "paper")
    assert "2 decided photo(s)" in str(e.value)
    assert "inat:30" in str(e.value) and "mo:31" in str(e.value)
    assert "inat:32" not in str(e.value)                     # undecided: nothing to refuse


def test_paper_mode_knows_a_v1_test_photo_by_its_record_or_by_the_photo_alone(site):
    s, live, rel, test, tmp = site
    s.seed([item("inat:40", INAT + "23", "R7 uv probe p=0.99")], "nightly")     # record listed
    decide(s, "inat:40", "exclude_photo", T0 + 30, "uv", "summary: shown")
    with pytest.raises(fx.ExportError, match="1 decided photo"):
        run(site, "a.sqlite", "paper")
    s.conn.execute("delete from decisions where item_id = (select id from items where "
                   "subject = 'inat:40')")
    s.conn.commit()
    s.seed([item("inat:32", INAT + "77", "R6 view habitat p=0.5")], "nightly")  # photo listed
    decide(s, "inat:32", "exclude_photo", T0 + 31, "habitat", "summary: shown")
    with pytest.raises(fx.ExportError, match="inat:32"):
        run(site, "b.sqlite", "paper")


def test_paper_mode_refuses_release_v1s_own_test_split_as_before(site):
    s, live, rel, test, tmp = site
    r = sqlite3.connect(rel)
    r.execute("insert into records values ('14', 'test')")
    r.execute("insert into photos values ('inat:9', '14')")
    r.commit()
    r.close()
    s.seed([item("inat:9", INAT + "14", "R7 uv probe p=0.99")], "x")
    decide(s, "inat:9", "exclude_photo", T0 + 9, "uv", "summary: shown")
    with pytest.raises(fx.ExportError, match="v1-test"):
        run(site, "copy.sqlite", "paper")


# --- serving mode -------------------------------------------------------------------------------

def test_serving_mode_includes_v1_test_photos_marked_test_and_counts_them(site):
    s, live, rel, test, tmp = site
    reopen(s)
    path, sha = hold_file(tmp / "hold.txt", ["99"])
    data, summary = run(site, "copy.sqlite", "serving", hold_keys=path, hold_sha256=sha)
    rows = rows_of(data)
    assert (rows["inat:30"]["v1_split"], rows["inat:30"]["record_key"],
            rows["inat:30"]["decision"], rows["inat:30"]["reason"]) == (
        "test", "20", "exclude_photo", "micrograph")
    assert (rows["mo:31"]["v1_split"], rows["mo:31"]["record_key"],
            rows["mo:31"]["decision"]) == ("test", "mo:21", "keep_photo")
    assert "inat:32" not in rows                              # undecided: never in the file
    assert summary["mode"] == "serving" and "v2.1" in summary["mode_rule"]
    assert (summary["v1_test_photos"], summary["v1_test_exclusions"],
            summary["v1_test_records"]) == (2, 1, 2)
    assert summary["by_split"] == {"none": 1, "test": 2, "train": 3, "val": 3}
    assert summary["records_by_split"] == {"none": 1, "test": 2, "train": 2, "val": 1}
    assert summary["v1_test_release_sha256"] == hashlib.sha256(test.read_bytes()).hexdigest()
    assert summary["sha256"] == hashlib.sha256(data).hexdigest()
    # every other row is the paper mode's, byte for byte
    paper = [l for l in GOLDEN.split("\n") if l]
    assert [l for l in data.decode().split("\n") if l and "\ttest\t" not in l] == paper


def test_serving_mode_matches_paper_mode_while_no_v1_test_photo_is_decided(site):
    s, live, rel, test, tmp = site
    path, sha = hold_file(tmp / "hold.txt", ["20"])
    data, summary = run(site, "copy.sqlite", "serving", hold_keys=path, hold_sha256=sha)
    assert data == GOLDEN.encode("utf-8") and summary["v1_test_photos"] == 0


def test_serving_mode_keeps_the_freeze_rules_same_bytes_twice_and_nothing_after_the_cut(site):
    s, live, rel, test, tmp = site
    reopen(s)
    path, sha = hold_file(tmp / "hold.txt", ["99"])
    one, _ = run(site, "a.sqlite", "serving", hold_keys=path, hold_sha256=sha)
    two, _ = run(site, "b.sqlite", "serving", hold_keys=path, hold_sha256=sha)
    decide(s, "inat:32", "exclude_photo", T0 + 9000, "habitat", "summary: shown")  # after cut
    decide(s, "inat:30", "keep_photo", T0 + 9001, None, "summary: flipped")       # after cut
    three, _ = run(site, "c.sqlite", "serving", hold_keys=path, hold_sha256=sha)
    assert one == two == three


# --- held records (MV_FEEDBACK_HOLD_KEYS) -------------------------------------------------------

@pytest.mark.parametrize("mode", ["paper", "serving"])
def test_a_decided_photo_on_a_held_record_refuses_either_mode_and_names_no_key(site, mode):
    s, live, rel, test, tmp = site
    reopen(s)
    path, sha = hold_file(tmp / "hold.txt", ["mo:21"])
    with pytest.raises(fx.ExportError, match="held out of every queue") as e:
        run(site, "copy.sqlite", mode, hold_keys=path, hold_sha256=sha)
    assert "mo:21" not in str(e.value) and "mo:31" not in str(e.value)


def test_a_held_record_outside_v1_test_refuses_the_serving_export_too(site):
    s, live, rel, test, tmp = site
    path, sha = hold_file(tmp / "hold.txt", ["11"])           # a v1 val record, for the test
    with pytest.raises(fx.ExportError, match="3 decided photo"):
        run(site, "copy.sqlite", "serving", hold_keys=path, hold_sha256=sha)


def test_serving_mode_refuses_to_run_without_the_hold_list(site):
    with pytest.raises(fx.ExportError, match="serving mode needs the hold list"):
        run(site, "copy.sqlite", "serving")


@pytest.mark.parametrize("mode", ["paper", "serving"])
def test_a_hold_list_that_cant_be_trusted_refuses_either_mode(site, mode):
    s, live, rel, test, tmp = site
    path, sha = hold_file(tmp / "hold.txt", ["20"])
    for kw in ({"hold_keys": path, "hold_sha256": "0" * 64},            # sha does not match
               {"hold_keys": path},                                     # no sha
               {"hold_keys": tmp / "missing.txt", "hold_sha256": sha},  # can't be read
               {"hold_sha256": sha}):                                   # sha, no list
        with pytest.raises(fx.ExportError, match="hold list"):
            run(site, "copy.sqlite", mode, **kw)


# --- the mode is an explicit argument -----------------------------------------------------------

def test_an_export_without_a_mode_is_refused(site):
    s, live, rel, test, tmp = site
    copy = backup(live, tmp / "copy.sqlite")
    with pytest.raises(TypeError, match="mode"):
        fx.export(copy, rel, CUT, KEY, live, v1_test_release=test)
    for bad in (None, "", "both", "Paper"):
        with pytest.raises(fx.ExportError, match="name the export's mode"):
            fx.export(copy, rel, CUT, KEY, live, mode=bad, v1_test_release=test)


def test_an_export_without_the_v1_test_release_is_refused(site):
    s, live, rel, test, tmp = site
    copy = backup(live, tmp / "copy.sqlite")
    with pytest.raises(TypeError, match="v1_test_release"):
        fx.export(copy, rel, CUT, KEY, live, mode="paper")
    with pytest.raises(fx.ExportError, match="name the v1-test release"):
        fx.export(copy, rel, CUT, KEY, live, mode="paper", v1_test_release=None)
    with pytest.raises(fx.ExportError, match="no v1-test release"):
        fx.export(copy, rel, CUT, KEY, live, mode="paper", v1_test_release=tmp / "nope.sqlite")
    with pytest.raises(fx.ExportError, match="not the v1-test release"):    # v1 given twice
        fx.export(copy, rel, CUT, KEY, live, mode="paper", v1_test_release=rel)


ARGS = ["feedback", "export-photo-exclusions", "--store-copy", "c.sqlite", "--release",
        "v1.sqlite", "--as-of", CUT, "--key-file", "k", "--out", "o.tsv"]


@pytest.mark.parametrize("extra", [[], ["--v1-test-release", "t.sqlite"],
                                   ["--mode", "paper"],
                                   ["--mode", "both", "--v1-test-release", "t.sqlite"]])
def test_the_command_line_refuses_a_missing_or_unknown_mode_or_v1_test_release(extra, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(ARGS + extra)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--mode" in err or "--v1-test-release" in err


def test_the_command_writes_the_tsv_and_a_summary_that_records_the_mode(site, monkeypatch):
    s, live, rel, test, tmp = site
    reopen(s)
    path, sha = hold_file(tmp / "hold.txt", ["99"])
    monkeypatch.setenv("MV_FEEDBACK_HOLD_KEYS", str(path))
    monkeypatch.setenv("MV_FEEDBACK_HOLD_SHA256", sha)
    (tmp / "key").write_bytes(KEY + b"\n")
    out = tmp / "serving.tsv"
    args = argparse.Namespace(
        action="export-photo-exclusions", key_file=str(tmp / "key"),
        store_copy=str(backup(live, tmp / "copy.sqlite")), release=str(rel), as_of=CUT,
        out=str(out), expect_key_fingerprint=None, mode="serving",
        v1_test_release=str(test), hold_keys=None, hold_sha256=None)
    cli.cmd_feedback(None, args)
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "serving" and summary["v1_test_photos"] == 2
    assert summary["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert summary["hold"] == {"checked": True, "held_keys": 1}
    args.mode, args.out = "paper", str(tmp / "paper.tsv")
    with pytest.raises(SystemExit, match="export refused: .*v1-test"):
        cli.cmd_feedback(None, args)
    assert not (tmp / "paper.tsv").exists()
