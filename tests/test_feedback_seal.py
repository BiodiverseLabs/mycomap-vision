"""The paper's seal on the Human Feedback queues (the coordinator and the paper coordinator,
2026-10-20): no queue may hold a v1-test record, one first green after dataset release v1's
label snapshot. Every way an item is added goes through Store.seed, which refuses such a
record, and one whose date the box cannot tell, counting both; the pinned v1 record keys are
accepted whatever their date. With the seal off (MV_PROSPECTIVE_SEAL=off) only the photo queues
(feedback.POST_SEAL_QUEUES) take post-seal records; every other queue keeps the paper's fixed
seal until submission (the paper coordinator, 15 Nov 2026)."""

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest
from test_feedback_api import make_site

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr
from mycomap_vision import prospective

SEAL = prospective.SEAL_DEFAULT                      # 2026-10-12T13:17:51+00:00


def add_records(conn, dated: dict):
    conn.executemany("insert into records (observation_id, source, scientific_name, "
                     "north_america, exported_at, validated_on) values (?, 'inat', 'A x', 1, "
                     "'now', ?)", list(dated.items()))
    conn.commit()


def queue_file(tmp_path, rows):
    head = ("item_id\titem_type\tgranularity\tselection_rule\tmodel_flagged\tsubject\tlink\t"
            "records\tcurrent\tproposed\tevidence\tin_v1\tv1_val_outcome\n")
    f = tmp_path / "q.tsv"
    f.write_text(head + "".join(rows), encoding="utf-8")
    return f


def record_row(n, key, link=""):
    return f"q{n}\tlabel_vs_own_dna\trecord\tR5\t0\t{key}\t{link}\t1\tA x\t\t\t\t\n"


def test_the_rule_for_one_record():
    dates = {"old": "2026-09-01", "new": "2026-10-14", "sealday": "2026-10-12", "nodate": None}
    gate = fb.SealGate(dates.get, frozenset({"pinned"}), SEAL)
    assert gate.judge("old") == "ok"
    assert gate.judge("new") == "sealed"
    assert gate.judge("sealday") == "unknown"        # before or after 13:17:51? the box can't tell
    assert gate.judge("nodate") == "unknown"
    assert gate.judge("missing") == "unknown"
    assert fb.SealGate(lambda k: "2026-10-12", frozenset({"pinned"}), SEAL).judge("pinned") == "ok"
    lifted = fb.SealGate(dates.get, frozenset(), None)                       # seal lifted
    assert lifted.judge("new", "non_specimen_photo") == "ok"
    assert lifted.judge("new") == "sealed"            # no item type: the strict way


def test_seeding_refuses_v1_test_and_unknown_records_and_counts_them(tmp_path, conn):
    add_records(conn, {"11": "2026-09-01", "12": "2026-10-15", "13": "2026-10-12"})
    q = queue_file(tmp_path, [
        record_row(1, "11"), record_row(2, "12"), record_row(3, "13"), record_row(4, "99"),
        "q5\tgroup_label\tname\tR2\t0\tLaccaria laccata group\t\t1\tLaccaria laccata group\t\t\t\t\n",
        # a photo item names its record by its link
        "q6\tnon_specimen_photo\tphoto\tR6 view slip p=0.9\t0\tinat:5\t"
        "https://www.inaturalist.org/observations/12\t1\tA x\t\t\t\t\n"])
    s = fb.Store(tmp_path / "f.sqlite")
    out = fb.seed_files(s, conn, queue=q)[0]
    assert out["added"] == 2                         # record 11 and the name-level group
    assert (out["refused_sealed"], out["refused_unknown"]) == (2, 2)
    assert out["refused_records"] == ["12", "13", "99"]
    queued = {r[0] for r in s.conn.execute("select subject from items")}
    assert queued == {"11", "Laccaria laccata group"}


def test_the_pinned_v1_keys_let_seal_day_records_in(tmp_path, conn):
    add_records(conn, {"13": "2026-10-12"})
    keys = tmp_path / "v1-keys.txt"
    keys.write_text("13\n", encoding="utf-8")
    s = fb.Store(tmp_path / "f.sqlite", gate=fb.SealGate.from_manifest(conn, v1_keys_path=keys))
    out = s.seed([{"item_type": "label_vs_own_dna", "subject": "13", "rule": "R5",
                   "record_keys": ["13"]}], "t")
    assert out["added"] == 1 and out["refused_unknown"] == 0


def test_with_the_seal_off_a_label_queue_still_refuses_post_seal_and_unknown_records(
        tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    add_records(conn, {"12": "2026-10-15"})
    q = queue_file(tmp_path, [record_row(1, "12"), record_row(2, "99")])
    out = fb.seed_files(fb.Store(tmp_path / "f.sqlite"), conn, queue=q)[0]
    assert out["added"] == 0 and out["refused_sealed"] == 1 and out["refused_unknown"] == 1


def test_a_moved_seal_moves_the_line_for_the_photo_queues_only(tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "2026-10-20T00:00:00+00:00")
    add_records(conn, {"12": "2026-10-15"})
    s = fb.Store(tmp_path / "f.sqlite", gate=fb.SealGate.from_manifest(conn))
    out = s.seed([record_item("12"), photo_item_of("12")], "t")
    assert (out["added"], out["refused_sealed"]) == (1, 1)
    assert queued(s) == {"inat:12"}


def test_a_reviewer_cannot_add_a_photo_item_for_a_sealed_record(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([{"item_type": "non_specimen_photo", "subject": "inat:1", "rule": "R6",
             "links": ["https://www.inaturalist.org/observations/20"], "current": "A x"}], "t")
    s.gate = fb.SealGate({"20": "2026-10-16"}.get, frozenset(), SEAL)
    with pytest.raises(fb.FeedbackError, match="sealed test"):
        pr.decide_record(s, "20", [{"photo_key": "inat:2", "decision": "exclude",
                                    "reason": "slip"}], {"sub": "a"}, {"inat:1", "inat:2"})
    assert [r[0] for r in s.conn.execute("select subject from items")] == ["inat:1"]


def test_the_server_opens_its_store_with_the_seal(conn, tmp_path):
    c, fstore, _ = make_site(conn, tmp_path)
    assert c.app.state.feedback is fstore and fstore.gate is not None
    assert fstore.gate.seal == SEAL
    assert fstore.gate.judge("100") == "ok"          # make_site's records: validated in 2026


def test_seal_check_reports_failing_items_and_their_decisions(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([{"item_type": "label_vs_own_dna", "subject": k, "rule": "R5", "record_keys": [k]}
            for k in ("old", "new", "nodate")] +
           [{"item_type": "group_label", "subject": "X group", "rule": "R2"}], "t")
    new_id = next(r[0] for r in s.conn.execute("select id from items where subject = 'new'"))
    s.decide(new_id, "unsure", {"sub": "a"})
    gate = fb.SealGate({"old": "2026-01-01", "new": "2026-10-14"}.get, frozenset(), SEAL)
    out = fb.seal_check(s, gate)
    q = out["queues"]["label_vs_own_dna"]
    assert (q["items"], q["sealed"], q["unknown"], q["decisions_on_failing"]) == (3, 1, 1, 1)
    assert out["queues"]["group_label"]["sealed"] == 0 and out["ok"] is False
    assert [x["record"] for x in out["sealed"]] == ["new"]
    assert fb.seal_check(s, fb.SealGate(lambda k: "2026-01-01", frozenset(), None))["ok"] is True


def test_the_pinned_v1_keys_file_is_written_once(tmp_path):
    db = sqlite3.connect(":memory:")
    db.execute("create table records (record_key text, included integer, split text)")
    db.executemany("insert into records values (?,?,?)", [("b", 1, "train"), ("a", 1, "val"),
                                                         ("c", 0, None)])
    out = fb.write_record_keys(SimpleNamespace(db=db, id="v1"), tmp_path / "k.txt")
    assert out["records"] == 2 and (tmp_path / "k.txt").read_text() == "a\nb\n"
    with pytest.raises(FileExistsError):
        fb.write_record_keys(SimpleNamespace(db=db, id="v1"), tmp_path / "k.txt")


# --- the held keys (the paper coordinator): a private list of records (a blind label check)
# held out of every queue whatever the seal says, until the list is emptied. Its records are only
# ever counted, never named, and a list that can't be trusted holds every record.

HOLD_SETTINGS = ("MV_FEEDBACK_HOLD_KEYS", "MV_FEEDBACK_HOLD_SHA256")


def hold_list(tmp_path, keys, name="hold.txt"):
    body = "".join(f"{k}\n" for k in keys).encode("utf-8")
    f = tmp_path / name
    f.write_bytes(body)
    return f, hashlib.sha256(body).hexdigest()


def record_item(*keys):
    return {"item_type": "label_vs_own_dna", "subject": keys[0], "rule": "R5",
            "record_keys": list(keys)}


def queued(s):
    return {r[0] for r in s.conn.execute("select subject from items")}


def photo_item_of(key, item_type="non_specimen_photo"):
    return {"item_type": item_type, "subject": f"inat:{key}", "rule": "R6",
            "links": [f"https://www.inaturalist.org/observations/{key}"]}


def test_a_held_record_is_judged_held_before_the_seal_and_held_is_a_rows_worst_verdict():
    dates = {"old": "2026-09-01", "new": "2026-10-20", "4242": "2026-09-01"}
    gate = fb.SealGate(dates.get, frozenset(), SEAL, held=frozenset({"4242"}))
    assert gate.judge("4242") == "held" and gate.judge("old") == "ok"
    lifted = fb.SealGate(dates.get, frozenset(), None, held=frozenset({"4242"}))
    assert lifted.judge("4242") == "held"                # the seal off holds it still
    assert lifted.judge("new", "photo_view") == "ok"
    assert gate.judge_row({"record_keys": ["old", "4242"]}) == ("held", "4242")
    assert gate.judge_row({"record_keys": ["new", "4242"]}) == ("held", "4242")
    assert gate.judge_row({"record_keys": ["old", "new"]}) == ("sealed", "new")
    assert gate.judge_row({"links": ["https://www.inaturalist.org/observations/4242"]})[0] == "held"


@pytest.mark.parametrize("seal", ["on", "off"])
def test_a_held_record_is_refused_by_seeding_whatever_the_seal_and_never_named(
        tmp_path, conn, monkeypatch, seal):
    if seal == "off":
        monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    else:
        monkeypatch.delenv("MV_PROSPECTIVE_SEAL", raising=False)
    add_records(conn, {"11": "2026-09-01", "4242": "2026-09-01", "12": "2026-10-15"})
    f, sha = hold_list(tmp_path, ["4242", "7777"])
    monkeypatch.setenv("MV_FEEDBACK_HOLD_KEYS", str(f))
    monkeypatch.setenv("MV_FEEDBACK_HOLD_SHA256", sha)
    s = fb.Store(tmp_path / "f.sqlite", gate=fb.SealGate.from_manifest(conn))
    assert s.gate.held == {"4242", "7777"} and not s.gate.hold_all
    out = s.seed([record_item("11"), record_item("4242"), record_item("12")], "t")
    assert out["refused_held"] == 1
    assert "4242" not in out["refused_records"] and "4242" not in json.dumps(out)
    assert "4242" not in queued(s) and "11" in queued(s)
    assert "12" not in queued(s) and out["refused_sealed"] == 1   # a label queue: sealed


def test_a_held_record_is_refused_even_when_it_is_on_the_pinned_v1_keys(tmp_path, conn):
    add_records(conn, {"13": "2026-10-12"})
    keys = tmp_path / "v1-keys.txt"
    keys.write_text("13\n", encoding="utf-8")
    f, sha = hold_list(tmp_path, ["13"])
    gate = fb.SealGate.from_manifest(conn, seal=SEAL, v1_keys_path=keys, hold_keys_path=f,
                                     hold_sha256=sha)
    assert "13" in gate.v1_keys and gate.judge("13") == "held"
    out = fb.Store(tmp_path / "f.sqlite", gate=gate).seed([record_item("13")], "t")
    assert (out["added"], out["refused_held"], out["refused_unknown"]) == (0, 1, 0)


def test_an_item_naming_one_ok_record_and_one_held_record_is_held(tmp_path):
    gate = fb.SealGate({"11": "2026-09-01", "4242": "2026-09-01"}.get, frozenset(), SEAL,
                       held=frozenset({"4242"}))
    s = fb.Store(tmp_path / "f.sqlite", gate=gate)
    out = s.seed([record_item("11", "4242")], "t")
    assert (out["added"], out["refused_held"], out["refused_records"]) == (0, 1, [])
    assert queued(s) == set()


@pytest.mark.parametrize("case", ["wrong sha256", "no sha256", "no file", "sha256 but no list"])
def test_a_hold_list_that_cannot_be_trusted_holds_every_record(tmp_path, conn, monkeypatch, case):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    for name in HOLD_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    add_records(conn, {"11": "2026-09-01"})
    f, sha = hold_list(tmp_path, ["4242"])
    path, digest = {"wrong sha256": (f, hashlib.sha256(b"another list").hexdigest()),
                    "no sha256": (f, None),
                    "no file": (tmp_path / "gone.txt", sha),
                    "sha256 but no list": (None, sha)}[case]
    if path is not None:
        monkeypatch.setenv("MV_FEEDBACK_HOLD_KEYS", str(path))
    if digest is not None:
        monkeypatch.setenv("MV_FEEDBACK_HOLD_SHA256", digest)
    gate = fb.SealGate.from_manifest(conn)                    # never raises at startup
    assert gate.hold_all and gate.hold_problem and "4242" not in gate.hold_problem
    assert gate.judge("11") == "held" and gate.judge("4242") == "held"
    out = fb.Store(tmp_path / "f.sqlite", gate=gate).seed([record_item("11")], "t")
    assert (out["added"], out["refused_held"]) == (0, 1)


def test_the_right_sha256_or_no_hold_list_at_all_holds_nothing_else(tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    for name in HOLD_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert not fb.SealGate.from_manifest(conn).hold_all          # no list: nothing held
    f, sha = hold_list(tmp_path, ["4242"])
    gate = fb.SealGate.from_manifest(conn, hold_keys_path=f, hold_sha256=sha.upper())
    assert not gate.hold_all and gate.judge("4242", "photo_view") == "held"
    assert gate.judge("11", "photo_view") == "ok"


def test_an_emptied_hold_list_releases_its_records(tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    add_records(conn, {"4242": "2026-09-01"})
    f, sha = hold_list(tmp_path, [])
    gate = fb.SealGate.from_manifest(conn, hold_keys_path=f, hold_sha256=sha)
    assert not gate.hold_all and gate.held == frozenset()
    out = fb.Store(tmp_path / "f.sqlite", gate=gate).seed([record_item("4242")], "t")
    assert (out["added"], out["refused_held"]) == (1, 0)


def test_seal_check_fails_on_a_held_item_and_names_it_by_item_id_only(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([record_item("old"), record_item("4242")], "t")
    held_id = next(r[0] for r in s.conn.execute("select id from items where subject = ?",
                                                ("4242",)))
    s.decide(held_id, "unsure", {"sub": "a"})
    gate = fb.SealGate(lambda k: "2026-01-01", frozenset(), None, held=frozenset({"4242"}))
    out = fb.seal_check(s, gate)
    q = out["queues"]["label_vs_own_dna"]
    assert (q["items"], q["held"], q["sealed"], q["unknown"], q["decisions_on_failing"]) == (
        2, 1, 0, 0, 1)
    assert out["held"] == [{"item": held_id, "type": "label_vs_own_dna"}]
    assert out["ok"] is False and out["held_keys"] == 1
    assert "4242" not in json.dumps(out)
    assert fb.seal_check(s, fb.SealGate(lambda k: "2026-01-01", frozenset(), None))["ok"] is True
    untrusted = fb.SealGate(lambda k: "2026-01-01", frozenset(), None, hold_all=True,
                            hold_problem="the hold list can't be read")
    out = fb.seal_check(s, untrusted)
    assert out["ok"] is False and out["hold_all"] and len(out["held"]) == 2


# --- which queues the setting opens (the paper coordinator, 15 Nov 2026): until submission only
# the photo queues, which show no species answer, take records first green after the paper's seal;
# public Human Feedback shows Vision's top 5 beside every other queue's names.

OTHER_TYPES = sorted(set(fb.TYPES) - set(fb.POST_SEAL_QUEUES))
POST_SEAL_DATES = {"12": "2026-10-15"}


def typed_item(item_type, key="12"):
    return {"item_type": item_type, "subject": key, "rule": "R", "record_keys": [key]}


def test_only_the_two_photo_queues_follow_the_setting():
    assert set(fb.POST_SEAL_QUEUES) == {"non_specimen_photo", "photo_view"}
    assert set(fb.POST_SEAL_QUEUES) <= set(fb.TYPES)


@pytest.mark.parametrize("item_type", fb.POST_SEAL_QUEUES)
def test_with_the_seal_off_a_photo_queue_takes_a_post_seal_record(tmp_path, item_type):
    s = fb.Store(tmp_path / "f.sqlite", gate=fb.SealGate(POST_SEAL_DATES.get, frozenset(), None))
    out = s.seed([photo_item_of("12", item_type)], "t")
    assert (out["added"], out["refused_sealed"], out["refused_unknown"]) == (1, 0, 0)


@pytest.mark.parametrize("item_type", OTHER_TYPES)
def test_with_the_seal_off_every_other_queue_refuses_a_post_seal_record(tmp_path, item_type):
    s = fb.Store(tmp_path / "f.sqlite", gate=fb.SealGate(POST_SEAL_DATES.get, frozenset(), None))
    out = s.seed([typed_item(item_type)], "t")
    assert (out["added"], out["refused_sealed"]) == (0, 1)
    assert queued(s) == set()


@pytest.mark.parametrize("item_type", fb.POST_SEAL_QUEUES)
def test_with_the_seal_on_the_photo_queues_still_refuse_a_post_seal_record(tmp_path, item_type):
    s = fb.Store(tmp_path / "f.sqlite", gate=fb.SealGate(POST_SEAL_DATES.get, frozenset(), SEAL))
    out = s.seed([photo_item_of("12", item_type)], "t")
    assert (out["added"], out["refused_sealed"]) == (0, 1)


@pytest.mark.parametrize("item_type", fb.POST_SEAL_QUEUES)
def test_with_the_seal_off_a_photo_queue_refuses_a_held_record(tmp_path, item_type):
    gate = fb.SealGate(POST_SEAL_DATES.get, frozenset(), None, held=frozenset({"12"}))
    out = fb.Store(tmp_path / "f.sqlite", gate=gate).seed([photo_item_of("12", item_type)], "t")
    assert (out["added"], out["refused_held"]) == (0, 1)


def test_a_pinned_v1_key_is_still_accepted_by_a_label_queue_with_the_seal_off(
        tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    add_records(conn, {"13": "2026-10-12", "14": "2026-10-12"})
    keys = tmp_path / "v1-keys.txt"
    keys.write_text("13\n", encoding="utf-8")
    gate = fb.SealGate.from_manifest(conn, v1_keys_path=keys)
    assert gate.seal is None and "13" in gate.v1_keys          # read with the seal off too
    out = fb.Store(tmp_path / "f.sqlite", gate=gate).seed(
        [record_item("13"), record_item("14")], "t")
    assert (out["added"], out["refused_unknown"]) == (1, 1)     # 14: the seal's day, not pinned


def test_a_judgement_with_no_item_type_is_the_strict_one():
    gate = fb.SealGate(POST_SEAL_DATES.get, frozenset(), "2026-10-20T00:00:00+00:00")
    assert gate.judge("12") == "sealed" and gate.judge("12", "non_specimen_photo") == "ok"
    assert gate.judge_row({"record_keys": ["12"]}) == ("sealed", "12")
    assert gate.judge_row(photo_item_of("12")) == ("ok", None)
    assert gate.judge_row(photo_item_of("12"), "recovered_label") == ("sealed", "12")


def test_seal_check_judges_each_queue_by_its_own_rule(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([record_item("12"), photo_item_of("12"), photo_item_of("12", "photo_view")], "t")
    out = fb.seal_check(s, fb.SealGate(POST_SEAL_DATES.get, frozenset(), None))
    assert out["queues"]["label_vs_own_dna"]["sealed"] == 1
    assert out["queues"]["non_specimen_photo"]["sealed"] == 0
    assert out["queues"]["photo_view"]["sealed"] == 0
    assert out["ok"] is False and [x["type"] for x in out["sealed"]] == ["label_vs_own_dna"]
    assert out["strict_seal"] == SEAL and out["seal"] is None
