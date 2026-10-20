"""The paper's seal on the Human Feedback queues (the coordinator and the paper coordinator,
2026-10-20): no queue may hold a v1-test record, one first green after dataset release v1's
label snapshot. Every way an item is added goes through Store.seed, which refuses such a
record, and one whose date the box cannot tell, counting both; the pinned v1 record keys are
accepted whatever their date; with the seal off (MV_PROSPECTIVE_SEAL=off) everything passes."""

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
    assert fb.SealGate(dates.get, frozenset(), None).judge("new") == "ok"   # seal lifted


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


def test_with_the_seal_off_everything_is_accepted(tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    add_records(conn, {"12": "2026-10-15"})
    q = queue_file(tmp_path, [record_row(1, "12"), record_row(2, "99")])
    out = fb.seed_files(fb.Store(tmp_path / "f.sqlite"), conn, queue=q)[0]
    assert out["added"] == 2 and out["refused_sealed"] == 0 and out["refused_unknown"] == 0


def test_a_moved_seal_moves_the_line(tmp_path, conn, monkeypatch):
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "2026-10-20T00:00:00+00:00")
    add_records(conn, {"12": "2026-10-15"})
    out = fb.seed_files(fb.Store(tmp_path / "f.sqlite"), conn,
                        queue=queue_file(tmp_path, [record_row(1, "12")]))[0]
    assert out["added"] == 1


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
    assert fb.seal_check(s, fb.SealGate(lambda k: None, frozenset(), None))["ok"] is True


def test_the_pinned_v1_keys_file_is_written_once(tmp_path):
    db = sqlite3.connect(":memory:")
    db.execute("create table records (record_key text, included integer, split text)")
    db.executemany("insert into records values (?,?,?)", [("b", 1, "train"), ("a", 1, "val"),
                                                         ("c", 0, None)])
    out = fb.write_record_keys(SimpleNamespace(db=db, id="v1"), tmp_path / "k.txt")
    assert out["records"] == 2 and (tmp_path / "k.txt").read_text() == "a\nb\n"
    with pytest.raises(FileExistsError):
        fb.write_record_keys(SimpleNamespace(db=db, id="v1"), tmp_path / "k.txt")
