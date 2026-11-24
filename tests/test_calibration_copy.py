"""`mv calibration copy` (calcopy.py; Steve, 2026-11-07): another model's calibration written as
the target's eval_runs rows, keyed exactly as serving reads them (evaluate.latest_calibration),
with where it came from; never overwriting the target's rows unless told; a dry run writes
nothing. Synthetic manifests and calibration only."""

import hashlib
import json
import sqlite3

import pytest

from mycomap_vision import calcopy, cli, evaluate, manifest
from mycomap_vision.identify import Identifier

SRC, DST = "step1-ft", "v2-ft"
BAL, PLACE = "nearest+mean-records-balanced", "nearest+mean-records-balanced+place"
S30, S1013 = "penalty-500-s30-2026-10-19", "penalty-500-2026-10-13"


def cal(t, floor=None):
    """A calibration block as mv compare saves one: per rank its temperature, n and sets."""
    out = {}
    for i, rank in enumerate(("species", "genus", "family")):
        c = {"temperature": t + i / 1000, "n": 100 + i, "nll": 1.0, "credit": "x"}
        if floor is not None:
            c["sets"] = {"method": "lac", "target": 0.8, "coverage": 0.81, "floor": floor + i,
                         "n": 120}
        out[rank] = c
    return out


def add_run(conn, backbone, method, calibration, setting=None, top1=0.5):
    report = {"backbone": backbone, "method": method, "all_photos": {"calibration": calibration},
              "first_photo_only": {"species": {"all": {"top1": top1}}}}
    if setting:
        report["place_setting"] = setting
    with conn:
        return conn.execute(
            "insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, "
            "n_reference, n_test, record_set, species_top1, report_json, code_version, created_at) "
            "values ('20261019-180004-36fd87', ?, ?, '2026-08-17', 28, 900, 90, 'abc', ?, ?, "
            "'c0de', '2026-10-19T19:38:31+00:00')",
            (backbone, method, top1, json.dumps(report))).lastrowid


@pytest.fixture
def man(tmp_path):
    """A live-format manifest (manifest.connect: WAL, the full schema) holding the source's
    rows: an older and a newer balanced row, a place row per setting, and a nearest model
    whose newest row has no calibration."""
    path = tmp_path / "manifest.sqlite"
    conn = manifest.connect(path)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    add_run(conn, SRC, BAL, cal(0.050, 0.01))
    add_run(conn, SRC, BAL, cal(0.027, 0.04))                     # newest: read
    add_run(conn, SRC, PLACE, cal(0.030, 0.05), S30)
    add_run(conn, SRC, PLACE, cal(0.033, 0.06), S1013)
    add_run(conn, SRC, "nearest", cal(0.020))
    add_run(conn, SRC, "nearest", {})                             # newest: none to copy
    add_run(conn, "other-ft", BAL, cal(0.9))
    conn.close()
    return path


def run(capsys, *args):
    code = cli.main(["calibration", "copy", *args])
    return code, capsys.readouterr().out


def reads(path, backbone, method):
    conn = sqlite3.connect(path)
    try:
        c = evaluate.latest_calibration(conn, backbone, method)
    finally:
        conn.close()
    return None if c is None else {k: c[k] for k in ("temperatures", "n", "sets")}


def count(path, backbone=None):
    conn = sqlite3.connect(path)
    try:
        sql, params = "select count(*) from eval_runs", ()
        if backbone:
            sql, params = sql + " where backbone = ?", (backbone,)
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


SETTINGS = [("shared", None), ("own", S30), ("own", S1013)]


@pytest.mark.parametrize("choice,setting", SETTINGS)
def test_a_copy_writes_exactly_the_rows_serving_reads(man, capsys, monkeypatch, choice, setting):
    monkeypatch.setenv("MV_PLACE_CALIBRATION", choice)
    if setting:
        monkeypatch.setenv("MV_PLACE_SETTING", setting)
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "beta", "--by", "R1",
                    "--manifest", str(man))
    assert code == 0, out
    for method in (BAL, PLACE, "nearest+mean+place", "nearest"):
        assert reads(man, DST, method) == reads(man, SRC, method), method
    assert reads(man, DST, BAL)["temperatures"]["species"] == pytest.approx(0.027)
    # One row per key serving reads: the balanced row, the place row of each setting; the
    # nearest model's newest row had no calibration, so nothing stands in for it.
    assert count(man, DST) == 3
    assert "skipped: row 6 (nearest): no calibration" in out


def test_a_copied_row_holds_the_calibration_and_where_it_came_from_and_no_scores(man, capsys):
    run(capsys, "--from", SRC, "--to", DST, "--note", "Steve 2026-11-07", "--by", "R1",
        "--manifest", str(man))
    conn = sqlite3.connect(man)
    rows = conn.execute("select comparison_id, species_top1, genus_top1, family_top1, "
                        "species_top1_first_photo, report_json from eval_runs where backbone = ?",
                        (DST,)).fetchall()
    conn.close()
    for cid, *scores, text in rows:
        report = json.loads(text)
        assert cid.endswith("-calcopy") and scores == [None] * 4
        assert set(report) <= {"backbone", "method", "place_setting", "all_photos", calcopy.MARK}
        assert set(report["all_photos"]) == {"calibration"}
        copy = report[calcopy.MARK]
        assert copy["from_backbone"] == SRC and copy["to_backbone"] == DST
        assert copy["note"] == "Steve 2026-11-07" and copy["by"] == "R1"
        assert copy["from_run_id"] in (2, 3, 4) and copy["copied_at"]


def test_copied_rows_are_never_listed_as_results(man, capsys):
    run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man))
    conn = sqlite3.connect(man)
    board = evaluate.scoreboard(conn)
    conn.close()
    assert board and DST not in {r["backbone"] for r in board}


def test_a_copy_refuses_to_overwrite_the_targets_rows_unless_told(man, capsys):
    conn = sqlite3.connect(man)
    add_run(conn, DST, BAL, cal(0.1, 0.2))                         # the target's own row
    conn.close()
    before = count(man)
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man))
    assert code == 2 and "refused: the target already has rows" in out and "row 8 " in out
    assert count(man) == before
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man),
                    "--replace")
    assert code == 0, out
    assert count(man) == before + 3 and count(man, DST) == 4      # added; the old row stays
    assert reads(man, DST, BAL) == reads(man, SRC, BAL)            # the copy is read first


def test_a_second_copy_is_refused_like_any_existing_row(man, capsys):
    assert run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man))[0] == 0
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man))
    assert code == 2 and count(man, DST) == 3


def test_a_dry_run_writes_nothing(man, capsys):
    digest = hashlib.sha256(man.read_bytes()).hexdigest()
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man),
                    "--dry-run")
    assert code == 0 and "dry run: nothing written" in out
    assert f"{DST} / {BAL}" in out and "species T 0.027000 floor 0.040000" in out
    assert hashlib.sha256(man.read_bytes()).hexdigest() == digest
    assert count(man, DST) == 0


def test_the_same_model_or_no_rows_is_refused(man, capsys):
    assert run(capsys, "--from", SRC, "--to", SRC, "--note", "n", "--manifest", str(man))[0] == 2
    code, out = run(capsys, "--from", "nobody-ft", "--to", DST, "--note", "n",
                    "--manifest", str(man))
    assert code == 2 and "nothing to copy" in out and count(man, DST) == 0


def test_rows_can_come_from_another_manifest_opened_read_only(man, tmp_path, capsys):
    target = tmp_path / "box" / "manifest.sqlite"
    target.parent.mkdir()
    manifest.connect(target).close()
    digest = hashlib.sha256(man.read_bytes()).hexdigest()
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(target),
                    "--from-manifest", str(man))
    assert code == 0, out
    assert reads(target, DST, BAL) == reads(man, SRC, BAL)
    assert hashlib.sha256(man.read_bytes()).hexdigest() == digest


# --- from a calibration file (the step-1 format) ---------------------------------------

def step1_file(tmp_path, model=SRC):
    def block(method, setting, temps, floors):
        return {"method": method, "place_setting": setting, "record_set": "fa494dffa022",
                "reference_records": 155648, "test_records": 12874,
                "values": {r: {"temperature": t, "floor": f, "target": 0.8 if r == "species"
                               else 0.9} for r, t, f in zip(("species", "genus", "family"),
                                                            temps, floors)},
                "achieved": {r: {"coverage": 0.8, "n": 12798, "mean_size": 3.0, "nll": 1.6,
                                 "n_nll": 12149} for r in ("species", "genus", "family")}}
    doc = {"model": model, "fit_on": "val", "answer_key": "species-rank-2026-10-18",
           "photos_only": block(BAL, None, (0.027429, 0.030419, 0.030419),
                                (0.042108, 0.244613, 0.508626)),
           "place": block(PLACE, S30, (0.030419, 0.033735, 0.033735),
                          (0.047013, 0.2432, 0.495271))}
    path = tmp_path / f"{model}.calibration-v1-val.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize("choice,expect", [("shared", 0.027429), ("own", 0.030419)])
def test_a_calibration_file_is_written_as_the_rows_serving_reads(man, tmp_path, capsys,
                                                                monkeypatch, choice, expect):
    monkeypatch.setenv("MV_PLACE_CALIBRATION", choice)
    monkeypatch.setenv("MV_PLACE_SETTING", S30)
    f = step1_file(tmp_path)
    code, out = run(capsys, "--from", SRC, "--to", DST, "--from-file", str(f), "--note", "n",
                    "--manifest", str(man))
    assert code == 0, out
    served = reads(man, DST, PLACE)                       # what the box's method reads
    assert served["temperatures"]["species"] == pytest.approx(expect)
    floor = 0.042108 if choice == "shared" else 0.047013
    assert served["sets"]["species"]["floor"] == pytest.approx(floor)
    assert served["n"]["species"] == 12149
    assert reads(man, DST, BAL)["temperatures"]["genus"] == pytest.approx(0.030419)
    conn = sqlite3.connect(man)
    copy = json.loads(conn.execute("select report_json from eval_runs where backbone = ? "
                                   "limit 1", (DST,)).fetchone()[0])[calcopy.MARK]
    conn.close()
    assert copy["from_file"] == f.name
    assert copy["from_file_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()


def test_a_calibration_file_of_another_model_is_refused(man, tmp_path, capsys):
    f = step1_file(tmp_path, model="someone-else-ft")
    code, out = run(capsys, "--from", SRC, "--to", DST, "--from-file", str(f), "--note", "n",
                    "--manifest", str(man))
    assert code == 2 and "is the calibration of 'someone-else-ft'" in out
    assert count(man, DST) == 0


def test_a_copied_calibration_says_so_where_confidence_is_explained():
    ident = Identifier.__new__(Identifier)
    ident.model = None
    ident.calibration = {"run_id": 1, "comparison_id": "20261124-000000-calcopy",
                         "temperatures": {"species": 0.03}, "n": {"species": 12149}, "sets": {},
                         "copied": {"from_backbone": SRC, "copied_at": "t", "note": "n"}}
    note = ident.confidence_note()
    assert "copied, not refitted on this model" in note and SRC not in note
    ident.calibration.pop("copied")
    assert "copied" not in ident.confidence_note()


def test_a_copy_that_leaves_the_served_method_uncalibrated_says_so(man, capsys, monkeypatch):
    # The step-1 rows on the laptop are of other methods than the one the box serves.
    monkeypatch.setenv("MV_DEFAULT_METHOD", PLACE)
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man),
                    "--methods", "nearest+mean,species-mean", "--dry-run")
    assert "nothing to copy" in out and code == 2
    conn = sqlite3.connect(man)
    add_run(conn, SRC, "species-mean", cal(0.04))
    conn.close()
    code, out = run(capsys, "--from", SRC, "--to", DST, "--note", "n", "--manifest", str(man),
                    "--methods", "species-mean", "--dry-run")
    assert code == 0 and f"  {PLACE}: null" in out
    assert f"warning: the served method (MV_DEFAULT_METHOD={PLACE}) would still read no " \
           "calibration" in out
