"""Scoreboard rows carried from the laptop (where the iNat baseline runs) to the server box."""
import io
import json
import sqlite3

import pytest

from test_inat_cv import FakeInat, result
from test_inat_cv_photos import TAXA, comparison

from mycomap_vision import evaluate, inat_cv, manifest, scoreboard_io


def laptop_and_box(conn, tmp_path):
    """The laptop's manifest with a comparison and the iNat baseline on it, and the box's
    copy made from the laptop before the baseline ran (as a release carries it)."""
    cmp = comparison(conn, tmp_path)
    box = sqlite3.connect(tmp_path / "box.sqlite")
    conn.backup(box)
    fake = FakeInat(TAXA, {"results": [result(1, "A x", "species", 90, 95)]})
    inat_cv.run(conn, cmp["comparison_id"], fake, tmp_path / "emb", log=lambda s: None)
    return cmp["comparison_id"], box


def inat_rows(c, cid):
    return sorted((r["method"], r["species_top1"], r["genus_top1"])
                  for r in evaluate.scoreboard(c, cid) if r["backbone"] == inat_cv.BACKBONE)


def test_the_inat_baseline_measured_on_the_laptop_joins_the_same_comparison_on_the_box(conn,
                                                                                      tmp_path):
    cid, box = laptop_and_box(conn, tmp_path)
    assert inat_rows(box, cid) == []
    payload = json.loads(json.dumps(scoreboard_io.export_runs(conn, cid, inat_cv.BACKBONE)))
    out = scoreboard_io.import_runs(box, payload)
    assert out == {"comparison_id": cid, "backbone": inat_cv.BACKBONE, "imported": 2,
                   "replaced": 0}
    assert inat_rows(box, cid) == inat_rows(conn, cid) and len(inat_rows(box, cid)) == 2


def test_importing_again_replaces_the_rows_instead_of_adding_them(conn, tmp_path):
    cid, box = laptop_and_box(conn, tmp_path)
    payload = scoreboard_io.export_runs(conn, cid, inat_cv.BACKBONE)
    scoreboard_io.import_runs(box, payload)
    assert scoreboard_io.import_runs(box, payload)["replaced"] == 2
    assert len(inat_rows(box, cid)) == 2


def test_rows_for_a_comparison_the_box_does_not_have_are_refused(conn, tmp_path):
    cid, _ = laptop_and_box(conn, tmp_path)
    empty = sqlite3.connect(tmp_path / "other.sqlite")
    with pytest.raises(scoreboard_io.ImportRefused, match="not on this machine"):
        scoreboard_io.import_runs(empty, scoreboard_io.export_runs(conn, cid, inat_cv.BACKBONE))


def test_rows_measured_on_other_records_are_refused_and_nothing_changes(conn, tmp_path):
    cid, box = laptop_and_box(conn, tmp_path)
    with box:
        box.execute("update eval_runs set record_set = 'other'")
    with pytest.raises(scoreboard_io.ImportRefused, match="other records"):
        scoreboard_io.import_runs(box, scoreboard_io.export_runs(conn, cid, inat_cv.BACKBONE))
    assert inat_rows(box, cid) == []


@pytest.mark.parametrize("damage, reason", [
    (lambda p: p.update(kind="something else"), "not a scoreboard export"),
    (lambda p: p.update(rows=[]), "no rows"),
    (lambda p: p["rows"][0].update(comparison_id="another"), "one comparison and model"),
    (lambda p: p["rows"][0].pop("report_json"), "columns"),
])
def test_a_damaged_export_is_refused(conn, tmp_path, damage, reason):
    cid, box = laptop_and_box(conn, tmp_path)
    payload = scoreboard_io.export_runs(conn, cid, inat_cv.BACKBONE)
    damage(payload)
    with pytest.raises(scoreboard_io.ImportRefused, match=reason):
        scoreboard_io.import_runs(box, payload)
    assert inat_rows(box, cid) == []


def test_mv_scoreboard_export_and_import_through_a_pipe(conn, tmp_path, monkeypatch, capsys):
    from mycomap_vision import cli, config
    cid, box = laptop_and_box(conn, tmp_path)
    conn.commit()
    box.close()
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.sqlite")
    assert cli.main(["scoreboard-export", "--comparison", cid]) == 0
    exported = capsys.readouterr().out
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "box.sqlite")
    monkeypatch.setattr("sys.stdin", io.StringIO(exported))
    assert cli.main(["scoreboard-import", "-"]) == 0
    assert '"imported": 2' in capsys.readouterr().out
    c = manifest.connect(tmp_path / "box.sqlite")
    assert len(inat_rows(c, cid)) == 2
    c.close()
