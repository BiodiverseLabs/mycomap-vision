"""The public model version is the training data's snapshot (Steve, 2026-10-11): the date
the DNA-validated labels were exported, .2/.3 for later builds on it, never a backbone."""

import json
import sqlite3

from test_release import exported

from mycomap_vision import modelcard


def test_the_snapshot_is_the_labels_export_date_and_data_counts_it(conn, tmp_path):
    exported(conn, "2026-10-06T18:01:55+00:00")
    data = modelcard.data_of(conn)
    assert data == {"snapshot": "2026-10-06", "records": 3, "species": 2, "genera": 2}


def test_a_release_reads_its_recorded_version_else_the_snapshot_date(conn, tmp_path):
    exported(conn, "2026-10-06T18:01:55+00:00")
    rel = tmp_path / "releases" / "20261008-015746-full-bioclip2-ft-20261007"
    rel.mkdir(parents=True)
    conn.execute("vacuum into ?", (str(rel / "manifest.sqlite"),))
    # A release made before versions were recorded: the bare snapshot date, never its id.
    (rel / "release.json").write_text(json.dumps({"id": rel.name}))
    card = modelcard.card(rel, None)
    assert card["version"] == "2026-10-06" and card["data"]["records"] == 3
    (rel / "release.json").write_text(json.dumps({"id": rel.name, "version": "2026-10-06.2"}))
    assert modelcard.card(rel, None)["version"] == "2026-10-06.2"
    # Anything that isn't a plain version string falls back to the date.
    (rel / "release.json").write_text(json.dumps({"version": "bioclip 2 / ft model!"}))
    assert modelcard.card(rel, None)["version"] == "2026-10-06"


def test_later_builds_on_the_same_snapshot_are_numbered():
    assert modelcard.version_for("2026-10-06", []) == "2026-10-06"
    assert modelcard.version_for("2026-10-06", ["2026-10-06"]) == "2026-10-06.2"
    assert modelcard.version_for("2026-10-06", ["2026-10-06", "2026-10-06.2",
                                                "2026-09-01"]) == "2026-10-06.3"
    assert modelcard.version_for("2026-10-20", ["2026-10-06.4"]) == "2026-10-20"
    assert modelcard.version_for(None, ["2026-10-06"]) is None


def test_a_manifest_without_records_has_no_snapshot(tmp_path):
    empty = sqlite3.connect(tmp_path / "m.sqlite")
    assert modelcard.data_of(empty) == {"snapshot": None, "records": 0, "species": 0,
                                        "genera": 0}
