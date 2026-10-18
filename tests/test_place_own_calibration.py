"""The place method's calibration (experiment v1-place-calibration, 2026-10-18): shared with
nearest+mean unless MV_PLACE_CALIBRATION=own asks for its own, and its own only from a row
scored under the place setting in force."""

import json
import sqlite3

import pytest

from mycomap_vision import evaluate, placeprior

B = "m1"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(evaluate.SCOREBOARD_SCHEMA)
    return c


def add_row(conn, method, temperature, place_setting=None):
    report = {"backbone": B, "method": method,
              "all_photos": {"calibration": {"species": {"temperature": temperature, "n": 10,
                                                         "nll": 1.0}}}}
    if place_setting:
        report["place_setting"] = place_setting
    conn.execute("insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, "
                 "n_reference, n_test, record_set, report_json, created_at) values "
                 "('c', ?, ?, '', 0, 1, 1, 'r', ?, '')", (B, method, json.dumps(report)))
    return conn.execute("select max(id) from eval_runs").fetchone()[0]


@pytest.fixture
def rows(conn):
    shared = add_row(conn, "nearest+mean", 0.0274)
    old = add_row(conn, "nearest+mean+place", 0.0304, "boost-2026-10-11")
    served = add_row(conn, "nearest+mean+place", 0.0337, "penalty-500-2026-10-13")
    return {"shared": shared, "old setting": old, "served setting": served}


def test_by_default_the_place_method_states_nearest_mean_s_calibration(conn, rows, monkeypatch):
    monkeypatch.delenv(placeprior.CALIBRATION_ENV, raising=False)
    monkeypatch.setenv(placeprior.SETTING_ENV, "penalty-500-2026-10-13")
    cal = evaluate.latest_calibration(conn, B, "nearest+mean+place")
    assert cal["run_id"] == rows["shared"] and cal["temperatures"]["species"] == 0.0274


def test_asked_for_its_own_it_reads_its_row_of_the_setting_in_force(conn, rows, monkeypatch):
    monkeypatch.setenv(placeprior.CALIBRATION_ENV, "own")
    monkeypatch.setenv(placeprior.SETTING_ENV, "penalty-500-2026-10-13")
    cal = evaluate.latest_calibration(conn, B, "nearest+mean+place")
    assert cal["run_id"] == rows["served setting"] and cal["temperatures"]["species"] == 0.0337
    monkeypatch.setenv(placeprior.SETTING_ENV, "boost-2026-10-11")
    assert evaluate.latest_calibration(conn, B, "nearest+mean+place")["run_id"] == \
        rows["old setting"]


def test_a_row_from_another_setting_is_not_its_own(conn, rows, monkeypatch):
    monkeypatch.setenv(placeprior.CALIBRATION_ENV, "own")
    monkeypatch.setenv(placeprior.SETTING_ENV, "penalty-300-2026-10-13")
    assert evaluate.latest_calibration(conn, B, "nearest+mean+place")["run_id"] == \
        rows["shared"]


def test_the_option_changes_nothing_for_a_method_with_its_own_calibration(conn, rows,
                                                                          monkeypatch):
    monkeypatch.setenv(placeprior.CALIBRATION_ENV, "own")
    assert evaluate.latest_calibration(conn, B, "nearest+mean")["run_id"] == rows["shared"]


def test_an_unknown_choice_is_refused(conn, rows, monkeypatch):
    monkeypatch.setenv(placeprior.CALIBRATION_ENV, "mine")
    with pytest.raises(ValueError, match="MV_PLACE_CALIBRATION"):
        evaluate.latest_calibration(conn, B, "nearest+mean+place")


def test_a_comparison_row_records_the_place_setting_it_was_scored_under(monkeypatch):
    monkeypatch.setenv(placeprior.SETTING_ENV, "penalty-500-2026-10-13")
    method = placeprior.NearestMeanPlace()
    assert method.scoreboard_extra() == {"place_setting": "penalty-500-2026-10-13"}
