import json

from mycomap_vision.records import (build_records, is_north_america, parse_export,
                                    parse_validation_date, save_records)


def row(oid, name="Amanita muscaria", **kw):
    base = {"observation_id": oid, "source": "iNaturalist", "scientific_name": name, "latitude": "39.1",
            "longitude": "-86.5", "country": "US", "continent": "North America",
            "validation_status_1": "yes", "validation_project_1": "Indiana",
            "validation_date_1": "9/2/2026 14:03"}
    base.update(kw)
    return base


def test_validation_dates_parse_from_org_text_format():
    assert parse_validation_date("9/2/2026 14:03") == "2026-09-02"
    assert parse_validation_date("12/31/2025") == "2025-12-31"
    assert parse_validation_date("2026-01-05T00:00:00") == "2026-01-05"
    assert parse_validation_date("13/40/2026") is None
    assert parse_validation_date("") is None


def test_a_1970_01_01_observation_date_is_exported_as_no_date(conn):
    records = build_records([row("1", observed_on="1970-01-01"),
                             row("2", observed_on="1970-01-02"),
                             row("3", observed_on="2025-09-01")], "t")
    assert [r["observed_on"] for r in records] == [None, "1970-01-02", "2025-09-01"]
    # A manifest that still holds the placeholder from an older export is cleaned by
    # the next export, which rewrites every record.
    conn.execute("insert into records (observation_id, source, north_america, exported_at, "
                 "observed_on) values ('1', 'inat', 1, 'old', '1970-01-01')")
    save_records(conn, records)
    got = dict(conn.execute("select observation_id, observed_on from records").fetchall())
    assert got == {"1": None, "2": "1970-01-02", "3": "2025-09-01"}


def test_a_1970_01_01_validation_date_is_no_date():
    assert parse_validation_date("1/1/1970") is None
    assert parse_validation_date("1/1/1970 00:00") is None
    assert parse_validation_date("1970-01-01T00:00:00") is None
    assert parse_validation_date("1/2/1970") == "1970-01-02"
    r = build_records([row("1", validation_date_1="1/1/1970")], "t")[0]
    assert r["validated_on"] is None


def test_north_america_uses_continent_then_country_then_coordinates():
    assert is_north_america("North America", "ES", None, None)
    assert not is_north_america("Europe", "US", None, None)
    assert is_north_america(None, "pr", None, None)
    assert is_north_america("", "", 45.0, -93.0)
    assert not is_north_america("", "", 48.0, 2.3)
    assert not is_north_america(None, None, None, None)


def test_only_green_slots_count_and_earliest_green_date_wins():
    rows = [row("100", validation_status_2="no", validation_project_2="Other",
                validation_date_2="1/1/2020",
                validation_status_3="yes", validation_project_3="Parent",
                validation_date_3="3/1/2026")]
    [rec] = build_records(rows, "now")
    assert json.loads(rec["green_projects"]) == ["Indiana", "Parent"]
    assert rec["validated_on"] == "2026-03-01"
    assert rec["source"] == "inat"
    assert rec["north_america"] == 1


def test_disagreeing_names_for_one_record_are_flagged_as_label_conflicts():
    rows = [row("200", "Russula A"), row("200", "Russula B"), row("300"), row("300")]
    recs = {r["observation_id"]: r for r in build_records(rows, "now")}
    assert recs["200"]["label_conflict"] == 1
    assert json.loads(recs["200"]["names_json"]) == ["Russula A", "Russula B"]
    assert recs["300"]["label_conflict"] == 0


def test_export_parsing_skips_the_header_line():
    text = 'row_to_json\n{"observation_id": "1", "note": "tab\\there"}\n'
    assert parse_export(text) == [{"observation_id": "1", "note": "tab\there"}]


def test_records_no_longer_green_leave_the_candidate_list(conn):
    save_records(conn, build_records([row("1"), row("2")], "t1"))
    save_records(conn, build_records([row("2", "Renamed")], "t2"))
    got = conn.execute("select observation_id, scientific_name from records").fetchall()
    assert [tuple(r) for r in got] == [("2", "Renamed")]
