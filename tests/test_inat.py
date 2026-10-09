from conftest import inat_obs

from mycomap_vision.inat import parse_observation, pending_ids, save_batch
from mycomap_vision.records import build_records, save_records

OPEN = "inaturalist-open-data.s3.amazonaws.com"
STATIC = "static.inaturalist.org"


def test_photos_come_back_in_position_order_with_owner_and_license():
    obs = inat_obs(5, photos=[(1, 22, None, STATIC), (0, 11, "cc-by-nc", OPEN)])
    row, photos = parse_observation(obs)
    assert row["photo_count"] == 2
    assert (row["inat_latitude"], row["inat_longitude"]) == (39.1, -86.5)
    assert [p["photo_id"] for p in photos] == [11, 22]
    assert photos[0]["license_class"] == "nc" and photos[0]["host"] == OPEN
    assert photos[1]["license_class"] == "arr" and photos[1]["license_code"] == ""
    assert all(p["owner_login"] == "alice" for p in photos)


def test_an_inat_date_of_1970_01_01_is_stored_as_no_date():
    assert parse_observation(inat_obs(5, observed_on="1970-01-01"))[0]["observed_on"] is None
    assert parse_observation(inat_obs(5))[0]["observed_on"] == "2025-09-01"


def test_ids_inat_does_not_return_are_marked_missing(conn):
    stats = save_batch(conn, ["5", "6"], [inat_obs(5, photos=[(0, 11, "cc0", OPEN)])], "t1")
    assert stats["ok"] == 1 and stats["missing"] == 1
    status = dict(conn.execute("select observation_id, status from inat_observations").fetchall())
    assert status == {"5": "ok", "6": "missing"}


def test_license_changes_are_recorded_in_history(conn):
    save_batch(conn, ["5"], [inat_obs(5, photos=[(0, 11, None, STATIC)])], "t1")
    stats = save_batch(conn, ["5"], [inat_obs(5, photos=[(0, 11, "cc-by", OPEN)])], "t2")
    assert stats["license_changes"] == 1
    hist = conn.execute("select license_code, seen_at from license_history "
                        "where photo_id = 11 order by seen_at").fetchall()
    assert [tuple(h) for h in hist] == [("", "t1"), ("cc-by", "t2")]
    photo = conn.execute("select license_class, host from photos where photo_id = 11").fetchone()
    assert tuple(photo) == ("open", OPEN)


def test_refetch_replaces_the_photo_list_when_owner_removed_a_photo(conn):
    save_batch(conn, ["5"], [inat_obs(5, photos=[(0, 11, "cc0", OPEN), (1, 12, "cc0", OPEN)])], "t1")
    save_batch(conn, ["5"], [inat_obs(5, photos=[(0, 12, "cc0", OPEN)])], "t2")
    links = conn.execute("select photo_id, position from observation_photos").fetchall()
    assert [tuple(l) for l in links] == [(12, 0)]


def test_pending_ids_skip_fetched_non_inat_and_other_continents(conn):
    rows = [
        {"source": src, "observation_id": oid, "scientific_name": "X", "continent": cont,
         "validation_status_1": "yes"}
        for oid, cont, src in [("1", "North America", "iNaturalist"),
                               ("2", "North America", "iNaturalist"),
                               ("3", "Europe", "iNaturalist"),
                               ("9", "North America", "MO Observations")]
    ]
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, ["1"], [inat_obs(1)], "t")
    assert pending_ids(conn, refresh=False, north_america_only=True) == ["2"]
    assert pending_ids(conn, refresh=True, north_america_only=False) == ["1", "2", "3"]
