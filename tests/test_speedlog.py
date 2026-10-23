"""The daily seconds-per-photo chart (speedlog.py): a day is its total time over its total
photos, days are US Eastern, and only averages ever leave the box."""
from datetime import datetime, timezone

from test_api import app_with_model, post_photos

from mycomap_vision.speedlog import SpeedLog, day_of, eastern_offset

DAY = 86400.0
OCT_22 = 1792627200.0        # 2026-10-22 00:00 UTC (21 Oct, 8 pm Eastern)
NOON = 16 * 3600              # 16:00 UTC: noon Eastern, well inside the day


def log_at(tmp_path, t):
    now = [t]
    return SpeedLog(tmp_path / "speed.sqlite", note=lambda s: None, clock=lambda: now[0]), now


def test_a_day_is_its_total_time_over_its_total_photos(tmp_path):
    log, now = log_at(tmp_path, OCT_22 + NOON)
    log.record(1, 3.0)                       # one slow photo
    log.record(9, 9.0)                       # nine at a second each
    now[0] += DAY
    log.record(2, 2.0)
    assert log.daily() == [{"day": "2026-10-22", "seconds_per_photo": 1.2},
                           {"day": "2026-10-23", "seconds_per_photo": 1.0}]


def test_days_are_us_eastern_and_kept_across_restarts(tmp_path):
    """A photo at 8 pm Eastern on 22 Oct is 00:00 UTC on the 23rd: it counts for the 22nd,
    as the people reading the chart see the day (Steve, 2026-10-22)."""
    log, _ = log_at(tmp_path, OCT_22 + 24 * 3600 + 1)      # 23 Oct 00:00:01 UTC, 20:00 EDT
    log.record(1, 1.0)
    again, _ = log_at(tmp_path, OCT_22 + 28 * 3600 + 1)    # 23 Oct 04:00:01 UTC, 00:00 EDT
    again.record(1, 3.0)
    assert [d["day"] for d in again.daily()] == ["2026-10-22", "2026-10-23"]


def test_the_eastern_day_follows_daylight_saving_time():
    utc = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()
    # EDT (-4) in summer, EST (-5) in winter.
    assert day_of(utc(2026, 7, 1, 3, 59)) == "2026-06-30"
    assert day_of(utc(2026, 7, 1, 4, 0)) == "2026-07-01"
    assert day_of(utc(2026, 12, 1, 4, 59)) == "2026-11-30"
    assert day_of(utc(2026, 12, 1, 5, 0)) == "2026-12-01"
    # 2026: EDT from 8 Mar 07:00 UTC to 1 Nov 06:00 UTC.
    assert eastern_offset(utc(2026, 3, 8, 6, 59)).total_seconds() == -5 * 3600
    assert eastern_offset(utc(2026, 3, 8, 7, 0)).total_seconds() == -4 * 3600
    assert eastern_offset(utc(2026, 11, 1, 5, 59)).total_seconds() == -4 * 3600
    assert eastern_offset(utc(2026, 11, 1, 6, 0)).total_seconds() == -5 * 3600


def test_only_the_last_days_asked_for_and_nothing_without_photos(tmp_path):
    log, now = log_at(tmp_path, OCT_22 + NOON)
    log.record(4, 4.0)
    log.record(0, 5.0)                       # a batch of stored vectors only: not a photo
    now[0] += 40 * DAY
    log.record(1, 2.0)
    assert [d["day"] for d in log.daily(30)] == ["2026-12-01"]


def test_identifications_fill_the_chart_and_it_shows_no_counts(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    assert client.get("/api/speed/daily").json() == {"days": []}
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    assert post_photos(client, [247, 10], "m1/nearest").status_code == 200
    [day] = client.get("/api/speed/daily").json()["days"]
    assert set(day) == {"day", "seconds_per_photo"}
    assert day["seconds_per_photo"] >= 0
    with client.app.state.speed_log._lock:
        photos = client.app.state.speed_log._conn.execute(
            "select photos from speed_daily").fetchone()[0]
    assert photos == 3
