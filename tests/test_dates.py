"""1970-01-01 is a placeholder, not a date (Steve, 2026-09-30: "treat this date as no date")."""

from datetime import date, datetime, timezone

import pytest

from mycomap_vision.dates import is_placeholder_date, real_date


@pytest.mark.parametrize("value", [
    "1970-01-01",
    "1970-01-01T00:00:00",
    "1970-01-01T00:00:00Z",
    "1970-01-01 00:00:00+00:00",
    "1970-01-01T00:00:00.000-05:00",
    " 1970-01-01 ",
    "1970:01:01 00:00:00",             # EXIF
    "1/1/1970",                        # .org's M/D/Y
    "01/01/1970",
    "1/1/1970 12:00:00 AM",
    "1970/1/1",
    date(1970, 1, 1),
    datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc),
])
def test_the_1970_01_01_placeholder_is_no_date_in_every_format(value):
    assert is_placeholder_date(value)
    assert real_date(value) is None


@pytest.mark.parametrize("value", [
    "2025-09-01", "1970-01-02", "1970-11-28", "1971-01-01", "1969-12-31T19:00:00-05:00",
    "12/31/1969", "1/10/1970", "11/1/1970", "2026:09:27 14:05:00", "19700101",
    "1970-01-011", date(2026, 1, 1),
])
def test_every_other_date_is_kept_exactly_as_it_came(value):
    assert not is_placeholder_date(value)
    assert real_date(value) == value


@pytest.mark.parametrize("value", [None, "", "   ", "unknown", 0, 1970])
def test_missing_or_unreadable_values_pass_through_untouched(value):
    assert not is_placeholder_date(value)
    assert real_date(value) == value
