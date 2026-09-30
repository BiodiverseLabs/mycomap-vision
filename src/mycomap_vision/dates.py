"""1970-01-01 is a placeholder, not a date: it counts as no date everywhere.

It is what an empty date becomes as a Unix timestamp of 0. mycomap.org holds
thousands of observations "observed" on it, and a camera whose clock was never
set stamps it into EXIF. Taken as real it would drag every such record to the
first week of January in the season score, so every date that enters Vision
(the .org export, iNat, a date typed on the identify page, a photo's EXIF)
passes through `real_date` on the way in.
"""

from __future__ import annotations

import re
from datetime import date, datetime

PLACEHOLDER = date(1970, 1, 1)

# The calendar day at the start of the text, in the formats dates arrive in:
# ISO ("1970-01-01", "1970-01-01T00:00:00Z", "1970-01-01 00:00:00+00:00"),
# EXIF ("1970:01:01 00:00:00"), and .org's M/D/Y ("1/1/1970", "01/01/1970 12:00 AM").
_YMD = re.compile(r"^\s*(\d{4})[-:/](\d{1,2})[-:/](\d{1,2})(?!\d)")
_MDY = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")


def is_placeholder_date(value) -> bool:
    """True when `value` (text, date or datetime) is the calendar day 1970-01-01."""
    if isinstance(value, datetime):
        return value.date() == PLACEHOLDER
    if isinstance(value, date):
        return value == PLACEHOLDER
    if not isinstance(value, str):
        return False
    m = _YMD.match(value)
    if m:
        y, mo, d = m.groups()
    else:
        m = _MDY.match(value)
        if not m:
            return False
        mo, d, y = m.groups()
    return (int(y), int(mo), int(d)) == (1970, 1, 1)


def real_date(value):
    """`value` unchanged, or None when it is the 1970-01-01 placeholder."""
    return None if is_placeholder_date(value) else value
