"""Place and date from a photo's EXIF, for the range-and-season score.

Used only to score the identification; never stored or logged. What the API
echoes back is rounded to 0.1 degree (about 10 km).
"""

from __future__ import annotations

from datetime import datetime

from PIL import Image

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
DATETIME = 306


def _degrees(value, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in value)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60 + s / 3600
    return -deg if str(ref).upper() in ("S", "W") else deg


def place_and_date(img: Image.Image) -> tuple[float | None, float | None, str | None]:
    """(latitude, longitude, ISO date) from EXIF; None for anything missing or odd."""
    try:
        exif = img.getexif()
    except Exception:
        return None, None, None
    lat = lon = None
    gps = exif.get_ifd(GPS_IFD) if exif else {}
    if gps:
        lat = _degrees(gps.get(2), gps.get(1))
        lon = _degrees(gps.get(4), gps.get(3))
        if lat is not None and not -90 <= lat <= 90 or lon is not None and not -180 <= lon <= 180:
            lat = lon = None
        if lat == 0 and lon == 0:            # an unset GPS block, not the Gulf of Guinea
            lat = lon = None
    when = None
    raw = (exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) if exif else None) or \
        (exif.get(DATETIME) if exif else None)
    if raw:
        try:
            when = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S").date().isoformat()
        except ValueError:
            when = None
    return lat, lon, when
