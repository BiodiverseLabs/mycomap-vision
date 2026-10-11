"""Photo licenses and iNat photo URLs."""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Usable for any purpose, including commercial, with attribution where required.
OPEN_LICENSES = {"cc0", "cc-by", "cc-by-sa", "pd"}


def license_class(code: str | None) -> str:
    """'open', 'nc' (non-commercial or no-derivatives) or 'arr' (all rights reserved).

    iNat reports an all-rights-reserved photo with no license code.
    """
    c = (code or "").strip().lower()
    if not c:
        return "arr"
    if c in OPEN_LICENSES:
        return "open"
    return "nc"


SIZES = ("square", "thumb", "small", "medium", "large", "original")
# Old photos (around 2012-13) have no extension: ".../2511038/square." serves fine.
_SIZE_RE = re.compile(r"/(square|thumb|small|medium|large|original)\.([A-Za-z0-9]*)(?:\?.*)?$")


def sized_url(url: str, size: str) -> str:
    """Rewrite an iNat photo URL to another size: .../square.jpg?123 -> .../medium.jpg."""
    if size not in SIZES:
        raise ValueError(f"unknown photo size {size!r}")
    m = _SIZE_RE.search(url)
    if not m:
        raise ValueError(f"not an iNat sized photo URL: {url}")
    return url[: m.start()] + f"/{size}.{m.group(2)}"


def url_extension(url: str) -> str:
    m = _SIZE_RE.search(url)
    return ((m.group(2) if m else "") or "jpg").lower()


def taken_down(url: str) -> bool:
    """iNat swaps a photo removed for copyright for a placeholder image."""
    return "/assets/copyright-infringement" in url


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""


# iNat's CC-licensed photos are served from the AWS Open Data bucket, which is meant
# for bulk download. All-rights-reserved photos are served from iNat's own static
# host, where iNat asks for no more than 5 GB an hour or 24 GB a day.
OPEN_DATA_HOST = "inaturalist-open-data.s3.amazonaws.com"
STATIC_HOST = "static.inaturalist.org"


def photo_relpath(photo_id: int, size: str, ext: str) -> str:
    """Where a photo lives under the data dir; 1,000 buckets keep directories small."""
    return f"photos/{size}/{photo_id % 1000:03d}/{photo_id}.{ext}"


_MAGIC = (
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"GIF87a", b"GIF89a",
)


def looks_like_image(head: bytes) -> bool:
    if any(head.startswith(m) for m in _MAGIC):
        return True
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


# Mushroom Observer serves each image at fixed sizes under /images/<size>/<id>.<ext>.
# Vision's size names map to MO's nearest: iNat's large is 1024 px, MO's "large" 960.
MO_HOST = "mushroomobserver.org"
MO_SIZES = {"square": "thumb", "thumb": "thumb", "small": "320", "medium": "640",
            "large": "960", "original": "orig"}
_MO_RE = re.compile(r"^https://mushroomobserver\.org/images/(thumb|320|640|960|1280|orig)/"
                    r"(\d+)\.([A-Za-z0-9]+)(?:\?.*)?$")


def mo_sized_url(url: str, size: str) -> str:
    """An MO image URL at another size: .../images/thumb/12.jpg?1 -> .../images/960/12.jpg."""
    if size not in MO_SIZES:
        raise ValueError(f"unknown photo size {size!r}")
    m = _MO_RE.match(url or "")
    if not m:
        raise ValueError(f"not a Mushroom Observer image URL: {url}")
    return f"https://{MO_HOST}/images/{MO_SIZES[size]}/{m.group(2)}.{m.group(3).lower()}"


def photo_url(source: str, url: str, size: str) -> str:
    """The URL of a photo at `size`, by the photo's source (photos.source), never by the
    look of the URL: an iNat photo through sized_url, an MO image through mo_sized_url."""
    if source == "inat":
        return sized_url(url, size)
    if source == "mo":
        return mo_sized_url(url, size)
    raise ValueError(f"no photos are fetched from source {source!r}")


def photo_extension(source: str, url: str) -> str:
    if source == "mo":
        m = _MO_RE.match(url or "")
        return (m.group(3) if m else "jpg").lower()
    return url_extension(url)
