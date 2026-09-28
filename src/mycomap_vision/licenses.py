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
_SIZE_RE = re.compile(r"/(square|thumb|small|medium|large|original)\.([A-Za-z0-9]+)(?:\?.*)?$")


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
    return (m.group(2) if m else "jpg").lower()


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
