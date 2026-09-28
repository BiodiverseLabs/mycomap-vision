"""Where photo files go: a local folder or an S3 bucket, under the same relative paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

CONTENT_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "gif": "image/gif", "webp": "image/webp"}


class PhotoStore(Protocol):
    location: str

    def put(self, relpath: str, body: bytes) -> None: ...

    def get(self, relpath: str) -> bytes: ...


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.location = str(self.root)

    def put(self, relpath: str, body: bytes) -> None:
        dest = self.root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(body)
        os.replace(tmp, dest)

    def get(self, relpath: str) -> bytes:
        return (self.root / relpath).read_bytes()


def parse_s3_url(url: str) -> tuple[str, str]:
    """'s3://bucket/some/prefix' -> ('bucket', 'some/prefix/'); the prefix may be empty."""
    if not url.startswith("s3://"):
        raise ValueError(f"not an s3:// URL: {url}")
    bucket, _, prefix = url[5:].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url}")
    prefix = prefix.strip("/")
    return bucket, (prefix + "/" if prefix else "")


class S3Store:
    def __init__(self, url: str, client=None):
        self.bucket, self.prefix = parse_s3_url(url)
        self.location = f"s3://{self.bucket}/{self.prefix}"
        if client is None:
            from .aws import s3_client
            client = s3_client()
        self.client = client   # boto3 clients are safe to share across threads

    def key(self, relpath: str) -> str:
        return self.prefix + relpath

    def put(self, relpath: str, body: bytes) -> None:
        ext = relpath.rsplit(".", 1)[-1].lower()
        self.client.put_object(Bucket=self.bucket, Key=self.key(relpath), Body=body,
                               ContentType=CONTENT_TYPES.get(ext, "application/octet-stream"))

    def get(self, relpath: str) -> bytes:
        with self.client.get_object(Bucket=self.bucket, Key=self.key(relpath))["Body"] as body:
            return body.read()

    def upload_file(self, path: Path, relpath: str) -> None:
        self.client.upload_file(str(path), self.bucket, self.key(relpath))


def open_store(dest: str | None, default_root: Path) -> PhotoStore:
    if dest and dest.startswith("s3://"):
        return S3Store(dest)
    return LocalStore(Path(dest) if dest else default_root)
