"""CEC North American Terrestrial Ecoregions (Level I) for a point, without GIS libraries
(paper E20's ecoregion slice; paper_attrs.py).

The polygons are CEC's (2010), as EPA serves them: NA_CEC_Eco_Level1.shp/.dbf in a
spherical Lambert azimuthal equal-area projection (radius 6,370,997 m, centred 45 N,
100 W; its .prj). This reads the shapefile's polygons and attribute table directly (ESRI
shapefile and dBASE III formats), projects points the same way, and finds the polygon
holding each point by the even-odd rule over every ring (holes included), with edges
binned by y so each point meets only the edges at its own height. A point in no polygon
is "(none)".
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RADIUS = 6_370_997.0
LAT0, LON0 = 45.0, -100.0
BINS = 2000


def laea(lat, lon, lat0: float = LAT0, lon0: float = LON0, radius: float = RADIUS):
    """Spherical Lambert azimuthal equal-area (Snyder 1987, eq. 24-2..24-4): metres."""
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    p0, l0 = math.radians(lat0), math.radians(lon0)
    dl = lam - l0
    k = np.sqrt(2.0 / (1.0 + math.sin(p0) * np.sin(phi) + math.cos(p0) * np.cos(phi) * np.cos(dl)))
    x = radius * k * np.cos(phi) * np.sin(dl)
    y = radius * k * (math.cos(p0) * np.sin(phi) - math.sin(p0) * np.cos(phi) * np.cos(dl))
    return x, y


def read_dbf(path: Path) -> list[dict]:
    """A dBASE III table's rows (text fields stripped; numbers as text)."""
    b = Path(path).read_bytes()
    n, hlen, rlen = struct.unpack("<IHH", b[4:12])
    fields, pos = [], 32
    while b[pos] != 0x0D:
        name = b[pos:pos + 11].split(b"\0", 1)[0].decode("latin-1")
        fields.append((name, b[pos + 16]))
        pos += 32
    rows = []
    for i in range(n):
        rec = b[hlen + i * rlen: hlen + (i + 1) * rlen]
        if rec[:1] == b"*":                    # deleted
            continue
        at, row = 1, {}
        for name, width in fields:
            row[name] = rec[at:at + width].decode("latin-1").strip()
            at += width
        rows.append(row)
    return rows


def read_polygons(path: Path) -> list[list[np.ndarray]]:
    """Each shape record's rings, as (n, 2) arrays, from a polygon shapefile (.shp)."""
    b = Path(path).read_bytes()
    pos, out = 100, []
    while pos < len(b):
        _num, length = struct.unpack(">II", b[pos:pos + 8])
        content = b[pos + 8: pos + 8 + 2 * length]
        pos += 8 + 2 * length
        shape = struct.unpack("<i", content[:4])[0]
        if shape == 0:
            out.append([])
            continue
        if shape not in (5, 15, 25):
            raise ValueError(f"{path}: shape type {shape} is not a polygon")
        nparts, npoints = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{nparts}i", content[44:44 + 4 * nparts])) + [npoints]
        pts = np.frombuffer(content, "<f8", 2 * npoints, 44 + 4 * nparts).reshape(-1, 2)
        out.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return out


@dataclass
class Regions:
    names: list[str]                 # one per shape record
    x1: np.ndarray                   # every edge of every ring
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    shape: np.ndarray                # each edge's shape record
    bins: list[np.ndarray]           # edges whose y-span reaches each bin
    y0: float
    dy: float

    @classmethod
    def load(cls, shp: Path, name_field: str = "NA_L1NAME", n_bins: int = BINS) -> "Regions":
        rows = read_dbf(Path(shp).with_suffix(".dbf"))
        polys = read_polygons(shp)
        if len(rows) != len(polys):
            raise ValueError(f"{shp}: {len(polys)} shapes but {len(rows)} attribute rows")
        e = [[], [], [], [], []]
        for i, rings in enumerate(polys):
            for ring in rings:
                a, b = ring[:-1], ring[1:]
                for lst, v in zip(e, (a[:, 0], a[:, 1], b[:, 0], b[:, 1],
                                      np.full(len(a), i, np.int32))):
                    lst.append(v)
        x1, y1, x2, y2, shape = (np.concatenate(v) for v in e)
        lo, hi = np.minimum(y1, y2), np.maximum(y1, y2)
        y0, y_top = float(lo.min()), float(hi.max())
        dy = (y_top - y0) / n_bins or 1.0
        b0 = np.clip(((lo - y0) / dy).astype(int), 0, n_bins - 1)
        b1 = np.clip(((hi - y0) / dy).astype(int), 0, n_bins - 1)
        span = b1 - b0 + 1
        edge = np.repeat(np.arange(len(x1)), span)
        bin_of = np.repeat(b0, span) + (np.arange(span.sum()) - np.repeat(np.cumsum(span) - span,
                                                                         span))
        order = np.argsort(bin_of, kind="stable")
        edge, bin_of = edge[order], bin_of[order]
        cuts = np.searchsorted(bin_of, np.arange(n_bins + 1))
        bins = [edge[cuts[i]:cuts[i + 1]] for i in range(n_bins)]
        return cls([r.get(name_field, "") for r in rows], x1, y1, x2, y2, shape, bins, y0, dy)

    def lookup_xy(self, x: np.ndarray, y: np.ndarray) -> list[str]:
        """The region holding each projected point (even-odd over its shape's rings)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = ["(none)"] * len(x)
        b = ((y - self.y0) / self.dy).astype(int)
        for k in np.unique(b):
            if k < 0 or k >= len(self.bins):
                continue
            pts = np.flatnonzero(b == k)
            ed = self.bins[k]
            if not len(ed):
                continue
            px, py = x[pts][:, None], y[pts][:, None]
            x1, y1, x2, y2 = self.x1[ed], self.y1[ed], self.x2[ed], self.y2[ed]
            spans = (y1 > py) != (y2 > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            cross = spans & (px < xint)
            sh = self.shape[ed]
            for j, p in enumerate(pts):
                hit = sh[cross[j]]
                if len(hit):
                    odd = np.flatnonzero(np.bincount(hit) % 2 == 1)
                    if len(odd):
                        out[p] = self.names[int(odd[0])]
        return out

    def lookup(self, lat, lon) -> list[str]:
        x, y = laea(lat, lon)
        return self.lookup_xy(x, y)
