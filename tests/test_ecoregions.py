"""Ecoregions for a point without GIS libraries (ecoregions.py): the shapefile and dBASE
readers, the equal-area projection, and the even-odd rule with a hole."""

import struct

import numpy as np
import pytest

from mycomap_vision import ecoregions as eco


def write_shapefile(base, shapes, names):
    """A minimal polygon shapefile (.shp) and its dBASE table (.dbf, one text field)."""
    recs = b""
    for i, rings in enumerate(shapes, 1):
        pts = np.concatenate(rings)
        parts = np.cumsum([0] + [len(r) for r in rings[:-1]])
        content = struct.pack("<i4d2i", 5, *pts.min(0), *pts.max(0), len(rings), len(pts))
        content += struct.pack(f"<{len(rings)}i", *parts) + pts.astype("<f8").tobytes()
        recs += struct.pack(">2i", i, len(content) // 2) + content
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2)
    header += struct.pack("<2i8d", 1000, 5, 0, 0, 0, 0, 0, 0, 0, 0)
    base.with_suffix(".shp").write_bytes(header + recs)
    width = 20
    hdr = struct.pack("<B3BIHH20x", 3, 26, 10, 13, len(names), 32 + 32 + 1, 1 + width)
    field = b"NA_L1NAME".ljust(11, b"\0") + b"C" + b"\0" * 4 + bytes([width, 0]) + b"\0" * 14
    rows = b"".join(b" " + n.encode().ljust(width) for n in names)
    base.with_suffix(".dbf").write_bytes(hdr + field + b"\r" + rows + b"\x1a")


def square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]], float)


def test_the_readers_and_the_even_odd_rule_with_a_hole(tmp_path):
    base = tmp_path / "eco"
    write_shapefile(base, [[square(0, 0, 10, 10), square(4, 4, 6, 6)], [square(20, 0, 30, 10)]],
                    ["FOREST", "PLAINS"])
    assert [r["NA_L1NAME"] for r in eco.read_dbf(base.with_suffix(".dbf"))] == ["FOREST", "PLAINS"]
    r = eco.Regions.load(base.with_suffix(".shp"), n_bins=7)
    got = r.lookup_xy(np.array([1, 5, 25, 15, 9.9]), np.array([1, 5, 5, 5, 9.9]))
    assert got == ["FOREST", "(none)", "PLAINS", "(none)", "FOREST"]     # (5, 5) is the hole


def test_the_projection_is_centred_on_45_north_100_west_and_symmetric():
    x, y = eco.laea([45.0, 45.0, 45.0], [-100.0, -90.0, -110.0])
    assert x[0] == pytest.approx(0, abs=1e-6) and y[0] == pytest.approx(0, abs=1e-6)
    assert x[1] == pytest.approx(-x[2]) and y[1] == pytest.approx(y[2])


def test_the_projection_keeps_area():
    lat, d = 45.0, 1e-3
    xs, ys = eco.laea([lat, lat, lat + d], [-90.0, -90.0 + d, -90.0])
    area = abs((xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]))
    true = (eco.RADIUS * np.radians(d)) ** 2 * np.cos(np.radians(lat))
    assert area == pytest.approx(true, rel=1e-3)
