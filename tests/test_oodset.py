"""The sealed not-a-fungus test set (oodset.py): only CC0 / CC BY photos, never one of the
earlier set's photos or observations nor a MycoMap record, every photo with its licence,
author, source and sha256, fungi among the near-OOD photos marked, and the manifest
written with LF line endings so its hash is stable."""

import csv
import io

import pytest
from PIL import Image

from mycomap_vision import oodset

JPEG = io.BytesIO()
Image.new("RGB", (8, 8), (10, 200, 10)).save(JPEG, "JPEG")
JPEG = JPEG.getvalue()


class Resp:
    def __init__(self, status=200, content=JPEG, data=None):
        self.status_code, self.content, self._data = status, content, data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def obs(oid, pid, licence="cc-by", host="inaturalist-open-data.s3.amazonaws.com",
        name="Quercus alba"):
    return {"id": oid, "taxon": {"id": 1, "name": name}, "user": {"login": f"user{oid}"},
            "photos": [{"id": pid, "license_code": licence,
                        "url": f"https://{host}/photos/{pid}/square.jpg"}]}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(oodset, "INAT_GROUPS", (
        oodset.InatGroup("dicots", "plant", "near-ood", 47124, 10),
        oodset.InatGroup("rusts", "plant-pathogen", "fungal-pathogen", 69968, 1)))
    monkeypatch.setattr(oodset, "OI_GROUPS", (
        oodset.OIGroup("oi-food", "food", ("/m/food",), 2),))


def fake_api(results_by_taxon, calls):
    def api(url, params=None):
        calls.append(params)
        return Resp(data={"results": results_by_taxon.get(params["taxon_id"], [])})
    return api


def write_oi(folder):
    folder.mkdir()
    (folder / "train-annotations-human-imagelabels-boxable.csv").write_text(
        "ImageID,Source,LabelName,Confidence\n"
        "a1,h,/m/food,1\na2,h,/m/food,1\na2,h,/m/052sf,1\na3,h,/m/food,1\na4,h,/m/food,0\n"
        "a5,h,/m/food,1\n", encoding="utf-8")
    (folder / "train-images-boxable-with-rotation.csv").write_text(
        "ImageID,License,Author,OriginalLandingURL\n"
        "a1,https://creativecommons.org/licenses/by/2.0/,Ann,https://flickr.com/a1\n"
        "a2,https://creativecommons.org/licenses/by/2.0/,Bo,https://flickr.com/a2\n"
        "a3,https://creativecommons.org/licenses/by-nc/2.0/,Cy,https://flickr.com/a3\n"
        "a4,https://creativecommons.org/licenses/by/2.0/,Di,https://flickr.com/a4\n"
        "a5,https://creativecommons.org/licenses/by/2.0/,Ed,https://flickr.com/a5\n",
        encoding="utf-8")


def test_the_set_takes_only_open_licences_from_new_sources(tmp_path, small):
    old = tmp_path / "old.csv"
    # The earlier set held observation 1 (another photo of it) and, alone, photo 12.
    old.write_text("file,photo_id,observation_id\nx.jpg,99,1\ny.jpg,12,\n", encoding="utf-8")
    write_oi(tmp_path / "oi")
    results = {47124: [obs(1, 11), obs(2, 12), obs(3, 13, licence="cc-by-nc"),
                       obs(4, 14, host="static.inaturalist.org"), obs(5, 15), obs(6, 16),
                       obs(7, 17)],
               69968: [obs(8, 18, name="Gymnosporangium juniperi-virginianae")]}
    calls = []
    out = oodset.build(tmp_path / "set", tmp_path / "oi", [old], mycomap_inat_ids={"5"},
                       v1_labels={"Gymnosporangium"}, api_get=fake_api(results, calls),
                       photo_get=lambda url: Resp(), log=lambda s: None)
    rows = list(csv.DictReader(open(tmp_path / "set" / "manifest.csv", encoding="utf-8")))
    dicots = {r["observation_id"] for r in rows if r["group"] == "dicots"}
    # 1: an earlier observation; 2: an earlier photo; 3: NC; 4: off the open bucket;
    # 5: a MycoMap record.
    assert dicots == {"6", "7"}
    assert {r["photo_id"] for r in rows if r["group"] == "oi-food"} == {"a1", "a5"}  # a2 mushroom, a3 NC, a4 unverified
    assert all(r["license"] in ("cc0", "cc-by") or "/licenses/by/" in r["license"] for r in rows)
    rust = next(r for r in rows if r["group"] == "rusts")
    assert rust["kind"] == "fungal-pathogen" and rust["in_v1_labels"] == "yes"
    for r in rows:
        assert len(r["sha256"]) == 64 and r["page_url"] and r["owner"] and r["photo_url"]
        assert (tmp_path / "set" / "photos" / r["file"]).read_bytes() == JPEG
    assert out["photos"] == 5 and out["kind"]["everyday"] == 2
    assert all(c["photo_license"] == "cc0,cc-by" and c["place_id"] == oodset.NA for c in calls)


def test_a_rerun_keeps_what_it_has_and_fetches_nothing_new(tmp_path, small):
    results = {47124: [obs(i, 100 + i) for i in range(1, 5)], 69968: [obs(9, 109)]}
    first = oodset.build(tmp_path / "set", None, [], set(), set(),
                         api_get=fake_api(results, []), photo_get=lambda url: Resp(),
                         log=lambda s: None)
    fetched = []
    again = oodset.build(tmp_path / "set", None, [], set(), set(),
                         api_get=fake_api(results, []),
                         photo_get=lambda url: fetched.append(url) or Resp(), log=lambda s: None)
    assert again["manifest_sha256"] == first["manifest_sha256"] and not fetched


def test_the_manifest_has_lf_line_endings_and_a_stable_order(tmp_path):
    rows = [{"file": "b.jpg", "source": "s"}, {"file": "a.jpg", "source": "s"}]
    sha = oodset.write_manifest(tmp_path / "m.csv", rows)
    body = (tmp_path / "m.csv").read_bytes()
    assert b"\r" not in body and body.index(b"a.jpg") < body.index(b"b.jpg")
    assert sha == oodset.sha256_bytes(body)
    assert oodset.write_manifest(tmp_path / "m.csv", rows[::-1]) == sha


def test_a_non_jpeg_answer_is_skipped():
    assert oodset.is_jpeg(JPEG) and not oodset.is_jpeg(b"<html>")
