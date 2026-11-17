"""E17, the blind review page (e17review.py; Steve, 2026-11-16, item schema v1.1).

The page must reveal no record id, observation link or source site, no person, no exact
date or place, and never Vision's answer. The loader takes only a packet that passes the
schema, R1-R4 and the photo checks, all or nothing; responses are insert-only; every e17
path is a plain 404 to anyone but a listed reviewer with the token."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import logging
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import e17_packet as P
from test_api import open_limits
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import cli, early_answers, e17review as E, identifylog
from mycomap_vision.api import create_app
from mycomap_vision.signin import SESSION_COOKIE, SigninConfig

TOKEN = "t0k3n-" + "A" * 20 + "_bcdefghij"          # 36 url-safe characters
TOKEN_SHA = hashlib.sha256(TOKEN.encode()).hexdigest()
REVIEWER = "reviewer-sub-1"
OTHER = "member-sub-2"
JS, CSS = "e17-Ab12Cd.js", "e17-Ef34Gh.css"


# --- the packet and the loader ---------------------------------------------------------

def packet(tmp_path, name="packet", **kw):
    folder = tmp_path / name
    shutil.rmtree(folder, ignore_errors=True)
    return folder, P.build(folder, **kw)


def refusals(folder, sha) -> list[E.Refusal]:
    with pytest.raises(E.LoadRefused) as e:
        E.check_packet(folder, sha)
    return e.value.refusals


def item_refusals(item, code="K7Q2M"):
    return E.check_item(item, code)


def base_item():
    return copy.deepcopy(P.ITEMS[0])


def test_the_synthetic_packet_loads_whole(tmp_path):
    folder, sha = packet(tmp_path)
    state, pk = E.load(folder, sha, tmp_path / "e17")
    assert state == "loaded" and len(pk.items) == 6
    store = E.Store(tmp_path / "e17")
    assert store.positions() == P.POSITIONS
    assert store.item("00042")["when"] == "not recorded"
    assert store.photo("K7Q2M", 2).read_bytes() == (folder / "photos/K7Q2M/2.jpg").read_bytes()
    assert E.load(folder, sha, tmp_path / "e17")[0] == "unchanged"


# Legitimate values R1-R4 must pass (schema v1.1, each asserted).
@pytest.mark.parametrize("field,value", [
    ("when", "September 2025"),
    ("when", "not recorded"),
    ("code", "K7Q2M"),
    ("code", "00042"),
    ("name", "Inventia sp. 'ABCD123-45'"),           # a BOLD process id inside a name
    ("name", "Cortinarius sp. 'IN01'"),
    ("name", "unnamed local sequence"),
    ("place", "Ontario"),
    ("length", 612),
    ("identity", 99.4),
    ("coverage", 100),
    ("accession", "MN123456.1"),
    ("accession", "NR_173001.1"),
])
def test_legitimate_values_pass_the_loader(field, value):
    it = base_item()
    code = "K7Q2M"
    if field == "when":
        it["when"] = value
    elif field == "code":
        it["code"] = code = value
    elif field == "place":
        it["place"] = value
    elif field == "length":
        it["sequences"][0]["length"] = value
    elif field in ("identity", "coverage"):
        it["sequences"][0]["local"][0][field] = value
    elif field == "accession":
        it["sequences"][0]["genbank"][0]["accession"] = value
    else:
        it["sequences"][0]["local"][0]["name"] = value
    assert item_refusals(it, code) == []
    if isinstance(value, str) and field not in ("code", "accession"):
        assert E.scan_string(value) is None


def test_not_recorded_is_allowed_for_when_only():
    it = base_item()
    it["place"] = "not recorded"
    assert item_refusals(it) == []                   # a place may say it, it is no date
    it["when"] = "Not recorded"
    assert [r.rule for r in item_refusals(it)] == ["schema:pattern"]
    for bad in ("Sept 2025", "September 25", "2025", "September 2025 ", "14 September 2025"):
        it["when"] = bad
        assert "schema:pattern" in [r.rule for r in item_refusals(it)], bad


# Values the loader must refuse (each asserted), with the rule that catches each.
@pytest.mark.parametrize("value,rules", [
    ("inat:123456789", {"R1", "R2"}),
    ("mo:45678", {"R2"}),
    ("obs #4567", {"R2"}),
    ("Inocybe isolate 1234567", {"R1"}),
    ("https://www.inaturalist.org/observations/1", {"R3"}),
    ("2025-09-14", {"R4"}),
    ("9/14/2025", {"R4"}),
    ("Fakeus exemplaris @someone", {"R3"}),
])
def test_ids_links_sites_and_full_dates_are_refused(value, rules):
    assert E.scan_string(value) in rules
    for where in ("name_under_review", "place", "local", "genbank"):
        it = base_item()
        if where == "local":
            it["sequences"][0]["local"][0]["name"] = value
            path = "sequences[0].local[0].name"
        elif where == "genbank":
            it["sequences"][0]["genbank"][0]["name"] = value
            path = "sequences[0].genbank[0].name"
        else:
            it[where] = value
            path = where
        got = [r for r in item_refusals(it) if r.rule in rules]
        assert got and got[0].path == path and got[0].code == "K7Q2M"


def test_accession_and_code_are_not_scanned_but_must_match_their_patterns():
    it = base_item()
    it["sequences"][0]["genbank"][0]["accession"] = "ZZ123456789.1"     # 9 digits: allowed
    assert item_refusals(it) == []
    it["sequences"][0]["genbank"][0]["accession"] = "inat:123456"
    assert [r.rule for r in item_refusals(it)] == ["schema:pattern"]


@pytest.mark.parametrize("where", ["item", "photo", "sequence", "local", "genbank"])
def test_an_extra_field_anywhere_in_an_item_is_refused(where):
    it = base_item()
    target = {"item": it, "photo": it["photos"][0], "sequence": it["sequences"][0],
              "local": it["sequences"][0]["local"][0],
              "genbank": it["sequences"][0]["genbank"][0]}[where]
    target["source"] = "x"
    assert "schema:additional" in [r.rule for r in item_refusals(it)]


@pytest.mark.parametrize("where", ["top", "item", "photo"])
def test_an_extra_field_anywhere_in_the_manifest_is_refused(tmp_path, where):
    def edit(m):
        {"top": m, "item": m["items"][0], "photo": m["photos"][0]}[where]["extra"] = 1
    folder, sha = packet(tmp_path, manifest_edit=edit)
    assert "schema:additional" in [r.rule for r in refusals(folder, sha)]


@pytest.mark.parametrize("change,rule", [
    (lambda it: it["sequences"][0]["local"][0].update(identity="99.4"), "schema:type"),
    (lambda it: it["sequences"][0]["local"][0].update(coverage=100.5), "schema:range"),
    (lambda it: it["sequences"][0]["local"][0].update(identity=-1), "schema:range"),
    (lambda it: it["sequences"][0].update(withdrawn=0), "schema:type"),
    (lambda it: it["sequences"][0].update(length="612"), "schema:type"),
    (lambda it: it["sequences"][0].update(length=True), "schema:type"),
    (lambda it: it["sequences"][0].update(n=3), "schema:contiguous"),
    (lambda it: it["photos"][1].update(n=5), "schema:contiguous"),
    (lambda it: it["photos"][0].update(width=800.0), "schema:type"),
    (lambda it: it.update(name_under_review=""), "schema:length"),
    (lambda it: it.update(name_under_review="F" * 201), "schema:length"),
    (lambda it: it.update(place="O" * 81), "schema:length"),
    (lambda it: it.update(schema="e17-item/2"), "schema:value"),
    (lambda it: it.update(code="K7Q2N"), "schema:file-name"),
    (lambda it: it.update(code="k7q2m"), "schema:pattern"),
    (lambda it: it["sequences"][0]["genbank"][0].update(accession="ZZ1.1"), "schema:pattern"),
    (lambda it: it["sequences"][0]["local"][1].update(hit_no=1), "schema:unique"),
    (lambda it: it.pop("place"), "schema:missing"),
    (lambda it: it["sequences"][0]["genbank"][0].pop("accession"), "schema:missing"),
    (lambda it: it.update(photos=[{"n": i + 1, "width": 10, "height": 10} for i in range(41)]),
     "schema:length"),
    (lambda it: it["sequences"][0].update(local=[P.hit(i + 1, "Fakeus x", 90, 90)
                                                 for i in range(51)]), "schema:length"),
])
def test_the_item_schema_is_applied_exactly(change, rule):
    it = base_item()
    change(it)
    assert rule in [r.rule for r in item_refusals(it)]


# Schema v1.2: each sequence says whether each search is stored.
@pytest.mark.parametrize("table", ["local", "genbank"])
def test_a_search_marked_none_must_have_no_hits(table):
    it = base_item()
    it["sequences"][0][f"{table}_search"] = "none"
    got = item_refusals(it)
    assert [(r.path, r.rule) for r in got] == [(f"sequences[0].{table}",
                                                "schema:hits-without-search")]
    it["sequences"][0][table] = []
    assert item_refusals(it) == []


@pytest.mark.parametrize("value", ["maybe", "Stored", "", True, None, 1])
def test_a_search_is_stored_or_none_and_nothing_else(value):
    it = base_item()
    it["sequences"][1]["genbank_search"] = value
    assert "schema:value" in [r.rule for r in item_refusals(it)]


@pytest.mark.parametrize("field", ["local_search", "genbank_search"])
def test_both_search_fields_are_required(field):
    it = base_item()
    del it["sequences"][0][field]
    assert [(r.path, r.rule) for r in item_refusals(it)] == [(f"sequences[0].{field}",
                                                             "schema:missing")]


def test_a_stored_search_may_have_found_no_hits():
    it = base_item()
    it["sequences"][0]["local"] = []
    it["sequences"][0]["genbank"] = []
    assert item_refusals(it) == []


def test_the_synthetic_packet_has_a_missing_search_and_a_search_with_no_hits(tmp_path):
    folder, sha = packet(tmp_path)
    pk = E.check_packet(folder, sha)
    seqs = [s for it in pk.items.values() for s in it["sequences"]]
    assert any(s["genbank_search"] == "none" and s["genbank"] == [] for s in seqs)
    assert any(s["local_search"] == "stored" and s["local"] == [] for s in seqs)


def test_length_may_be_null_and_lists_may_be_empty():
    it = base_item()
    it["sequences"][0]["length"] = None
    it["sequences"][1]["local"] = []
    it["photos"] = []
    assert item_refusals(it) == []


@pytest.mark.parametrize("edit,rule", [
    (lambda m: m.update(count=7), "schema:count"),
    (lambda m: m["items"][1].update(position=m["items"][0]["position"]), "schema:unique"),
    (lambda m: m["items"][0].update(position=99), "schema:range"),
    (lambda m: m.update(schema="e17-packet/2"), "schema:value"),
    (lambda m: m["photos"][0].update(path="photos/K7Q2M/../x.jpg"), "schema:pattern"),
])
def test_the_manifest_schema_is_applied(tmp_path, edit, rule):
    folder, sha = packet(tmp_path, manifest_edit=edit)
    assert rule in [r.rule for r in refusals(folder, sha)]


def test_the_pinned_manifest_sha_must_match(tmp_path):
    folder, sha = packet(tmp_path)
    assert [r.rule for r in refusals(folder, "0" * 64)] == ["sha:pinned"]
    assert [r.rule for r in refusals(folder, "not-a-sha")] == ["sha:pinned"]


def test_a_listed_file_whose_sha_differs_is_refused(tmp_path):
    folder, sha = packet(tmp_path)
    p = folder / "items" / "00042.json"
    p.write_text(p.read_text(encoding="utf-8").replace("Quebec", "Quebek"), encoding="utf-8")
    got = refusals(folder, sha)
    assert [(r.code, r.path, r.rule) for r in got] == [("00042", "items/00042.json", "sha:file")]


def test_an_unlisted_file_is_refused_without_naming_it(tmp_path):
    folder, sha = packet(tmp_path)
    (folder / "photos" / "K7Q2M" / "inat-123456789.jpg").write_bytes(P.jpeg(10, 10))
    got = refusals(folder, sha)
    assert [r.rule for r in got] == ["file:unlisted"]
    assert "123456789" not in got[0].line() and "inat" not in got[0].line()


def test_a_missing_listed_file_is_refused(tmp_path):
    folder, sha = packet(tmp_path)
    (folder / "photos" / "00042" / "2.jpg").unlink()
    assert [(r.path, r.rule) for r in refusals(folder, sha)] == [("photos/00042/2.jpg",
                                                                  "file:missing")]


def test_photos_listed_in_the_manifest_and_the_items_must_agree(tmp_path):
    def drop_photo(m):
        m["photos"] = [p for p in m["photos"] if p["path"] != "photos/00042/2.jpg"]
    folder, sha = packet(tmp_path, manifest_edit=drop_photo)
    (folder / "photos" / "00042" / "2.jpg").unlink()
    assert [(r.code, r.rule) for r in refusals(folder, sha)] == [("00042", "photo:not-in-manifest")]


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def _with_segment(data: bytes, segment: bytes) -> bytes:
    return data[:2] + segment + data[2:]


def _exif_gps() -> bytes:
    img = Image.new("RGB", (800, 600), (200, 0, 0))
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (40.0, 7.0, 24.44), 3: "W", 4: (82.0, 58.0, 48.0)})
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


@pytest.mark.parametrize("make,rule", [
    (lambda w, h: _exif_gps(), "photo:app1-exif-xmp"),
    (lambda w, h: _with_segment(P.jpeg(w, h), _segment(
        0xE1, b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta/>")), "photo:app1-exif-xmp"),
    (lambda w, h: _with_segment(P.jpeg(w, h), _segment(0xED, b"Photoshop 3.0\x008BIM")),
     "photo:app13-iptc"),
    (lambda w, h: _with_segment(P.jpeg(w, h), _segment(0xFE, b"a comment")), "photo:comment"),
    (lambda w, h: P.jpeg(w + 2, h), "photo:dimensions"),
    (lambda w, h: (lambda b: (b.seek(0), Image.new("RGB", (w, h)).save(b, "PNG"),
                              b.getvalue())[-1])(io.BytesIO()), "photo:jpeg"),
    (lambda w, h: P.jpeg(w, h) + b"\x00" * E.MAX_PHOTO_BYTES, "photo:bytes"),
])
def test_photos_must_be_bare_jpegs_of_the_stated_size(tmp_path, make, rule):
    def photo(code, n, w, h):
        return make(w, h) if (code, n) == ("00042", 1) else None
    folder, sha = packet(tmp_path, photo_bytes=photo)
    got = refusals(folder, sha)
    assert rule in [r.rule for r in got]
    assert all(r.code == "00042" and r.path == "photos[0]" for r in got)


def test_the_test_jpegs_carry_no_metadata_segments():
    problems, size = E.jpeg_check(P.jpeg(640, 480))
    assert problems == [] and size == (640, 480)


def test_a_refused_load_changes_nothing_and_prints_code_path_rule_never_the_value(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "e17"
    monkeypatch.setenv("MV_E17_DIR", str(root))
    good, good_sha = packet(tmp_path, "good")
    assert cli.main(["e17", "load", str(good), "--sha", good_sha]) == 0
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
              if p.is_file() and p.name != E.DB_FILE}

    items = copy.deepcopy(P.ITEMS)
    items[2]["sequences"][0]["local"][1]["name"] = "Inocybe isolate 1234567"
    bad, bad_sha = packet(tmp_path, "bad", items=items)
    capsys.readouterr()
    assert cli.main(["e17", "load", str(bad), "--sha", bad_sha, "--replace"]) == 1
    err = capsys.readouterr().err
    assert "B3XQ9\tsequences[0].local[1].name\tR1" in err
    assert "1234567" not in err and "Inocybe" not in err
    after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
             if p.is_file() and p.name != E.DB_FILE}
    assert after == before
    assert E.Store(root).packet_sha() == good_sha


def test_a_first_load_that_is_refused_loads_nothing(tmp_path):
    items = copy.deepcopy(P.ITEMS)
    items[0]["place"] = "obs #4567"
    folder, sha = packet(tmp_path, items=items)
    with pytest.raises(E.LoadRefused):
        E.load(folder, sha, tmp_path / "e17")
    assert not E.Store(tmp_path / "e17").loaded()
    assert not (tmp_path / "e17" / "items").exists()


def test_another_packet_replaces_the_loaded_one_only_when_asked(tmp_path):
    root = tmp_path / "e17"
    a, a_sha = packet(tmp_path, "a")
    E.load(a, a_sha, root)
    items = copy.deepcopy(P.ITEMS[:3])
    b, b_sha = packet(tmp_path, "b", items=items)
    with pytest.raises(E.LoadRefused):
        E.load(b, b_sha, root)
    assert E.Store(root).packet_sha() == a_sha
    assert E.load(b, b_sha, root, replace=True)[0] == "loaded"
    store = E.Store(root)
    assert store.packet_sha() == b_sha and set(store.positions()) == {"K7Q2M", "00042", "B3XQ9"}
    assert not (root / "photos" / "PLM4T").exists()
    assert not [p for p in root.iterdir() if p.name.startswith(".")]


# --- the store -------------------------------------------------------------------------

def loaded_store(tmp_path) -> E.Store:
    folder, sha = packet(tmp_path)
    E.load(folder, sha, tmp_path / "e17")
    return E.Store(tmp_path / "e17")


def test_responses_are_insert_only_in_the_database_itself(tmp_path):
    store = loaded_store(tmp_path)
    store.add("K7Q2M", REVIEWER, {"verdict": "wrong"})
    conn = sqlite3.connect(store.db)
    with pytest.raises(sqlite3.DatabaseError, match="insert-only"):
        conn.execute("update responses set verdict = 'supported'")
    with pytest.raises(sqlite3.DatabaseError, match="insert-only"):
        conn.execute("delete from responses")
    conn.close()
    assert [r["verdict"] for r in store.history()] == ["wrong"]


def test_the_current_answer_is_the_latest_row(tmp_path):
    store = loaded_store(tmp_path)
    store.add("K7Q2M", REVIEWER, {"verdict": "wrong", "better_name": "Fakeus amoenus"})
    store.add("K7Q2M", REVIEWER, {"verdict": "supported"})
    store.add("00042", REVIEWER, {"notes": "first look"})
    assert store.latest("K7Q2M")["verdict"] == "supported"
    assert store.latest("K7Q2M")["better_name"] is None
    assert store.answered() == {"K7Q2M"}               # notes alone are not an answer
    assert store.latest("PLM4T") is None


def test_export_writes_the_current_answer_per_code_and_every_row(tmp_path):
    store = loaded_store(tmp_path)
    store.add("K7Q2M", REVIEWER, {"verdict": "wrong", "shown_by": "both", "recognised": True})
    store.add("K7Q2M", REVIEWER, {"verdict": "cannot_tell", "notes": "a, \"quoted\" note"})
    store.add("00042", OTHER, {"verdict": "supported"})
    n_latest, n_history = E.export(store.root, tmp_path / "out")
    assert (n_latest, n_history) == (6, 3)
    import csv
    latest = list(csv.DictReader(open(tmp_path / "out" / "latest.csv", encoding="utf-8")))
    assert [r["code"] for r in latest] == ["00042", "PLM4T", "K7Q2M", "W2N6D", "B3XQ9", "ZQ8R1"]
    k = next(r for r in latest if r["code"] == "K7Q2M")
    assert k["verdict"] == "cannot_tell" and k["notes"] == 'a, "quoted" note' and k["sub"] == REVIEWER
    assert next(r for r in latest if r["code"] == "PLM4T")["verdict"] == ""
    history = list(csv.DictReader(open(tmp_path / "out" / "history.csv", encoding="utf-8")))
    assert [(r["code"], r["verdict"]) for r in history] == [
        ("K7Q2M", "wrong"), ("K7Q2M", "cannot_tell"), ("00042", "supported")]
    assert history[0]["recognised"] == "1" and history[0]["shown_by"] == "both"


@pytest.mark.parametrize("body", [
    {"verdict": "maybe"}, {"shown_by": "dna"}, {"recognised": "yes"}, {"sub": "someone"},
    {"better_name": "x" * 201}, {"notes": 5}, [], "supported",
])
def test_a_malformed_response_is_refused(body):
    assert E.clean_answer(body) is None


# --- serving ---------------------------------------------------------------------------

def bundle(tmp_path) -> Path:
    dist = tmp_path / "dist"
    (dist / "e17review" / "assets").mkdir(parents=True)
    (dist / "e17review" / ".vite").mkdir()
    (dist / "index.html").write_text("<!doctype html><title>site shell</title><div id=root>"
                                     "</div>", encoding="utf-8")
    (dist / "e17review" / "assets" / JS).write_text("console.log('e17')", encoding="utf-8")
    (dist / "e17review" / "assets" / CSS).write_text("body{}", encoding="utf-8")
    (dist / "e17review" / ".vite" / "manifest.json").write_text(json.dumps({
        "src/e17/main.tsx": {"file": f"assets/{JS}", "css": [f"assets/{CSS}"], "isEntry": True,
                             "src": "src/e17/main.tsx", "name": "e17"}}), encoding="utf-8")
    return dist


class Models:
    """The model loader: any call is a failure of the e17 paths."""

    def __init__(self):
        self.calls = []

    def __call__(self, name):
        self.calls.append(name)
        raise AssertionError("an e17 path asked for a model")


def site(conn, tmp_path, mode="identify", token_sha=TOKEN_SHA, reviewers=(REVIEWER,),
         load=True, e17=True):
    root = tmp_path / "e17"
    if load:
        folder, sha = packet(tmp_path)
        E.load(folder, sha, root)
    signin = SigninConfig(mode=mode, origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET) if mode != "off" else SigninConfig()
    settings = E.Settings(token_sha, frozenset(reviewers), root) if e17 else None
    models = Models()
    log = identifylog.IdentifyLog(tmp_path / "identify-log.sqlite", note=lambda s: None,
                                  start=False)
    early = early_answers.Store(tmp_path / "early-answers.sqlite", note=lambda s: None,
                                start=False)
    app = create_app(tmp_path / "manifest.sqlite", tmp_path / "emb", backbone_loader=models,
                     limits=open_limits(), signin=signin, background=False, preload=[],
                     fungus_gate=None, photo_tagger=None, web_dist=bundle(tmp_path),
                     note=lambda s: None, identify_log=log, early_store=early, e17=settings)
    app.state.e17_models = models
    return app


def browser(app, sub=None) -> TestClient:
    c = TestClient(app, follow_redirects=False)
    if sub is not None:
        signer = SigninConfig(mode="identify", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                              secret=SECRET).sessions
        c.cookies.set(SESSION_COOKIE, signer.sign({"sub": sub, "name": "R. Viewer",
                                                   "exp": time.time() + 3600}))
    return c


def routes(token=TOKEN):
    """Every e17 route, as (method, path, json body)."""
    return [("GET", f"/review/{token}", None),
            ("GET", f"/review/{token}/assets/{JS}", None),
            ("GET", f"/api/e17/{token}/items", None),
            ("GET", f"/api/e17/{token}/item/K7Q2M", None),
            ("GET", f"/api/e17/{token}/photo/K7Q2M/1", None),
            ("POST", f"/api/e17/{token}/response/K7Q2M", {"verdict": "supported"})]


def call(c, method, path, body):
    return c.post(path, json=body) if method == "POST" else c.get(path)


def assert_e17_headers(r):
    for k, v in E.HEADERS.items():
        assert r.headers.get(k) == v, (k, r.request.url)


def assert_plain_404(r):
    assert r.status_code == 404, (r.request.url, r.status_code)
    assert r.text == E.NOT_FOUND_BODY
    assert r.headers["content-type"].startswith("text/plain")
    assert_e17_headers(r)


def test_a_listed_reviewer_with_the_token_reaches_every_route(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    for method, path, body in routes():
        r = call(c, method, path, body)
        assert r.status_code == 200, (path, r.status_code, r.text)
        assert_e17_headers(r)


@pytest.mark.parametrize("who,token", [
    (REVIEWER, "x" + TOKEN[1:]),                        # a wrong token
    (REVIEWER, TOKEN[:31]),                             # too short to be one
    (None, TOKEN),                                      # no session
    (OTHER, TOKEN),                                     # signed in, not on the list
])
def test_a_wrong_token_no_session_or_another_member_gets_the_same_plain_404(conn, tmp_path,
                                                                             who, token):
    c = browser(site(conn, tmp_path), who)
    for method, path, body in routes(token):
        assert_plain_404(call(c, method, path, body))
    assert E.Store(tmp_path / "e17").history() == []


@pytest.mark.parametrize("missing", ["token", "reviewers", "packet", "all"])
def test_with_any_setting_missing_every_e17_path_is_a_404(conn, tmp_path, missing):
    app = site(conn, tmp_path, token_sha=None if missing == "token" else TOKEN_SHA,
               reviewers=() if missing == "reviewers" else (REVIEWER,),
               load=missing != "packet", e17=missing != "all")
    c = browser(app, REVIEWER)
    for method, path, body in routes():
        assert_plain_404(call(c, method, path, body))


def test_settings_come_from_the_environment(monkeypatch, tmp_path):
    for k in ("MV_E17_TOKEN_SHA256", "MV_E17_REVIEWERS", "MV_E17_DIR"):
        monkeypatch.delenv(k, raising=False)
    assert not E.Settings.from_settings(tmp_path / "manifest.sqlite").enabled
    assert E.default_dir(tmp_path / "manifest.sqlite") == tmp_path / "e17-review"
    monkeypatch.setenv("MV_E17_TOKEN_SHA256", TOKEN_SHA.upper())
    monkeypatch.setenv("MV_E17_REVIEWERS", f" {REVIEWER} , ,{OTHER}")
    monkeypatch.setenv("MV_E17_DIR", str(tmp_path / "elsewhere"))
    s = E.Settings.from_settings()
    assert s.enabled and s.reviewers == {REVIEWER, OTHER} and s.root == tmp_path / "elsewhere"
    assert s.token_ok(TOKEN) and not s.token_ok(TOKEN + "x")
    monkeypatch.setenv("MV_E17_TOKEN_SHA256", "abc")              # not a sha256: off
    assert not E.Settings.from_settings().enabled


def test_the_box_keeps_the_store_under_the_release_root_state(monkeypatch, tmp_path):
    from mycomap_vision import config
    monkeypatch.delenv("MV_E17_DIR", raising=False)
    monkeypatch.setattr(config, "RELEASE_ROOT", tmp_path / "srv")
    assert E.default_dir() == tmp_path / "srv" / "state" / "e17-review"


def test_with_sign_in_off_nothing_e17_answers(conn, tmp_path):
    c = browser(site(conn, tmp_path, mode="off"), REVIEWER)    # a cookie, but no sessions here
    for method, path, body in routes():
        assert_plain_404(call(c, method, path, body))


def test_the_whole_site_gate_still_needs_a_session_for_e17_paths(conn, tmp_path):
    app = site(conn, tmp_path, mode="all")
    gate = SigninConfig(mode="all", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                        secret=SECRET)
    for method, path, _ in routes() + routes("wrong-token"):
        assert gate.needs_session(method, path), path
    c = browser(app)
    page = c.get(f"/review/{TOKEN}")
    assert page.status_code == 302 and page.headers["location"].startswith("/auth/signin")
    assert_e17_headers(page)
    api = c.get(f"/api/e17/{TOKEN}/items")
    assert api.status_code == 401
    assert_e17_headers(api)
    # The same answer for a token that is wrong: the gate tells nothing about the token.
    assert c.get("/review/wrong-token").status_code == 302
    assert browser(app, OTHER).get(f"/api/e17/{TOKEN}/items").status_code == 404
    assert browser(app, REVIEWER).get(f"/api/e17/{TOKEN}/items").status_code == 200


def test_the_identify_only_gate_leaves_e17_to_check_its_own_session(conn, tmp_path):
    gate = SigninConfig(mode="identify", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                        secret=SECRET)
    assert not gate.needs_session("GET", f"/api/e17/{TOKEN}/items")
    c = browser(site(conn, tmp_path, mode="identify"))
    assert_plain_404(c.get(f"/api/e17/{TOKEN}/items"))


def test_the_page_is_its_own_document_with_nothing_from_another_host(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    r = c.get(f"/review/{TOKEN}")
    html = r.text
    assert r.headers["content-type"].startswith("text/html")
    assert '<meta name="robots" content="noindex,nofollow">' in html
    assert '<meta name="referrer" content="no-referrer">' in html
    assert "<title>Review</title>" in html
    assert f'src="/review/{TOKEN}/assets/{JS}"' in html
    assert f'href="/review/{TOKEN}/assets/{CSS}"' in html
    for banned in ("http://", "https://", "fonts.", "/api/me", "site shell", "<style",
                   "<script>"):
        assert banned not in html, banned


def test_assets_are_only_the_bundle_files(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    js = c.get(f"/review/{TOKEN}/assets/{JS}")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    assert c.get(f"/review/{TOKEN}/assets/{CSS}").headers["content-type"].startswith("text/css")
    for name in ("missing.js", "..%2F..%2Findex.html", "manifest.json", ".vite%2Fmanifest.json",
                 ""):
        assert_plain_404(c.get(f"/review/{TOKEN}/assets/{name}"))
    assert_plain_404(c.get(f"/review/{TOKEN}/assets"))         # no listing
    assert_plain_404(c.get(f"/review/{TOKEN}/"))
    assert_plain_404(c.get("/review"))
    assert_plain_404(c.get("/review/"))


def test_the_site_shell_never_serves_the_bundle_or_answers_e17_paths(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    assert "<title>site shell</title>" in c.get("/").text
    assert c.get(f"/e17review/assets/{JS}").status_code == 404
    assert c.get("/e17review/.vite/manifest.json").status_code == 404
    for path in ("/review/anything/else", "/api/e17/", "/api/e17/x/y/z"):
        assert_plain_404(c.get(path))
        assert_plain_404(c.post(path, json={}))


def test_items_list_codes_positions_answered_and_the_manifest_count(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    c.post(f"/api/e17/{TOKEN}/response/PLM4T", json={"verdict": "wrong"})
    c.post(f"/api/e17/{TOKEN}/response/00042", json={"notes": "later"})
    body = c.get(f"/api/e17/{TOKEN}/items").json()
    assert body["count"] == 6
    assert [(i["code"], i["position"], i["answered"]) for i in body["items"]] == [
        ("00042", 1, False), ("PLM4T", 2, True), ("K7Q2M", 3, False), ("W2N6D", 4, False),
        ("B3XQ9", 5, False), ("ZQ8R1", 6, False)]
    assert set(body) == {"count", "items"}
    assert all(set(i) == {"code", "position", "answered"} for i in body["items"])


def test_an_item_is_exactly_its_own_fields_and_the_latest_response(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    body = c.get(f"/api/e17/{TOKEN}/item/K7Q2M").json()
    want = next(it for it in P.ITEMS if it["code"] == "K7Q2M")
    assert body == {**want, "response": None}
    c.post(f"/api/e17/{TOKEN}/response/K7Q2M", json={
        "verdict": "wrong", "better_name": "Fakeus sp. 'IN07'", "shown_by": "sequences",
        "recognised": True, "looked_up": False, "notes": "closest to IN07",
        "client_saved_at": "2026-11-16T21:14:00.000Z"})
    body = c.get(f"/api/e17/{TOKEN}/item/K7Q2M").json()
    resp = body.pop("response")
    assert body == want
    assert set(resp) == set(E.RESPONSE_FIELDS)          # never who saved it
    assert resp["verdict"] == "wrong" and resp["recognised"] is True
    assert REVIEWER not in json.dumps(resp)
    assert_plain_404(c.get(f"/api/e17/{TOKEN}/item/NOPE1"))
    assert_plain_404(c.get(f"/api/e17/{TOKEN}/item/k7q2m"))


def test_photos_are_served_by_code_and_number_only(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    r = c.get(f"/api/e17/{TOKEN}/photo/K7Q2M/3")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == (tmp_path / "packet" / "photos" / "K7Q2M" / "3.jpg").read_bytes()
    assert_e17_headers(r)
    for bad in ("K7Q2M/4", "K7Q2M/0", "K7Q2M/1.jpg", "ZQ8R1/1", "NOPE1/1", "K7Q2M/..%2F1"):
        assert_plain_404(c.get(f"/api/e17/{TOKEN}/photo/{bad}"))


def test_a_response_is_a_new_row_with_the_sessions_sub_never_the_bodys(conn, tmp_path):
    c = browser(site(conn, tmp_path), REVIEWER)
    url = f"/api/e17/{TOKEN}/response/K7Q2M"
    assert c.post(url, json={"verdict": "wrong", "sub": OTHER}).status_code == 400
    assert c.post(url, json={"verdict": "supported"}).status_code == 200
    assert c.post(url, json={"verdict": "cannot_tell", "shown_by": "both"}).status_code == 200
    assert c.post(url, content=b'{"verdict":"wrong"}',
                  headers={"content-type": "text/plain"}).status_code == 415
    assert c.post(url, content=b"not json",
                  headers={"content-type": "application/json"}).status_code == 400
    rows = E.Store(tmp_path / "e17").history()
    assert [(r["sub"], r["verdict"]) for r in rows] == [(REVIEWER, "supported"),
                                                        (REVIEWER, "cannot_tell")]
    assert_plain_404(c.post(f"/api/e17/{TOKEN}/response/NOPE1", json={"verdict": "wrong"}))


def test_no_e17_route_identifies_logs_or_keeps_an_answer(conn, tmp_path):
    app = site(conn, tmp_path)
    log, early = app.state.identify_log, app.state.early_store
    recorded = []
    log.record = lambda *a, **k: recorded.append(("log", a, k))
    early.record = lambda *a, **k: recorded.append(("early", a, k))

    def counts():
        out = {}
        for path in (tmp_path / "identify-log.sqlite", tmp_path / "early-answers.sqlite"):
            db = sqlite3.connect(path)
            tables = [r[0] for r in db.execute("select name from sqlite_master where type='table'")]
            out[path.name] = {t: db.execute(f"select count(*) from {t}").fetchone()[0]
                              for t in tables}
            db.close()
        return out
    before = counts()
    for who in (REVIEWER, OTHER, None):
        c = browser(app, who)
        for method, path, body in routes() + routes("wrong-" + TOKEN):
            call(c, method, path, body)
    assert recorded == []
    assert counts() == before
    assert app.state.e17_models.calls == []


def test_the_module_imports_nothing_that_identifies_logs_or_keeps_answers():
    src = Path(E.__file__).read_text(encoding="utf-8")
    relative, absolute = set(), set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                relative.update([node.module] if node.module else [a.name for a in node.names])
            else:
                absolute.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            absolute.update(a.name.split(".")[0] for a in node.names)
    assert relative == {"config"}
    banned = {"identify", "byobservation", "early_answers", "identifylog", "answercache",
              "feedback", "models", "embed", "fastsearch", "serving", "evaluate", "finetune",
              "methods", "prior", "likely", "torch", "numpy", "mycomap_vision"}
    assert not (absolute | relative) & banned
    # And what importing it actually loads of the package.
    out = subprocess.run([sys.executable, "-c",
                          "import sys, mycomap_vision.e17review; print(' '.join(sorted("
                          "m for m in sys.modules if m.startswith('mycomap_vision'))))"],
                         capture_output=True, text=True, check=True,
                         cwd=Path(E.__file__).parents[2]).stdout.split()
    assert out == ["mycomap_vision", "mycomap_vision.config", "mycomap_vision.e17review"]


def test_the_token_stays_out_of_the_access_log():
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "GET", f"/review/{TOKEN}/assets/{JS}", "1.1", 200),
                            None)
    E.TokenRedactor().filter(rec)
    assert TOKEN not in rec.getMessage() and "/review/(token)/assets/" in rec.getMessage()
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "POST", f"/api/e17/{TOKEN}/response/K7Q2M", "1.1",
                             200), None)
    E.TokenRedactor().filter(rec)
    assert TOKEN not in rec.getMessage()


def test_cli_export_and_status(tmp_path, monkeypatch, capsys):
    root = tmp_path / "e17"
    monkeypatch.setenv("MV_E17_DIR", str(root))
    folder, sha = packet(tmp_path)
    assert cli.main(["e17", "load", str(folder), "--sha", sha]) == 0
    E.Store(root).add("K7Q2M", REVIEWER, {"verdict": "wrong"})
    assert cli.main(["e17", "status"]) == 0
    assert "6 items, 1 answered" in capsys.readouterr().out
    assert cli.main(["e17", "export", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "latest.csv").is_file()
    assert (tmp_path / "out" / "history.csv").is_file()
    assert cli.main(["e17", "new-token"]) == 0
    lines = capsys.readouterr().out.splitlines()[-2:]
    token = lines[0].rsplit(" ", 1)[1]
    assert lines[1] == f"MV_E17_TOKEN_SHA256={hashlib.sha256(token.encode()).hexdigest()}"
    assert E.TOKEN_RE.match(token)
