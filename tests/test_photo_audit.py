"""The photo review's rescoring audit (photo_audit.py; Steve, 2026-11-20): out-of-fold tagger
scores per decided photo and tab, loaded with `mv feedback load-audit-scores`, read apart from
the queue, and used by the Summary's Decided list of a tab: "Strongest disagreement first" and
the outcome "High score, not in this tab".

Disagreement for tab T: 1 - p_T for a photo left out for T's reason, p_T for one kept or left
out for another reason. Loading changes no item, decision, flag, band, suggestion or status,
and the kept item lists stay what a fresh read gives."""

import csv
import hashlib
import json
import random
import threading
import time

import pytest
from test_feedback_api import as_user
from test_feedback_lists import STEVE, Names, check, fill_with, make_queue
from test_photo_review import photo_item, site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_lists as fl
from mycomap_vision import photo_audit as pa
from mycomap_vision import photo_review as pr

INAT = "https://www.inaturalist.org/observations/"


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(pa, "PAUSE", 0.0)          # the load's pause between chunks, off in tests


def item(i, record, rule, status="decided", decision=None, reason=None):
    last = {"decision": decision, "value": reason} if decision else None
    return {"id": i, "subject": f"inat:{i}", "rule": rule, "status": status,
            "links": [INAT + record], "current": "A x", "last_decision": last}


UV = "R11.1 v2 uv p=0.80 (high)"


def queue():
    """The UV tab's decided photos: 1 and 2 left out as UV, 3 kept, 4 left out as a slip,
    5 open; 6-9 never flagged for UV: 6 a slip left out as a slip, 7 kept, 8 a micrograph left
    out as UV (in the tab by its reason), 9 a slip left out as a slip."""
    return [
        item(1, "10", UV, decision="exclude_photo", reason="uv"),
        item(2, "10", "R11.1 v2 uv p=0.60 (medium)", decision="exclude_photo", reason="uv"),
        item(3, "11", UV, decision="keep_photo"),
        item(4, "11", UV, decision="exclude_photo", reason="slip"),
        item(5, "12", UV, status="open"),
        item(6, "13", "R9 v2 slip p=0.90 (high)", decision="exclude_photo", reason="slip"),
        item(7, "13", "R9 v2 specimen p=0.90 (high)", decision="keep_photo"),
        item(8, "14", "R6 micrograph probe p=0.70", decision="exclude_photo", reason="uv"),
        item(9, "15", "R9 v2 slip p=0.70 (high)", decision="exclude_photo", reason="slip"),
    ]


SCORES = {1: 0.97, 2: 0.03, 3: 0.88, 4: 0.40, 5: 0.99, 6: 0.91, 7: 0.20, 8: 0.65}   # 9: none
AUDIT = pr.Audit("uv", SCORES, 0.85, "R11.1", 1)


def keys(tiles):
    return [t["photo_key"] for t in tiles]


# --- the rule ------------------------------------------------------------------------------
def test_disagreement_is_one_minus_p_for_the_tabs_reason_and_p_otherwise():
    out = {"decision": "exclude_photo", "value": "uv"}
    assert pr.audit_disagreement(out, "uv", 0.03) == pytest.approx(0.97)
    assert pr.audit_disagreement(out, "slip", 0.03) == pytest.approx(0.03)
    assert pr.audit_disagreement({"decision": "keep_photo"}, "uv", 0.88) == pytest.approx(0.88)
    assert pr.audit_disagreement({"decision": "exclude_photo", "value": "slip"}, "uv", 0.4) == \
        pytest.approx(0.4)
    assert pr.audit_disagreement({"decision": "exclude_photo", "value": "phylo_tree"}, "tree",
                                 0.1) == pytest.approx(0.9)
    # a retired reason counts as its successor's tab
    assert pr.audit_disagreement({"decision": "exclude_photo", "value": "drawing_micro"},
                                 "drawing", 0.2) == pytest.approx(0.8)


def test_strongest_disagreement_first_on_the_decided_list():
    tiles = pr.flagged_tiles(queue(), "uv", "decided", "disagreement", audit=AUDIT)
    # 2: out as UV, p 0.03 -> 0.97; 3: kept, 0.88; 8: out as UV (by reason), 0.65 -> 0.35;
    # 4: out as slip, 0.40; 1: out as UV, 0.97 -> 0.03
    assert keys(tiles) == ["inat:2", "inat:3", "inat:4", "inat:8", "inat:1"]
    d = [t["audit"]["disagreement"] for t in tiles]
    assert d == sorted(d, reverse=True)
    two = tiles[0]["audit"]
    assert two == {"tab": "uv", "p": 0.03, "comparator": "R11.1", "disagreement": 0.97,
                   "high": 0.85}


def test_a_decided_tile_with_no_score_goes_last_and_the_outcome_filter_still_applies():
    q = queue() + [item(10, "16", UV, decision="exclude_photo", reason="uv")]
    tiles = pr.flagged_tiles(q, "uv", "decided", "disagreement", audit=AUDIT)
    assert keys(tiles)[-1] == "inat:10" and "audit" not in tiles[-1]
    kept = pr.flagged_tiles(q, "uv", "decided", "disagreement", outcome="as_flagged", audit=AUDIT)
    assert keys(kept) == ["inat:2", "inat:1", "inat:10"]          # left out as UV, as flagged
    inc = pr.flagged_tiles(q, "uv", "decided", "disagreement", outcome="included", audit=AUDIT)
    assert keys(inc) == ["inat:3"]


def test_high_score_not_in_this_tab_lists_the_photos_the_tab_may_have_missed():
    counts = {}
    tiles = pr.flagged_tiles(queue(), "uv", "decided", "disagreement", outcome="high_not_in_tab",
                             counts=counts, audit=AUDIT)
    # 6 (a slip left out as slip, p 0.91) only: 7 scores low; 8 is in the tab by its reason;
    # 9 has no score; 3 is in the tab (flagged for UV)
    assert keys(tiles) == ["inat:6"]
    t = tiles[0]
    assert t["not_in_tab"] and t["outcome"] is None
    assert t["suggested_reason"] == "uv"                        # a flip to Exclude says UV
    assert [f["tab"] for f in t["flags"]] == ["slip"]             # what it was flagged as
    assert counts == {"as_flagged": 2, "included": 1, "moved": 1, "not_flagged": 1,
                      "high_not_in_tab": 1}
    # every count is what the list shows with that option picked
    for code, n in counts.items():
        assert len(pr.flagged_tiles(queue(), "uv", "decided", outcome=code, audit=AUDIT)) == n
    # the tab's own list never holds them
    assert "inat:6" not in keys(pr.flagged_tiles(queue(), "uv", "decided", audit=AUDIT))


def test_not_in_this_tab_follows_the_runs_high_cut():
    low_cut = pr.Audit("uv", SCORES, 0.15, "R11.1")
    assert keys(pr.flagged_tiles(queue(), "uv", "decided", "disagreement",
                                 outcome="high_not_in_tab", audit=low_cut)) == ["inat:6", "inat:7"]


def test_without_an_audit_nothing_changes_and_the_audit_options_are_refused():
    for tab, status in (("uv", "decided"), ("uv", "open"), ("all", "decided")):
        assert pr.flagged_tiles(queue(), tab, status, audit=AUDIT if tab == "uv" and status == "open" else None) == \
            pr.flagged_tiles(queue(), tab, status)
    plain = pr.flagged_tiles(queue(), "uv", "decided")
    with_audit = pr.flagged_tiles(queue(), "uv", "decided", audit=AUDIT)
    assert [{k: v for k, v in t.items() if k != "audit"} for t in with_audit] == plain
    for kw in ({"sort": "disagreement"}, {"outcome": "high_not_in_tab"}):
        with pytest.raises(fb.FeedbackError, match="audit"):
            pr.flagged_tiles(queue(), "uv", "decided", **kw)
        with pytest.raises(fb.FeedbackError, match="audit"):            # Open, All: none
            pr.flagged_tiles(queue(), "uv", "open", **kw, audit=AUDIT)
        with pytest.raises(fb.FeedbackError, match="audit"):
            pr.flagged_tiles(queue(), "all", "decided", **kw, audit=AUDIT)
        with pytest.raises(fb.FeedbackError, match="audit"):            # another tab's scores
            pr.flagged_tiles(queue(), "slip", "decided", **kw, audit=AUDIT)
    counts = {}
    pr.flagged_tiles(queue(), "uv", "decided", counts=counts)
    assert "high_not_in_tab" not in counts


# --- the loader ------------------------------------------------------------------------------
def gate_for(store, v1=None, held=frozenset(), seal=None, validated=lambda k: "2020-01-01"):
    recs = {pr.record_of_item(i) for i in store.items_of(pr.ITEM_TYPE)}
    return fb.SealGate(validated, frozenset(recs if v1 is None else v1), seal, held)


def write_run(path, store, tabs=("slip", "uv"), only=None, edit=None, manifest_edit=None):
    """A TSV and manifest as r12_audit.py writes them, for the store's decided photo items."""
    rng = random.Random(3)
    rows = []
    for it in store.items_of(pr.ITEM_TYPE):
        if it["status"] != "decided" or (only and it["id"] not in only):
            continue
        for tab in sorted(tabs):
            rows.append({"item_id": it["id"], "photo_key": it["subject"],
                         "record_key": pr.record_of_item(it), "group": "train", "tab": tab,
                         "label": "slip", "p": f"{rng.random():.4f}",
                         "p_image": f"{rng.random():.4f}" if tab in ("uv", "slip") else "",
                         "comparator": "R11.1" if tab == "uv" else "v2.1-336",
                         "comparator_p": f"{rng.random():.4f}", "fold": rng.randint(0, 4),
                         "model": "abc123def456"})
    if edit:
        rows = edit(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pa.FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    man = {"format": pa.FORMAT, "run_id": "r12-audit-test", "tsv_sha256": sha, "rows": len(rows),
           "tabs": {t: {"cuts": {"high": 0.7, "medium": 0.4, "low": 0.1}} for t in tabs}}
    if manifest_edit:
        man = manifest_edit(man)
    path.with_suffix(".json").write_text(json.dumps(man), encoding="utf-8")
    return sha, rows


def queue_state(store):
    """Everything about the queue a person sees: items as items_of gives them (flags, bands,
    suggestions and status come from these), every decision, the seeds."""
    return (store.items_of(pr.ITEM_TYPE),
            store.conn.execute("select * from decisions order by id").fetchall(),
            store.conn.execute("select * from seeds order by name").fetchall(),
            [pr.flagged_tiles(store.items_of(pr.ITEM_TYPE), t, s)
             for t in ("all", "slip", "uv", "micrograph") for s in ("open", "decided")])


def test_a_load_adds_the_scores_and_changes_nothing_in_the_queue(tmp_path):
    s = make_queue(tmp_path)
    before = queue_state(s)
    fp = pa.fingerprint(s.conn)
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)
    out = pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s), loaded_by="test")
    assert out["rows"] == out["rows_loaded"] == len(rows) and out["items"] == len(rows) // 2
    assert queue_state(s) == before and pa.fingerprint(s.conn) == fp
    run = pa.latest_run(s.conn)
    assert run["run_id"] == "r12-audit-test" and run["tabs"] == ["slip", "uv"]
    assert run["high"] == {"slip": 0.7, "uv": 0.7}
    assert run["comparators"] == {"slip": "v2.1-336", "uv": "R11.1"}
    got = pa.scores(s.conn, run["id"], "uv")
    want = {int(r["item_id"]): float(r["p"])
            for r in rows if r["tab"] == "uv"}
    assert got == want
    with pytest.raises(pa.LoadRefused, match="already loaded"):
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s))


def test_the_kept_lists_stay_a_fresh_read_after_a_load_from_another_process(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names))
    check(lists, s, names, "full")
    other = fb.Store(s.path)                                    # the CLI's own connection
    sha, _ = write_run(tmp_path / "run" / "scores.tsv", other)
    pa.load(other, tmp_path / "run" / "scores.tsv", sha, gate_for(other))
    check(lists, s, names, "full")                              # another process: read whole,
    check(lists, s, names, "same")                              # and equal to a fresh read
    s.decide(next(i["id"] for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "open"),
             "keep_photo", STEVE)
    check(lists, s, names, "patch")                             # saves patch as before
    # loaded on the store's own connection: nothing for the lists to read at all
    s2 = make_queue(tmp_path / "b")
    lists2 = fl.ItemLists(s2, fill_with(names))
    check(lists2, s2, names, "full")
    sha2, _ = write_run(tmp_path / "b" / "run" / "scores.tsv", s2)
    pa.load(s2, tmp_path / "b" / "run" / "scores.tsv", sha2, gate_for(s2))
    check(lists2, s2, names, "same")


def test_a_write_that_would_change_the_queue_is_undone_and_refused(tmp_path):
    s = make_queue(tmp_path)
    before = queue_state(s)
    sha, _ = write_run(tmp_path / "run" / "scores.tsv", s)
    target = next(i["id"] for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "open")

    def sneak(conn):                                            # a bug that closes an item
        conn.execute("update items set status = 'decided' where id = ?", (target,))
    with pytest.raises(pa.LoadRefused, match="more than the audit"):
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s), _in_write=sneak)
    assert queue_state(s) == before
    assert s.conn.execute("select count(*) from photo_audit_scores").fetchone()[0] == 0
    assert s.conn.execute("select count(*) from photo_audit_runs").fetchone()[0] == 0
    assert pa.latest_run(s.conn) is None

    def decide(conn):                                           # or writes a decision
        conn.execute("insert into decisions (item_id, decision, applies, by_user, at) "
                     "values (?, 'keep_photo', 'vision', 'x', 1)", (target,))
    with pytest.raises(pa.LoadRefused, match="more than the audit"):
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s), _in_write=decide)
    assert queue_state(s) == before
    # and with nothing sneaked in, the same file loads
    pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s))
    assert pa.latest_run(s.conn) is not None


def test_a_load_in_several_writes_counts_only_once_complete(tmp_path, monkeypatch):
    s = make_queue(tmp_path)
    monkeypatch.setattr(pa, "CHUNK", 7)
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)
    calls = []

    def fail_late(conn):
        calls.append(1)
        assert pa.latest_run(conn) is None                      # not counted while loading
        if len(calls) == 3:
            raise RuntimeError("disk full")
    with pytest.raises(RuntimeError):
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s), _in_write=fail_late)
    assert s.conn.execute("select count(*) from photo_audit_scores").fetchone()[0] == 0
    out = pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s))
    assert s.conn.execute("select count(*) from photo_audit_scores").fetchone()[0] == len(rows)
    assert pa.latest_run(s.conn)["id"] == out["run"]


def refused(tmp_path, s, match, gate=None, sha=None, **kw):
    edit = kw.pop("edit", None)
    medit = kw.pop("manifest_edit", None)
    tabs = kw.pop("tabs", ("slip", "uv"))
    got, _ = write_run(tmp_path / "run" / "scores.tsv", s, tabs=tabs, edit=edit,
                       manifest_edit=medit)
    before = queue_state(s)
    with pytest.raises(pa.LoadRefused, match=match):
        pa.load(s, tmp_path / "run" / "scores.tsv", sha or got, gate or gate_for(s), **kw)
    assert queue_state(s) == before
    assert pa.latest_run(s.conn) is None


def test_the_loader_refuses_a_wrong_file(tmp_path):
    s = make_queue(tmp_path)
    refused(tmp_path, s, "sha256", sha="0" * 64)
    refused(tmp_path, s, "another TSV", manifest_edit=lambda m: {**m, "tsv_sha256": "1" * 64})
    refused(tmp_path, s, "format", manifest_edit=lambda m: {**m, "format": "something else"})
    refused(tmp_path, s, "sorted", edit=lambda r: list(reversed(r)))
    refused(tmp_path, s, "sorted", edit=lambda r: r[:1] + r)            # a row twice
    refused(tmp_path, s, "unknown tab", edit=lambda r: [{**r[0], "tab": "nope"}] + r[1:])
    refused(tmp_path, s, "outside 0..1", edit=lambda r: [{**r[0], "p": "1.5"}] + r[1:])
    refused(tmp_path, s, "not a number", edit=lambda r: [{**r[0], "p": "x"}] + r[1:])
    refused(tmp_path, s, "no rows", edit=lambda r: [])


def test_the_loader_refuses_items_the_store_does_not_have(tmp_path):
    s = make_queue(tmp_path)
    refused(tmp_path, s, "not photo items", edit=lambda r: r + [{**r[-1], "item_id": 99_999,
                                                                    "tab": "slip"}])
    other_type = s.conn.execute("select id from items where item_type != ? order by id desc",
                                (pr.ITEM_TYPE,)).fetchone()[0]
    refused(tmp_path, s, "not photo items",
            edit=lambda r: sorted(r + [{**r[0], "item_id": other_type, "tab": "slip"}],
                                  key=lambda x: (int(x["item_id"]), x["tab"])))
    refused(tmp_path, s, "other photos", edit=lambda r: [{**x, "photo_key": "inat:elsewhere"}
                                                         if x["item_id"] == r[0]["item_id"] else x
                                                         for x in r])


def test_the_loader_refuses_held_v1_test_and_unsealed_items(tmp_path):
    s = make_queue(tmp_path)
    decided = [i for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "decided"]
    rec = pr.record_of_item(decided[0])
    # a held record (the E17 hold list): the seal-check itself fails on it
    refused(tmp_path, s, "seal-check", gate=gate_for(s, held=frozenset({rec})))
    # a seal-check that fails for any reason refuses the load: here a record past the seal
    strict = fb.SealGate(lambda k: "2026-11-01" if k == rec else "2020-01-01", frozenset({"x"}),
                         "2026-10-12T13:17:51+00:00")
    refused(tmp_path, s, "seal-check", gate=strict)
    # a v1-test record
    refused(tmp_path, s, "v1-test", v1_test_keys=frozenset({rec}))
    # not a v1 train/val record, and no v1-test list: refused, or left out with only_v1
    v1 = {pr.record_of_item(i) for i in decided} - {rec}
    refused(tmp_path, s, "not on v1 train/val", gate=gate_for(s, v1=v1))
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)
    out = pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s, v1=v1), only_v1=True)
    skipped = {i["id"] for i in decided if pr.record_of_item(i) == rec}
    assert out["skipped_not_v1"] == len(skipped) > 0
    loaded = set(pa.scores(s.conn, out["run"], "uv"))
    assert loaded and not loaded & skipped


def test_the_held_check_never_names_a_held_record(tmp_path):
    s = make_queue(tmp_path)
    decided = [i for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "decided"]
    rec = pr.record_of_item(decided[0])
    sha, _ = write_run(tmp_path / "run" / "scores.tsv", s)
    with pytest.raises(pa.LoadRefused) as e:
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s, held=frozenset({rec})))
    assert rec not in str(e.value)


def test_a_v1_test_key_list_is_pinned_by_its_sha(tmp_path):
    p = tmp_path / "v1test.txt"
    p.write_bytes(b"123\n456\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert pa.read_key_list(p, sha) == {"123", "456"}
    for bad in (None, "0" * 64):
        with pytest.raises(pa.LoadRefused, match="sha256"):
            pa.read_key_list(p, bad)


def test_a_dry_run_checks_everything_and_writes_nothing(tmp_path):
    s = make_queue(tmp_path)
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)
    tables = s.conn.execute("select name from sqlite_master order by name").fetchall()
    out = pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s), dry_run=True)
    assert out["dry_run"] and out["rows_loaded"] == len(rows)
    assert s.conn.execute("select name from sqlite_master order by name").fetchall() == tables
    with pytest.raises(pa.LoadRefused):
        pa.load(s, tmp_path / "run" / "scores.tsv", "0" * 64, gate_for(s), dry_run=True)


# --- over the real app ------------------------------------------------------------------------
def test_the_api_sorts_by_disagreement_filters_not_in_tab_and_keeps_its_cache_right(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item("inat:2000", "102", "R11.1 v2 uv p=0.91 (high)"),
                 photo_item("inat:2001", "102", "R11.1 v2 uv p=0.81 (high)"),
                 photo_item("inat:2002", "103", "R9 v2 slip p=0.71 (high)")], "uv")
    as_user(c, "admin-1")
    ids = {i["subject"]: i["id"] for i in fstore.items_of(pr.ITEM_TYPE)}
    assert c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:2000"], "decision": "exclude", "reason": "uv"},
        {"item_id": ids["inat:2001"], "decision": "keep"},
        {"item_id": ids["inat:2002"], "decision": "exclude", "reason": "slip"},
        {"item_id": ids["inat:1000"], "decision": "exclude", "reason": "micrograph"}]}
    ).status_code == 200

    def flagged(**p):
        return c.get("/api/feedback/photo-review/flagged",
                     params={"tab": "uv", "status": "decided", **p})
    # no run loaded: the audit options are refused, and nothing else changes
    assert c.get("/api/feedback/photo-review/audit").json() == {"run": None, "tabs": []}
    assert flagged(sort="disagreement").status_code == 400
    assert flagged(outcome="high_not_in_tab").status_code == 400
    plain = flagged().json()
    assert all("audit" not in t for t in plain["tiles"])
    slip_before = c.get("/api/feedback/photo-review/flagged",
                        params={"tab": "slip", "status": "decided"}).json()
    open_before = c.get("/api/feedback/photo-review/flagged", params={"tab": "uv"}).json()

    p = {ids["inat:2000"]: 0.10, ids["inat:2001"]: 0.95, ids["inat:2002"]: 0.92,
         ids["inat:1000"]: 0.30}

    def edit(rows):
        return [{**r, "p": f"{p[int(r['item_id'])]:.4f}"} if r["tab"] == "uv" else r for r in rows]
    other = fb.Store(fstore.path)                               # the CLI: another connection
    sha, _ = write_run(tmp_path / "run" / "scores.tsv", other, edit=edit)
    pa.load(other, tmp_path / "run" / "scores.tsv", sha, gate_for(other))

    info = c.get("/api/feedback/photo-review/audit").json()
    assert info["run"]["run_id"] == "r12-audit-test"
    assert {t["code"]: t["high"] for t in info["tabs"]} == {"slip": 0.7, "uv": 0.7}
    body = flagged(sort="disagreement").json()
    # 2001 kept at 0.95 -> 0.95; 2000 out as UV at 0.10 -> 0.90
    assert [t["photo_key"] for t in body["tiles"]] == ["inat:2001", "inat:2000"]
    assert body["tiles"][0]["audit"]["p"] == 0.95 and body["tiles"][1]["audit"]["disagreement"] == 0.9
    for t in body["tiles"]:                         # read for the page's tiles only
        row = other.conn.execute("select p_image, comparator_p from photo_audit_scores "
                                 "where tab = 'uv' and item_id = ?", (t["item_id"],)).fetchone()
        assert (t["audit"]["p_image"], t["audit"]["comparator_p"]) == (round(row[0], 4), round(row[1], 4))
    assert body["outcomes"]["high_not_in_tab"] == 1
    missed = flagged(sort="disagreement", outcome="high_not_in_tab").json()
    assert [t["photo_key"] for t in missed["tiles"]] == ["inat:2002"]   # a slip at UV 0.92
    assert missed["tiles"][0]["not_in_tab"] and missed["tiles"][0]["suggested_reason"] == "uv"
    # the same lists as before the load, the audit aside; Open and other tabs untouched
    again = flagged().json()
    assert [{k: v for k, v in t.items() if k != "audit"} for t in again["tiles"]] == plain["tiles"]
    slip_now = c.get("/api/feedback/photo-review/flagged",
                     params={"tab": "slip", "status": "decided"}).json()
    assert [{k: v for k, v in t.items() if k != "audit"} for t in slip_now["tiles"]] ==         slip_before["tiles"]
    assert all(t["audit"]["tab"] == "slip" for t in slip_now["tiles"])
    assert c.get("/api/feedback/photo-review/flagged", params={"tab": "uv"}).json() == open_before
    assert flagged(tab="all", sort="disagreement").status_code == 400

    # Steve changes a decision the usual way: the disagreement follows the decision now
    assert c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:2001"], "decision": "exclude", "reason": "uv"}]}).status_code == 200
    body = flagged(sort="disagreement").json()
    assert [t["photo_key"] for t in body["tiles"]] == ["inat:2000", "inat:2001"]
    assert body["tiles"][1]["audit"]["disagreement"] == pytest.approx(0.05)
    # a Contributor reads the same list (these photos are not in the manifest, so a viewer below
    # Contributor sees none of them: test_feedback_read_only.py)
    as_user(c, "leader-1")
    ro = flagged(sort="disagreement").json()
    assert [t["photo_key"] for t in ro["tiles"]] == ["inat:2000", "inat:2001"]


# --- the command line -------------------------------------------------------------------------
def test_mv_feedback_load_audit_scores_dry_run_load_and_refusal(conn, tmp_path, monkeypatch, capsys):
    import argparse

    from mycomap_vision import cli
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    monkeypatch.delenv("MV_FEEDBACK_HOLD_KEYS", raising=False)
    monkeypatch.delenv("MV_FEEDBACK_HOLD_SHA256", raising=False)
    s = make_queue(tmp_path)
    recs = sorted({k for it in s.conn.execute("select * from items")
                   for k in fb.row_records(fb.Store._item(it))})
    v1 = tmp_path / "v1-keys.txt"
    v1.write_text("\n".join(recs) + "\n", encoding="utf-8")
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)

    def run(**kw):
        a = {"action": "load-audit-scores", "tsv": str(tmp_path / "run" / "scores.tsv"),
             "expect_sha256": sha, "manifest": None, "dry_run": False, "v1_test_keys": None,
             "v1_test_sha256": None, "only_v1": False, "v1_keys": str(v1), "hold_keys": None,
             "hold_sha256": None, "store": str(s.path)} | kw
        cli.cmd_feedback(conn, argparse.Namespace(**a))
        return json.loads(capsys.readouterr().out)
    out = run(dry_run=True)
    assert out["dry_run"] and out["rows_loaded"] == len(rows)
    assert pa.latest_run(s.conn) is None
    out = run()
    assert out["run"] and pa.latest_run(s.conn)["rows"] == len(rows)
    with pytest.raises(SystemExit, match="load refused: this TSV is already loaded"):
        run()
    with pytest.raises(SystemExit, match="load refused: the TSV's sha256"):
        run(expect_sha256="0" * 64)


# --- review fixes (2026-11-21) ----------------------------------------------------------------
def test_saves_on_another_connection_go_through_quickly_while_a_load_runs(tmp_path, monkeypatch):
    """Fix 1: the queue's whole-table checks run outside the write lock, each chunk is small and
    followed by a pause, so a reviewer's save never waits behind them. The checks are made slow
    here (a big store's), and every save must still finish well inside that time."""
    s = make_queue(tmp_path)
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)
    plain = pa.fingerprint

    def big_store_fingerprint(conn):
        time.sleep(0.4)
        return plain(conn)
    monkeypatch.setattr(pa, "fingerprint", big_store_fingerprint)
    monkeypatch.setattr(pa, "CHUNK", 20)
    monkeypatch.setattr(pa, "PAUSE", 0.05)
    server = fb.Store(s.path)                                  # the site's own connection
    targets = [i["id"] for i in server.items_of(pr.ITEM_TYPE) if i["status"] == "open"]
    done, waits, errors = threading.Event(), [], []

    def reviewer():
        k = 0
        while not done.is_set():
            t0 = time.perf_counter()
            try:
                server.decide(targets[k % len(targets)], "exclude_photo" if k % 2 else "keep_photo",
                              STEVE, value="slip" if k % 2 else None)
            except Exception as e:                              # noqa: BLE001
                errors.append(repr(e))
            waits.append(time.perf_counter() - t0)
            k += 1
            time.sleep(0.01)
    th = threading.Thread(target=reviewer)
    th.start()
    try:
        out = pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s))
    finally:
        done.set()
        th.join()
    assert not errors
    assert len(waits) >= 10
    assert max(waits) < 0.3, sorted(waits)[-5:]
    assert out["rows_loaded"] == len(rows) and out["queue_written_by_others_during_load"]
    assert s.conn.execute("select count(*) from photo_audit_scores").fetchone()[0] == len(rows)


def _two_runs(tmp_path):
    s = make_queue(tmp_path)
    sha1, rows1 = write_run(tmp_path / "r1" / "scores.tsv", s,
                            tabs=("micrograph", "slip", "uv", "dried"))
    pa.load(s, tmp_path / "r1" / "scores.tsv", sha1, gate_for(s))
    return s, rows1


def test_the_servers_score_cache_keeps_p_only_for_a_few_tabs(tmp_path):
    """Fix 2: per tab only item id -> p is kept, for at most CACHE_TABS tabs."""
    s, rows = _two_runs(tmp_path)
    cache = pa.ScoreCache(s.conn)
    for tab in ("micrograph", "slip", "uv", "dried"):
        a = cache.get(tab)
        assert a is not None and a.tab == tab
        assert all(type(k) is int and type(v) is float for k, v in a.scores.items())
        assert a.scores == {int(r["item_id"]): float(r["p"]) for r in rows if r["tab"] == tab}
    assert pa.CACHE_TABS == 3 and len(cache.entries) <= 3
    assert list(cache.entries) == ["slip", "uv", "dried"]       # the least recent went first
    assert cache.get("habitat") is None                         # not scored by the run
    some = [int(r["item_id"]) for r in rows if r["tab"] == "uv"][:3]
    det = pa.details(s.conn, cache.get("uv").run, "uv", some)
    assert set(det) == set(some)
    for r in rows:
        if r["tab"] == "uv" and int(r["item_id"]) in det:
            assert det[int(r["item_id"])] == (float(r["p_image"]), float(r["comparator_p"]))


def test_a_new_run_replaces_every_cached_tab(tmp_path):
    s, _ = _two_runs(tmp_path)
    cache = pa.ScoreCache(s.conn)
    old_uv, old_slip = cache.get("uv"), cache.get("slip")

    def shift(rows):
        return [{**r, "p": f"{1 - float(r['p']):.4f}"} for r in rows]
    other = fb.Store(s.path)
    sha2, rows2 = write_run(tmp_path / "r2" / "scores.tsv", other, tabs=("slip", "uv", "dried"),
                            edit=shift)
    new = pa.load(other, tmp_path / "r2" / "scores.tsv", sha2, gate_for(other))["run"]
    uv = cache.get("uv")
    assert uv.run == new != old_uv.run
    assert uv.scores == {int(r["item_id"]): float(r["p"]) for r in rows2 if r["tab"] == "uv"}
    assert all(a is None or a.run == new for a in cache.entries.values())
    assert "slip" not in cache.entries                           # dropped, read again when asked
    assert cache.get("slip").run == new and cache.get("slip").scores != old_slip.scores
    assert cache.get("micrograph") is None                       # not in the new run


def test_an_unfinished_load_of_the_same_file_is_replaced(tmp_path):
    """Fix 3: a 'loading' run left by a load that died mid-way does not block the file."""
    s = make_queue(tmp_path)
    sha, rows = write_run(tmp_path / "run" / "scores.tsv", s)
    pa.ensure_schema(s.conn)
    with s.conn:
        stale = s.conn.execute(
            "insert into photo_audit_runs (run_id, tsv_sha256, manifest_sha256, manifest, "
            "loaded_at) values ('r12-audit-test', ?, 'x', '{}', 1)", (sha,)).lastrowid
        s.conn.execute("insert into photo_audit_scores (run, tab, item_id, p) "
                       "values (?, 'habitat', 1, 0.5)", (stale,))       # a tab the file lacks
    out = pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s))
    assert out["replaced_unfinished_run"] == stale
    assert s.conn.execute("select count(*) from photo_audit_runs").fetchone()[0] == 1
    assert s.conn.execute("select count(*) from photo_audit_scores where tab = 'habitat'"
                          ).fetchone()[0] == 0
    assert pa.latest_run(s.conn)["id"] == out["run"]
    assert s.conn.execute("select count(*) from photo_audit_scores").fetchone()[0] == len(rows)
    with pytest.raises(pa.LoadRefused, match="already loaded"):          # complete: still refused
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s))


def test_no_v1_keys_or_nothing_to_load_is_refused(tmp_path):
    """Fix 4: with no v1 keys (MV_FEEDBACK_V1_KEYS unset) --only-v1 would load an empty run."""
    s = make_queue(tmp_path)
    refused(tmp_path, s, "no v1 record keys", gate=gate_for(s, v1=frozenset()), only_v1=True)
    refused(tmp_path, s, "no row would load", gate=gate_for(s, v1={"not-a-record"}), only_v1=True)
    sha, _ = write_run(tmp_path / "run" / "scores.tsv", s)
    with pytest.raises(pa.LoadRefused, match="no row would load"):        # the dry run says so too
        pa.load(s, tmp_path / "run" / "scores.tsv", sha, gate_for(s, v1={"not-a-record"}),
                only_v1=True, dry_run=True)


def test_the_cli_refuses_when_no_v1_keys_are_named(conn, tmp_path, monkeypatch):
    import argparse

    from mycomap_vision import cli
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    for k in ("MV_FEEDBACK_V1_KEYS", "MV_FEEDBACK_HOLD_KEYS", "MV_FEEDBACK_HOLD_SHA256"):
        monkeypatch.delenv(k, raising=False)
    s = make_queue(tmp_path)
    sha, _ = write_run(tmp_path / "run" / "scores.tsv", s)
    a = {"action": "load-audit-scores", "tsv": str(tmp_path / "run" / "scores.tsv"),
         "expect_sha256": sha, "manifest": None, "dry_run": False, "v1_test_keys": None,
         "v1_test_sha256": None, "only_v1": True, "v1_keys": None, "hold_keys": None,
         "hold_sha256": None, "store": str(s.path)}
    with pytest.raises(SystemExit, match="no v1 record keys"):
        cli.cmd_feedback(conn, argparse.Namespace(**a))
    assert pa.latest_run(s.conn) is None


def test_not_in_this_tab_sorts_by_the_tabs_score_by_default():
    """Fix 6: those photos have no flag in the tab, so Strongest first is their audit p."""
    q = queue() + [item(11, "17", "R9 v2 slip p=0.95 (high)", decision="exclude_photo",
                        reason="slip")]
    a = pr.Audit("uv", {**SCORES, 11: 0.87}, 0.15, "R11.1", 1)
    tiles = pr.flagged_tiles(q, "uv", "decided", outcome="high_not_in_tab", audit=a)
    assert keys(tiles) == ["inat:6", "inat:11", "inat:7"]           # 0.91, 0.87, 0.20
    up = pr.flagged_tiles(q, "uv", "decided", "p_asc", outcome="high_not_in_tab", audit=a)
    assert keys(up) == ["inat:7", "inat:11", "inat:6"]
