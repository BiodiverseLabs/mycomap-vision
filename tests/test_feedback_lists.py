"""The queues' item lists kept across saves (feedback_lists.py; 2026-11-20): after a decision,
a batch, a reopen or a system close only what changed is read again, and the list is then
exactly what a fresh read gives (Store.items_of plus the names filled): the same items, in the
same order, with the same fields, for every reviewer's projects and for read-only viewers. A
seed reads the queue whole again. Also Store.items_of itself, read as one query: it still
gives every queue exactly what the old two-query read gave."""

import json
import random

import pytest
from test_feedback_api import as_user
from test_photo_review import site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_lists as fl
from mycomap_vision import photo_review as pr
from mycomap_vision import photo_views as pv
from mycomap_vision import recovered_labels as rl

INAT = "https://www.inaturalist.org/observations/"
STEVE = {"sub": "u1", "name": "Steve"}
NAMED = (pr.ITEM_TYPE, pv.ITEM_TYPE, rl.ITEM_TYPE)
SCOPES = (None, {"P1"}, {"P2", "P3"}, {"nobody's"}, fb.PUBLIC_ITEMS)
PROJECT_SETS = ([], ["P1"], ["P2"], ["P1", "P3"], ["P3", "P4 with a long name " * 4], ["P4"])


# --- the old read, kept here as the reference ---------------------------------------------
def old_item(r) -> dict:
    d = dict(r)
    for k in ("record_keys", "links", "projects"):
        d[k] = json.loads(d[k] or "[]")
    d["evidence"] = json.loads(d["evidence"] or "{}")
    d["model_flagged"] = bool(d["model_flagged"])
    split = fb.writing_split.of_item(d)
    if split is not None:
        d["list"], d["list_reason"] = split.kind, split.reason
        if split.proposal and split.proposal != (d.get("proposed") or ""):
            d["proposed_seeded"] = d.get("proposed")
            d["proposed"] = split.proposal
    d["blind"] = bool(d["blind"])
    if d["blind"]:
        d["hidden_until_decided"] = d["status"] == "open"
    return d


def old_items_of(store: fb.Store, item_type: str, projects=None) -> list[dict]:
    """Store.items_of as it was before 2026-11-20 (two queries, sqlite3.Row, five parses)."""
    rows = store._visible("select * from items where item_type = ? order by id",
                          (item_type,), projects)
    last = {r["item_id"]: dict(r) for r in store.conn.execute(
        "select d.* from items i join decisions d on d.item_id = i.id where i.item_type = ? "
        f"and {fb.LATEST_DECISION}", (item_type,))}
    return [old_item(r) | {"last_decision": last.get(r["id"])} for r in rows]


# --- a small queue shaped like the live one -----------------------------------------------
class Names:
    """The served manifest's record names (ReviewPhotos.names_of), changeable."""

    def __init__(self):
        self.names = {str(10_000 + k): f"Dna name {k}" for k in range(0, 400, 3)}
        self.asked = 0

    def __call__(self, keys):
        self.asked += len(keys)
        return {k: self.names[k] for k in keys if k in self.names}


def fill_with(names):
    def fill(item_type, items):
        return pr.fill_names(items, names) if item_type in NAMED else items
    return fill


def photo_rows(rng, n, start=0, item_type=pr.ITEM_TYPE):
    rules = ["R6 micrograph probe p=0.{:02d}", "R9 v2 slip p=0.{:02d} (high)",
             "R6 uv probe p=0.{:02d}", "nightly v2.1-336 drawing p=0.{:02d}0 (high)",
             "gate not_fungus p(fungus)=0.{:02d}"]
    out = []
    for k in range(start, start + n):
        rec = str(10_000 + k // 2)
        out.append({"item_type": item_type, "subject": f"inat:{k}", "granularity": "photo",
                    "rule": "; ".join(r.format(rng.randint(10, 99))
                                      for r in rng.sample(rules, rng.choice([1, 1, 2]))),
                    "links": [INAT + rec], "record_keys": [rec] if k % 5 == 0 else [],
                    "current": "" if k % 4 == 0 else f"Label {k % 17}",
                    "evidence": {"evidence": "probe", "p": rng.random(),
                                 **({"crops": [[1, 2, 3, 4]]} if k % 7 == 0 else {})},
                    "projects": rng.choice(PROJECT_SETS)})
    return out


def make_queue(tmp_path, n=240):
    rng = random.Random(5)
    s = fb.Store(tmp_path / "hf.sqlite")
    s.seed(photo_rows(rng, n), "photos")
    s.seed(photo_rows(rng, 30, 5000, rl.ITEM_TYPE), "crops")
    s.seed(photo_rows(rng, 20, 7000, pv.ITEM_TYPE), "views")
    s.seed([{"item_type": "photos_unlike_label", "subject": "300", "rule": "M1", "blind": True,
             "record_keys": ["300"], "current": "Ä ünïcode", "projects": ["P1"],
             "evidence": {"in_v1": "train", "nested": {"a": [1, {"b": None}]}}},
            {"item_type": "name_writing_split", "subject": "Inocybe sp. 'a1'",
             "rule": "R1", "current": "Inocybe sp. 'a1' (2); Inocybe sp. a1 (1)",
             "proposed": "Inocybe sp. 'a1'", "projects": []}], "names")
    ids = [it["id"] for it in s.items_of(pr.ITEM_TYPE)]
    for k, item_id in enumerate(ids):                   # most of the queue already decided
        if k % 3:
            s.decide(item_id, "exclude_photo" if k % 5 else "keep_photo", STEVE,
                     value="micrograph" if k % 5 else None)
    return s


def check(lists, store, names, how=None):
    """Every named queue, every scope: the kept list equals a fresh read."""
    fill = fill_with(names)
    first = None
    for t in NAMED:
        got = lists.of(t)
        if t == pr.ITEM_TYPE:
            first = lists.last[t]
            assert how is None or first == how, (first, how)
        assert got == fill(t, old_items_of(store, t))
        assert [list(i) for i in got] == [list(i) for i in fill(t, store.items_of(t))]  # key order
        for p in SCOPES[1:]:
            assert lists.of(t, p) == fill(t, old_items_of(store, t, p)), (t, p)
    return first


def open_ids(store, n=None):
    return [it["id"] for it in store.items_of(pr.ITEM_TYPE) if it["status"] == "open"][:n]


def undecided_ids(store, n=None):
    """Open items nobody ever decided (Store.close_by_system closes only those)."""
    return [it["id"] for it in store.items_of(pr.ITEM_TYPE) if it["last_decision"] is None][:n]


def decided_ids(store, n=None):
    return [it["id"] for it in store.items_of(pr.ITEM_TYPE) if it["status"] == "decided"][:n]


# --- Store.items_of -----------------------------------------------------------------------
def test_items_of_gives_every_queue_exactly_what_the_old_read_gave(tmp_path):
    s = make_queue(tmp_path)
    s.reopen(decided_ids(s, 1)[0])
    s.close_by_system(undecided_ids(s, 1)[0], "keep_photo", "mycomap.org", "mycomap.org")
    for t in fb.TYPES:
        for p in SCOPES:
            assert s.items_of(t, p) == old_items_of(s, t, p), (t, p)
    blind = s.items_of("photos_unlike_label")[0]
    assert blind["hidden_until_decided"] is True and blind["evidence"]["nested"]["a"][1] == {"b": None}
    split = s.items_of("name_writing_split")[0]
    assert "list" in split and split["projects"] == []


def test_items_that_share_a_project_list_each_get_their_own(tmp_path):
    s = make_queue(tmp_path)
    a, b = [i for i in s.items_of(pr.ITEM_TYPE) if i["projects"] == ["P1", "P3"]][:2]
    a["projects"].append("changed")
    assert b["projects"] == ["P1", "P3"]
    assert s.items_of(pr.ITEM_TYPE)[0]["projects"] is not s.items_of(pr.ITEM_TYPE)[0]["projects"]


def test_items_by_id_reads_those_items_as_items_of_does(tmp_path):
    s = make_queue(tmp_path)
    every = {i["id"]: i for i in s.items_of(pr.ITEM_TYPE)}
    want = sorted(every)[::7]
    assert s.items_by_id(want + want[:2], pr.ITEM_TYPE) == [every[i] for i in want]
    other = s.items_of(rl.ITEM_TYPE)[0]["id"]
    assert s.items_by_id([other], pr.ITEM_TYPE) == []          # another queue's item: not this one's


# --- the kept lists -----------------------------------------------------------------------
def test_after_each_save_the_kept_list_is_a_fresh_read_and_only_a_seed_reads_it_whole(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names))
    check(lists, s, names, "full")
    check(lists, s, names, "same")                              # nothing changed: nothing read

    one = open_ids(s, 1)[0]
    s.decide(one, "exclude_photo", STEVE, value="slip")         # one decision
    check(lists, s, names, "patch")

    page = open_ids(s, 6)                                       # Save page
    pr.batch_decide(s, [{"item_id": i, "decision": "exclude" if k % 2 else "keep",
                         "reason": "micrograph"} for k, i in enumerate(page)], STEVE)
    check(lists, s, names, "patch")

    pr.batch_decide(s, [{"item_id": page[0], "decision": "exclude", "reason": "uv"}], STEVE)
    check(lists, s, names, "patch")                             # a decided photo decided again

    s.reopen(page[1])                                           # writes no decision
    check(lists, s, names, "patch")
    assert next(i for i in lists.of(pr.ITEM_TYPE) if i["id"] == page[1])["status"] == "open"

    s.reopen(page[2])                                           # a reopen and a decision at once
    s.decide(open_ids(s, 1)[0], "keep_photo", STEVE)
    check(lists, s, names, "patch")

    target = undecided_ids(s, 1)[0]                             # the 10-minute round's close
    assert s.close_by_system(target, "keep_photo", "mycomap.org", "mycomap.org")
    check(lists, s, names, "patch")

    d = s.get(page[3])["decisions"][-1]["id"]                   # applied: in last_decision
    s.mark_applied(d, {"sub": "admin"})
    check(lists, s, names, "patch")
    s.mark_applied_by(s.get(page[4])["decisions"][-1]["id"], "mycomap.org")
    check(lists, s, names, "patch")

    other = fb.Store(s.path)                                    # another process's decision
    other.decide(open_ids(s, 1)[0], "exclude_photo", STEVE, value="slip")
    check(lists, s, names, "patch")

    pv_item = lists.of(pv.ITEM_TYPE)[0]["id"]                   # another queue's batch
    pv.batch_decide(s, [{"item_id": pv_item, "views": ["gills"]}], STEVE)
    check(lists, s, names, "patch")

    rng = random.Random(9)                                      # a seed: new items, a flag added
    flagged = next(i for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "open")
    s.seed(photo_rows(rng, 4, 9000) + [{**photo_rows(rng, 1)[0], "subject": flagged["subject"],
                                        "rule": "R11.1 v2 uv p=0.043 (very_low)"}], "nightly")
    check(lists, s, names, "full")
    assert "R11.1" in next(i for i in lists.of(pr.ITEM_TYPE) if i["id"] == flagged["id"])["rule"]

    rec = pr.record_of_item(lists.of(pr.ITEM_TYPE)[0])          # the record view's save
    pr.decide_record(s, rec, [{"photo_key": "inat:unflagged-1", "decision": "exclude",
                               "reason": "slip"}], STEVE, {"inat:unflagged-1"})
    check(lists, s, names, "full")                              # it seeds an item first
    s.decide(open_ids(s, 1)[0], "keep_photo", STEVE)
    check(lists, s, names, "patch")


def test_a_long_random_round_of_saves_stays_equal_to_fresh_reads(tmp_path):
    s = make_queue(tmp_path, n=120)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names))
    rng = random.Random(11)
    hows = []
    for step in range(120):
        op = rng.choice(["decide", "batch", "batch", "reopen", "close", "applied", "seed"]
                        if step % 25 else ["seed"])
        if op == "decide" and open_ids(s):
            s.decide(rng.choice(open_ids(s)), "exclude_photo", STEVE, value="slip")
        elif op == "batch":
            ids = rng.sample(open_ids(s) + decided_ids(s), 5)
            pr.batch_decide(s, [{"item_id": i, "decision": rng.choice(["keep", "exclude"]),
                                 "reason": rng.choice(["micrograph", "uv", "slip"])}
                                for i in ids], STEVE)
        elif op == "reopen":
            s.reopen(rng.choice(decided_ids(s)))
        elif op == "close" and undecided_ids(s):
            assert s.close_by_system(rng.choice(undecided_ids(s)), "keep_photo", "mycomap.org",
                                     "mycomap.org")
        elif op == "applied":
            item = s.get(rng.choice(decided_ids(s)))
            s.mark_applied(item["decisions"][-1]["id"], STEVE)
        elif op == "seed":
            s.seed(photo_rows(rng, 2, 20_000 + step * 2), f"seed-{step}")
        if rng.random() < 0.2 and lists.lists:                 # a new release's names
            names.names[str(10_000 + rng.randrange(200))] = f"New name {step}"
            for kept in lists.lists.values():
                kept.filled_at = -1e9                           # past NAMES_TTL
        hows.append(check(lists, s, names))
    assert hows.count("patch") > 60 and "full" in hows


def test_names_are_filled_for_the_changed_items_and_read_again_after_a_while(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    now = [0.0]
    lists = fl.ItemLists(s, fill_with(names), names_ttl=600, clock=lambda: now[0])
    lists.of(pr.ITEM_TYPE)
    asked = names.asked
    blank = next(i for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "open" and not i["current"])
    s.decide(blank["id"], "keep_photo", STEVE)
    names.names[pr.record_of_item(blank)] = "Renamed since"
    got = next(i for i in lists.of(pr.ITEM_TYPE) if i["id"] == blank["id"])
    assert got["current"] == "Renamed since" and names.asked - asked == 1   # its record alone
    others = {pr.record_of_item(i) for i in s.items_of(pr.ITEM_TYPE)
              if not i["current"] and i["id"] != blank["id"]}
    for k in others:
        names.names[k] = "From a new release"
    now[0] = 601
    s.decide(open_ids(s, 1)[0], "keep_photo", STEVE)
    assert lists.of(pr.ITEM_TYPE) == fill_with(names)(pr.ITEM_TYPE, s.items_of(pr.ITEM_TYPE))
    assert lists.last[pr.ITEM_TYPE] == "names"


def test_a_list_handed_out_is_never_changed_by_a_later_save(tmp_path):
    s = make_queue(tmp_path)
    lists = fl.ItemLists(s, fill_with(Names()))
    before = lists.of(pr.ITEM_TYPE)
    snapshot = [dict(i) for i in before]
    s.decide(open_ids(s, 1)[0], "keep_photo", STEVE)
    after = lists.of(pr.ITEM_TYPE)
    assert after is not before and [dict(i) for i in before] == snapshot


# --- through the API ----------------------------------------------------------------------
def test_the_summary_after_save_page_reads_only_the_saved_photos_and_shows_them_saved(conn,
                                                                                         tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    lists = c.app.state.feedback_lists
    as_user(c, "leader-1")
    tiles = c.get("/api/feedback/photo-review/flagged", params={"tab": "micrograph"}).json()["tiles"]
    assert len(tiles) == 2 and lists.last[pr.ITEM_TYPE] == "full"
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": tiles[0]["item_id"], "decision": "exclude", "reason": "micrograph"}]})
    assert r.status_code == 200
    left = c.get("/api/feedback/photo-review/flagged", params={"tab": "micrograph"}).json()
    assert [t["item_id"] for t in left["tiles"]] == [tiles[1]["item_id"]]
    assert lists.last[pr.ITEM_TYPE] == "patch"
    done = c.get("/api/feedback/photo-review/flagged",
                 params={"tab": "micrograph", "status": "decided"}).json()["tiles"]
    assert [(t["item_id"], t["decision"]) for t in done] == [
        (tiles[0]["item_id"], {"decision": "exclude_photo", "reason": "micrograph"})]
    as_user(c, "member-1")                                      # another project's: none of it
    assert c.get("/api/feedback/photo-review/flagged").json()["total"] == 0
    c.cookies.clear()                                           # read-only: every photo item
    ro = c.get("/api/feedback/photo-review/flagged", params={"tab": "micrograph"}).json()
    assert [t["item_id"] for t in ro["tiles"]] == [tiles[1]["item_id"]]
    as_user(c, "admin-1")                                       # a reopen brings it back
    assert c.post(f"/api/feedback/items/{tiles[0]['item_id']}/reopen").status_code == 200
    back = c.get("/api/feedback/photo-review/flagged", params={"tab": "micrograph"}).json()
    assert {t["item_id"] for t in back["tiles"]} == {t["item_id"] for t in tiles}
    assert lists.last[pr.ITEM_TYPE] == "patch"
    summary = {t["type"]: t for t in c.get("/api/feedback/summary").json()}[pr.ITEM_TYPE]
    assert (summary["open"], summary["records_open"]) == (3, 2)


def test_a_reviewer_refused_a_save_leaves_the_kept_list_as_it_was(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    first = c.get("/api/feedback/photo-review/flagged").json()
    c.cookies.clear()
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": first["tiles"][0]["item_id"], "decision": "exclude", "reason": "micrograph"}]})
    assert r.status_code == 401
    as_user(c, "member-1")
    assert c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": first["tiles"][0]["item_id"], "decision": "exclude",
         "reason": "micrograph"}]}).status_code == 400              # not one they review
    as_user(c, "admin-1")
    assert c.get("/api/feedback/photo-review/flagged").json() == first


@pytest.mark.parametrize("change", ["items", "decision removed"])
def test_a_change_that_cannot_be_traced_reads_the_queue_whole(tmp_path, change):
    s = make_queue(tmp_path)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names))
    lists.of(pr.ITEM_TYPE)
    if change == "items":                                       # an item added without a seed
        s.conn.execute("insert into items (item_type, subject, granularity, rule, seed, "
                       "created_at) values (?, 'inat:x', 'photo', 'R6 uv probe p=0.5', 'x', 0)",
                       (pr.ITEM_TYPE,))
    else:
        s.conn.execute("delete from decisions where id = (select min(id) from decisions)")
    s.conn.commit()
    check(lists, s, names, "full")
