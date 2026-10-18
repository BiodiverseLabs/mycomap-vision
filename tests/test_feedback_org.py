"""Human Feedback decisions about a whole name open a Name Update on mycomap.org
(feedback_org.py), against a fake of .org's answers (its branch feat/name-update-requests,
routes/visionNameUpdates.ts and services/nameUpdateRequests.ts): which decisions send one and
with which names, what an unreachable or refusing .org leaves, the item's status block and
row chip, applied only when .org says done, withdrawing on reopen, and who may do any of it."""

import json

import pytest
import requests
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_api
from mycomap_vision import feedback_org as fo

ORG = "https://www.mycomap.org"


# --- a fake of .org's Vision routes ------------------------------------------------------------

class FakeNameUpdates:
    """POST/GET/DELETE /api/vision/name-updates as .org answers them, plus Human Feedback's
    roles. `down` makes every call fail as if .org were unreachable."""

    OPEN = ("searching", "waiting", "running", "needs_attention")

    def __init__(self):
        self.requests: dict[int, dict] = {}
        self.calls: list[tuple] = []
        self.down = False
        self.next_id = 7
        self.refuse_names: set[str] = set()
        self.roles_by_user = {
            "admin-1": {"admin": True, "staff": True, "projects": []},
            "leader-1": {"admin": False, "staff": True,
                         "projects": [{"id": "1", "name": "p1", "role": "leader"}]},
            "member-2": {"admin": False, "staff": False,
                         "projects": [{"id": "2", "name": "p2", "role": "contributor"}]},
            "nobody-1": {"admin": False, "staff": False, "projects": []},
        }

    def _net(self):
        if self.down:
            raise requests.ConnectionError("mycomap.org is down")

    def roles(self, user_id):
        return self.roles_by_user.get(user_id)

    def record(self, key):
        return None

    def view(self, r):
        return {k: r[k] for k in ("id", "status", "url", "note", "declined_by",
                                  "declined_reason", "old_name_left", "batches", "totals")}

    def name_update_post(self, body):
        self.calls.append(("POST", body))
        self._net()
        if not body.get("decision_id") or body.get("kind") not in (
                "respelling", "accepted_name", "rename", "same_taxon"):
            return 400, {"code": "bad_request", "message": "bad body"}
        if set(body["old_names"]) & self.refuse_names or body["new_name"] in body["old_names"]:
            return 422, {"code": "refused", "message": "Mycena zzz is not in MycoMap's names list"}
        for r in self.requests.values():
            if r["decision_id"] == body["decision_id"]:
                if sorted(r["old_names"]) != sorted(body["old_names"]):
                    return 409, {"code": "conflict", "message": "a different change"}
                if r["status"] == "refused":
                    r["status"] = "searching"
                    return 202, self.view(r)
                return 200, self.view(r)
        for r in self.requests.values():
            if r["status"] in self.OPEN and set(r["old_names"]) & set(body["old_names"]):
                return 409, {"code": "open_request", "message": f"Name Update request {r['id']} "
                             "for this name is still open on mycomap.org.", "request_id": r["id"]}
        rid, self.next_id = self.next_id, self.next_id + 1
        self.requests[rid] = {"id": rid, "status": "searching", "decision_id": body["decision_id"],
                              "old_names": body["old_names"], "new_name": body["new_name"],
                              "url": f"{ORG}/admin/name-update?request={rid}", "note": None,
                              "declined_by": None, "declined_reason": None,
                              "old_name_left": None, "batches": [], "totals": {
                                  "total": 0, "fixed": 0, "set_aside": 0, "problems": 0,
                                  "to_fix": 0}, "started": False, "body": body}
        return 202, self.view(self.requests[rid])

    def name_update_get(self, rid):
        self.calls.append(("GET", rid))
        self._net()
        r = self.requests.get(int(rid))
        return self.view(r) if r else None

    def name_update_find(self, decision_id):
        self.calls.append(("FIND", decision_id))
        self._net()
        r = next((r for r in self.requests.values() if r["decision_id"] == decision_id), None)
        return self.view(r) if r else None

    def name_update_delete(self, rid):
        self.calls.append(("DELETE", rid))
        self._net()
        r = self.requests.get(int(rid))
        if r is None:
            return 404, {"code": "not_found"}
        if r["status"] == "withdrawn":
            return 200, self.view(r)
        if r["status"] not in ("searching", "waiting", "refused") or r["started"]:
            return 409, {"code": "started", "message": "A person has already run this update on "
                         "mycomap.org, so it cannot be withdrawn.", "status": r["status"]}
        r["status"] = "withdrawn"
        return 200, self.view(r)

    # .org moving a request along -------------------------------------------------------------
    def searched(self, rid, total):
        r = self.requests[rid]
        r.update(status="waiting", totals={**r["totals"], "total": total, "to_fix": total})

    def run(self, rid):
        self.requests[rid].update(status="running", started=True)

    def finish(self, rid, fixed):
        r = self.requests[rid]
        r.update(status="done", totals={**r["totals"], "fixed": fixed, "to_fix": 0},
                 old_name_left=0)

    def gets(self):
        return [c for c in self.calls if c[0] == "GET"]


# --- an app with only the Human Feedback routes -------------------------------------------------

def seed(store):
    store.seed([
        {"item_type": "name_writing_split", "subject": "Mycena sp. 'IN10'; Mycena \"sp-IN10\"",
         "rule": "R1", "current": "Mycena sp. 'IN10' (456) | Mycena \"sp-IN10\" (1)",
         "proposed": "Mycena sp. 'IN10'", "records": 457, "projects": ["p1"]},
        {"item_type": "name_retired", "subject": "Amanita retired", "rule": "R4",
         "current": "Amanita retired", "proposed": "Amanita accepted", "projects": ["p1"]},
        {"item_type": "name_inactive_on_inat", "subject": "Ramaria rubricarnata verna",
         "rule": "R5", "current": "Ramaria rubricarnata verna", "proposed": "",
         "projects": ["p1"]},
        {"item_type": "crosswalk", "subject": "Russula sp. 'IN26' <-> Russula fragilis",
         "rule": "crosswalk: dna", "current": "Russula sp. 'IN26'", "proposed": "Russula fragilis",
         "projects": ["p1"]},
        {"item_type": "vision_names_same_taxon", "subject": "555", "rule": "M3",
         "record_keys": ["555"], "current": "Pluteus cervinus", "proposed": "Pluteus atricapillus",
         "projects": ["p1"]},
        {"item_type": "group_label", "subject": "Laccaria laccata group", "rule": "R2",
         "current": "Laccaria laccata group", "projects": ["p1"]},
        {"item_type": "label_vs_own_dna", "subject": "556", "rule": "R6", "record_keys": ["556"],
         "current": "Inocybe a", "proposed": "Inocybe b", "projects": ["p1"]},
        {"item_type": "name_retired", "subject": "Boletus other", "rule": "R4",
         "current": "Boletus other", "proposed": "Boletus newer", "projects": ["p2"]},
    ], "test")


def make(tmp_path, name_info=None, monkeypatch=None):
    store = fb.Store(tmp_path / "state" / "human-feedback.sqlite")
    seed(store)
    org = FakeNameUpdates()
    done = []
    app = FastAPI()

    def person(request: Request):
        sub = request.headers.get("x-test-user")
        return {"sub": sub, "name": f"Name of {sub}"} if sub else None
    reqs = feedback_api.add_routes(app, person=person, store=store, org=org,
                                   name_info=name_info, on_name_update_done=lambda: done.append(1),
                                   note=lambda s: None)
    return TestClient(app), store, org, reqs, done


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv("MV_PUBLIC_ORIGIN", "https://vision.mycomap.org")
    return make(tmp_path)


def iid(store, item_type, subject=None):
    items = store.items(item_type)["items"] + store.items(item_type, "decided")["items"]
    return next(i["id"] for i in items if subject is None or i["subject"] == subject)


def as_(sub):
    return {"x-test-user": sub}


ADMIN = as_("admin-1")


def decide(c, item_id, decision, value=None, who=ADMIN, note=None):
    body = {"decision": decision}
    if value is not None:
        body["value"] = value
    if note is not None:
        body["note"] = note
    return c.post(f"/api/feedback/items/{item_id}/decision", json=body, headers=who)


# --- which decisions open a Name Update, with which names -----------------------------------

def item(item_type, current, proposed="", **extra):
    return {"item_type": item_type, "current": current, "proposed": proposed, **extra}


def test_writings_are_read_from_current_without_their_record_counts():
    assert fo.writings("Mycena sp. 'IN10' (456) | Mycena \"sp-IN10\" (1)") == [
        "Mycena sp. 'IN10'", 'Mycena "sp-IN10"']
    assert fo.writings("A b (1,204) | A  b var. c (3)") == ["A b", "A  b var. c"]
    assert fo.writings("") == []


def test_one_writing_for_all_sends_every_other_writing_as_a_respelling():
    it = item("name_writing_split", "A b (5) | A  b (2) | A b. (1)", "A b")
    p = fo.plan(it, "use_writing", "A b")
    assert (p.kind, p.old_names, p.new_name) == ("respelling", ["A  b", "A b."], "A b")
    other = fo.plan(it, "use_writing", "A b.")              # the value chosen, not the proposal
    assert other.old_names == ["A b", "A  b"] and other.new_name == "A b."


def test_retired_and_inactive_names_accept_or_rename_the_whole_name():
    for t in ("name_retired", "name_inactive_on_inat"):
        it = item(t, "Old name", "Accepted name")
        assert fo.plan(it, "accept").body() == {"kind": "accepted_name",
                                                "old_names": ["Old name"],
                                                "new_name": "Accepted name"}
        # The accepted name shown before saving can be corrected (typos come from source).
        assert fo.plan(it, "accept", "Accepted nome").new_name == "Accepted nome"
        assert fo.plan(it, "rename", "Typed name").body() == {
            "kind": "rename", "old_names": ["Old name"], "new_name": "Typed name"}
        assert fo.plan(it, "keep") is None


def test_accept_is_refused_where_inat_gives_no_successor():
    it = item("name_inactive_on_inat", "Ramaria rubricarnata verna", "")
    with pytest.raises(fb.FeedbackError, match="use Rename"):
        fo.plan(it, "accept")
    assert fo.plan(it, "rename", "Ramaria rubricarnata").new_name == "Ramaria rubricarnata"


def test_crosswalk_same_keeps_one_of_the_two_names_and_renames_the_other():
    it = item("crosswalk", "Russula sp. 'IN26'", "Russula fragilis")
    p = fo.plan(it, "same", "Russula fragilis")
    assert (p.kind, p.old_names, p.new_name) == ("same_taxon", ["Russula sp. 'IN26'"],
                                                 "Russula fragilis")
    assert fo.plan(it, "same", "Russula sp. 'IN26'").old_names == ["Russula fragilis"]
    with pytest.raises(fb.FeedbackError, match="one of"):
        fo.plan(it, "same", "Russula third")
    assert fo.plan(it, "nested") is None and fo.plan(it, "different") is None


def test_vision_names_same_taxon_sends_the_less_used_name_to_the_more_used():
    it = item("vision_names_same_taxon", "Label name", "Vision name")
    counts = {"Label name": 40, "Vision name": 3}
    info = lambda names: {n: {"records": counts.get(n, 0), "genus_known": True}  # noqa: E731
                          for n in names}
    assert fo.plan(it, "same", name_info=info).body() == {
        "kind": "same_taxon", "old_names": ["Vision name"], "new_name": "Label name"}
    counts.update({"Vision name": 40})                   # a tie: the label goes to Vision's name
    assert fo.plan(it, "same", name_info=info).old_names == ["Label name"]
    assert fo.plan(it, "same").old_names == ["Label name"]          # no counts at all
    assert fo.plan(it, "crosswalk") is None and fo.plan(it, "different") is None


def test_record_level_decisions_open_no_name_update():
    for t, decisions in {
            "group_label": ("rename", "keep_group"),
            "label_vs_own_dna": ("label_right", "rename", "bad_sequence", "unsure"),
            "photos_unlike_label": ("rename", "exclude_photos", "hard_example"),
            "reference_label_doubt": ("label_right", "rename", "exclude_record"),
            "non_specimen_photo": ("exclude_photo", "keep_photo"),
            "crosspost_twin": ("same_find", "different_finds"),
            "twin_location_conflict": ("fix_mo_location", "fix_inat_location", "both_right"),
            "name_writing_split": ("different",)}.items():
        for d in decisions:
            assert fo.plan(item(t, "A b", "C d"), d, "E f") is None, (t, d)


def test_every_whole_name_decision_in_the_catalogue_has_a_kind():
    """A new whole-name option added to feedback.TYPES must be listed in KINDS or nowhere."""
    for (t, d) in fo.KINDS:
        assert any(o.code == d for o in fb.TYPES[t].options), (t, d)


def test_a_plan_that_changes_nothing_or_too_much_is_refused():
    with pytest.raises(fb.FeedbackError, match="already the name"):
        fo.plan(item("name_retired", "Same", "Same"), "accept")
    many = " | ".join(f"A b{i} (1)" for i in range(12))
    with pytest.raises(fb.FeedbackError, match="at most 10"):
        fo.plan(item("name_writing_split", many, "A b0"), "use_writing", "A new")
    with pytest.raises(fb.FeedbackError):
        fo.plan(item("name_writing_split", "A b (3)", "A b"), "use_writing", "A b")


def test_warnings_flag_an_unknown_target_a_code_formal_pair_and_another_rank():
    held = {"A c": 4, "Echinoderma sp. 'PR01'": 2}
    info = lambda names: {n: {"records": held.get(n, 0),  # noqa: E731
                              "genus_known": not n.startswith("Echinoderrma")} for n in names}
    p = fo.plan(item("name_retired", "Echinoderma sp. 'PR01'", "Echinoderrma sp. 'PR01'"),
                "accept", name_info=info)
    assert p.warnings == ["No record Vision holds is called Echinoderrma sp. 'PR01', and iNat "
                          "has no genus Echinoderrma: check the spelling before every record "
                          "gets it"]
    p = fo.plan(item("name_retired", "Amanita retired", "Amanita acepted"), "accept",
                name_info=info)                     # a known genus, a misspelt epithet
    assert p.warnings == ["No record Vision holds is called Amanita acepted: check the "
                          "spelling before every record gets it"]
    p = fo.plan(item("vision_names_same_taxon", "Russula sp. 'IN26'", "Russula fragilis"), "same")
    assert any("provisional code with a formal name" in w for w in p.warnings)
    p = fo.plan(item("vision_names_same_taxon", "Pseudohydnum gelatinosum",
                     "Pseudohydnum gelatinosum ssp. pusillum"), "same")
    assert any("different ranks" in w for w in p.warnings)
    assert fo.plan(item("name_retired", "A b", "A c"), "accept", name_info=info).warnings == []


# --- deciding sends the request --------------------------------------------------------------

def test_a_whole_name_decision_is_sent_to_org_with_who_decided_and_the_link_back(site):
    c, store, org, _, _ = site
    i = iid(store, "name_writing_split")
    r = decide(c, i, "use_writing", "Mycena sp. 'IN10'", note="one name, Vision right")
    assert r.status_code == 200 and r.json()["status"] == "decided"
    (_, body), = [x for x in org.calls if x[0] == "POST"]
    did = store.get(i)["decisions"][-1]["id"]
    assert body == {"decision_id": str(did), "kind": "respelling",
                    "old_names": ['Mycena "sp-IN10"'], "new_name": "Mycena sp. 'IN10'",
                    "decided_by": {"org_user_id": "admin-1", "name": "Name of admin-1"},
                    "reason": "one name, Vision right",
                    "item_url": f"https://vision.mycomap.org/feedback?item={i}"}
    req = r.json()["org_request"]
    assert req["status"] == "searching" and req["request_id"] == 7
    assert req["url"] == f"{ORG}/admin/name-update?request=7"
    assert [s["state"] for s in req["steps"]] == ["done", "current", "todo", "todo"]
    assert req["chip"]["text"] == "on mycomap.org: searching"
    row = store.conn.execute("select * from org_requests where decision_id = ?", (did,)).fetchone()
    assert row["request_id"] == 7 and row["status"] == "searching" and row["sent_at"]


def test_record_level_and_note_decisions_send_nothing(site):
    c, store, org, _, _ = site
    assert decide(c, iid(store, "group_label"), "rename", "Laccaria sp. 'PNW01'").status_code == 200
    assert decide(c, iid(store, "label_vs_own_dna"), "label_right").status_code == 200
    assert decide(c, iid(store, "crosswalk"), "nested").status_code == 200
    assert decide(c, iid(store, "name_retired", "Amanita retired"), "keep").status_code == 200
    assert not [x for x in org.calls if x[0] == "POST"]
    assert store.conn.execute("select count(*) from org_requests").fetchone()[0] == 0
    out = decide(c, iid(store, "vision_names_same_taxon"), "different").json()
    assert out["org_request"] is None


def test_a_decision_that_cannot_make_an_update_is_refused_and_not_saved(site):
    c, store, org, _, _ = site
    i = iid(store, "name_inactive_on_inat")
    r = decide(c, i, "accept")
    assert r.status_code == 400 and "use Rename" in r.json()["detail"]
    r = decide(c, iid(store, "crosswalk"), "same", "Russula third")
    assert r.status_code == 400 and "one of" in r.json()["detail"]
    assert store.get(i)["status"] == "open" and store.get(i)["decisions"] == []
    assert not [x for x in org.calls if x[0] == "POST"]


def test_unreachable_org_saves_the_decision_as_not_sent_and_send_again_sends_it(site):
    c, store, org, _, _ = site
    org.down = True
    i = iid(store, "name_retired", "Amanita retired")
    r = decide(c, i, "accept")
    assert r.status_code == 200 and r.json()["status"] == "decided"
    req = r.json()["org_request"]
    assert req["status"] == "not_sent" and req["can_send"] is True
    assert req["chip"] == {"text": "not sent to mycomap.org yet", "tone": "bad"}
    assert req["steps"][0] == {"step": "sent", "state": "stopped", "label": "Not sent yet"}
    assert "could not be reached" in req["note"]
    org.down = False
    again = c.post(f"/api/feedback/items/{i}/org-request", headers=ADMIN)
    assert again.status_code == 200
    assert again.json()["org_request"]["status"] == "searching"
    posts = [b for k, b in org.calls if k == "POST"]
    assert len(posts) == 2 and posts[0] == posts[1]          # the same request, sent again
    r = c.post(f"/api/feedback/items/{i}/org-request", headers=ADMIN)
    assert r.status_code == 409 and "already on mycomap.org" in r.json()["detail"]


def test_without_a_key_nothing_is_sent_and_the_item_says_so(tmp_path):
    store = fb.Store(tmp_path / "f.sqlite")
    seed(store)
    client = feedback_api.OrgClient(base_url=ORG, key="")
    reqs = fo.OrgRequests(store, client)
    i = iid(store, "name_retired", "Amanita retired")
    store.decide(i, "accept", {"sub": "admin-1"})
    row = reqs.send(store.get(i)["decisions"][-1]["id"])
    assert row["status"] == "not_sent" and "no mycomap.org key" in row["note"]


def test_a_refusal_by_org_is_kept_with_its_message_and_can_be_sent_again(site):
    c, store, org, _, _ = site
    org.refuse_names.add("Amanita retired")
    i = iid(store, "name_retired", "Amanita retired")
    req = decide(c, i, "accept").json()["org_request"]
    assert req["status"] == "not_taken" and "not in MycoMap's names list" in req["note"]
    assert req["can_send"] and req["chip"]["text"] == "not taken by mycomap.org"
    org.refuse_names.clear()
    assert c.post(f"/api/feedback/items/{i}/org-request",
                  headers=ADMIN).json()["org_request"]["status"] == "searching"


def test_another_open_request_for_the_name_is_not_taken_until_it_closes(site):
    c, store, org, _, _ = site
    org.requests[3] = {"id": 3, "status": "waiting", "decision_id": "x", "old_names":
                       ["Amanita retired"], "new_name": "A", "url": "u", "note": None,
                       "declined_by": None, "declined_reason": None, "old_name_left": None,
                       "batches": [], "totals": {}, "started": False}
    i = iid(store, "name_retired", "Amanita retired")
    req = decide(c, i, "accept").json()["org_request"]
    assert req["status"] == "not_taken" and "request 3" in req["note"]


def test_send_again_needs_a_decided_item_whose_decision_opens_an_update(site):
    c, store, _, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    assert c.post(f"/api/feedback/items/{i}/org-request", headers=ADMIN).status_code == 409
    g = iid(store, "group_label")
    decide(c, g, "rename", "Laccaria sp. 'PNW01'")
    r = c.post(f"/api/feedback/items/{g}/org-request", headers=ADMIN)
    assert r.status_code == 409 and "opens no Name Update" in r.json()["detail"]


# --- following the request ---------------------------------------------------------------------

def test_opening_the_item_reads_org_and_shows_the_four_steps(site):
    c, store, org, _, _ = site
    i = iid(store, "name_writing_split")
    decide(c, i, "use_writing", "Mycena sp. 'IN10'")
    org.searched(7, 1)
    req = c.get(f"/api/feedback/items/{i}", headers=ADMIN).json()["org_request"]
    assert req["status"] == "waiting"
    assert [(s["label"], s["state"]) for s in req["steps"]] == [
        ("Sent", "done"), ("Searched: 1 record", "done"), ("Waiting for a person", "current"),
        ("Done", "todo")]
    assert req["chip"] == {"text": "on mycomap.org: waiting for a person", "tone": "busy"}
    assert req["totals"]["to_fix"] == 1 and req["can_check"] is True
    listed = c.get("/api/feedback/items", params={"type": "name_writing_split",
                                                  "status": "decided"}, headers=ADMIN).json()
    assert listed["items"][0]["org_chip"]["text"] == "on mycomap.org: waiting for a person"
    org.run(7)
    req = c.get(f"/api/feedback/items/{i}", headers=ADMIN).json()["org_request"]
    assert req["chip"]["text"] == "on mycomap.org: running"
    assert req["steps"][3] == {"step": "done", "state": "current", "label": "Running"}


def test_done_on_org_is_the_only_automatic_way_to_applied(site):
    c, store, org, _, done = site
    i = iid(store, "name_writing_split")
    decide(c, i, "use_writing", "Mycena sp. 'IN10'")
    did = store.get(i)["decisions"][-1]["id"]
    for step in (lambda: org.searched(7, 1), lambda: org.run(7)):
        step()
        c.get(f"/api/feedback/items/{i}", headers=ADMIN)
        assert store.get(i)["decisions"][-1]["applied"] == 0
    org.requests[7]["status"] = "needs_attention"
    org.requests[7]["note"] = "1 record on mycomap.org still says the old name"
    req = c.get(f"/api/feedback/items/{i}", headers=ADMIN).json()["org_request"]
    assert req["chip"]["text"] == "on mycomap.org: needs attention" and "still says" in req["note"]
    assert store.get(i)["decisions"][-1]["applied"] == 0 and done == []
    org.finish(7, 1)
    req = c.get(f"/api/feedback/items/{i}", headers=ADMIN).json()["org_request"]
    d = store.get(i)["decisions"][-1]
    assert d["id"] == did and d["applied"] == 1
    assert d["applied_by"] == "mycomap.org Name Update 7" and d["applied_at"]
    assert req["chip"] == {"text": "applied on mycomap.org · 1 renamed", "tone": "good"}
    assert [s["state"] for s in req["steps"]] == ["done"] * 4
    assert done == [1]                              # the name-change log is read at once
    c.get(f"/api/feedback/items/{i}", headers=ADMIN)
    assert done == [1] and len(org.gets()) == 4     # finished: not read again


def test_the_ten_minute_check_reads_only_unfinished_requests(site):
    c, store, org, reqs, done = site
    a = iid(store, "name_writing_split")
    b = iid(store, "name_retired", "Amanita retired")
    decide(c, a, "use_writing", "Mycena sp. 'IN10'")
    decide(c, b, "accept")
    org.down = True
    decide(c, iid(store, "crosswalk"), "same", "Russula fragilis")      # not sent: not checked
    org.down = False
    org.finish(7, 1)
    assert reqs.poll_once() == {"checked": 2, "done": 1}
    assert store.get(a)["decisions"][-1]["applied"] == 1 and done == [1]
    assert reqs.poll_once() == {"checked": 1, "done": 0}
    org.down = True
    assert reqs.poll_once() == {"checked": 1, "done": 0}          # down: kept as it was
    assert reqs.row(store.get(b)["decisions"][-1]["id"])["status"] == "searching"
    assert fo.POLL_SECONDS == 600


def test_a_declined_request_shows_who_declined_and_why(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.requests[7].update(status="declined", declined_by="Steve Russell",
                           declined_reason="Amanita accepted is a misapplied name")
    req = c.get(f"/api/feedback/items/{i}", headers=ADMIN).json()["org_request"]
    assert req["status"] == "declined" and req["declined_by"] == "Steve Russell"
    assert req["declined_reason"] == "Amanita accepted is a misapplied name"
    assert req["chip"] == {"text": "declined on mycomap.org", "tone": "bad"}
    assert req["steps"][2] == {"step": "waiting", "state": "stopped", "label": "Declined"}
    assert store.get(i)["decisions"][-1]["applied"] == 0


def test_a_refused_search_can_be_sent_again(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.requests[7].update(status="refused", note="not in MycoMap's names list")
    req = c.get(f"/api/feedback/items/{i}", headers=ADMIN).json()["org_request"]
    assert req["status"] == "refused" and req["can_send"] and req["can_check"] is False
    assert req["chip"]["text"] == "refused by mycomap.org"
    assert req["steps"][1] == {"step": "searched", "state": "stopped", "label": "Search refused"}
    again = c.post(f"/api/feedback/items/{i}/org-request", headers=ADMIN).json()["org_request"]
    assert again["status"] == "searching" and again["request_id"] == 7


# --- reopening withdraws -----------------------------------------------------------------------

def test_reopening_an_item_whose_request_is_waiting_withdraws_it(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.searched(7, 4)
    r = c.post(f"/api/feedback/items/{i}/reopen", headers=ADMIN)
    assert r.status_code == 200 and r.json()["status"] == "open"
    assert ("DELETE", 7) in org.calls and org.requests[7]["status"] == "withdrawn"
    assert r.json()["org_request"]["chip"]["text"] == "withdrawn from mycomap.org"
    again = decide(c, i, "rename", "Amanita other").json()["org_request"]
    assert again["status"] == "searching" and again["request_id"] == 8


def test_a_request_a_person_has_run_cannot_be_withdrawn_by_reopening(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.searched(7, 4)
    org.run(7)
    c.get(f"/api/feedback/items/{i}", headers=ADMIN)           # Vision has seen it running
    r = c.post(f"/api/feedback/items/{i}/reopen", headers=ADMIN)
    assert r.status_code == 409 and "already run" in r.json()["detail"]
    assert store.get(i)["status"] == "decided" and ("DELETE", 7) not in org.calls


def test_a_run_started_since_the_last_check_refuses_the_reopen(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.searched(7, 4)
    org.run(7)                                     # Vision still thinks it is searching
    r = c.post(f"/api/feedback/items/{i}/reopen", headers=ADMIN)
    assert r.status_code == 409 and store.get(i)["status"] == "decided"
    assert fo.OrgRequests(store, org).for_item(i)["status"] == "running"


def test_reopening_while_org_is_down_is_refused_so_nothing_is_left_behind(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.down = True
    r = c.post(f"/api/feedback/items/{i}/reopen", headers=ADMIN)
    assert r.status_code == 409 and "could not be reached" in r.json()["detail"]
    assert store.get(i)["status"] == "decided"


def test_reopening_a_decision_never_sent_withdraws_it_wherever_it_got(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    org.down = True
    decide(c, i, "accept")
    org.down = False
    r = c.post(f"/api/feedback/items/{i}/reopen", headers=ADMIN)
    assert r.status_code == 200 and not [x for x in org.calls if x[0] == "DELETE"]
    assert fo.OrgRequests(store, org).for_item(i)["status"] == "withdrawn"
    # A POST whose answer was lost did reach .org: found by the decision and withdrawn.
    b = iid(store, "name_writing_split")
    decide(c, b, "use_writing", "Mycena sp. 'IN10'")
    did = store.get(b)["decisions"][-1]["id"]
    with store.conn:
        store.conn.execute("update org_requests set status = 'not_sent', request_id = null "
                           "where decision_id = ?", (did,))
    assert c.post(f"/api/feedback/items/{b}/reopen", headers=ADMIN).status_code == 200
    assert ("FIND", str(did)) in org.calls and org.requests[7]["status"] == "withdrawn"


def test_reopening_a_finished_item_leaves_org_alone(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")
    decide(c, i, "accept")
    org.finish(7, 2)
    c.get(f"/api/feedback/items/{i}", headers=ADMIN)
    assert c.post(f"/api/feedback/items/{i}/reopen", headers=ADMIN).status_code == 200
    assert not [x for x in org.calls if x[0] == "DELETE"]


# --- the preview and who may do what -----------------------------------------------------------

def test_the_preview_shows_the_names_before_saving_and_flags_unknown_ones(tmp_path):
    info = lambda names: {n: {"records": {"Amanita retired": 12, "Amanita accepted": 5}.get(  # noqa: E731
                              n, 0), "genus_known": True} for n in names}
    c, store, org, _, _ = make(tmp_path, name_info=info)
    i = iid(store, "name_retired", "Amanita retired")
    r = c.post(f"/api/feedback/items/{i}/org-preview", json={"decision": "accept"},
               headers=ADMIN).json()
    assert r == {"opens": True, "kind": "accepted_name", "old_names": ["Amanita retired"],
                 "new_name": "Amanita accepted", "warnings": [],
                 "records": {"Amanita retired": 12, "Amanita accepted": 5}}
    r = c.post(f"/api/feedback/items/{i}/org-preview",
               json={"decision": "accept", "value": "Amanita acepted"}, headers=ADMIN).json()
    assert r["warnings"][0].startswith("No record Vision holds is called Amanita acepted")
    assert c.post(f"/api/feedback/items/{i}/org-preview", json={"decision": "keep"},
                  headers=ADMIN).json() == {"opens": False}
    n = iid(store, "name_inactive_on_inat")
    assert c.post(f"/api/feedback/items/{n}/org-preview", json={"decision": "accept"},
                  headers=ADMIN).status_code == 400
    assert not org.calls                            # a preview never reaches .org


def test_who_may_preview_decide_send_and_reopen(site):
    c, store, org, _, _ = site
    i = iid(store, "name_retired", "Amanita retired")               # project p1
    p2 = iid(store, "name_retired", "Boletus other")                # project p2
    routes = [("post", f"/api/feedback/items/{i}/org-preview", {"decision": "accept"}),
              ("post", f"/api/feedback/items/{i}/org-request", None),
              ("post", f"/api/feedback/items/{i}/decision", {"decision": "accept"}),
              ("post", f"/api/feedback/items/{i}/reopen", None)]
    for method, url, body in routes:
        assert getattr(c, method)(url, json=body).status_code == 401, url
        assert getattr(c, method)(url, json=body, headers=as_("nobody-1")).status_code == 403
        assert getattr(c, method)(url, json=body, headers=as_("member-2")).status_code == 404
    assert not org.calls
    r = decide(c, i, "accept", who=as_("leader-1"))                 # a leader of p1
    assert r.status_code == 200 and r.json()["org_request"]["status"] == "searching"
    assert org.calls[-1][1]["decided_by"] == {"org_user_id": "leader-1",
                                              "name": "Name of leader-1"}
    r = decide(c, p2, "accept", who=as_("member-2"))                # a contributor of p2
    assert r.status_code == 200 and r.json()["org_request"]["request_id"] == 8


def test_manual_mark_applied_stays_for_org_changes_that_are_not_name_updates(site):
    c, store, org, _, _ = site
    g = iid(store, "group_label")
    decide(c, g, "rename", "Laccaria sp. 'PNW01'")
    did = store.get(g)["decisions"][-1]["id"]
    assert c.post(f"/api/feedback/decisions/{did}/applied", headers=as_("leader-1")).status_code == 403
    assert c.post(f"/api/feedback/decisions/{did}/applied", headers=ADMIN).status_code == 200
    assert store.get(g)["decisions"][-1]["applied_by"] == "admin-1"


def test_mark_applied_by_applies_once():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        store = fb.Store(Path(d) / "f.sqlite")
        seed(store)
        i = iid(store, "name_retired", "Amanita retired")
        store.decide(i, "accept", {"sub": "a"})
        did = store.get(i)["decisions"][-1]["id"]
        assert store.mark_applied_by(did, "mycomap.org Name Update 7") is True
        assert store.mark_applied_by(did, "mycomap.org Name Update 9") is False
        assert store.get(i)["decisions"][-1]["applied_by"] == "mycomap.org Name Update 7"
        store.conn.close()


# --- the client's calls ------------------------------------------------------------------------

class Session:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.seen = status, body if body is not None else {}, []

    def _r(self):
        status, body = self.status, self.body

        class R:
            status_code = status

            def json(self):
                return body

            def raise_for_status(self):
                if status >= 400:
                    raise requests.HTTPError(str(status))
        return R()

    def get(self, url, params=None, timeout=None, headers=None):
        self.seen.append(("GET", url, params, headers))
        return self._r()

    def request(self, method, url, json=None, timeout=None, headers=None):
        self.seen.append((method, url, json, headers))
        return self._r()


def test_the_client_sends_the_vision_key_and_returns_refusals_as_answers():
    s = Session(422, {"code": "refused", "message": "no"})
    c = feedback_api.OrgClient(base_url=ORG + "/", key="k1", session=s)
    assert c.name_update_post({"decision_id": "1"}) == (422, {"code": "refused", "message": "no"})
    assert s.seen[-1] == ("POST", f"{ORG}/api/vision/name-updates", {"decision_id": "1"},
                          {"Authorization": "Bearer k1"})
    s.status, s.body = 409, {"code": "started"}
    assert c.name_update_delete(7) == (409, {"code": "started"})
    assert s.seen[-1][:2] == ("DELETE", f"{ORG}/api/vision/name-updates/7")
    s.status, s.body = 200, {"id": 7, "status": "waiting"}
    assert c.name_update_get(7) == {"id": 7, "status": "waiting"}
    assert c.name_update_find("12") == {"id": 7, "status": "waiting"}
    assert s.seen[-1][1:3] == (f"{ORG}/api/vision/name-updates", {"decision": "12"})
    assert c.name_changes(5, 100) and s.seen[-1][1:3] == (f"{ORG}/api/vision/name-changes",
                                                         {"after": 5, "limit": 100})
    assert c.name_change_records(9, 3) and s.seen[-1][1:3] == (
        f"{ORG}/api/vision/name-changes/9/records", {"after": 3})
    s.status = 404
    assert c.name_update_get(8) is None
    nokey = feedback_api.OrgClient(base_url=ORG, key="", session=s)
    n = len(s.seen)
    assert nokey.name_update_post({}) == (None, {}) and nokey.name_changes(0) is None
    assert len(s.seen) == n                          # without a key nothing is sent


def test_the_item_link_back_needs_the_sites_origin(monkeypatch):
    monkeypatch.delenv("MV_PUBLIC_ORIGIN", raising=False)
    assert feedback_api.item_url(3) is None
    monkeypatch.setenv("MV_PUBLIC_ORIGIN", "https://vision.mycomap.org/")
    assert feedback_api.item_url(3) == "https://vision.mycomap.org/feedback?item=3"


def test_the_table_is_additive_on_a_store_made_before_it(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript(fb.SCHEMA.split("-- A whole-name decision")[0])
    old.execute("insert into items (item_type, subject, granularity, rule, seed, created_at) "
                "values ('name_retired', 'X y', 'name', 'R4', 's', 0)")
    old.commit()
    old.close()
    store = fb.Store(path)
    assert store.items("name_retired")["total"] == 1
    assert {r[1] for r in store.conn.execute("pragma table_info(org_requests)")} >= {
        "decision_id", "request_id", "status", "url", "counts", "note", "checked_at"}
    json.dumps(store.summary())


def test_the_catalogue_says_which_options_open_a_name_update(site):
    c, _, _, _, _ = site
    cat = {t["type"]: {o["code"]: o["name_update"] for o in t["options"]}
           for t in c.get("/api/feedback/catalogue", headers=ADMIN).json()}
    assert cat["name_writing_split"] == {"use_writing": "respelling", "different": None}
    assert cat["name_retired"] == {"accept": "accepted_name", "keep": None, "rename": "rename"}
    assert cat["crosswalk"]["same"] == "same_taxon" and cat["crosswalk"]["nested"] is None
    assert cat["vision_names_same_taxon"]["same"] == "same_taxon"
    assert set(cat["group_label"].values()) == {None}
    assert set(cat["label_vs_own_dna"].values()) == {None}
    labels = {(t["type"], o["code"]): o["label"]
              for t in c.get("/api/feedback/catalogue", headers=ADMIN).json() for o in t["options"]}
    assert labels[("name_retired", "rename")] == "Rename every record of this name to…"
    assert labels[("group_label", "rename")] == "Rename the record(s) to…"


def test_the_change_list_says_which_org_changes_are_already_name_updates(site):
    c, store, org, _, _ = site
    decide(c, iid(store, "name_retired", "Amanita retired"), "accept")
    decide(c, iid(store, "group_label"), "rename", "Laccaria sp. 'PNW01'")
    lines = c.get("/api/feedback/changes.tsv", headers=ADMIN).text.splitlines()
    head = lines[0].split("\t")
    assert head[-1] == "name_update" and len(lines) == 3
    rows = {r.split("\t")[1]: r.split("\t")[-1] for r in lines[1:]}
    assert rows == {"name_retired": "request 7: searching", "group_label": ""}


def test_names_vision_holds_are_counted_with_their_writings(conn):
    from mycomap_vision.records import build_records, save_records
    rows = [{"source": "iNaturalist", "observation_id": str(100 + i), "scientific_name": n,
             "genus": n.split()[0], "family": "F", "continent": "North America",
             "validation_status_1": "yes", "validation_date_1": "1/1/2026"}
            for i, n in enumerate(["Mycena sp. 'IN10'", 'Mycena "sp-IN10"', "Amanita x"])]
    save_records(conn, build_records(rows, "t"))
    info = fo.manifest_name_info(conn, lambda g: g in ("Mycena", "Amanita"))
    out = info(["Mycena sp. 'IN10'", "Amanita acepted", "Zzyzx y"])
    assert out["Mycena sp. 'IN10'"] == {"records": 2, "genus_known": True}
    assert out["Amanita acepted"] == {"records": 0, "genus_known": True}
    assert out["Zzyzx y"] == {"records": 0, "genus_known": False}
    assert fo.manifest_name_info(conn)(["Amanita x"])["Amanita x"] == {"records": 1,
                                                                        "genus_known": None}
