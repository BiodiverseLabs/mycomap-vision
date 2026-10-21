"""Human Feedback decisions about a whole name open a Name Update on mycomap.org (Steve,
2026-10-18: "one name decision everywhere").

Name Update changes every record of an old name, so only decisions about a WHOLE name go
there; a decision about particular records ("Rename the record(s) to…", label right, bad
sequence, crosswalk "nested") stays with Validate Records as before:

    name_writing_split      use_writing    respelling     every other writing -> the value
    species_rank_split      use_name       same_taxon     every other label -> the value
    name_retired            accept         accepted_name  the retired name -> the accepted one
    name_inactive_on_inat   accept         accepted_name  the inactive name -> iNat's current
    name_retired / ..._inat rename         rename         the name -> the value typed
    crosswalk               same           same_taxon     the other name -> the value chosen
    vision_names_same_taxon same           same_taxon     the less used name -> the more used

The decision is saved first and sent after (POST /api/vision/name-updates, Vision key). If
mycomap.org cannot be reached, the item says "not sent yet" and offers Send again. Sending
changes nothing outside .org: .org searches every record of each old name into an ordinary
Name Update, and a person with Name Update rights there presses Fix (or declines, with a
reason shown here). Vision reads the request when the item opens and every 10 minutes while
it is unfinished; when .org says done, the decision is applied ("mycomap.org Name Update
<id>"), the only automatic way to applied. Reopening an item whose request nobody has run
yet withdraws it (DELETE); one a person has started cannot be reopened from here.

What a person can type is checked before saving: the new name is shown, editable, and a name
Vision does not know (no live label, a genus iNat does not have) is flagged, because proposed
names come from source verbatim, typos included (the miss review, 2026-10-18).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

import requests

from . import feedback as fb

KINDS: dict[tuple[str, str], str] = {
    ("name_writing_split", "use_writing"): "respelling",
    ("species_rank_split", "use_name"): "same_taxon",
    ("name_retired", "accept"): "accepted_name",
    ("name_retired", "rename"): "rename",
    ("name_inactive_on_inat", "accept"): "accepted_name",
    ("name_inactive_on_inat", "rename"): "rename",
    ("crosswalk", "same"): "same_taxon",
    ("vision_names_same_taxon", "same"): "same_taxon",
}

MAX_OLD_NAMES = 10                 # .org refuses more in one update
# .org's request statuses still moving: read again every POLL_SECONDS.
UNFINISHED = ("searching", "waiting", "running", "needs_attention")
POLL_SECONDS = 600
# Statuses with nothing open on .org (Vision's own: not_sent, not_taken).
SEND_AGAIN = ("not_sent", "not_taken", "refused")
# Withdrawn by DELETE when the item is reopened: nobody has run them.
WITHDRAWABLE = ("searching", "waiting", "refused")
APPLIED_BY = "mycomap.org Name Update {id}"

_COUNT = re.compile(r"\s*\(\s*[\d,]+\s*\)\s*$")
_INFRA = re.compile(r"\s(var\.|subsp\.|ssp\.|f\.|forma|subvar\.)\s", re.IGNORECASE)


def writings(current: str | None) -> list[str]:
    """The writings of a name_writing_split item: "Name A (17) | Name B (1)" -> both names."""
    out = []
    for part in (current or "").split(" | "):
        name = _COUNT.sub("", part).strip()
        if name and name not in out:
            out.append(name)
    return out


def is_infraspecific(name: str) -> bool:
    return bool(_INFRA.search(f" {name.strip()} "))


def is_provisional(name: str) -> bool:
    from .name_equiv import parts
    return parts(name).provisional


# name -> {"records": records Vision holds under it, "genus_known": iNat has its genus (None:
# not asked)}
NameInfo = Callable[[list[str]], dict[str, dict]]


@dataclass(frozen=True)
class Plan:
    """What a decision asks .org for, before it is saved (shown to the reviewer)."""
    kind: str
    old_names: list[str]
    new_name: str
    warnings: list[str]

    def body(self) -> dict:
        return {"kind": self.kind, "old_names": self.old_names, "new_name": self.new_name}


def plan(item: dict, decision: str, value: str | None = None,
         name_info: NameInfo | None = None) -> Plan | None:
    """The Name Update a decision opens, or None when it opens none (a record-level decision,
    a note). Raises FeedbackError for one that cannot make an update as it stands."""
    kind = KINDS.get((item["item_type"], decision))
    if kind is None:
        return None
    value = (value or "").strip()
    current, proposed = (item.get("current") or "").strip(), (item.get("proposed") or "").strip()
    t = item["item_type"]
    if t in ("name_writing_split", "species_rank_split"):
        new = value or proposed
        if t == "species_rank_split" and new not in writings(current):
            raise fb.FeedbackError(f"the name to keep is one of {', '.join(writings(current))}")
        old = [w for w in writings(current) if w != new]
    elif t in ("name_retired", "name_inactive_on_inat"):
        new = value if (decision == "rename" or value) else proposed
        if not new:
            raise fb.FeedbackError(f"{current} has no accepted name to use (iNat's successor is "
                                   "a split or missing): use Rename and type the name")
        old = [current]
    elif t == "crosswalk":
        pair = [n for n in (current, proposed) if n]
        if value not in pair:
            raise fb.FeedbackError(f"Same lineage: the name to keep is one of {' or '.join(pair)}")
        new, old = value, [n for n in pair if n != value]
    else:                                              # vision_names_same_taxon
        new, old = vision_direction(current, value or proposed, name_info)
    old = [o for o in old if o]
    if not new or not old:
        raise fb.FeedbackError("this decision names no old name and new name to change")
    if new in old:
        raise fb.FeedbackError(f"{new} is already the name: nothing to change")
    if len(old) > MAX_OLD_NAMES:
        raise fb.FeedbackError(f"at most {MAX_OLD_NAMES} old names go into one Name Update")
    return Plan(kind, old, new, warnings(kind, old, new, name_info))


def vision_direction(label: str, vision: str, name_info: NameInfo | None) -> tuple[str, list[str]]:
    """vision_names_same_taxon: the less used name goes to the more used one (a tie, or no
    counts: the label goes to Vision's name, as the option says)."""
    info = name_info([label, vision]) if name_info and label and vision else {}
    a, b = (info.get(label) or {}).get("records", 0), (info.get(vision) or {}).get("records", 0)
    return (label, [vision]) if a > b else (vision, [label])


def warnings(kind: str, old: list[str], new: str, name_info: NameInfo | None) -> list[str]:
    out = []
    info = name_info(old + [new]) if name_info else {}
    target = info.get(new) or {}
    if info and not target.get("records"):
        # The cache knows genera only, so a name no record carries is always worth a look.
        genus = (new.split() or [""])[0]
        out.append(f"No record Vision holds is called {new}"
                   + (f", and iNat has no genus {genus}" if target.get("genus_known") is False
                      else "") + ": check the spelling before every record gets it")
    if kind == "same_taxon":
        for o in old:
            if is_provisional(o) != is_provisional(new):
                out.append(f"{o} and {new} pair a provisional code with a formal name: that is "
                           "usually a crosswalk question, decided at source")
            if is_infraspecific(o) != is_infraspecific(new):
                out.append(f"{o} and {new} are at different ranks (one is infraspecific)")
    return out


# ---------------------------------------------------------------------------------------------
# The requests

def request_view(row: dict | None) -> dict | None:
    """The item's "On mycomap.org" block: status, link, the four steps and the row's chip."""
    if row is None:
        return None
    counts = json.loads(row["counts"] or "{}") if isinstance(row["counts"], str) else row["counts"]
    totals = counts.get("totals") or {}
    status = row["status"]
    found = totals.get("total")
    order = ["sent", "searched", "waiting", "done"]
    reached = {"not_sent": -1, "not_taken": -1, "searching": 0, "refused": 0, "waiting": 1,
               "withdrawn": 1, "declined": 2, "running": 2, "needs_attention": 2,
               "done": 3}.get(status, -1)

    def state(i):
        return "done" if i <= reached else ("current" if i == reached + 1 else "todo")
    steps = [{"step": s, "state": state(i)} for i, s in enumerate(order)]
    steps[0]["label"] = "Not sent yet" if status == "not_sent" else (
        "Not taken" if status == "not_taken" else "Sent")
    steps[1]["label"] = ("Search refused" if status == "refused" else
                         f"Searched: {found:,} record{'' if found == 1 else 's'}"
                         if found is not None and reached >= 1 else "Searching")
    steps[2]["label"] = {"declined": "Declined", "withdrawn": "Withdrawn"}.get(
        status, "Waiting for a person")
    steps[3]["label"] = {"running": "Running", "needs_attention": "Needs attention"}.get(
        status, "Done")
    if status in ("declined", "withdrawn"):
        steps[2]["state"], steps[3]["state"] = "stopped", "todo"
    if status in ("refused", "not_sent", "not_taken"):
        steps[reached + 1]["state"] = "stopped"
    fixed = int(totals.get("fixed") or 0)
    chip = {"not_sent": "not sent to mycomap.org yet", "not_taken": "not taken by mycomap.org",
            "searching": "on mycomap.org: searching", "refused": "refused by mycomap.org",
            "waiting": "on mycomap.org: waiting for a person",
            "running": "on mycomap.org: running", "needs_attention":
            "on mycomap.org: needs attention", "declined": "declined on mycomap.org",
            "withdrawn": "withdrawn from mycomap.org",
            "done": f"applied on mycomap.org · {fixed:,} renamed"}.get(status, status)
    tone = ("good" if status == "done" else "bad" if status in (
        "not_sent", "not_taken", "refused", "declined", "needs_attention") else
        "quiet" if status == "withdrawn" else "busy")
    return {"status": status, "request_id": row["request_id"], "url": row["url"],
            "note": row["note"], "totals": totals, "batches": counts.get("batches") or [],
            "declined_by": counts.get("declined_by"),
            "declined_reason": counts.get("declined_reason"),
            "old_names": json.loads(row["payload"] or "{}").get("old_names") or [],
            "new_name": json.loads(row["payload"] or "{}").get("new_name"),
            "sent_at": row["sent_at"], "checked_at": row["checked_at"], "steps": steps,
            "chip": {"text": chip, "tone": tone}, "can_send": status in SEND_AGAIN,
            "can_check": row["request_id"] is not None and status in UNFINISHED}


class OrgRequests:
    """The org_requests table (feedback.SCHEMA) and the calls that keep it in step with .org.
    `org` is feedback_api.OrgClient (or a fake with name_update_post/get/find/delete);
    `item_url(item_id)` the item's address for .org's link back; `on_done()` runs after a
    request is found done (Vision reads the name-change log at once)."""

    def __init__(self, store: fb.Store, org, item_url: Callable[[int], str | None] = lambda i: None,
                 on_done: Callable[[], object] | None = None, note=print):
        self.store, self.org, self.item_url, self.on_done, self.note = (
            store, org, item_url, on_done, note)

    # --- reading -------------------------------------------------------------------------------
    def row(self, decision_id: int) -> dict | None:
        r = self.store.conn.execute("select * from org_requests where decision_id = ?",
                                    (decision_id,)).fetchone()
        return dict(r) if r else None

    def latest_decision(self, item_id: int) -> dict | None:
        r = self.store.conn.execute("select * from decisions where item_id = ? order by id desc "
                                    "limit 1", (item_id,)).fetchone()
        return dict(r) if r else None

    def for_item(self, item_id: int) -> dict | None:
        """The request of the item's latest decision (None: it opened none)."""
        d = self.latest_decision(item_id)
        return self.row(d["id"]) if d else None

    def chips(self, item_ids: list[int]) -> dict[int, dict]:
        """item id -> the row chip of its latest decision's request, for a page of items."""
        if not item_ids:
            return {}
        marks = ",".join("?" * len(item_ids))
        rows = self.store.conn.execute(
            f"select d.item_id, o.* from org_requests o join decisions d on d.id = o.decision_id "
            f"where d.item_id in ({marks}) and {fb.LATEST_DECISION}", item_ids).fetchall()
        return {r["item_id"]: request_view(dict(r))["chip"] for r in rows}

    def unfinished(self) -> list[int]:
        return [r[0] for r in self.store.conn.execute(
            f"select decision_id from org_requests where request_id is not null and status in "
            f"({','.join('?' * len(UNFINISHED))}) order by decision_id", UNFINISHED)]

    # --- writing -------------------------------------------------------------------------------
    def _save(self, decision_id: int, **fields) -> dict:
        cols = ["status", "request_id", "url", "counts", "note", "payload", "sent_at",
                "checked_at"]
        old = self.row(decision_id) or {"counts": "{}", "payload": "{}"}
        new = {c: fields.get(c, old.get(c)) for c in cols}
        for c in ("counts", "payload"):
            if not isinstance(new[c], str):
                new[c] = json.dumps(new[c] or {})
        with self.store._lock, self.store.conn:
            self.store.conn.execute(
                "insert into org_requests (decision_id, status, request_id, url, counts, note, "
                "payload, sent_at, checked_at) values (?,?,?,?,?,?,?,?,?) on conflict "
                "(decision_id) do update set status = excluded.status, request_id = "
                "excluded.request_id, url = excluded.url, counts = excluded.counts, note = "
                "excluded.note, payload = excluded.payload, sent_at = excluded.sent_at, "
                "checked_at = excluded.checked_at",
                (decision_id, *(new[c] for c in cols)))
        return self.row(decision_id)

    def _take(self, decision_id: int, answer: dict, extra: dict | None = None) -> dict:
        """Store .org's view of a request; apply the decision when it is done."""
        counts = {k: answer.get(k) for k in ("totals", "batches", "declined_by",
                                             "declined_reason", "old_name_left",
                                             "searched_at", "finished_at")}
        status = str(answer.get("status") or "searching")
        row = self._save(decision_id, status=status, request_id=answer.get("id"),
                         url=answer.get("url"), counts=counts, note=answer.get("note"),
                         checked_at=time.time(), **(extra or {}))
        if status == "done" and self.store.mark_applied_by(
                decision_id, APPLIED_BY.format(id=answer.get("id"))):
            self.note(f"[feedback] decision {decision_id} applied by mycomap.org Name Update "
                      f"{answer.get('id')}")
            if self.on_done:
                try:
                    self.on_done()
                except Exception as e:  # noqa: BLE001 - the 10-minute check reads it again
                    self.note(f"[feedback] reading the name-change log failed: "
                              f"{type(e).__name__}: {e}")
        return row

    def send(self, decision_id: int, person: dict | None = None,
             name_info: NameInfo | None = None) -> dict | None:
        """Send a whole-name decision to .org (again). None when the decision opens no Name
        Update. Never raises for .org: an unreachable .org leaves "not_sent", a refusal
        "not_taken" with .org's message."""
        d = self.store.conn.execute("select * from decisions where id = ?",
                                    (decision_id,)).fetchone()
        if d is None:
            raise KeyError(decision_id)
        item = self.store.get(d["item_id"])
        old = self.row(decision_id)
        if old and old["status"] not in SEND_AGAIN:
            return old                              # already on .org: nothing to send
        if old and old["payload"] not in (None, "", "{}"):
            payload = json.loads(old["payload"])    # the same request, never a new reading
        else:
            p = plan(item, d["decision"], d["value"], name_info)
            if p is None:
                return None
            by_user = str(d["by_user"])
            payload = {"decision_id": str(decision_id), **p.body(),
                       "decided_by": {"org_user_id": by_user, "name": d["by_name"] or by_user}}
            if d["note"]:
                payload["reason"] = d["note"]
            url = self.item_url(item["id"])
            if url:
                payload["item_url"] = url
        sent_at = time.time()
        try:
            code, answer = self.org.name_update_post(payload)
        except requests.RequestException as e:
            return self._save(decision_id, status="not_sent", payload=payload, sent_at=sent_at,
                              note=f"mycomap.org could not be reached ({type(e).__name__})",
                              checked_at=sent_at)
        if code is None:
            return self._save(decision_id, status="not_sent", payload=payload, sent_at=sent_at,
                              note="Vision has no mycomap.org key set", checked_at=sent_at)
        if code in (200, 202):
            return self._take(decision_id, answer, {"payload": payload, "sent_at": sent_at})
        if code >= 500:
            return self._save(decision_id, status="not_sent", payload=payload, sent_at=sent_at,
                              note=f"mycomap.org answered {code}", checked_at=sent_at)
        message = str((answer or {}).get("message") or f"mycomap.org answered {code}")
        return self._save(decision_id, status="not_taken", payload=payload, sent_at=sent_at,
                          note=message, request_id=None, url=None, checked_at=sent_at)

    def check(self, decision_id: int) -> dict | None:
        """Read the request again from .org. Leaves the row as it was if .org is down."""
        row = self.row(decision_id)
        if row is None or row["request_id"] is None:
            return row
        try:
            answer = self.org.name_update_get(row["request_id"])
        except requests.RequestException:
            return row
        if answer is None:                          # gone from .org (a reset test database)
            return self._save(decision_id, status="not_taken", request_id=None, url=None,
                              note="mycomap.org no longer has this request: send it again",
                              checked_at=time.time())
        return self._take(decision_id, answer)

    def check_item(self, item_id: int) -> dict | None:
        d = self.latest_decision(item_id)
        row = self.row(d["id"]) if d else None
        if row and row["status"] in UNFINISHED:
            row = self.check(d["id"])
        return row

    def poll_once(self) -> dict:
        """Every unfinished request read again (the 10-minute check)."""
        out = {"checked": 0, "done": 0}
        for decision_id in self.unfinished():
            row = self.check(decision_id)
            out["checked"] += 1
            out["done"] += int(bool(row) and row["status"] == "done")
        return out

    def withdraw_for_reopen(self, item_id: int) -> dict | None:
        """Before an item reopens: withdraw its request if nobody has run it. Raises
        Unwithdrawable when a person has started it, or .org cannot be reached to stop it."""
        d = self.latest_decision(item_id)
        row = self.row(d["id"]) if d else None
        if row is None or row["status"] in ("withdrawn", "declined", "done", "not_taken"):
            return row
        if row["status"] in ("running", "needs_attention"):
            raise Unwithdrawable("A person has already run this update on mycomap.org, so the "
                                 "item cannot be reopened here. Open it in Name Update.")
        request_id = row["request_id"]
        try:
            if request_id is None:                  # never answered: maybe it arrived anyway
                found = self.org.name_update_find(str(d["id"]))
                if found is None:
                    return self._save(d["id"], status="withdrawn", note="never reached "
                                      "mycomap.org", checked_at=time.time())
                request_id = found["id"]
            code, answer = self.org.name_update_delete(request_id)
        except requests.RequestException as e:
            raise Unwithdrawable("mycomap.org could not be reached to withdraw the Name Update "
                                 f"({type(e).__name__}): try again shortly") from None
        if code is None:
            raise Unwithdrawable("Vision has no mycomap.org key set, so the Name Update cannot "
                                 "be withdrawn")
        if code == 404:
            return self._save(d["id"], status="withdrawn", note="mycomap.org no longer has it",
                              checked_at=time.time())
        if code == 409:
            if row["request_id"] is not None:
                self.check(d["id"])
            raise Unwithdrawable(str((answer or {}).get("message") or "A person has already run "
                                     "this update on mycomap.org, so it cannot be withdrawn."))
        if code >= 400:
            raise Unwithdrawable(f"mycomap.org answered {code} to the withdrawal: try again")
        return self._take(d["id"], answer)


class Unwithdrawable(RuntimeError):
    """The item's Name Update cannot be withdrawn (a person ran it, or .org is unreachable)."""


# ---------------------------------------------------------------------------------------------
# Names Vision knows, from the served manifest (and the taxonomy cache)

def manifest_name_info(conn, genus_known: Callable[[str], bool | None] | None = None) -> NameInfo:
    """name -> {"records": records under the name in the served manifest (its writings
    counted together, names.variant_key), "genus_known": whether iNat has its genus (the
    taxonomy cache; None when not asked)}."""
    from . import names

    def info(wanted: list[str]) -> dict[str, dict]:
        keys = {n: names.variant_key(n) for n in wanted}
        counts: dict[str, int] = {}
        for name, n in conn.execute("select scientific_name, count(*) from records "
                                    "group by scientific_name"):
            k = names.variant_key(name or "")
            if k in keys.values():
                counts[k] = counts.get(k, 0) + n
        out = {}
        for n, k in keys.items():
            records = counts.get(k, 0)
            genus = (n.split() or [""])[0]
            out[n] = {"records": records,
                      "genus_known": genus_known(genus) if genus_known else None}
        return out
    return info
