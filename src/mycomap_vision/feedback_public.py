"""Human Feedback as a read-only viewer sees it (Steve, 2026-10-22).

Anyone signed out, and signed-in mycomap.org people who are not admins, project leaders or
contributors, may browse Human Feedback: its sections, queues, Summary and record views, and
item detail. They change nothing (every write refuses them: feedback_api.access), and see:

- every photo item (the photo review's queue and Photo views), whatever its project, and of
  the record and name items only those that belong to no project (feedback.PUBLIC_ITEMS);
- each decision's outcome and date, never who decided, their note, or whether they had seen
  Vision's answer (PERSONAL);
- blind items still blind: reviewer_view masks them as for reviewers, and revealing is a write;
- only photos anyone may see under the photo permissions now (Creative Commons, or the owner
  said yes): every other photo is a placeholder without its photographer's name, and
  /api/feedback/photo/ serves them no bytes (photo_review.ReviewPhotos.for_viewer with
  reviewer=False);
- of mycomap.org's live record, only the name, source, projects and sequences with their
  public BLAST link: no admin links, and no error text.

Reviewers get exactly what they had before.
"""

from __future__ import annotations

# Who decided, what they wrote, whether they had looked: never to a read-only viewer.
PERSONAL = frozenset({"by", "by_user", "by_name", "note", "applied_by", "seen_answer",
                      "declined_by", "declined_reason", "revealed", "error"})
LIVE_KEYS = ("record_key", "observation", "name", "source", "projects", "unavailable")
SEQUENCE_KEYS = ("id", "category_id", "trash", "name", "length", "approved", "genbank", "url",
                 "blast_link")
ORG_REQUEST_KEYS = ("status", "steps", "chip", "tone", "totals")


def read_only(value):
    """`value` without any PERSONAL key, at any depth."""
    if isinstance(value, dict):
        return {k: read_only(v) for k, v in value.items() if k not in PERSONAL}
    if isinstance(value, list):
        return [read_only(v) for v in value]
    return value


def live_record(rec: dict) -> dict:
    """mycomap.org's record as a read-only viewer may see it (its blast_link already the
    public one: feedback_api.blast_link with staff False)."""
    out = {k: rec[k] for k in LIVE_KEYS if k in rec}
    if isinstance(rec.get("sequences"), list):
        out["sequences"] = [{k: s[k] for k in SEQUENCE_KEYS if k in s}
                            for s in rec["sequences"] if isinstance(s, dict)]
    return out


def org_request(view: dict | None) -> dict | None:
    """The "On mycomap.org" block without its link, notes or who declined."""
    return None if view is None else {k: view[k] for k in ORG_REQUEST_KEYS if k in view}


def photo(tile: dict) -> dict:
    """A photo or tile: a photo not shown publicly keeps no photographer's name."""
    if tile.get("shown") == "public":
        return tile
    return {k: v for k, v in tile.items() if k != "owner_login"}
