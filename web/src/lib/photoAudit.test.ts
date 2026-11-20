import { test } from "node:test";
import assert from "node:assert/strict";
import type { FlaggedTile, PhotoAudit, PhotoAuditInfo } from "./api";
import { photoFlaggedQuery } from "./api";
import { AUDIT_SORT, auditAvailable, auditLabel, auditOutcomeInUse, auditTitle, NOT_IN_TAB_FILTER,
         outcomeOptions, sortInUse, sortOptions } from "./photoAudit";
import { OUTCOME_FILTERS, outcomeCount, SORTS } from "./photoReview";
import { overlay } from "./photoSummary";

const info: PhotoAuditInfo = {
  run: { run_id: "r12-audit-20261121T120000Z", loaded_at: 1, rows: 10, items: 5 },
  tabs: [{ code: "uv", high: 0.85, comparator: "R11.1" }, { code: "slip", high: 0.7, comparator: "v2.1-336" }],
};

const uv: PhotoAudit = { tab: "uv", p: 0.0312, p_image: 0.1, comparator: "R11.1", comparator_p: 0.41,
                         disagreement: 0.9688, high: 0.85 };

test("the audit's options are offered on a scored tab's Decided Summary only", () => {
  assert.equal(auditAvailable(info, "summary", "decided", "uv"), true);
  assert.equal(auditAvailable(info, "summary", "decided", "slip"), true);
  assert.equal(auditAvailable(info, "summary", "decided", "micrograph"), false);   // not scored
  assert.equal(auditAvailable(info, "summary", "open", "uv"), false);
  assert.equal(auditAvailable(info, "summary", "decided", "all"), false);
  assert.equal(auditAvailable(info, "full", "decided", "uv"), false);
  assert.equal(auditAvailable({ run: null, tabs: [] }, "summary", "decided", "uv"), false);
  assert.equal(auditAvailable(undefined, "summary", "decided", "uv"), false);
});

test("the sort list gains Strongest disagreement first, and a pick falls back where it is not offered", () => {
  assert.deepEqual(sortOptions(SORTS, false).map((s) => s.label), SORTS.map((s) => s.label));
  const withAudit = sortOptions(SORTS, true);
  assert.equal(withAudit.length, SORTS.length + 1);
  assert.equal(withAudit[withAudit.length - 1].label, "Strongest disagreement first");
  assert.equal(withAudit[withAudit.length - 1].code, "disagreement");
  assert.equal(SORTS[0].label, "Strongest flag first");                // the usual list unchanged
  assert.equal(sortInUse("disagreement", true), "disagreement");
  assert.equal(sortInUse("disagreement", false), "p");
  assert.equal(sortInUse("p_asc", false), "p_asc");
  assert.equal(AUDIT_SORT.code, "disagreement");
});

test("the Outcome list gains High score, not in this tab, and it is asked for only where offered", () => {
  assert.deepEqual(outcomeOptions(OUTCOME_FILTERS, false), OUTCOME_FILTERS);
  const withAudit = outcomeOptions(OUTCOME_FILTERS, true);
  assert.deepEqual(withAudit.slice(0, OUTCOME_FILTERS.length), OUTCOME_FILTERS);   // kept, in order
  assert.equal(withAudit[withAudit.length - 1].label, "High score, not in this tab");
  assert.equal(withAudit[withAudit.length - 1].code, "high_not_in_tab");
  assert.match(NOT_IN_TAB_FILTER.title, /may have missed/);
  assert.equal(auditOutcomeInUse("high_not_in_tab", true), "high_not_in_tab");
  assert.equal(auditOutcomeInUse("high_not_in_tab", false), "all");
  assert.equal(auditOutcomeInUse("moved", false), "moved");
  // its count shows like the others'
  assert.equal(outcomeCount({ as_flagged: 3, high_not_in_tab: 1234 }, "high_not_in_tab"), " (1,234)");
});

test("the query sends the audit's sort and outcome as the server takes them", () => {
  const qs = new URLSearchParams(photoFlaggedQuery({ tab: "uv", status: "decided", sort: "disagreement",
                                                     outcome: "high_not_in_tab" }));
  assert.equal(qs.get("sort"), "disagreement");
  assert.equal(qs.get("outcome"), "high_not_in_tab");
  assert.equal(qs.get("tab"), "uv");
  assert.equal(qs.get("status"), "decided");
});

test("a tile's audit line reads audit <tab> p, with the served tagger's p", () => {
  assert.equal(auditLabel(uv), "audit UV 0.03 · served 0.41");
  assert.equal(auditLabel({ ...uv, tab: "spore_print", p: 0.9, comparator_p: null }), "audit spore print 0.90");
  assert.equal(auditLabel({ ...uv, tab: "gate", p: 0.5 }), "audit not a fungus 0.50 · served 0.41");
  const t = auditTitle(uv);
  assert.match(t, /never trained on this photo/);
  assert.match(t, /photo alone 0\.10/);
  assert.match(t, /R11\.1\) 0\.41/);
  assert.match(t, /disagreement with the decision 0\.97/);
  assert.doesNotMatch(auditTitle({ ...uv, p_image: null }), /photo alone/);
});

test("the photo viewer lists the audit line after the decision", () => {
  const tile = {
    label: "Mycena", link: "https://www.inaturalist.org/observations/10", record_key: "10", photo_key: "inat:1",
    position: 1, record_photos: 3, flags: [], owner_login: null, license_code: "cc-by", license_class: "cc",
    status: "decided", decision: { decision: "exclude_photo", reason: "uv" }, audit: uv,
  } as Partial<FlaggedTile> as FlaggedTile;
  const lines = overlay(tile, [{ code: "uv", label: "UV reaction" }]).lines;
  assert.equal(lines[lines.length - 1], "audit UV 0.03 · served 0.41");
  assert.equal(lines[lines.length - 2], "Decided: excluded (UV reaction)");
  const none = overlay({ ...tile, audit: undefined }).lines;
  assert.ok(!none.some((l) => l.startsWith("audit ")));
});
