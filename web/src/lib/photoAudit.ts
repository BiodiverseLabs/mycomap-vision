// The photo review's rescoring audit (Steve, 2026-11-20; the server's photo_audit.py): every
// decided photo scored by the photo-type tagger with a model that never trained on it. On a tab's
// Decided Summary it adds a sort, "Strongest disagreement first", and an Outcome option, "High
// score, not in this tab"; each tile shows its score ("audit UV 0.03 · served 0.41").
//
// Disagreement for tab T: 1 - p_T for a photo left out for T's reason, p_T for one kept or left
// out for another reason. Both options are offered only where they work: Summary, Decided, a
// single tab the loaded run scored. Elsewhere a pick falls back to the usual sort and to "All
// outcomes", and comes back when the reviewer returns there.

import type { FeedbackStatus, PhotoAudit, PhotoAuditInfo, PhotoOutcomeFilter, PhotoReviewSort,
              PhotoReviewTab } from "./api";

export const DISAGREEMENT_SORT: PhotoReviewSort = "disagreement";
export const NOT_IN_TAB: PhotoOutcomeFilter = "high_not_in_tab";

export const AUDIT_SORT: { code: PhotoReviewSort; label: string } = {
  code: DISAGREEMENT_SORT, label: "Strongest disagreement first",
};

export const NOT_IN_TAB_FILTER: { code: PhotoOutcomeFilter; label: string; title: string } = {
  code: NOT_IN_TAB, label: "High score, not in this tab",
  title: "Decided photos never flagged for this tab and not left out for its reason that the audit scores high for it: photos this tab may have missed",
};

/** Are the audit's sort and filter offered here? Summary, Decided, one tab the run scored. */
export function auditAvailable(info: PhotoAuditInfo | null | undefined, view: "summary" | "full",
                               status: FeedbackStatus, tab: PhotoReviewTab): boolean {
  return view === "summary" && status === "decided" && tab !== "all"
    && !!info?.run && info.tabs.some((t) => t.code === tab);
}

/** The Sort choices: the usual ones, and the disagreement sort where the audit is available. */
export function sortOptions<T extends { code: PhotoReviewSort }>(base: T[], available: boolean):
    (T | typeof AUDIT_SORT)[] {
  return available ? [...base, AUDIT_SORT] : base;
}

/** The sort the lists are asked for: the pick, except the disagreement sort where it is not
 *  available (then the usual "Strongest flag first"). */
export const sortInUse = (sort: PhotoReviewSort, available: boolean): PhotoReviewSort =>
  sort === DISAGREEMENT_SORT && !available ? "p" : sort;

/** The Outcome choices: the usual ones, and "High score, not in this tab" where available. */
export function outcomeOptions<T extends { code: PhotoOutcomeFilter }>(base: T[], available: boolean):
    (T | typeof NOT_IN_TAB_FILTER)[] {
  return available ? [...base, NOT_IN_TAB_FILTER] : base;
}

/** The outcome the lists are asked for: "High score, not in this tab" only where available. */
export const auditOutcomeInUse = (outcome: PhotoOutcomeFilter, available: boolean): PhotoOutcomeFilter =>
  outcome === NOT_IN_TAB && !available ? "all" : outcome;

/** A tab's name, short, for the score line. */
export const AUDIT_TAB_NAMES: Record<string, string> = {
  micrograph: "micrograph", slip: "slip", habitat: "habitat", people: "people", gate: "not a fungus",
  uv: "UV", tree: "tree/BLAST", drawing: "drawing", chemical: "chemical", spore_print: "spore print",
  dried: "dried", specimen: "specimen", other: "other",
};

const two = (p: number) => p.toFixed(2);

/** The tile's line: "audit UV 0.03 · served 0.41" (the served tagger's p when there is one). */
export function auditLabel(a: PhotoAudit): string {
  const served = a.comparator_p == null ? "" : ` · served ${two(a.comparator_p)}`;
  return `audit ${AUDIT_TAB_NAMES[a.tab] ?? a.tab} ${two(a.p)}${served}`;
}

/** The line's hover: what the numbers are. */
export function auditTitle(a: PhotoAudit): string {
  const name = AUDIT_TAB_NAMES[a.tab] ?? a.tab;
  const parts = [`Audit score for ${name}: ${two(a.p)}, from a model that never trained on this photo`];
  if (a.p_image != null) parts.push(`the photo alone ${two(a.p_image)} (the score also weighs who took it)`);
  if (a.comparator_p != null) parts.push(`served tagger${a.comparator ? ` (${a.comparator})` : ""} ${two(a.comparator_p)}`);
  parts.push(`disagreement with the decision ${two(a.disagreement)}`);
  return parts.join("; ") + ".";
}
