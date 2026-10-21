// Human Feedback (/feedback): the review queue of places where Vision and the reference data
// disagree, for mycomap.org admins, project leaders and contributors. The page's rules live
// here as plain functions so they can be tested (feedback.test.ts).
//
// Fairness: model-flagged queues are mixed with a sample of finds Vision answered right, and
// those items must look exactly like the others until they are decided. Nothing here shows
// which items they are.

import type { BlastLink, FeedbackApplies, FeedbackDecisionBody, FeedbackItem, FeedbackOption,
              FeedbackSummary, LiveSequence, Me, NameContextEntry, NameRole, OrgChip, OrgRequest, OrgStep,
              RefreshedRecord, Revealed, RevealedAnswer } from "./api";
import { ApiError } from "./api";

export const FEEDBACK_PATH = "/feedback";
export const FEEDBACK_NAV = { href: FEEDBACK_PATH, label: "Human Feedback" };
export const PAGE_SIZE = 25;

/** The menu shows Human Feedback only to signed-in people (the server decides who may use it). */
export const showsFeedbackNav = (me: Me | null | undefined): boolean => !!me?.user;

/** What /api/feedback/me's refusal means for the page. */
export function feedbackAccess(error: unknown): "signin" | "forbidden" | "error" | null {
  if (!error) return null;
  if (error instanceof ApiError && error.status === 401) return "signin";
  if (error instanceof ApiError && error.status === 403) return "forbidden";
  return "error";
}

/** Where each decision takes effect, in the reviewer's words. */
export const APPLIES_WORDS: Record<FeedbackApplies, string> = {
  org: "on mycomap.org",
  vision: "in Vision",
  note: "note only",
};

export const appliesWords = (a: string): string =>
  (APPLIES_WORDS as Record<string, string>)[a] ?? a;

/** A sequence's BLAST link, exactly as the server chose it for this viewer: the admin results
 *  form for staff, the public job page for everyone else, nothing when there is none. The
 *  admin address is never built here. */
export function blastAnchor(seq: Pick<LiveSequence, "blast_link"> | { blast_link?: BlastLink | null }):
    { href: string; label: string; target: "_blank"; rel: "noopener" } | null {
  const link = seq.blast_link;
  if (!link || !link.href) return null;
  return {
    href: link.href,
    label: link.kind === "admin" ? "BLAST results (admin)" : "BLAST results",
    target: "_blank",
    rel: "noopener",
  };
}

/** "123" → "iNat 123"; "mo:45" → "MO 45". */
export function recordLabel(key: string): string {
  return key.startsWith("mo:") ? `MO ${key.slice(3)}` : `iNat ${key}`;
}

/** The badges on an item's row. Only what every item of its kind shares. */
export function itemBadges(item: Pick<FeedbackItem, "model_flagged">): string[] {
  return item.model_flagged ? ["model-flagged"] : [];
}

/** The rule that picked the item, as reviewers see it. While a model-flagged item is open the
 *  rule is held back for all of them alike (so none stands out); once decided it is shown,
 *  worded for reviewers. */
export function shownRule(item: Pick<FeedbackItem, "rule" | "status" | "model_flagged">): string | null {
  if (item.status === "open" && item.model_flagged) return null;
  const r = (item.rule ?? "")
    .replace(/\bblind\s+sample\b/gi, "check sample")
    .replace(/\bblind\b\s*/gi, "")
    .replace(/\s*\(\s*seed[^)]*\)/gi, "")
    .trim();
  return r || null;
}

const HIDDEN_KEY = /blind|seed|hidden/i;

const humanKey = (k: string) => {
  const s = k.replace(/_/g, " ").replace(/\bv1\b/g, "v1").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
};

const asText = (v: unknown): string => {
  if (v == null) return "";
  if (typeof v === "string") return v.trim();
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
};

/** The item's facts as a small key/value list: the label now and the proposal, then the
 *  evidence, leaving out empty values and anything kept for the analysis. */
export function evidenceRows(item: Pick<FeedbackItem, "current" | "proposed" | "evidence">): [string, string][] {
  const rows: [string, string][] = [];
  if (asText(item.current)) rows.push(["Current", asText(item.current)]);
  if (asText(item.proposed)) rows.push(["Proposed", asText(item.proposed)]);
  for (const [k, v] of Object.entries(item.evidence ?? {})) {
    if (HIDDEN_KEY.test(k)) continue;
    const t = asText(v);
    if (t) rows.push([humanKey(k), t]);
  }
  return rows;
}

/** "a, b ,, c" → ["a", "b", "c"], each once. */
export function parsePhotoKeys(s: string): string[] {
  const out: string[] = [];
  for (const k of s.split(/[,\s]+/)) {
    const t = k.trim();
    if (t && !out.includes(t)) out.push(t);
  }
  return out;
}

/** Whether an option asks for photo keys (the photos to leave out of Vision). */
export const takesPhotoKeys = (option: Pick<FeedbackOption, "code">) =>
  option.code === "exclude_photos";

/** The value box's starting text: the item's proposal, for an option that needs a value or
 *  opens a Name Update (whose new name is shown, editable, before saving: proposed names come
 *  from source verbatim, typos included). */
export const prefillValue = (option: Pick<FeedbackOption, "needs_value" | "name_update">,
                             item: Pick<FeedbackItem, "proposed">): string =>
  option.needs_value || option.name_update ? (item.proposed ?? "").trim() : "";

/** Whether the decision form shows the value box, and what it is called. */
export function valueBox(option: Pick<FeedbackOption, "needs_value" | "name_update">):
    { label: string; required: boolean } | null {
  if (option.name_update) return { label: "New name", required: option.needs_value };
  return option.needs_value ? { label: "Value", required: true } : null;
}

/** The POST body for a decision, or what is missing. */
export function decisionBody(option: FeedbackOption,
                             input: { value?: string; note?: string; photoKeys?: string }):
    { body: FeedbackDecisionBody } | { error: string } {
  const value = (input.value ?? "").trim();
  if (option.needs_value && !value) return { error: "Enter the value this decision needs." };
  const body: FeedbackDecisionBody = { decision: option.code };
  if (option.needs_value || (option.name_update && value)) body.value = value;
  const note = (input.note ?? "").trim();
  if (note) body.note = note;
  if (takesPhotoKeys(option)) {
    const keys = parsePhotoKeys(input.photoKeys ?? "");
    if (!keys.length) return { error: "List the photos to leave out." };
    body.photo_keys = keys;
  }
  return { body };
}

/** An option's label by its code (the code itself for one no longer offered). */
export const decisionLabel = (code: string, options: FeedbackOption[] | undefined) =>
  options?.find((o) => o.code === code)?.label ?? code;

export type Verdict = { tone: "disagrees" | "agrees" | "unknown"; text: string };

/** What a refresh found for one record: still disagrees, now agrees, or Vision couldn't check. */
export function refreshVerdict(r: Pick<RefreshedRecord, "still_disagrees" | "vision">): Verdict {
  if (r.vision && r.vision.available === false) {
    return { tone: "unknown",
             text: `Vision can't re-check this record${r.vision.reason ? ` (${r.vision.reason})` : ""}` };
  }
  return storedVerdict(r.still_disagrees);
}

/** The same for a refresh kept in the item's history (still_disagrees as 1 / 0 / null). */
export function storedVerdict(still: boolean | number | null | undefined): Verdict {
  if (still == null) return { tone: "unknown", text: "Vision can't re-check this record" };
  return still ? { tone: "disagrees", text: "Still disagrees" } : { tone: "agrees", text: "Now agrees" };
}

/** Epoch seconds (or an ISO string) as "2026-10-18 14:05 UTC". */
export function whenText(at: number | string | null | undefined): string {
  if (at == null || at === "") return "";
  const d = typeof at === "number" ? new Date(at * 1000) : new Date(at);
  if (Number.isNaN(d.getTime())) return String(at);
  return `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** "26–50 of 312" for a page of the list. */
export function pageText(offset: number, shown: number, total: number): string {
  if (!total || !shown) return `0 of ${total}`;
  return `${offset + 1}–${offset + shown} of ${total.toLocaleString("en-US")}`;
}

/** Truthy flags from the server come as booleans or 0/1. */
export const yes = (v: unknown) => v === true || v === 1 || v === "1";

// --- a whole-name decision's Name Update on mycomap.org (feedback_org.py) ---------------------

const CHIP: Record<OrgChip["tone"], string> = {
  good: "bg-myco-green/10 text-myco-green",
  bad: "border border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200",
  busy: "border border-sky-300 bg-sky-50 text-sky-800 dark:border-sky-700 dark:bg-sky-950 dark:text-sky-200",
  quiet: "border border-[#A87146]/30 text-muted-foreground dark:border-border",
};

/** The row chip's classes, by the tone the server gave it. */
export const orgChipClass = (tone: OrgChip["tone"]) => CHIP[tone] ?? CHIP.quiet;

/** A step's mark and text classes: done and current filled, stopped hollow and amber. */
export function stepLook(state: OrgStep["state"]): { mark: string; markClass: string; textClass: string } {
  switch (state) {
    case "done": return { mark: "●", markClass: "text-myco-green", textClass: "" };
    case "current": return { mark: "●", markClass: "text-sky-600", textClass: "font-medium" };
    case "stopped": return { mark: "○", markClass: "text-amber-700 dark:text-amber-300", textClass: "font-medium" };
    default: return { mark: "○", markClass: "text-muted-foreground", textClass: "text-muted-foreground" };
  }
}

/** A step's words: "Sent" gains the time it was sent (UTC, like History). */
export function stepText(step: OrgStep, org: Pick<OrgRequest, "sent_at">): string {
  if (step.step === "sent" && step.label === "Sent" && org.sent_at) {
    const d = new Date(org.sent_at * 1000);
    return `Sent ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" })} UTC`;
  }
  return step.label;
}

/** The ?item= of a link back from mycomap.org's Name Update, or null. */
export function linkedItem(search: string): number | null {
  const raw = new URLSearchParams(search).get("item");
  const n = raw && /^\d{1,9}$/.test(raw) ? Number(raw) : NaN;
  return Number.isInteger(n) && n > 0 ? n : null;
}

/** The History line's words for a decision with a Name Update, e.g.
 *  ", Name Update request 7: waiting for a person". */
export function historyOrgText(org: Pick<OrgRequest, "request_id" | "chip"> | null | undefined): string {
  if (!org) return "";
  const what = org.chip.text.replace(/^on mycomap\.org: /, "").replace(/ (on|from|by|to) mycomap\.org/, "");
  return org.request_id ? `, Name Update request ${org.request_id}: ${what}` : `, ${what}`;
}

// --- the page's two sections ------------------------------------------------------------------

/** The page's sections, in order (Steve, 2026-10-18): mechanical fixes a rule found, then
 *  whether a record's label is right. */
export const SECTIONS: { code: string; title: string; intro: string }[] = [
  { code: "cleanup", title: "Data cleanup",
    intro: "Fix what a rule found: photos that show no specimen, one name written several ways, "
      + "retired or inactive names, one find posted twice or placed in two places." },
  { code: "validation", title: "Data validation",
    intro: "Is the record's label right? Look at the photos, its DNA and BLAST results, and decide "
      + "as a curator. Some queues are reviewed blind: Vision's answer stays hidden unless you "
      + "choose to see it." },
];

/** Queues in a section this page does not know yet, kept rather than dropped. */
export const OTHER_SECTION = { code: "other", title: "Other", intro: "" };

export interface SectionGroup<T> {
  code: string;
  title: string;
  intro: string;
  /** Open items across the section's queues. */
  open: number;
  types: T[];
}

/** The stat cards grouped by their `section`: the known sections in order, each queue in the
 *  order the summary lists it, a section with no queues left out, and queues of an unknown
 *  (or missing) section last under "Other". */
export function groupBySection<T extends Pick<FeedbackSummary, "type" | "open"> & { section?: string | null }>(
    summary: T[]): SectionGroup<T>[] {
  const known = new Set(SECTIONS.map((s) => s.code));
  const groups = [...SECTIONS, OTHER_SECTION].map((s) => ({
    ...s,
    types: summary.filter((t) => (s === OTHER_SECTION ? !known.has(t.section ?? "") : t.section === s.code)),
  }));
  return groups
    .filter((g) => g.types.length > 0)
    .map((g) => ({ ...g, open: g.types.reduce((n, t) => n + (t.open || 0), 0) }));
}

// --- "Show Vision's answer" on a blind item ----------------------------------------------------
// The button is the same on every open blind item, so it says nothing about why an item is in
// the queue. Seeing the answer is recorded, and the decision then stays out of the blind check.

export const REVEAL_BUTTON = "Show Vision's answer";
export const REVEAL_CONFIRM = "Your decision on this record will be counted as made with Vision's answer seen. Show it?";
export const REVEAL_SEEN = "Seen — this decision won't count in the blind check.";

/** Vision's answer in one line: "Vision's top 5: A x 0.71 · B y 0.40 …", or why there is none. */
export function formatTop5(answer: Pick<RevealedAnswer, "available" | "top5" | "reason"> | null | undefined): string {
  if (!answer || !answer.available) {
    const why = (answer?.reason ?? "").trim();
    return why ? `Vision has no answer for this record: ${why}` : "Vision has no answer for this record.";
  }
  const top = (answer.top5 ?? []).slice(0, 5)
    .filter(([name]) => !!name)
    .map(([name, score]) => (typeof score === "number" && Number.isFinite(score) ? `${name} ${score.toFixed(2)}` : name));
  return top.length ? `Vision's top 5: ${top.join(" · ")}` : "Vision has no answer for this record.";
}

/** The note under a revealed answer; null when the answer has not been seen. */
export const revealNote = (revealed: Revealed | null | undefined): string | null =>
  revealed ? REVEAL_SEEN : null;

/** Why "Show Vision's answer" failed, in the reviewer's words. */
export function revealProblem(error: unknown): string {
  if (error instanceof ApiError && error.status === 409) {
    return "Nothing is hidden on this item any more (it may have been decided elsewhere).";
  }
  if (error instanceof ApiError && error.status === 401) return error.message;
  return "Vision's answer could not be shown. Try again.";
}


/** What a queue's card says (Steve, 2026-10-18): a photo queue counts photos and the
 *  observations they are in; every other queue counts its items. */
export function statCardLines(t: Pick<FeedbackSummary, "open" | "decided" | "records_open" |
                                       "records_decided">): { big: string; unit: string; sub: string } {
  const n = (x: number) => x.toLocaleString("en-US");
  if (t.records_open === undefined) {
    return { big: n(t.open), unit: "", sub: `open · ${n(t.decided)} decided` };
  }
  const obs = (x: number) => `${n(x)} observation${x === 1 ? "" : "s"}`;
  return {
    big: n(t.open),
    unit: t.open === 1 ? "photo" : "photos",
    sub: `open in ${obs(t.records_open)} · ${n(t.decided)} decided` +
         (t.records_decided ? ` (${obs(t.records_decided)} done)` : ""),
  };
}

// --- which queue the page opens (Steve, 2026-10-21: "Initial load of this page takes a really
// long time"): never one by itself. Opening the first queue with open items meant the 16,000-
// photo review and its photos on every visit.

/** Where the queue a viewer last opened is remembered (their browser only). */
export const QUEUE_STORAGE_KEY = "mycomap-vision.feedback.queue";

type QueueStore = Pick<Storage, "getItem" | "setItem"> | null | undefined;

/** The queue this viewer last opened, or null (none, or storage unavailable). */
export function readLastQueue(storage: () => QueueStore): string | null {
  try {
    const v = storage()?.getItem(QUEUE_STORAGE_KEY);
    return v && /^[a-z0-9_]{1,64}$/.test(v) ? v : null;
  } catch {
    return null;
  }
}

/** Remembers the queue opened; does nothing when storage is unavailable. */
export function writeLastQueue(storage: () => QueueStore, type: string): void {
  try {
    storage()?.setItem(QUEUE_STORAGE_KEY, type);
  } catch {
    // Not remembered: the next visit asks again.
  }
}

/** A ?queue= deep link's queue, or null. */
export function queueParam(search: string): string | null {
  const v = new URLSearchParams(search).get("queue");
  return v && /^[a-z0-9_]{1,64}$/.test(v) ? v : null;
}

/** The queue the page shows: the one the viewer picked here, else a ?queue= link, else a linked
 *  item's (?item=; nothing while it loads), else the one they last opened, else none ("Pick a
 *  queue above"). Only a queue the summary lists; nothing until the summary is loaded. */
export function openQueue(o: {
  picked: string | null; param: string | null; linkedPending: boolean; linkedType: string | null;
  stored: string | null; types: readonly string[] | null | undefined;
}): string | null {
  if (!o.types?.length) return null;
  const known = (t: string | null): t is string => !!t && o.types!.includes(t);
  if (known(o.picked)) return o.picked;
  if (known(o.param)) return o.param;
  if (o.linkedPending) return null;
  if (known(o.linkedType)) return o.linkedType;
  if (known(o.stored)) return o.stored;
  return null;
}

// --- The names behind a name item (Steve, 2026-10-21) ---------------------------------------
// "Names iNaturalist no longer uses" and "Names retired on MycoMap's synonym list" show each
// name with its iNat taxon, linked, and MycoMap's synonym list around it (name_context.py).

export const NAME_ITEM_TYPES = ["name_retired", "name_inactive_on_inat"];

export const isNameItem = (item: Pick<FeedbackItem, "item_type">): boolean =>
  NAME_ITEM_TYPES.includes(item.item_type);

const ROLE_WORDS: Record<NameRole, string> = {
  records: "on the records", proposed: "proposed", "synonym list": "on MycoMap's synonym list",
};

/** "on the records · on MycoMap's synonym list". */
export const roleText = (roles: NameRole[]): string => roles.map((r) => ROLE_WORDS[r] ?? r).join(" · ");

export interface InatLink { text: string; href: string; strong?: boolean }

/** The iNat links for one name: its own taxon (active or not), the current taxa an inactive one
 *  points to, iNat's active writing of the same name; iNat's search when there is no taxon. */
export function inatLinks(entry: Pick<NameContextEntry, "inat">): InatLink[] {
  const { inat } = entry;
  const out: InatLink[] = [];
  if (inat.taxon) {
    out.push({ text: `iNat taxon ${inat.taxon.id}${inat.taxon.active ? "" : " (inactive)"}`, href: inat.taxon.url });
  }
  for (const t of inat.current) {
    out.push({ text: `iNat's current name: ${t.name}${t.active ? "" : " (also inactive)"}`, href: t.url, strong: true });
  }
  for (const t of inat.near) out.push({ text: `iNat writes it ${t.name}`, href: t.url, strong: true });
  if (!inat.taxon) out.push({ text: "Search iNaturalist", href: inat.search_url });
  return out;
}

/** A name item with nothing proposed: iNat names no current taxon, so "use iNat's name" has
 *  nothing to use. The row says so instead of showing the name alone. */
export const noProposal = (item: Pick<FeedbackItem, "item_type" | "proposed">): boolean =>
  isNameItem(item) && !(item.proposed ?? "").trim();

export const NO_PROPOSAL_TEXT = "no current name on iNaturalist";

/** The options a reviewer can pick: without a proposal, not the one that uses it. */
export function usableOptions<T extends Pick<FeedbackOption, "code">>(
  item: Pick<FeedbackItem, "item_type" | "proposed">, options: T[]): T[] {
  return noProposal(item) ? options.filter((o) => o.code !== "accept") : options;
}
