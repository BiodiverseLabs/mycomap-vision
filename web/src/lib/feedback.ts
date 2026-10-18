// Human Feedback (/feedback): the review queue of places where Vision and the reference data
// disagree, for mycomap.org admins, project leaders and contributors. The page's rules live
// here as plain functions so they can be tested (feedback.test.ts).
//
// Fairness: model-flagged queues are mixed with a sample of finds Vision answered right, and
// those items must look exactly like the others until they are decided. Nothing here shows
// which items they are.

import type { BlastLink, FeedbackApplies, FeedbackDecisionBody, FeedbackItem, FeedbackOption,
              LiveSequence, Me, RefreshedRecord } from "./api";
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

/** The value box's starting text: the item's proposal, for an option that needs a value. */
export const prefillValue = (option: Pick<FeedbackOption, "needs_value">,
                             item: Pick<FeedbackItem, "proposed">): string =>
  option.needs_value ? (item.proposed ?? "").trim() : "";

/** The POST body for a decision, or what is missing. */
export function decisionBody(option: FeedbackOption,
                             input: { value?: string; note?: string; photoKeys?: string }):
    { body: FeedbackDecisionBody } | { error: string } {
  const value = (input.value ?? "").trim();
  if (option.needs_value && !value) return { error: "Enter the value this decision needs." };
  const body: FeedbackDecisionBody = { decision: option.code };
  if (option.needs_value) body.value = value;
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
