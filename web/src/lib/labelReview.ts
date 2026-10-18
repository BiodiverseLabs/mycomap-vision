// The Human Feedback label review (Steve, 2026-10-18): the BLIND "Photos unlike the label" queue
// one record at a time. Each card shows the record, its label and every photo; the reviewer
// keeps or leaves out photos with a click (or a key) and saves one outcome for the record.
// The page's rules live here as plain functions so they can be tested (labelReview.test.ts).
//
// Blind rule: finds Vision answered right are mixed into this queue, so until a card is decided
// it shows nothing about Vision or why the record is here, and nothing that differs between
// cards. maskedCard is the one place that decides what an open card may show.

import type { LabelReviewCard, LabelReviewDecisionBody, LabelReviewOutcome, LabelReviewPhoto } from "./api";
import { evidenceRows, shownRule } from "./feedback";
import type { Choice, Choices, KeyPress, KeyState } from "./photoReview";

export const LABEL_REVIEW_TYPE = "photos_unlike_label";
export const LABEL_PAGE_SIZE = 10;
/** The reason "Leave out the chosen photos" starts with. */
export const DEFAULT_REASON = "other_organism";

export const OUTCOME_ORDER: LabelReviewOutcome[] = ["exclude_photos", "hard_example", "rename"];

/** The outcome buttons, in the reviewer's words. */
export const OUTCOME_BUTTONS: Record<LabelReviewOutcome, string> = {
  exclude_photos: "Leave out the chosen photos",
  hard_example: "Label and photos are right (a hard example)",
  rename: "Rename the record to…",
};

/** What the page says above the cards. The queue's own description says why records were
 *  picked (Vision's answer), which a blind review must not, so the page says what to do. */
export const LABEL_REVIEW_ABOUT = "Look at each record's photos against its label. Leave out photos that show "
  + "another organism or no usable specimen, say the label and photos are right, or rename the record.";

// --- the blind rule ----------------------------------------------------------------------------

/** A card is masked while it is open, whatever the server's flag says. */
export const isMasked = (card: Pick<LabelReviewCard, "masked" | "status">): boolean =>
  card.masked !== false || card.status !== "decided";

const maskedPhoto = (p: LabelReviewPhoto): LabelReviewPhoto => {
  const out: LabelReviewPhoto = { photo_key: p.photo_key, position: p.position, shown: p.shown, excluded: false };
  if (p.license_class !== undefined) out.license_class = p.license_class;
  if (p.thumb !== undefined) out.thumb = p.thumb;
  if (p.full !== undefined) out.full = p.full;
  if (p.hidden_reason !== undefined) out.hidden_reason = p.hidden_reason;
  return out;
};

/** What a card may show. A masked (open) card keeps only the record, its link, its label and
 *  its photos, by an allowlist: no rule, evidence, decision or any other field, even when the
 *  server sent one. A decided card is shown as sent. */
export function maskedCard(card: LabelReviewCard): LabelReviewCard {
  if (!isMasked(card)) return card;
  return {
    item_id: card.item_id,
    record_key: card.record_key,
    label: card.label,
    link: card.link,
    status: card.status,
    masked: true,
    photos: (card.photos ?? []).map(maskedPhoto),
  };
}

/** "What Vision said" on a decided card: the rule worded for reviewers, and the evidence. Null
 *  on a masked card. */
export function visionSaid(card: LabelReviewCard): { rule: string | null; rows: [string, string][] } | null {
  const c = maskedCard(card);
  if (isMasked(c)) return null;
  return {
    rule: shownRule({ rule: c.rule ?? "", status: "decided", model_flagged: true }),
    rows: evidenceRows({ current: null, proposed: null, evidence: c.evidence ?? {} }),
  };
}

// --- photos and choices ----------------------------------------------------------------------

/** The card's photos in position order (a photo without a position last, as sent). */
export function sortedPhotos<P extends Pick<LabelReviewPhoto, "position">>(photos: P[]): P[] {
  return photos
    .map((p, i) => ({ p, i }))
    .sort((a, b) => (a.p.position ?? Infinity) - (b.p.position ?? Infinity) || a.i - b.i)
    .map(({ p }) => p);
}

/** How many of the card's photos are left out (a choice for another photo does not count). */
export const excludedCount = (card: Pick<LabelReviewCard, "photos">, choices: Choices): number =>
  card.photos.filter((p) => choices[p.photo_key] === "exclude").length;

// --- which outcomes may be saved -------------------------------------------------------------

export interface OutcomeInput { reason?: string; value?: string; note?: string }

export type OutcomeState = { enabled: true } | { enabled: false; why: string };

export const WHY_LEFT_OUT = "You've left photos out: save that outcome, or keep every photo.";
export const WHY_NONE_LEFT_OUT = "To leave photos out, exclude at least one first.";
export const WHY_NO_REASON = "Pick why the photos are left out.";
export const WHY_NO_NAME = "Type the name to rename the record to.";

/** Each outcome: whether it may be saved with these choices, and why not. Leaving photos out
 *  is its own outcome; a hard example and a rename keep every photo. */
export function outcomeStates(card: Pick<LabelReviewCard, "photos">, choices: Choices, input: OutcomeInput = {},
                              reasonCodes?: string[]): Record<LabelReviewOutcome, OutcomeState> {
  const out = excludedCount(card, choices);
  const reason = (input.reason ?? "").trim();
  const name = (input.value ?? "").trim();
  const reasonOk = !!reason && (!reasonCodes || reasonCodes.length === 0 || reasonCodes.includes(reason));
  return {
    exclude_photos: out === 0 ? { enabled: false, why: WHY_NONE_LEFT_OUT }
      : reasonOk ? { enabled: true } : { enabled: false, why: WHY_NO_REASON },
    hard_example: out > 0 ? { enabled: false, why: WHY_LEFT_OUT } : { enabled: true },
    rename: out > 0 ? { enabled: false, why: WHY_LEFT_OUT }
      : name ? { enabled: true } : { enabled: false, why: WHY_NO_NAME },
  };
}

/** The short line under the buttons: why the disabled ones are disabled, each reason once. */
export function outcomeHint(states: Record<LabelReviewOutcome, OutcomeState>): string {
  const whys: string[] = [];
  for (const o of OUTCOME_ORDER) {
    const s = states[o];
    if (!s.enabled && !whys.includes(s.why)) whys.push(s.why);
  }
  return whys.join(" ");
}

/** The POST body for saving one outcome: only the card's chosen photos, in position order; the
 *  reason only with exclude_photos, the name only with rename. Or why it can't be saved. */
export function saveBody(card: Pick<LabelReviewCard, "photos">, choices: Choices, outcome: LabelReviewOutcome,
                         input: OutcomeInput = {}, reasonCodes?: string[]):
    { body: LabelReviewDecisionBody } | { error: string } {
  if (!OUTCOME_ORDER.includes(outcome)) return { error: "Choose one of the outcomes." };
  const state = outcomeStates(card, choices, input, reasonCodes)[outcome];
  if (!state.enabled) return { error: state.why };
  const photos: LabelReviewDecisionBody["photos"] = [];
  for (const p of sortedPhotos(card.photos)) {
    const c = choices[p.photo_key];
    if (c === "keep" || c === "exclude") photos.push({ photo_key: p.photo_key, decision: c });
  }
  const body: LabelReviewDecisionBody = { outcome, photos };
  if (outcome === "exclude_photos") body.reason = (input.reason ?? "").trim();
  if (outcome === "rename") body.value = (input.value ?? "").trim();
  const note = (input.note ?? "").trim();
  if (note) body.note = note;
  return { body };
}

// --- the outcome in words --------------------------------------------------------------------

type Reasons = { code: string; label: string }[];

const reasonLabel = (code: string | null | undefined, reasons: Reasons = []): string | null =>
  code ? (reasons.find((r) => r.code === code)?.label ?? code) : null;

const photosWord = (n: number) => `${n} photo${n === 1 ? "" : "s"}`;

/** An outcome as the saved line and the decided view say it. */
export function outcomeWords(outcome: string, detail: { left_out?: number; reason?: string | null;
                                                         value?: string | null } = {}, reasons?: Reasons): string {
  switch (outcome) {
    case "exclude_photos": {
      const why = reasonLabel(detail.reason, reasons);
      return `left out ${photosWord(detail.left_out ?? 0)}${why ? ` (${why})` : ""}`;
    }
    case "hard_example":
      return "label and photos are right (a hard example)";
    case "rename":
      return detail.value ? `renamed to ${detail.value}` : "renamed";
    default:
      return outcome;
  }
}

/** The one line a saved card collapses to. */
export function savedLine(body: LabelReviewDecisionBody, reasons?: Reasons): string {
  const left = body.photos.filter((p) => p.decision === "exclude").length;
  return `Saved: ${outcomeWords(body.outcome, { left_out: left, reason: body.reason, value: body.value }, reasons)}`;
}

/** A stored note as the server keeps it ("reason: <code>; <note>"), split back apart. */
export function splitStoredNote(note: string | null | undefined): { reason: string | null; note: string | null } {
  const s = (note ?? "").trim();
  const m = /^reason:\s*([^;]+?)\s*(?:;\s*([\s\S]*))?$/.exec(s);
  if (m) return { reason: m[1], note: (m[2] ?? "").trim() || null };
  return { reason: null, note: s || null };
}

/** A decided card's stored outcome: its words, the note, who decided. Null while masked. */
export function decidedSummary(card: LabelReviewCard, reasons?: Reasons):
    { words: string; note: string | null; by: string | null; at: number | string | null } | null {
  const c = maskedCard(card);
  if (isMasked(c) || !c.decision) return null;
  const d = c.decision;
  const { reason, note } = splitStoredNote(d.note);
  const left = d.photo_keys?.length ?? c.photos.filter((p) => p.excluded).length;
  return { words: outcomeWords(d.decision, { left_out: left, reason, value: d.value }, reasons),
           note, by: d.by ?? null, at: d.at ?? null };
}

// --- keys ------------------------------------------------------------------------------------

export type LabelKeyAction =
  | { type: "record"; step: 1 | -1 }
  | { type: "photo"; step: 1 | -1 }
  | { type: "choose"; choice: Choice }
  | { type: "outcome"; outcome: "exclude_photos" | "hard_example" }
  | { type: "rename" }
  | { type: "help" }
  | { type: "close" };

/** The keys. On the list: j/k next/previous record, ←/→ photo, Shift+K keep (k alone is
 *  "previous record"), x exclude, e save "leave out", h save a hard example, r the rename field,
 *  ? help. In the photo viewer: ←/→ photo, k or K keep, x exclude, Esc close. None while typing. */
export function labelKeyAction(e: KeyPress, s: KeyState): LabelKeyAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey || s.typing) return null;
  const k = e.key;
  if (s.lightbox) {
    if (k === "Escape") return { type: "close" };
    if (k === "ArrowLeft") return { type: "photo", step: -1 };
    if (k === "ArrowRight") return { type: "photo", step: 1 };
    if (k === "k" || k === "K") return { type: "choose", choice: "keep" };
    if (k === "x" || k === "X") return { type: "choose", choice: "exclude" };
    return null;
  }
  switch (k) {
    case "j": return { type: "record", step: 1 };
    case "k": return { type: "record", step: -1 };
    case "K": return { type: "choose", choice: "keep" };
    case "x": case "X": return { type: "choose", choice: "exclude" };
    case "ArrowLeft": return { type: "photo", step: -1 };
    case "ArrowRight": return { type: "photo", step: 1 };
    case "e": case "E": return { type: "outcome", outcome: "exclude_photos" };
    case "h": case "H": return { type: "outcome", outcome: "hard_example" };
    case "r": case "R": return { type: "rename" };
    case "?": return { type: "help" };
    case "Escape": return s.help ? { type: "help" } : null;
    default: return null;
  }
}

/** The key help, as shown. */
export const LABEL_KEY_HELP: [string, string][] = [
  ["j / k", "next / previous record"],
  ["← / →", "previous / next photo"],
  ["Shift+K", "keep the photo (K in the photo viewer)"],
  ["X", "exclude the photo"],
  ["E", "save: leave out the chosen photos"],
  ["H", "save: label and photos are right"],
  ["R", "type the name to rename to"],
  ["Esc", "close the photo viewer"],
  ["?", "show or hide these keys"],
];
