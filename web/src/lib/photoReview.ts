// The Human Feedback photo review (Steve, 2026-10-18): the "Photos that show no specimen" queue
// reviewed one record at a time. Each card shows every photo of the record; the reviewer keeps
// or leaves out each one with a click (or a key) and saves the record. The page's rules live
// here as plain functions so they can be tested (photoReview.test.ts).
//
// Photo rule: a photo the server sends as "hidden" never gets an image URL here, and an admin's
// "private" photo is only ever Vision's own same-origin copy, shown by a plain <img>.

import type { PhotoFlag, PhotoReviewDecisionBody, PhotoReviewSort, PhotoReviewTab, ReviewPhoto,
              ReviewRecord } from "./api";

export const PHOTO_REVIEW_TYPE = "non_specimen_photo";
export const REVIEW_PAGE_SIZE = 10;

export const SORTS: { code: PhotoReviewSort; label: string }[] = [
  { code: "p", label: "Strongest flag first" },
  { code: "p_asc", label: "Weakest first" },
  { code: "flagged", label: "Most flagged photos" },
];

export const MIN_FLAGGED = [1, 2, 3, 5] as const;

export type Choice = "keep" | "exclude";
/** A plain Keep / Exclude map by photo key (the label review's, and the shared tiles'). */
export type Choices = Record<string, Choice>;

/** One photo's choice on a photo-review card (Steve, 2026-10-18: a record's photos can be of
 *  several kinds, so each left-out photo carries its own reason). `suggested` marks an
 *  exclusion the page proposed in "suggest" mode that the reviewer has not confirmed yet; any
 *  choice the reviewer makes clears it. A kept photo keeps its reason so it comes back if the
 *  photo is excluded again; it is never sent. */
export interface PhotoChoice { decision: Choice; reason: string | null; suggested?: boolean }
/** A record's choices so far, by photo key. A photo with no choice is not sent. */
export type PhotoChoices = Record<string, PhotoChoice>;
/** Reasons the reviewer picked, by photo key: kept across tabs and pages, so changing the tab
 *  never resets them. */
export type PickedReasons = Record<string, string>;

/** How a card starts: "none" preselects nothing (the default); "suggest" starts every open
 *  photo flagged in the tab shown as a suggested Exclude with its own suggested reason. */
export type PreselectMode = "none" | "suggest";
/** Where a viewer's mode is remembered (their browser only). */
export const PRESELECT_STORAGE_KEY = "mycomap-vision.photo-review.suggest-exclude";

type ModeStore = Pick<Storage, "getItem" | "setItem"> | null | undefined;

/** The viewer's remembered mode; "none" when nothing is stored or storage is unavailable (a
 *  private window, blocked site data: the getter itself may throw). */
export function readPreselectMode(storage: () => ModeStore): PreselectMode {
  try {
    return storage()?.getItem(PRESELECT_STORAGE_KEY) === "suggest" ? "suggest" : "none";
  } catch {
    return "none";
  }
}

/** Remembers the viewer's mode; does nothing when storage is unavailable. */
export function writePreselectMode(storage: () => ModeStore, mode: PreselectMode): void {
  try {
    storage()?.setItem(PRESELECT_STORAGE_KEY, mode);
  } catch {
    // Not remembered: the page still works in the mode picked.
  }
}

/** The reason a photo's exclusion starts at: the one the reviewer picked for it, else its own
 *  flag's (suggested_reason), else none (an unflagged photo: the reviewer picks). */
export function reasonFor(photo: Pick<ReviewPhoto, "photo_key" | "suggested_reason">,
                          picked: PickedReasons = {}): string | null {
  return picked[photo.photo_key] ?? photo.suggested_reason ?? null;
}

/** The record's photos in position order (a photo without a position last, as sent). */
export function sortedPhotos(record: Pick<ReviewRecord, "photos">): ReviewPhoto[] {
  return record.photos
    .map((p, i) => ({ p, i }))
    .sort((a, b) => (a.p.position ?? Infinity) - (b.p.position ?? Infinity) || a.i - b.i)
    .map(({ p }) => p);
}

/** A photo as the page names it: "photo 3" (its place in the record when it has no position). */
export function photoName(record: Pick<ReviewRecord, "photos">, photoKey: string): string {
  const sorted = sortedPhotos(record);
  const i = sorted.findIndex((p) => p.photo_key === photoKey);
  return `photo ${sorted[i]?.position ?? i + 1}`;
}

/** A decided photo's stored choice; nothing for an open or unflagged one. */
export function storedChoice(photo: Pick<ReviewPhoto, "status" | "decision">): Choice | null {
  if (photo.status !== "decided" || !photo.decision) return null;
  if (photo.decision.decision === "exclude_photo") return "exclude";
  if (photo.decision.decision === "keep_photo") return "keep";
  return null;
}

/** The photos "Exclude all flagged" leaves out: the tab's flagged photos, every flagged one in All. */
export function flaggedForBulk(photo: Pick<ReviewPhoto, "flagged" | "in_tab">, tab: PhotoReviewTab): boolean {
  return tab === "all" ? photo.flagged : photo.in_tab;
}

/** "Suggest" mode proposes leaving out an OPEN photo flagged in the tab shown; never a decided
 *  one, one flagged only for other tabs, or an unflagged one. */
export const suggestsExclude = (photo: Pick<ReviewPhoto, "flagged" | "in_tab" | "status">,
                                tab: PhotoReviewTab): boolean =>
  photo.status === "open" && flaggedForBulk(photo, tab);

/** The choices under a preselection mode: the reviewer's own choices stay, earlier suggestions
 *  go, and in "suggest" mode each suggestible photo with no choice starts as a suggested Exclude
 *  with its own reason. Used when a card starts and when the mode is switched. */
export function withMode(record: Pick<ReviewRecord, "photos">, tab: PhotoReviewTab, mode: PreselectMode,
                         choices: PhotoChoices, picked: PickedReasons = {}): PhotoChoices {
  const out: PhotoChoices = {};
  for (const [k, c] of Object.entries(choices)) if (!c.suggested) out[k] = c;
  if (mode !== "suggest") return out;
  for (const p of record.photos) {
    if (out[p.photo_key] || !suggestsExclude(p, tab)) continue;
    out[p.photo_key] = { decision: "exclude", reason: reasonFor(p, picked), suggested: true };
  }
  return out;
}

/** What a card starts with: a decided photo shows its stored decision (and reason); in the
 *  default mode nothing else is preselected; in "suggest" mode see withMode. */
export function initialChoices(record: Pick<ReviewRecord, "photos">, tab: PhotoReviewTab,
                               mode: PreselectMode = "none", picked: PickedReasons = {}): PhotoChoices {
  const out: PhotoChoices = {};
  for (const p of record.photos) {
    const c = storedChoice(p);
    if (c) out[p.photo_key] = { decision: c, reason: c === "exclude" ? (p.decision?.reason ?? null) : null };
  }
  return withMode(record, tab, mode, out, picked);
}

/** One click on Keep or Exclude. On a suggestion: Keep turns it into a kept photo, Exclude
 *  confirms it. Otherwise it picks the choice, or clears it when it was already picked. An
 *  exclusion keeps the photo's reason, else starts at reasonFor (none for an unflagged photo). */
export function chooseDecision(choices: PhotoChoices, photo: Pick<ReviewPhoto, "photo_key" | "suggested_reason">,
                               decision: Choice, picked: PickedReasons = {}): PhotoChoices {
  const next = { ...choices };
  const key = photo.photo_key;
  const c = next[key];
  if (c && c.decision === decision && !c.suggested) {
    delete next[key];
    return next;
  }
  next[key] = { decision, reason: c?.reason ?? (decision === "exclude" ? reasonFor(photo, picked) : null) };
  return next;
}

/** The reviewer picks a photo's reason: it is their choice from then on (a suggestion is
 *  confirmed). Only an excluded photo has a reason to pick. */
export function setReason(choices: PhotoChoices, photoKey: string, reason: string): PhotoChoices {
  const c = choices[photoKey];
  if (!c || c.decision !== "exclude") return choices;
  return { ...choices, [photoKey]: { decision: "exclude", reason } };
}

/** R: the excluded photo's next reason, in the review's order (the first when it has none). */
export function cycleReason(choices: PhotoChoices, photoKey: string, reasonCodes: string[]): PhotoChoices {
  const c = choices[photoKey];
  if (!c || c.decision !== "exclude" || !reasonCodes.length) return choices;
  const i = reasonCodes.indexOf(c.reason ?? "");
  return setReason(choices, photoKey, reasonCodes[(i + 1) % reasonCodes.length]);
}

/** "Keep all": every photo of the record kept. */
export function bulkKeepAll(record: Pick<ReviewRecord, "photos">, choices: PhotoChoices = {}): PhotoChoices {
  const out: PhotoChoices = {};
  for (const p of record.photos) out[p.photo_key] = { decision: "keep", reason: choices[p.photo_key]?.reason ?? null };
  return out;
}

/** "Exclude all flagged": those photos left out, each with its own reason (one already on it,
 *  else its suggested one); the others keep their choice (or none). */
export function bulkExcludeFlagged(record: Pick<ReviewRecord, "photos">, tab: PhotoReviewTab,
                                   choices: PhotoChoices = {}, picked: PickedReasons = {}): PhotoChoices {
  const out: PhotoChoices = { ...choices };
  for (const p of record.photos) {
    if (!flaggedForBulk(p, tab)) continue;
    out[p.photo_key] = { decision: "exclude", reason: choices[p.photo_key]?.reason ?? reasonFor(p, picked) };
  }
  return out;
}

/** The first excluded photo (in position order) with no reason yet, or null. Save is off
 *  while there is one. */
export function missingReason(record: Pick<ReviewRecord, "photos">, choices: PhotoChoices): ReviewPhoto | null {
  for (const p of sortedPhotos(record)) {
    const c = choices[p.photo_key];
    if (c?.decision === "exclude" && !(c.reason ?? "").trim()) return p;
  }
  return null;
}

/** The POST body for saving a record: only the photos with a choice (suggestions included), in
 *  position order; each exclusion carries its own reason. Or what is missing, naming the photo. */
export function saveBody(record: Pick<ReviewRecord, "photos">, choices: PhotoChoices, note?: string,
                         reasonCodes?: string[]): { body: PhotoReviewDecisionBody } | { error: string } {
  const photos: PhotoReviewDecisionBody["photos"] = [];
  for (const p of sortedPhotos(record)) {
    const c = choices[p.photo_key];
    if (!c) continue;
    if (c.decision === "keep") {
      photos.push({ photo_key: p.photo_key, decision: "keep" });
      continue;
    }
    const why = (c.reason ?? "").trim();
    if (!why) return { error: `Pick why ${photoName(record, p.photo_key)} is left out.` };
    if (reasonCodes && reasonCodes.length > 0 && !reasonCodes.includes(why)) {
      return { error: `Pick a listed reason for ${photoName(record, p.photo_key)}.` };
    }
    photos.push({ photo_key: p.photo_key, decision: "exclude", reason: why });
  }
  if (!photos.length) return { error: "Keep or exclude at least one photo first." };
  const body: PhotoReviewDecisionBody = { photos };
  const n = (note ?? "").trim();
  if (n) body.note = n;
  return { body };
}

/** How many photos have a choice, suggestions included (Save is off at 0). */
export const chosenCount = (record: Pick<ReviewRecord, "photos">, choices: PhotoChoices): number =>
  record.photos.filter((p) => choices[p.photo_key]).length;

/** How many of them are still only suggestions. */
export const suggestedCount = (record: Pick<ReviewRecord, "photos">, choices: PhotoChoices): number =>
  record.photos.filter((p) => choices[p.photo_key]?.suggested).length;

/** The plain Keep / Exclude map the shared tiles and viewer draw. */
export function decisionsOf(choices: PhotoChoices): Choices {
  const out: Choices = {};
  for (const [k, c] of Object.entries(choices)) out[k] = c.decision;
  return out;
}

/** The one line a saved card collapses to: how many left out (how many of them as suggested),
 *  how many kept. */
export function savedLine(body: PhotoReviewDecisionBody, choices: PhotoChoices): string {
  const out = body.photos.filter((p) => p.decision === "exclude");
  const asSuggested = out.filter((p) => choices[p.photo_key]?.suggested).length;
  const kept = body.photos.length - out.length;
  return `Saved: ${out.length} left out${asSuggested ? ` (${asSuggested} as suggested)` : ""}, ${kept} kept`;
}

/** The label review's plain toggle: picks it, or clears it when it was already picked. */
export function toggleChoice(choices: Choices, photoKey: string, choice: Choice): Choices {
  const next = { ...choices };
  if (next[photoKey] === choice) delete next[photoKey];
  else next[photoKey] = choice;
  return next;
}

/** A flag as the tile labels it: the probe or view and its p; the gate in plain words. */
export function flagLabel(flag: PhotoFlag): string {
  const p = flag.p == null ? null : flag.p.toFixed(2);
  if (flag.tab === "gate") return p == null ? "not a fungus" : `not a fungus, p(fungus)=${p}`;
  return p == null ? flag.kind : `${flag.kind} p=${p}`;
}

export type Outline = "strong" | "light" | "none";

/** A strong outline for a photo flagged in the tab shown, a light one for a photo flagged only
 *  for other tabs. */
export function outlineOf(photo: Pick<ReviewPhoto, "flagged" | "in_tab">): Outline {
  if (photo.in_tab) return "strong";
  return photo.flagged ? "light" : "none";
}

/** The tile's labels: this tab's flags (strong) and the other tabs' (light). In All every flag
 *  counts as this tab's. */
export function photoLabels(photo: Pick<ReviewPhoto, "flags" | "in_tab">, tab: PhotoReviewTab):
    { strong: string[]; light: string[] } {
  const strong: string[] = [];
  const light: string[] = [];
  for (const f of photo.flags ?? []) {
    const mine = photo.in_tab && (tab === "all" || f.tab === tab);
    (mine ? strong : light).push(flagLabel(f));
  }
  return { strong, light };
}

/** The image URL a tile may load, or null. Never one for a hidden photo; a private photo only
 *  from Vision's own same-origin route; a public one only over https. */
export function tileImage(photo: Pick<ReviewPhoto, "shown" | "thumb" | "full">,
                          size: "thumb" | "full" = "thumb"): string | null {
  if (photo.shown === "hidden") return null;
  const url = (size === "full" ? photo.full || photo.thumb : photo.thumb) ?? "";
  if (photo.shown === "private") return url.startsWith("/api/feedback/photo/") ? url : null;
  if (photo.shown === "public") return /^https:\/\//i.test(url) ? url : null;
  return null;
}

/** An image, or the placeholder (position, why it is hidden, the record's link). A photo whose
 *  image failed to load (`failed`: e.g. a private copy Vision does not hold, a 404) is a
 *  placeholder too, and stays decidable. */
export const tileKind = (photo: Pick<ReviewPhoto, "shown" | "thumb" | "full">,
                         failed = false): "image" | "placeholder" =>
  !failed && tileImage(photo, "thumb") ? "image" : "placeholder";

/** What a placeholder says: why the photo is hidden, or that its image failed to load. */
export function placeholderText(photo: Pick<ReviewPhoto, "shown" | "hidden_reason">, failed = false): string {
  if (photo.shown === "hidden") return photo.hidden_reason || "not shown here";
  if (failed) return "photo not available here";
  return "no usable photo URL";
}

/** "Open on MO" for a Mushroom Observer record, "Open on iNat" otherwise. */
export const openOnLabel = (recordKey: string) => (recordKey.startsWith("mo:") ? "Open on MO" : "Open on iNat");

/** One step within [0, n), stopping at the ends. */
export function step(i: number, n: number, delta: number): number {
  if (n <= 0) return 0;
  return Math.min(n - 1, Math.max(0, i + delta));
}

/** Where focus goes after a save: the next unsaved card after `from`, else the first unsaved
 *  one before it, else -1 (the page is done). */
export function nextUnsaved(keys: string[], saved: ReadonlySet<string>, from: number): number {
  for (let i = from + 1; i < keys.length; i++) if (!saved.has(keys[i])) return i;
  for (let i = 0; i <= Math.min(from, keys.length - 1); i++) if (!saved.has(keys[i])) return i;
  return -1;
}

/** The offset to load when every card on the page is saved: the same one in Open (the saved
 *  records drop out, the next ones move up), the next page in Decided (they stay listed). */
export function offsetWhenPageDone(status: "open" | "decided", offset: number, limit: number,
                                   total: number): number {
  if (status === "open") return offset;
  return offset + limit < total ? offset + limit : offset;
}

/** The offset "Next page" goes to. In Open the records saved on this page drop out of the
 *  list, so the next page starts that many earlier. */
export function nextPageOffset(status: "open" | "decided", offset: number, limit: number,
                               savedOnPage: number): number {
  return offset + limit - (status === "open" ? Math.min(limit, savedOnPage) : 0);
}

// --- keys -------------------------------------------------------------------------------------

export type KeyAction =
  | { type: "record"; step: 1 | -1 }
  | { type: "photo"; step: 1 | -1 }
  | { type: "choose"; choice: Choice }
  | { type: "reason" }
  | { type: "keepAll" }
  | { type: "excludeFlagged" }
  | { type: "save" }
  | { type: "help" }
  | { type: "close" };

export interface KeyState {
  /** An input, textarea, select or editable element has focus: keys are its own. */
  typing: boolean;
  /** A button or link has focus: Enter is its own. */
  onControl?: boolean;
  lightbox: boolean;
  help?: boolean;
}

export interface KeyPress { key: string; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean }

/** The keys. On the list: j/k next/previous record, ←/→ photo, Shift+K keep (k alone is
 *  "previous record"), x exclude, r next reason, a keep all, f exclude all flagged, s or Enter
 *  save, ? help. In the lightbox: ←/→ photo, k or K keep, x exclude, r next reason, Esc close. */
export function keyAction(e: KeyPress, s: KeyState): KeyAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey || s.typing) return null;
  const k = e.key;
  if (s.lightbox) {
    if (k === "Escape") return { type: "close" };
    if (k === "ArrowLeft") return { type: "photo", step: -1 };
    if (k === "ArrowRight") return { type: "photo", step: 1 };
    if (k === "k" || k === "K") return { type: "choose", choice: "keep" };
    if (k === "x" || k === "X") return { type: "choose", choice: "exclude" };
    if (k === "r" || k === "R") return { type: "reason" };
    return null;
  }
  switch (k) {
    case "j": return { type: "record", step: 1 };
    case "k": return { type: "record", step: -1 };
    case "K": return { type: "choose", choice: "keep" };
    case "x": case "X": return { type: "choose", choice: "exclude" };
    case "r": case "R": return { type: "reason" };
    case "ArrowLeft": return { type: "photo", step: -1 };
    case "ArrowRight": return { type: "photo", step: 1 };
    case "a": case "A": return { type: "keepAll" };
    case "f": case "F": return { type: "excludeFlagged" };
    case "s": case "S": return { type: "save" };
    case "Enter": return s.onControl ? null : { type: "save" };
    case "?": return { type: "help" };
    case "Escape": return s.help ? { type: "help" } : null;
    default: return null;
  }
}

/** The key help, as shown. */
export const KEY_HELP: [string, string][] = [
  ["j / k", "next / previous record"],
  ["← / →", "previous / next photo"],
  ["Shift+K", "keep the photo (K in the photo viewer)"],
  ["X", "exclude the photo, with its suggested reason (else pick one)"],
  ["R", "next reason for an excluded photo"],
  ["A", "keep all"],
  ["F", "exclude all flagged"],
  ["S or Enter", "save the record"],
  ["Esc", "close the photo viewer; in a reason list, Enter or Esc hands the keys back"],
  ["?", "show or hide these keys"],
];
