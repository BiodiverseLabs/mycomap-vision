// The photo review's Summary view (Steve, 2026-10-21): the "Photos that show no specimen" queue
// as a grid of its flagged photos, one tile each, to scan quickly. Every open tile starts as
// Exclude with its suggested reason; its Include / Exclude flipper IS the saved decision:
//
// - a flip (or a reason picked in the viewer) saves at once, with the note "summary: flipped";
// - the default Exclude is saved only for a tile actually shown: at least 60% of it inside the
//   viewport for a second AND its image loaded (or its full photo loaded in the viewer). Such
//   tiles are collected and sent every couple of seconds, with the note "summary: shown". A
//   hidden or failed photo is never saved by itself; a flip still saves it.
// - nothing is sent twice: a tile whose current state is the one last saved is not due.
//
// With Auto-save off (lib/reviewSave.ts) the same decisions are held as unsaved changes instead:
// nothing is sent until the reviewer presses a tile's Save or "Save page"; a tile not flipped is
// then saved with the note "summary: saved". Discard drops them.
//
// The page's rules live here as plain functions so they can be tested (photoSummary.test.ts).

import type { FlaggedTile, PhotoBatchBody, PhotoBatchDecision, PhotoReviewTab } from "./api";
import { recordLabel } from "./feedback";
import { flagLabel, type Choice } from "./photoReview";

/** The Summary's photos per page: the default, and the choices (Steve, 2026-10-21), each
 *  remembered per viewer. The server serves at most the largest. */
export const SUMMARY_PAGE_SIZE = 150;
export const PAGE_SIZES = [150, 300, 500] as const;
export type PageSize = (typeof PAGE_SIZES)[number];
/** How often the shown tiles' default decisions are sent. */
export const AUTOSAVE_MS = 2000;
/** A tile counts as seen when this much of it has been in view for SEEN_MS. */
export const SEEN_RATIO = 0.6;
export const SEEN_MS = 1000;

export const NOTE_FLIPPED = "summary: flipped";
export const NOTE_SHOWN = "summary: shown";
/** A starting Exclude the reviewer saved by hand (Auto-save off). */
export const NOTE_SAVED = "summary: saved";

/** The tabs, as the server names them (photo_review.TABS); the record view's response carries
 *  them with counts, the Summary's does not. */
export const SUMMARY_TABS: { code: Exclude<PhotoReviewTab, "all">; title: string }[] = [
  { code: "micrograph", title: "Micrographs and microscope set-ups" },
  { code: "slip", title: "Field slips and labels" },
  { code: "habitat", title: "Habitat and scenes" },
  { code: "people", title: "People" },
  { code: "gate", title: "Not a fungus (gate)" },
  { code: "uv", title: "UV reactions" },
  { code: "tree", title: "Phylogenetic trees and BLAST" },
  { code: "drawing", title: "Drawings" },
  { code: "chemical", title: "Chemical spot tests" },
  { code: "spore_print", title: "Spore prints" },
  { code: "dried", title: "Dried specimens" },
  { code: "specimen", title: "Probably shows the specimen" },
  { code: "other", title: "Other" },
];

type Store = Pick<Storage, "getItem" | "setItem"> | null | undefined;

function readKey<T extends string>(storage: () => Store, key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = storage()?.getItem(key);
    return allowed.includes(v as T) ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeKey(storage: () => Store, key: string, value: string): void {
  try {
    storage()?.setItem(key, value);
  } catch {
    // Not remembered: the page still works with the choice made.
  }
}

// --- per-viewer settings ------------------------------------------------------------------------

/** Summary (the flagged photos as a grid) or Full (record by record). */
export type ReviewView = "summary" | "full";
export const VIEW_STORAGE_KEY = "mycomap-vision.photo-review.view";
export const readView = (storage: () => Store): ReviewView =>
  readKey(storage, VIEW_STORAGE_KEY, ["summary", "full"] as const, "summary");
export const writeView = (storage: () => Store, v: ReviewView) => writeKey(storage, VIEW_STORAGE_KEY, v);

/** The Summary's tile size: small, medium (the default, about 6 across on a desktop) or large. */
export type TileSize = "s" | "m" | "l";
export const TILE_SIZES: { code: TileSize; label: string; px: number }[] = [
  { code: "s", label: "Small", px: 120 },
  { code: "m", label: "Medium", px: 170 },
  { code: "l", label: "Large", px: 240 },
];
export const TILE_SIZE_STORAGE_KEY = "mycomap-vision.photo-review.tile-size";
export const readTileSize = (storage: () => Store): TileSize =>
  readKey(storage, TILE_SIZE_STORAGE_KEY, ["s", "m", "l"] as const, "m");
export const writeTileSize = (storage: () => Store, s: TileSize) => writeKey(storage, TILE_SIZE_STORAGE_KEY, s);

export const PAGE_SIZE_STORAGE_KEY = "mycomap-vision.photo-review.page-size";
export const readPageSize = (storage: () => Store): PageSize =>
  Number(readKey(storage, PAGE_SIZE_STORAGE_KEY, PAGE_SIZES.map(String), String(SUMMARY_PAGE_SIZE))) as PageSize;
export const writePageSize = (storage: () => Store, n: PageSize) => writeKey(storage, PAGE_SIZE_STORAGE_KEY, String(n));

/** The grid's columns for a tile size: as many tiles of about that width as fit, never fewer
 *  than two across (a phone). `gapPx` is the grid's gap. */
export function gridColumns(size: TileSize, gapPx = 8): string {
  const px = TILE_SIZES.find((t) => t.code === size)?.px ?? 170;
  return `repeat(auto-fill, minmax(min(${px}px, calc(50% - ${gapPx / 2}px)), 1fr))`;
}

// --- marks: each tile's Include / Exclude -------------------------------------------------------

/** A tile's current choice. "keep" is shown as Include. A kept tile keeps its reason so it comes
 *  back if the tile is excluded again (it is never sent with a keep). `readOnly`: a decided
 *  tile's stored decision. `reclassified`: a decided tile the reviewer changed full screen
 *  (Steve, 2026-10-21), saved like a flip; the latest decision wins on the server. */
export interface Mark { decision: Choice; reason: string | null; readOnly?: boolean; reclassified?: boolean }
/** By item id. */
export type Marks = Record<number, Mark>;

type MarkTile = Pick<FlaggedTile, "item_id" | "status" | "decision" | "suggested_reason" | "suggested_decision">;

/** How a page starts: an open tile as Exclude with its suggested reason, in every band, or as
 *  Include when the server suggests keeping it (tagger v2 reads it as a specimen photo); a
 *  decided tile shows its stored decision, read-only. */
export function initialMarks(tiles: MarkTile[]): Marks {
  const out: Marks = {};
  for (const t of tiles) {
    if (t.status === "open") {
      out[t.item_id] = t.suggested_decision === "keep"
        ? { decision: "keep", reason: t.suggested_reason ?? null }
        : { decision: "exclude", reason: t.suggested_reason ?? null };
      continue;
    }
    const d = t.decision?.decision;
    if (d === "exclude_photo") out[t.item_id] = { decision: "exclude", reason: t.decision?.reason ?? null, readOnly: true };
    else if (d === "keep_photo") out[t.item_id] = { decision: "keep", reason: null, readOnly: true };
  }
  return out;
}

/** The page's marks: the reviewer's own (`edits`, kept across pages this visit) over the
 *  starting ones; a decided tile always shows what is stored. */
export function mergeMarks(fresh: Marks, edits: Marks): Marks {
  const out: Marks = {};
  for (const [k, m] of Object.entries(fresh)) {
    const id = Number(k);
    const e = edits[id];
    out[id] = m.readOnly ? (e?.reclassified ? e : m) : e ?? m;
  }
  return out;
}

/** Include or Exclude a tile (a read-only one does not change). */
export function setDecision(marks: Marks, id: number, decision: Choice): Marks {
  const m = marks[id];
  if (!m || m.readOnly || m.decision === decision) return marks;
  return { ...marks, [id]: { decision, reason: m.reason } };
}

/** The flipper: Exclude → Include → Exclude. */
export function toggle(marks: Marks, id: number): Marks {
  const m = marks[id];
  if (!m) return marks;
  return setDecision(marks, id, m.decision === "exclude" ? "keep" : "exclude");
}

/** The reason picked in the viewer; only an excluded tile has one to pick. */
export function setReason(marks: Marks, id: number, reason: string): Marks {
  const m = marks[id];
  if (!m || m.readOnly || m.decision !== "exclude" || !reason || m.reason === reason) return marks;
  return { ...marks, [id]: { decision: "exclude", reason } };
}

/** A quick reason button (QUICK_REASONS) in the viewer: Exclude, for that reason, whether the
 *  tile was included or excluded for another reason. It saves like a flip (markTouched). */
export function quickMark(marks: Marks, id: number, reason: string): Marks {
  const m = marks[id];
  if (!m || m.readOnly || !reason || (m.decision === "exclude" && m.reason === reason)) return marks;
  return { ...marks, [id]: { decision: "exclude", reason } };
}

/** A change made full screen (Steve, 2026-10-21: "on the decided tab I should have the ability to
 *  open and reclassify an image"): on a decided tile it unlocks the stored decision and applies
 *  `op` (setDecision, setReason or quickMark) to it, marked `reclassified`; a change that ends
 *  where the stored decision is changes nothing. On an open tile it is `op` itself. The grid
 *  never reclassifies: a decided tile is changed only from the viewer. */
export function reclassify(marks: Marks, id: number, op: (m: Marks) => Marks): Marks {
  const m = marks[id];
  if (!m?.readOnly) return op(marks);
  const stored = { decision: m.decision, reason: m.reason };
  const unlocked = { ...marks, [id]: { ...stored, reclassified: true } };
  const next = op(unlocked);
  if (next === unlocked || sameState(next[id], stored)) return marks;
  return { ...next, [id]: { ...next[id], reclassified: true } };
}

// --- saving -------------------------------------------------------------------------------------

/** What the server holds or is being sent, for one tile. */
export interface Sent { decision: Choice; reason: string | null }

/** One tile's saving this visit. `seen` and `loaded`: shown long enough, and its image loaded
 *  (both needed before the default Exclude is saved). `touched`: the reviewer flipped it or
 *  picked its reason. `saved`: what the server holds from this visit; `sending`: in flight;
 *  `failed`: why the last send was refused (not sent again until a retry or a flip).
 *  `dropped`: its unsaved change was discarded (Auto-save off): its starting Exclude is not
 *  pending again until it is flipped or its page is loaded afresh. */
export interface SaveState {
  seen?: boolean;
  loaded?: boolean;
  touched?: boolean;
  dropped?: boolean;
  saved?: Sent;
  sending?: Sent;
  failed?: string;
}
export type SaveStates = Record<number, SaveState>;

/** Two states the server would store the same way (a keep has no reason). */
export const sameState = (a: Sent | null | undefined, b: Sent | null | undefined): boolean =>
  !!a && !!b && a.decision === b.decision && (a.decision === "keep" || (a.reason ?? "") === (b.reason ?? ""));

const patch = (states: SaveStates, id: number, p: Partial<SaveState>): SaveStates =>
  ({ ...states, [id]: { ...states[id], ...p } });

/** The tile has been in view long enough. */
export const markSeen = (states: SaveStates, id: number): SaveStates =>
  states[id]?.seen ? states : patch(states, id, { seen: true });

/** The tile's image loaded. */
export const markLoaded = (states: SaveStates, id: number): SaveStates =>
  states[id]?.loaded ? states : patch(states, id, { loaded: true });

/** The reviewer changed the tile (a flip or a reason): it saves whether shown or not, and a
 *  refused one is tried again. */
export const markTouched = (states: SaveStates, id: number): SaveStates =>
  patch(states, id, { touched: true, failed: undefined, dropped: undefined });

/** Retry a refused tile. */
export const retry = (states: SaveStates, id: number): SaveStates =>
  states[id]?.failed ? patch(states, id, { failed: undefined }) : states;

export type DueTile = Pick<FlaggedTile, "item_id" | "status">;

/** The decisions due to be sent now: open tiles whose current mark is not the one last saved,
 *  not in flight or refused, that the reviewer changed, or that were shown (seen AND loaded).
 *  An exclusion with no reason yet waits for one.
 *
 *  `manual` (Auto-save off): the same tiles are the unsaved changes that Save writes; a refused
 *  one is included (Save is its retry), and a starting Exclude carries NOTE_SAVED. */
export function dueDecisions(tiles: DueTile[], marks: Marks, states: SaveStates,
                             opts: { manual?: boolean } = {}): PhotoBatchDecision[] {
  const manual = !!opts.manual;
  const out: PhotoBatchDecision[] = [];
  const seen = new Set<number>();
  for (const t of tiles) {
    const m = marks[t.item_id];
    if ((t.status !== "open" && !m?.reclassified) || seen.has(t.item_id)) continue;
    seen.add(t.item_id);
    const s = states[t.item_id] ?? {};
    if (!m || m.readOnly || s.sending || (s.failed && !manual)) continue;
    if (sameState(m, s.saved)) continue;
    if (!s.touched && (s.dropped || !(s.seen && s.loaded))) continue;
    if (m.decision === "exclude" && !(m.reason ?? "").trim()) continue;
    const note = s.touched ? NOTE_FLIPPED : manual ? NOTE_SAVED : NOTE_SHOWN;
    out.push(m.decision === "exclude"
      ? { item_id: t.item_id, decision: "exclude", reason: m.reason!.trim(), note }
      : { item_id: t.item_id, decision: "keep", note });
  }
  return out;
}

/** What the timer, hiding the page and leaving it send by themselves: the due decisions with
 *  Auto-save on, nothing at all with it off. */
export const autoDue = (autosave: boolean, tiles: DueTile[], marks: Marks, states: SaveStates): PhotoBatchDecision[] =>
  (autosave ? dueDecisions(tiles, marks, states) : []);

/** Auto-save off: the unsaved changes, which "Save page" writes in one batch. */
export const pendingDecisions = (tiles: DueTile[], marks: Marks, states: SaveStates): PhotoBatchDecision[] =>
  dueDecisions(tiles, marks, states, { manual: true });

/** Auto-save off: one tile's own Save. Pressing it is seeing it, so a tile not yet counted as
 *  shown (or one whose change was discarded) is written too. */
export function tileDecision(tile: DueTile, marks: Marks, states: SaveStates): PhotoBatchDecision[] {
  const s = states[tile.item_id] ?? {};
  return dueDecisions([tile], marks, { ...states, [tile.item_id]: { ...s, seen: true, loaded: true, dropped: undefined } },
                      { manual: true });
}

/** Discard (or leaving with changes unsaved): the reviewer's flips on these tiles are undone and
 *  their starting Excludes stop being pending. What is saved, or in flight, stays. */
export function discardChanges(edits: Marks, states: SaveStates, ids: Iterable<number>): { edits: Marks; states: SaveStates } {
  const e = { ...edits };
  let s = states;
  for (const id of ids) {
    if (s[id]?.sending) continue;
    delete e[id];
    s = patch(s, id, { touched: undefined, failed: undefined, dropped: true });
  }
  return { edits: e, states: s };
}

/** A page loaded afresh: its discarded tiles count again once they are shown again. */
export function freshPage(states: SaveStates, ids: Iterable<number>): SaveStates {
  let s = states;
  for (const id of ids) {
    if (s[id]?.dropped) s = patch(s, id, { dropped: undefined, seen: undefined, loaded: undefined });
  }
  return s;
}

/** The POST body for decisions (null when there is nothing to send). */
export const batchBody = (decisions: PhotoBatchDecision[]): PhotoBatchBody | null =>
  decisions.length ? { decisions } : null;

const sentOf = (d: PhotoBatchDecision): Sent =>
  ({ decision: d.decision, reason: d.decision === "exclude" ? d.reason ?? null : null });

/** The decisions are on their way. */
export function beginSend(states: SaveStates, decisions: PhotoBatchDecision[]): SaveStates {
  let out = states;
  for (const d of decisions) out = patch(out, d.item_id, { sending: sentOf(d), failed: undefined });
  return out;
}

/** The server wrote them: each is now the saved state. */
export function sendDone(states: SaveStates, decisions: PhotoBatchDecision[]): SaveStates {
  let out = states;
  for (const d of decisions) out = patch(out, d.item_id, { saved: sentOf(d), sending: undefined, failed: undefined });
  return out;
}

/** The server refused the batch (whole or nothing): none of them is saved. */
export function sendFailed(states: SaveStates, decisions: PhotoBatchDecision[], why: string): SaveStates {
  let out = states;
  for (const d of decisions) out = patch(out, d.item_id, { sending: undefined, failed: why || "Not saved." });
  return out;
}

export type TileStatus = "decided" | "saving" | "saved" | "failed" | "needs-reason" | "unsaved" | "pending";

/** The tile's status mark. `manual` (Auto-save off): a tile holding an unsaved change (flipped,
 *  or shown with its starting Exclude) is "pending", marked "unsaved". */
export function tileStatus(mark: Mark | undefined, state: SaveState | undefined, manual = false): TileStatus {
  if (!mark || mark.readOnly) return "decided";
  const s = state ?? {};
  if (s.sending) return "saving";
  if (s.failed) return "failed";
  if (sameState(mark, s.saved)) return "saved";
  if (mark.decision === "exclude" && !(mark.reason ?? "").trim()) return "needs-reason";
  if (manual && (s.touched || (!s.dropped && s.seen && s.loaded))) return "pending";
  return "unsaved";
}

export const STATUS_TEXT: Record<TileStatus, string> = {
  decided: "",
  saving: "saving…",
  saved: "saved ✓",
  failed: "failed",
  "needs-reason": "pick a reason",
  unsaved: "",
  pending: "unsaved",
};

/** "Saved this visit: N excluded, M included": what the server holds from this visit, by each
 *  tile's latest saved state. */
export function visitTotals(states: SaveStates): { excluded: number; included: number } {
  let excluded = 0;
  let included = 0;
  for (const s of Object.values(states)) {
    if (s.saved?.decision === "exclude") excluded++;
    else if (s.saved?.decision === "keep") included++;
  }
  return { excluded, included };
}

export const visitLine = (t: { excluded: number; included: number }): string =>
  `Saved this visit: ${t.excluded.toLocaleString("en-US")} excluded, ${t.included.toLocaleString("en-US")} included`;

/** The offset Next goes to: in Open the tiles saved from this page drop out, so it starts that
 *  many earlier. */
export function nextOffset(status: "open" | "decided", offset: number, limit: number, savedOnPage: number): number {
  return offset + limit - (status === "open" ? Math.min(limit, Math.max(0, savedOnPage)) : 0);
}

// --- the viewer's words -------------------------------------------------------------------------

/** "iNaturalist photo 1000" / "Mushroom Observer photo 456" from a photo key. */
export function photoSourceText(photoKey: string): string {
  const [src, id] = photoKey.includes(":") ? [photoKey.slice(0, photoKey.indexOf(":")), photoKey.slice(photoKey.indexOf(":") + 1)]
    : ["", photoKey];
  if (src === "inat") return `iNaturalist photo ${id}`;
  if (src === "mo") return `Mushroom Observer photo ${id}`;
  return `photo ${photoKey}`;
}

/** "photo 2 of 5" (position counts from 0); "photo 2" when the count is unknown. */
export function photoOfText(position: number | null | undefined, count: number | null | undefined): string {
  const n = position == null ? "?" : String(position + 1);
  return count ? `photo ${n} of ${count}` : `photo ${n}`;
}

/** The licence as shown: its code ("CC-BY-NC"), else "all rights reserved" for one with none. */
export function licenceText(code: string | null | undefined, cls?: string | null): string | null {
  const c = (code ?? "").trim();
  if (c) return c.toUpperCase();
  return cls === "arr" ? "all rights reserved" : null;
}

/** A decided tile's stored decision in words. */
export function decidedText(t: Pick<FlaggedTile, "decision">, reasons: { code: string; label: string }[] = []): string | null {
  const d = t.decision;
  if (!d) return null;
  if (d.decision === "keep_photo") return "Decided: included";
  if (d.decision === "exclude_photo") {
    const why = d.reason ? reasons.find((r) => r.code === d.reason)?.label ?? d.reason : null;
    return `Decided: excluded${why ? ` (${why})` : ""}`;
  }
  return `Decided: ${d.decision}`;
}

export type OverlayTile = Pick<FlaggedTile, "label" | "link" | "record_key" | "photo_key" | "position" |
  "record_photos" | "flags" | "owner_login" | "license_code" | "license_class" | "status" | "decision">;

/** The viewer's overlay: the species name, the observation link, then its lines: source and id,
 *  photo n of N, each flag (the gate as "not a fungus, p(fungus)=…"), owner and licence, and a
 *  decided tile's stored decision. */
export function overlay(t: OverlayTile, reasons: { code: string; label: string }[] = []):
    { name: string; link: { href: string; text: string }; lines: string[] } {
  const lines = [`${photoSourceText(t.photo_key)} · ${photoOfText(t.position, t.record_photos)}`];
  for (const f of t.flags ?? []) lines.push(flagLabel(f));
  const lic = licenceText(t.license_code, t.license_class);
  const who = [t.owner_login ? `by ${t.owner_login}` : null, lic].filter(Boolean).join(" · ");
  if (who) lines.push(who);
  if (t.status === "decided") {
    const d = decidedText(t, reasons);
    if (d) lines.push(d);
  }
  return { name: t.label || "(no name)", link: { href: t.link, text: recordLabel(t.record_key) }, lines };
}

/** A reason code in words. */
export const reasonLabel = (code: string | null | undefined, reasons: { code: string; label: string }[]) =>
  (code ? reasons.find((r) => r.code === code)?.label ?? code : null);

// --- keys ---------------------------------------------------------------------------------------

export type GridAction =
  | { type: "move"; delta: number }
  | { type: "open" }
  | { type: "choose"; choice: Choice }
  | { type: "quick"; index: number }
  | { type: "step"; delta: 1 | -1 }
  | { type: "help" }
  | { type: "close" };

export interface GridKeyState {
  /** An input, textarea, select or editable element has focus: keys are its own. */
  typing: boolean;
  /** A button or link has focus: Enter is its own. */
  onControl?: boolean;
  lightbox: boolean;
  help?: boolean;
  /** Tiles per row, for ↑ / ↓. */
  columns: number;
}

export interface GridKeyPress { key: string; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean }

/** The keys. On the grid: arrows move between tiles, Enter opens the viewer, K includes and X
 *  excludes the focused tile (each saves), ? help. In the viewer: ← / → the page's tiles, K / X,
 *  Esc close. */
export function gridKey(e: GridKeyPress, s: GridKeyState): GridAction | null {
  if (e.ctrlKey || e.metaKey || e.altKey || s.typing) return null;
  const k = e.key;
  if (s.lightbox) {
    if (k === "Escape") return { type: "close" };
    if (k === "ArrowLeft") return { type: "step", delta: -1 };
    if (k === "ArrowRight") return { type: "step", delta: 1 };
    if (k === "k" || k === "K") return { type: "choose", choice: "keep" };
    if (k === "x" || k === "X") return { type: "choose", choice: "exclude" };
    if (/^[1-5]$/.test(k)) return { type: "quick", index: Number(k) - 1 };
    return null;
  }
  const cols = Math.max(1, Math.floor(s.columns) || 1);
  switch (k) {
    case "ArrowLeft": return { type: "move", delta: -1 };
    case "ArrowRight": return { type: "move", delta: 1 };
    case "ArrowUp": return { type: "move", delta: -cols };
    case "ArrowDown": return { type: "move", delta: cols };
    case "Enter": return s.onControl ? null : { type: "open" };
    case "k": case "K": return { type: "choose", choice: "keep" };
    case "x": case "X": return { type: "choose", choice: "exclude" };
    case "?": return { type: "help" };
    case "Escape": return s.help ? { type: "help" } : null;
    default: return null;
  }
}

/** Focus after a move, within [0, n), stopping at the ends. */
export function moveFocus(i: number, n: number, delta: number): number {
  if (n <= 0) return 0;
  return Math.min(n - 1, Math.max(0, i + delta));
}

/** The key help, as shown. */
export const SUMMARY_KEY_HELP: [string, string][] = [
  ["← → ↑ ↓", "move between photos"],
  ["Enter", "open the photo full screen"],
  ["K", "include the photo (saves with Auto-save on)"],
  ["X", "exclude the photo (saves with Auto-save on)"],
  ["← / → (full screen)", "previous / next photo on the page"],
  ["1–5 (full screen)", "exclude as Field slip, Microscopy, Spore print, UV, Chemical"],
  ["Esc", "close the full-screen photo"],
  ["?", "show or hide these keys"],
];
