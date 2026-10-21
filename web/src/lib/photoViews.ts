import type { PhotoViewDecision, PhotoViewTile, PhotoViewsPage } from "./api";

// Photo views (Steve, 2026-10-21): the reviewer confirms or corrects what each photo shows.
// Each tile starts with the tagger's suggestions switched on; tapping a tag toggles it; a
// photo is saved with its whole tag set (Save on the tile, or Save page for every open tile
// on the page). "No specimen" sends the photo to the exclusion review instead. Nothing is
// written until a Save. The page: pages/PhotoViews.tsx.

export const PHOTO_VIEW_TYPE = "photo_view";
export const VIEWS_PAGE_SIZE = 60;

/** A tile's choice: the views switched on, or "shows no specimen". */
export interface ViewChoice { views: string[]; noSpecimen: boolean }
export type ViewChoices = Record<number, ViewChoice>;
type Views = PhotoViewsPage["views"];

const ordered = (codes: Iterable<string>, views: Views): string[] => {
  const on = new Set(codes);
  return views.map((v) => v.code).filter((c) => on.has(c));
};

/** What the server holds for the tile, or null while it is undecided. */
export function savedChoice(t: Pick<PhotoViewTile, "decision">, views: Views): ViewChoice | null {
  const d = t.decision;
  if (!d) return null;
  if (d.decision === "no_specimen") return { views: [], noSpecimen: true };
  if (d.decision === "confirm_views") return { views: ordered(d.views ?? [], views), noSpecimen: false };
  return null;
}

/** How a tile starts: its saved choice when decided, else the tagger's suggestions on. */
export function startChoice(t: Pick<PhotoViewTile, "decision" | "suggested">, views: Views): ViewChoice {
  return savedChoice(t, views) ?? { views: ordered(t.suggested.map((s) => s.view), views), noSpecimen: false };
}

export const choiceOf = (choices: ViewChoices, t: PhotoViewTile, views: Views): ViewChoice =>
  choices[t.item_id] ?? startChoice(t, views);

/** Toggle one view on the tile (and clear "no specimen"). */
export function toggleView(choices: ViewChoices, t: PhotoViewTile, code: string, views: Views): ViewChoices {
  if (!views.some((v) => v.code === code)) return choices;
  const c = choiceOf(choices, t, views);
  const on = new Set(c.views);
  if (on.has(code)) on.delete(code); else on.add(code);
  return { ...choices, [t.item_id]: { views: ordered(on, views), noSpecimen: false } };
}

/** "Shows no specimen" on or off; on clears the views, off brings the suggestions back. */
export function setNoSpecimen(choices: ViewChoices, t: PhotoViewTile, on: boolean, views: Views): ViewChoices {
  return { ...choices, [t.item_id]: on ? { views: [], noSpecimen: true }
    : { views: ordered(t.suggested.map((s) => s.view), views), noSpecimen: false } };
}

const same = (a: ViewChoice | null, b: ViewChoice | null) =>
  !!a && !!b && a.noSpecimen === b.noSpecimen && a.views.join(",") === b.views.join(",");

/** A tile still to save: open and not saved this visit, or its choice differs from what is
 *  saved (on the server, or this visit). */
export function needsSave(t: PhotoViewTile, choices: ViewChoices, saved: ViewChoices, views: Views): boolean {
  const now = choiceOf(choices, t, views);
  const held = saved[t.item_id] ?? savedChoice(t, views);
  return !same(now, held);
}

/** The decisions a Save writes: for `ids` (one tile), or every tile still to save. */
export function batchFor(tiles: PhotoViewTile[], choices: ViewChoices, saved: ViewChoices, views: Views,
                         ids?: number[]): PhotoViewDecision[] {
  const want = ids ? new Set(ids) : null;
  return tiles
    .filter((t) => (want ? want.has(t.item_id) : true) && needsSave(t, choices, saved, views))
    .map((t) => {
      const c = choiceOf(choices, t, views);
      return c.noSpecimen ? { item_id: t.item_id, no_specimen: true as const } : { item_id: t.item_id, views: c.views };
    });
}

/** After a successful save: what the server now holds for each sent tile. */
export function markSaved(saved: ViewChoices, sent: PhotoViewDecision[]): ViewChoices {
  const out = { ...saved };
  for (const d of sent) {
    out[d.item_id] = "no_specimen" in d ? { views: [], noSpecimen: true } : { views: [...d.views], noSpecimen: false };
  }
  return out;
}

/** The tags a tile shows: every suggestion (on or off) and every view switched on, in the
 *  views' order, each with whether it is on and whether the tagger suggested it. */
export function chips(t: PhotoViewTile, c: ViewChoice, views: Views): { code: string; label: string; on: boolean; suggested: boolean }[] {
  const sug = new Set(t.suggested.map((s) => s.view));
  const on = new Set(c.views);
  return views.filter((v) => sug.has(v.code) || on.has(v.code))
    .map((v) => ({ code: v.code, label: v.label, on: on.has(v.code), suggested: sug.has(v.code) }));
}

/** The page's one line: "3 to save" / "All saved". */
export function saveLine(n: number): string {
  return n === 0 ? "All saved" : `${n} to save`;
}

/** Leaving the page or paging with unsaved tiles asks first. */
export const LEAVE_PROMPT = "Some photos on this page are not saved. Leave without saving them?";
