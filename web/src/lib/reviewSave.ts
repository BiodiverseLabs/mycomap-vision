// The Human Feedback review pages' Auto-save setting (Steve, 2026-10-21): a toggle at the top of
// the photo review, OFF by default and remembered per viewer.
//
// - On: the pages save as they always have (the Summary saves a tile once it has been shown and
//   each flip at once; the Full view saves a record with its own Save).
// - Off: nothing is written until the reviewer saves. Flips and the starting Excludes are held as
//   unsaved changes, marked on each photo; each photo (Summary) or record (Full) has its own Save,
//   and "Save page (N changes)" writes them all. Leaving the page, paging, or changing the tab, a
//   filter or the queue with unsaved changes asks first; Discard drops them.
//
// The label review and the other queues have no Auto-save: they write only when a Save (or an
// outcome) button is pressed, which is already the Off behaviour.
//
// The rules live here as plain functions so they can be tested (reviewSave.test.ts); the
// Summary's own pieces are in photoSummary.ts.

import type { PhotoReviewDecisionBody, ReviewPhoto, ReviewRecord } from "./api";
import { saveBody, storedChoice, type PhotoChoices } from "./photoReview";

type Store = Pick<Storage, "getItem" | "setItem"> | null | undefined;

export const AUTOSAVE_STORAGE_KEY = "mycomap-vision.review.autosave";

/** The viewer's Auto-save: off unless they turned it on ("on" stored). Storage unavailable (a
 *  private window, blocked site data: the getter itself may throw) gives off. */
export function readAutosave(storage: () => Store): boolean {
  try {
    return storage()?.getItem(AUTOSAVE_STORAGE_KEY) === "on";
  } catch {
    return false;
  }
}

/** Remembers the viewer's Auto-save; does nothing when storage is unavailable. */
export function writeAutosave(storage: () => Store, on: boolean): void {
  try {
    storage()?.setItem(AUTOSAVE_STORAGE_KEY, on ? "on" : "off");
  } catch {
    // Not remembered: the page still works with the setting picked.
  }
}

const changes = (n: number) => `${n.toLocaleString("en-US")} unsaved change${n === 1 ? "" : "s"}`;

/** The page-level button. */
export const savePageLabel = (n: number): string => `Save page (${n.toLocaleString("en-US")} change${n === 1 ? "" : "s"})`;

/** The line beside it. */
export const unsavedLine = (n: number): string => (n > 0 ? changes(n) : "No unsaved changes");

/** The question asked before unsaved changes are left behind. */
export const leaveText = (n: number): string =>
  `You have ${changes(n)} on this page. Leave without saving them?`;

/** The question asked when Auto-save is turned on with changes waiting. */
export const autosaveOnText = (n: number): string =>
  `Turn on Auto-save? Your ${changes(n)} will be saved now.`;

/** Whether to go ahead with leaving (paging, a tab, a filter, another queue): yes when nothing
 *  is unsaved, without asking; otherwise only when the reviewer says so. */
export function confirmLeave(unsaved: number, ask: (message: string) => boolean): boolean {
  if (!(unsaved > 0)) return true;
  return ask(leaveText(unsaved));
}

/** The browser's own "Leave site?" (beforeunload): asked only while changes are unsaved. */
export function guardUnload(e: { preventDefault: () => void; returnValue?: unknown }, unsaved: number): boolean {
  if (!(unsaved > 0)) return false;
  e.preventDefault();
  e.returnValue = "";               // older browsers ask only when this is set
  return true;
}

/** Unsaved changes by source (the photo review's view), so the queue picker above it can ask
 *  before closing it. One per page; the sources clear themselves when they go. */
export function createUnsaved() {
  const by = new Map<string, number>();
  return {
    set(source: string, n: number) {
      if (n > 0) by.set(source, n);
      else by.delete(source);
    },
    total(): number {
      let n = 0;
      for (const v of by.values()) n += v;
      return n;
    },
  };
}
export const reviewUnsaved = createUnsaved();

// --- the Full (record) view -----------------------------------------------------------------------

type RecordPhotos = Pick<ReviewRecord, "photos">;

/** The photos of a record whose choice is not what the server holds: a choice on an open photo
 *  (a suggested Exclude too: it is written by Save), or a decided photo changed. A photo with no
 *  choice is not sent, so it is no change. */
export function changedPhotos(record: RecordPhotos, choices: PhotoChoices): ReviewPhoto[] {
  return record.photos.filter((p) => {
    const c = choices[p.photo_key];
    if (!c) return false;
    const s = storedChoice(p);
    if (!s) return true;
    if (c.decision !== s) return true;
    return c.decision === "exclude" && (c.reason ?? "") !== (p.decision?.reason ?? "");
  });
}

export interface PageSave { key: string; index: number; body: PhotoReviewDecisionBody }

/** What "Save page" writes in the Full view: every record not yet saved with at least one
 *  change, each through its own record route (a record is whole or nothing). A record that
 *  cannot be saved yet (a reason missing) is held back with why. `changes`: the photos changed. */
export function pageSaves(records: (RecordPhotos & Pick<ReviewRecord, "record_key">)[],
                          choices: Record<string, PhotoChoices>, opts: {
                            saved?: Record<string, unknown>; busy?: Record<string, boolean>;
                            notes?: Record<string, string>; reasonCodes?: string[];
                          } = {}): { saves: PageSave[]; blocked: { key: string; error: string }[]; changes: number } {
  const saves: PageSave[] = [];
  const blocked: { key: string; error: string }[] = [];
  let n = 0;
  records.forEach((r, index) => {
    const key = r.record_key;
    if (opts.saved?.[key] || opts.busy?.[key]) return;
    const mine = choices[key] ?? {};
    const changed = changedPhotos(r, mine).length;
    if (!changed) return;
    n += changed;
    const b = saveBody(r, mine, opts.notes?.[key], opts.reasonCodes);
    if ("error" in b) blocked.push({ key, error: b.error });
    else saves.push({ key, index, body: b.body });
  });
  return { saves, blocked, changes: n };
}
