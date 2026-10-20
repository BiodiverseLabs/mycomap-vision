// Every run has its own link (Steve, 2026-10-20): when the Identify page gets an answer, the
// server keeps the run and the page's address becomes /r/<id>. Anyone given the link can open
// it, signed in or not; nobody else can find it (a 96-bit id, not indexed, listed nowhere).
// The server: src/mycomap_vision/runs.py.

/** A run's id: 16 url-safe characters. */
export const RUN_ID = /^[A-Za-z0-9_-]{16}$/;

export const runPath = (id: string) => `/r/${id}`;

/** The run id in a /r/<id> path, or null. */
export function runIdFrom(path: string): string | null {
  const m = /^\/r\/([^/?#]+)$/.exec(path);
  return m && RUN_ID.test(m[1]) ? m[1] : null;
}

export const isRunPath = (path: string) => runIdFrom(path) != null;

/** Where a run's own photo copy is served (1024 px, no camera details). */
export const runPhotoUrl = (id: string, n: number) => `/api/runs/${id}/photos/${n}.jpg`;

/** Runs answered in this tab: their /r/<id> stays on the Identify page that made them, with
 *  its photos and fields, rather than reopening as a link. */
export const liveRuns = new Set<string>();

/** The line under Copy link: what is kept, and who can open it. No expiry (Steve). */
export function keptLine(kind: "photos" | "observation"): string {
  return kind === "photos"
    ? "This result has its own link. Vision keeps it, with smaller copies of your photos (no camera "
      + "details or exact place), until you delete it. Only people you give the link to can open it."
    : "This result has its own link. Only people you give the link to can open it.";
}

/** One address per photo the answer scored, in its order: a photos run's own copies, or an
 *  observation's photos at its source ("" where the source's photo may not be shown here). */
export function runPhotos(run: { id: string; kind: "photos" | "observation"; photos: number;
                                 observation?: { photos_shown: (string | null)[] } | null }): string[] {
  if (run.kind === "photos") return Array.from({ length: run.photos }, (_, n) => runPhotoUrl(run.id, n));
  return (run.observation?.photos_shown ?? []).map((u) => u ?? "");
}
