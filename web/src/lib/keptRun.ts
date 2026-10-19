// The run behind an answer to an iNat or MO observation (POST /api/identify/observation
// `answered`): Vision runs an observation once and keeps the answer for everyone who asks
// again (the Identify page, mycomap.org's Validate Records, the app) until someone presses
// Run again (Steve, 2026-10-19). The page says when it was run, who asked, and whether
// Vision's reference set has changed since.

/** When, who asked and under which answers version a kept answer was given. */
export interface Answered {
  /** UTC, ISO 8601. */
  at: string;
  /** "site" | "mycomap.org" | "app" | "api"; null for an answer kept before 2026-10-19. */
  by: string | null;
  answers_version: string;
  /** Vision still answers from what this answer was given from. */
  current: boolean;
}

const ASKED: Record<string, string> = {
  site: "on this site",
  "mycomap.org": "from mycomap.org",
  app: "from the MycoMap app",
  api: "through the Vision API",
};

/** Who asked for the run, in words ("" when not known). */
export const askedBy = (by: string | null | undefined) => (by ? ASKED[by] ?? "" : "");

/** The run's line: when and who, then what keeping it means for this answer. */
export function runDetails(answered: Answered, cached: boolean, when: (iso: string) => string): {
  ran: string;
  note: string | null;
} {
  const who = askedBy(answered.by);
  const ran = `Run ${cached ? when(answered.at) : "just now"}${who ? ` ${who}` : ""}.`;
  if (!cached) return { ran, note: "Kept: opening this observation again shows this answer until someone runs it again." };
  return {
    ran,
    note: answered.current
      ? "Kept from that run, nothing was run again. Vision's model and reference set haven't changed since."
      : "Kept from that run, nothing was run again. Vision's reference set has changed since, so running it again may answer differently.",
  };
}

/** A date and time in the reader's own time zone. */
export const localWhen = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
