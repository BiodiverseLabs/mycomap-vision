/**
 * The specimen panels under the ID area of a result (Steve, 2026-10-18), in this order:
 *   1. Your collections: the signed-in person's own DNA-verified records, most alike first.
 *   2. The closest DNA-verified specimens from other collectors (the server leaves the
 *      person's own records out of `specimens`, so nothing shows twice).
 *   3. Best match per photo: each photo's single closest DNA-verified record.
 * The page holds no logic; it renders what these return.
 */
import type { IdentifyResult, Specimen } from "@/lib/api";
import type { Answer } from "@/lib/answer";

/** Where people connect their iNaturalist and Mushroom Observer accounts. */
export const CONNECT_URL = "https://mycomap.org/profile";

export type YoursPanel =
  | { kind: "none" }
  | { kind: "connect" }
  | { kind: "matches"; specimens: Specimen[]; records: number };

/** Panel 1. Nothing when signed out (no `yours`) or when none of the person's records are
 *  in the reference; one line when they have no account connected. */
export function yoursPanel(r: Pick<IdentifyResult, "yours">): YoursPanel {
  const y = r.yours;
  if (!y) return { kind: "none" };
  if (!y.linked) return { kind: "connect" };
  if (!y.specimens.length) return { kind: "none" };
  return { kind: "matches", specimens: y.specimens, records: y.records };
}

/** Panel 2's heading: "from other collectors" only when panel 1 is shown beside it. */
export function othersHeading(r: Pick<IdentifyResult, "yours">, found: string): string {
  return yoursPanel(r).kind === "matches"
    ? `Closest DNA-verified ${found} from other collectors`
    : `Closest DNA-verified ${found}`;
}

/** Panel 3: one row per photo scored, with its closest record. None for a one-photo find
 *  (it would repeat the combined lists) and none for a photo left out (nothing scored). */
export function bestPerPhoto(r: Pick<IdentifyResult, "per_photo">): { photo: number; specimen: Specimen }[] {
  if (r.per_photo.length < 2) return [];
  return r.per_photo.flatMap((p) => (p.specimens[0] ? [{ photo: p.photo, specimen: p.specimens[0] }] : []));
}

/** The line under panel 1's heading. */
export function yoursLine(records: number, shown: number): string {
  const n = records.toLocaleString("en-US");
  return records <= shown
    ? `Your ${n} DNA-verified record${records === 1 ? "" : "s"} in Vision's reference, most alike first.`
    : `The ${shown} most alike of your ${n} DNA-verified records in Vision's reference.`;
}

/** The species a card is tagged "same species as the answer" against: the headline, only when
 *  it is a settled species with a stated chance (not a genus, not a look-alike). */
export function answerSpecies(a: Answer | null): string | null {
  const h = a?.headline;
  return h && h.rank === "species" && !a.lookAlike ? h.name : null;
}
