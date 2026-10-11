// What the Identify page says about the not-a-fungus gate (fungusgate.py on the server).

import type { FindGate, GateVerdict, IdentifyResult, PhotoResult } from "./api";

export interface GateBanner {
  verdict: "unsure" | "not_fungus";
  title: string;
  body: string;
}

/** The banner above a result, or null when the find looks like a fungus (or the gate
 *  didn't run). */
export function gateBanner(r: Pick<IdentifyResult, "gate">): GateBanner | null {
  const g: FindGate | null | undefined = r.gate;
  if (!g || g.verdict === "fungus") return null;
  if (g.looks_like === "slime") {
    return {
      verdict: g.verdict,
      title: g.message ?? "This looks like a slime mould.",
      body:
        "Slime moulds aren't fungi, but MycoMap holds some DNA-verified ones; the closest are " +
        "below, without a stated chance.",
    };
  }
  if (g.verdict === "not_fungus") {
    return {
      verdict: "not_fungus",
      title: g.message ?? "This doesn't look like a fungus.",
      body:
        "MycoMap Vision only knows DNA-verified fungi, so it gives no name and no chance for " +
        "these photos. If it is a fungus, add a clear photo of the cap, the underside and the stem.",
    };
  }
  return {
    verdict: "unsure",
    title: g.message ?? "Not sure this is a fungus.",
    body:
      "No chance is stated: the names below are only the closest DNA-verified look-alikes, " +
      "and the stated chances hold only for photos of fungi.",
  };
}

/** True when this photo was left out of the answer as not a fungus. */
export function leftOut(p: Pick<PhotoResult, "gate">): boolean {
  return p.gate?.verdict === "not_fungus";
}

/** The short label under a photo's tile, or null for a photo that counted normally. */
export function photoGateLabel(p: Pick<PhotoResult, "gate">): string | null {
  if (p.gate?.verdict === "not_fungus") return "not a fungus";
  if (p.gate?.verdict === "unsure") return "fungus?";
  return null;
}

/** True when the page must show names without any chance (no %, no bars). */
export function chanceWithheld(r: Pick<IdentifyResult, "confidence_withheld" | "gate">): boolean {
  return !!r.confidence_withheld || (!!r.gate && r.gate.verdict !== "fungus");
}

/** The hints the page lists under a result: the server's, less the gate's own hint when
 *  the banner already says it (the API keeps it for callers without a banner). */
export function pageHints(r: Pick<IdentifyResult, "gate" | "hints">): string[] {
  const banner = gateBanner(r);
  if (!banner) return r.hints;
  const said = banner.title.replace(/\.$/, "");
  return r.hints.filter((h) => !h.startsWith(said));
}

export type NextSteps = { title: string; steps: string[] };

/** "{sequencing}" in a step is the link to MycoMap's free DNA sequencing. */
export const WHAT_NEXT: Record<GateVerdict, NextSteps> = {
  fungus: {
    title: "Suggestions, then specimens, then DNA",
    steps: [
      "Compare your find with the closest specimens above: cap, underside and stem.",
      "If the species is unclear, the genus often isn't. Go by the finest rank with a firm answer.",
      "DNA settles it. MycoMap offers {sequencing}; once validated, your find joins the reference set.",
    ],
  },
  unsure: {
    title: "Check it's a fungus, then compare, then DNA",
    steps: [
      "Make sure the photos show the fungus itself: the cap from above, the underside and the stem, close and sharp.",
      "Compare your find with the closest look-alikes above. They come with no chance, because the photos may not show a fungus.",
      "If it is a fungus, DNA settles it. MycoMap offers {sequencing}; once validated, your find joins the reference set.",
    ],
  },
  not_fungus: {
    title: "Nothing to compare yet",
    // The gate's notice above already says why: no name, no chance, only fungi known.
    steps: [
      "If it is a fungus, start over with clear, close photos of the cap from above, the underside and the stem.",
      "Sure it's a fungus? MycoMap offers {sequencing}; once validated, your find joins the reference set.",
    ],
  },
};

/**
 * The page-wide next steps follow the most hopeful result: names to compare when any
 * model's photos look like a fungus, look-alikes when the gate is only unsure, and none
 * when every result is "not a fungus". No gate (not deployed) counts as a fungus.
 */
/** Next steps for a find the gate says looks like a slime mould (gate.looks_like "slime"). */
export const WHAT_NEXT_SLIME: NextSteps = {
  title: "Looks like a slime mould: compare, then DNA",
  steps: [
    "Compare your find with the closest DNA-verified look-alikes above; the slime moulds among them are the ones to check.",
    "They come with no chance: slime moulds aren't fungi, and MycoMap holds only some DNA-verified ones.",
    "DNA settles it. MycoMap offers {sequencing}; once validated, your find joins the reference set.",
  ],
};

export function whatNext(results: Pick<IdentifyResult, "gate">[]): NextSteps {
  const verdicts = results.map((r) => r.gate?.verdict ?? "fungus");
  const kind: GateVerdict = verdicts.includes("fungus") || !verdicts.length ? "fungus"
    : verdicts.includes("unsure") ? "unsure" : "not_fungus";
  if (kind !== "fungus" && results.some((r) => r.gate?.looks_like === "slime")) return WHAT_NEXT_SLIME;
  return WHAT_NEXT[kind];
}
