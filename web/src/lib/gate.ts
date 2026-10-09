// What the Identify page says about the not-a-fungus gate (fungusgate.py on the server).

import type { FindGate, IdentifyResult, PhotoResult } from "./api";

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
