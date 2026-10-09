// What the public pages show: plain model names, the served model's latest measured
// results, and waiting in line instead of an error when the server is busy.

import type { Me, ScoreRun } from "./api";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"];
const BASE_NAME: Record<string, string> = { "bioclip-2": "BioCLIP 2" };
/** The Picek group's classifier recipe replicated on our records (retrain.py), named
 *  `picek-<preset>-<run>`: the architecture each preset trains. */
const PICEK_NAME: Record<string, string> = {
  "fungitastic-beit-b384": "BEiT",
  "fungitastic-beit-b224": "BEiT at 224 px",
  "vit-b384-ce": "ViT-B",
  "df20-vit-l384": "ViT-L",
};
/** Published classifiers run as outside baselines (published.py): the authors' models,
 *  trained on Danish records, never retrained by us. */
const EXTERNAL_NAME: Record<string, string> = {
  "external:fungitastic-beit-b384": "FungiTastic BEiT-B (Picek et al., Danish records)",
  "external:fungitastic-vit-b384": "FungiTastic ViT-B (Picek et al., Danish records)",
  "external:df20-vit-l384": "Danish Fungi 2020 ViT-L (Picek et al., Danish records)",
};

/** A model as a visitor should read it. Fine-tuned models are named `<base>-ft-<YYYYMMDD>-…`
 *  (finetune.py); the internal name stays available for a tooltip. */
export function modelName(backbone: string): string {
  if (backbone === "external:inat-cv") return "iNaturalist's computer vision";
  const picek = /^picek-(.+?)-(\d{8}-\d{6}|smoke)$/.exec(backbone);
  if (picek && PICEK_NAME[picek[1]]) {
    const test = picek[2] === "smoke" ? " (test model)" : "";
    return `Danish Fungi method (${PICEK_NAME[picek[1]]}, Picek et al.), trained on our ` +
      `DNA-verified records${test}`;
  }
  if (backbone in EXTERNAL_NAME) return EXTERNAL_NAME[backbone];
  const ft = /^(.+)-ft-(\d{4})(\d{2})\d{2}-\d+$/.exec(backbone);
  if (ft) {
    const base = BASE_NAME[ft[1]] ?? ft[1];
    return `${base}, fine-tuned on DNA-verified records (${MONTHS[Number(ft[3]) - 1]} ${ft[2]})`;
  }
  if (backbone.endsWith("-ft-sample")) {
    const base = backbone.slice(0, -"-ft-sample".length);
    return `${BASE_NAME[base] ?? base}, fine-tuned on a sample (test model)`;
  }
  return BASE_NAME[backbone] ?? backbone;
}

export interface Headline {
  comparisonId: string;
  cutoff: string;
  nTest: number;
  nReference: number;
  ours: ScoreRun;
  /** iNaturalist's vision on the same records, with location when it was given. */
  inat: ScoreRun | null;
}

/** The newest comparison that scored a served model with `method`, and iNat on the same
 *  records when that comparison has it. Runs come newest first (the scoreboard's order). */
export function headline(runs: ScoreRun[], served: string[], method = "nearest"): Headline | null {
  const ours = runs.find((r) => served.includes(r.backbone) && r.method === method);
  if (!ours) return null;
  const same = runs.filter((r) => r.comparison_id === ours.comparison_id);
  const inat = same.find((r) => r.backbone === "external:inat-cv" && r.method === "combined-max")
    ?? same.find((r) => r.backbone === "external:inat-cv") ?? null;
  return { comparisonId: ours.comparison_id, cutoff: ours.cutoff, nTest: ours.n_test,
           nReference: ours.n_reference, ours, inat };
}

/** The likely list's wording, in one place: Steve has not ruled on it yet. */
export const LIKELY_COPY = {
  coverage: (rank: string, coverage: number) =>
    `The right ${rank} is in this list ${coverageWords(coverage)}, in tests on the newest ` +
    "DNA-verified records.",
  /** Shown under the species list: the stated coverage counts these as misses. */
  noReference: "A species with no DNA-verified reference yet can't appear in any list.",
  capped: "More names qualified than are shown.",
  alsoPossible: "Also possible, less likely:",
};

/** How often a likely list held the right name, in words. */
export function coverageWords(coverage: number): string {
  const tenths = Math.round(coverage * 10);
  if (coverage === 0.5) return "about half the time";
  if (Math.abs(coverage * 10 - tenths) < 1e-9) return `about ${tenths} times in 10`;
  return `about ${Math.round(coverage * 100)}% of the time`;
}

/** Whether this visitor can identify now. "sign-in": the site asks for a mycomap.org
 *  account to identify and they have none yet, so the page asks them to sign in before they
 *  pick photos (a sign-in leaves the page, and the photos would be lost). When who they are
 *  can't be told, the server decides (it answers 401, and the page offers sign-in then). */
export type IdentifyGate = "checking" | "ready" | "sign-in";

export function identifyGate(me: Me | null | undefined, failed = false): IdentifyGate {
  if (failed) return "ready";
  if (!me) return "checking";
  return me.signin === "off" || me.user ? "ready" : "sign-in";
}

/** Send, and when the server says it is busy (503 with Retry-After), wait in line and send
 *  again, up to `tries` times; `onWait(try, seconds)` lets the page say so. Anything else
 *  (a refusal, a rate limit, an answer) is returned as it came. */
export async function sendWaitingInLine(
  send: () => Promise<Response>,
  onWait: (attempt: number, seconds: number) => void = () => {},
  tries = 8,
  sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<Response> {
  for (let attempt = 1; ; attempt++) {
    const res = await send();
    const after = res.headers.get("Retry-After");
    if (res.status !== 503 || !after || attempt >= tries) return res;
    const seconds = Math.min(30, Math.max(2, Number(after) || 10));
    onWait(attempt, seconds);
    await sleep(seconds * 1000);
  }
}

/** The Paper in Progress tab is for signed-in mycomap.org members only (Steve, 2026-10-09);
 *  the server refuses the draft to anyone else. An open (sign-in off) site shows it. */
export function showsPaper(me: Me | null | undefined): boolean {
  return !!me && (me.signin === "off" || !!me.user);
}
