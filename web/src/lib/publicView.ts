// What the public pages show: plain model names, the served model's latest measured
// results, and waiting in line instead of an error when the server is busy.

import type { ScoreRun } from "./api";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"];
const BASE_NAME: Record<string, string> = { "bioclip-2": "BioCLIP 2" };

/** A model as a visitor should read it. Fine-tuned models are named `<base>-ft-<YYYYMMDD>-…`
 *  (finetune.py); the internal name stays available for a tooltip. */
export function modelName(backbone: string): string {
  if (backbone === "external:inat-cv") return "iNaturalist's computer vision";
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
