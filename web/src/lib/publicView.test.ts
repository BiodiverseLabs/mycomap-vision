import { test } from "node:test";
import assert from "node:assert/strict";
import { headline, identifyGate, modelName, sendWaitingInLine } from "./publicView";
import type { Me, ScoreRun } from "./api";

test("a signed-out visitor is asked to sign in before picking photos, only when identifying needs it", () => {
  const me = (signin: Me["signin"], user: Me["user"]): Me => ({ signin, user });
  const someone = { id: "7", name: "Ann" };
  assert.equal(identifyGate(me("identify", null)), "sign-in");
  assert.equal(identifyGate(me("all", null)), "sign-in");
  assert.equal(identifyGate(me("identify", someone)), "ready");
  assert.equal(identifyGate(me("off", null)), "ready", "an open site needs no account");
  assert.equal(identifyGate(undefined), "checking");
  assert.equal(identifyGate(undefined, true), "ready", "unknown: the server decides");
});

test("visitors read a model's name, not its file name", () => {
  assert.equal(modelName("bioclip-2-ft-20261007-165400"),
    "BioCLIP 2, fine-tuned on DNA-verified records (October 2026)");
  assert.equal(modelName("bioclip-2"), "BioCLIP 2");
  assert.equal(modelName("bioclip-2-ft-sample"), "BioCLIP 2, fine-tuned on a sample (test model)");
  assert.equal(modelName("external:inat-cv"), "iNaturalist's computer vision");
  assert.equal(modelName("dinov3-l16-512"), "dinov3-l16-512");
});

const run = (id: number, comparison_id: string, backbone: string, method: string,
             species = 0.3): ScoreRun => ({
  id, comparison_id, backbone, method, cutoff: "2026-09-07", test_days: 28, n_reference: 100,
  n_test: 10, record_set: "s", species_top1: species, genus_top1: 0.7, family_top1: 0.8,
  species_top1_first_photo: 0.2, code_version: "x", created_at: "t",
});

test("the headline is the served model's newest comparison, with iNat on the same records", () => {
  const runs = [
    run(9, "new", "bioclip-2-ft-x", "species-mean"),
    run(8, "new", "bioclip-2-ft-x", "nearest", 0.345),
    run(7, "new", "external:inat-cv", "vision-max"),
    run(6, "new", "external:inat-cv", "combined-max", 0.4),
    run(5, "old", "bioclip-2-ft-x", "nearest", 0.2),
    run(4, "old", "bioclip-2", "nearest"),
  ];
  const h = headline(runs, ["bioclip-2-ft-x"])!;
  assert.equal(h.comparisonId, "new");
  assert.equal(h.ours.species_top1, 0.345);
  assert.equal(h.inat?.method, "combined-max", "iNat with location, as people use it");
  assert.equal(headline(runs.slice(0, 2), ["bioclip-2-ft-x"])!.inat, null, "no iNat run: none shown");
  assert.equal(headline(runs, ["not-served"]), null);
});

const answer = (status: number, retryAfter?: string) =>
  new Response("{}", { status, headers: retryAfter ? { "Retry-After": retryAfter } : {} });

test("a busy server is waited for in line instead of shown as an error", async () => {
  const replies = [answer(503, "10"), answer(503, "10"), answer(200)];
  const waits: number[][] = [];
  const slept: number[] = [];
  const res = await sendWaitingInLine(async () => replies.shift()!, (a, s) => waits.push([a, s]),
    8, async (ms) => { slept.push(ms); });
  assert.equal(res.status, 200);
  assert.deepEqual(waits, [[1, 10], [2, 10]]);
  assert.deepEqual(slept, [10_000, 10_000]);
});

test("a rate limit, a refusal or a 503 without Retry-After is not retried", async () => {
  for (const reply of [answer(429, "60"), answer(401), answer(503)]) {
    let sent = 0;
    const res = await sendWaitingInLine(async () => { sent++; return reply; }, () => {}, 8,
      async () => {});
    assert.equal(sent, 1);
    assert.equal(res.status, reply.status);
  }
});

test("the line is given up after its tries, and the busy answer is shown then", async () => {
  let sent = 0;
  const res = await sendWaitingInLine(async () => { sent++; return answer(503, "10"); }, () => {},
    3, async () => {});
  assert.equal(sent, 3);
  assert.equal(res.status, 503);
});
