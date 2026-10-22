import { test } from "node:test";
import assert from "node:assert/strict";
import { coverageWords, headline, headlineModel, identifyGate, LIKELY_COPY, longDate, modelName, recentWeekLine, releaseNote, sendWaitingInLine, shareWords, showsMembersPages } from "./publicView";
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

test("a likely list's coverage reads as a plain frequency", () => {
  assert.equal(coverageWords(0.9), "about 9 times in 10");
  assert.equal(coverageWords(0.5), "about half the time");
  assert.equal(coverageWords(0.55), "about 55% of the time");
});

test("the list's sentence states the tested coverage for its rank", () => {
  assert.equal(LIKELY_COPY.coverage("genus", 0.9),
    "The right genus is in this list about 9 times in 10, in tests on the newest DNA-verified records.");
  assert.match(LIKELY_COPY.coverage("species", 0.55), /about 55% of the time/);
});

test("visitors read a model's name, its image size and its training data, not its file name", () => {
  // The site's model and the paper's headline model (Steve, 2026-10-21).
  assert.equal(modelName("bioclip-2-336-ft-20261018-171420"),
    "BioCLIP 2, 336 px, fine-tuned (public + non-public training data)");
  assert.equal(modelName("bioclip-2-336-ft-20261020-173106"),
    "BioCLIP 2, 336 px, fine-tuned (open data only)");
  assert.equal(modelName("bioclip-2-ft-20261013-003428"),
    "BioCLIP 2, 224 px, fine-tuned (public + non-public training data)");
  assert.equal(modelName("bioclip-2-ft-20261014-003539"), "BioCLIP 2, 224 px, fine-tuned (open data only)");
  assert.equal(modelName("bioclip-2-ft-20261007-165400"),
    "BioCLIP 2, 224 px, fine-tuned (public + non-public training data)");
  assert.equal(modelName("bioclip-2"), "BioCLIP 2, 224 px, frozen");
  assert.equal(modelName("bioclip-2-336"), "BioCLIP 2, 336 px, frozen");
  assert.equal(modelName("bioclip-2-ft-sample"), "BioCLIP 2, 224 px, fine-tuned on a sample (test model)");
  assert.equal(modelName("external:inat-cv"), "iNaturalist's computer vision");
  assert.equal(modelName("external:df20-vit-l384"),
    "Danish Fungi 2020 ViT-L, 384 px (Picek et al., Danish records)");
  assert.equal(modelName("external:fungitastic-beit-b384"),
    "FungiTastic BEiT-B, 384 px (Picek et al., Danish records)");
  assert.equal(modelName("dinov3-l16-512"), "DINOv3 large, 512 px, frozen");
  assert.equal(modelName("eva02-l14-448"), "EVA-02 large, 448 px, frozen");
  assert.equal(modelName("some-new-model"), "some-new-model");
  assert.equal(modelName("picek-fungitastic-beit-b384-20261012-080000"),
    "Danish Fungi method (BEiT-B, 384 px, Picek et al.), trained on our DNA-verified records");
  assert.equal(modelName("picek-fungitastic-beit-b384-smoke"),
    "Danish Fungi method (BEiT-B, 384 px, Picek et al.), trained on our DNA-verified records (test model)");
  assert.equal(modelName("picek-unknown-preset-20261012-080000"),
    "picek-unknown-preset-20261012-080000");
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

test("members-only pages show only to signed-in members", () => {
  const me = (signin: Me["signin"], user: Me["user"]): Me => ({ signin, user });
  const someone = { id: "7", name: "Ann" };
  assert.equal(showsMembersPages(me("identify", null)), false, "open site, signed out");
  assert.equal(showsMembersPages(me("all", null)), false);
  assert.equal(showsMembersPages(me("identify", someone)), true);
  assert.equal(showsMembersPages(me("all", someone)), true);
  assert.equal(showsMembersPages(me("off", null)), true, "local site without sign-in");
  assert.equal(showsMembersPages(undefined), false, "not known yet");
});

test("dates read as written, with no time zone shift", () => {
  assert.equal(longDate("2026-09-07"), "7 September 2026");
  assert.equal(longDate("2026-10-06T23:30:00+00:00"), "6 October 2026");
  assert.equal(longDate("not a date"), "not a date");
});

test("the provisional share is said from the data, never hard-coded as half", () => {
  assert.equal(shareWords(10864, 19208), "More than half");
  assert.equal(shareWords(50, 100), "About half");
  assert.equal(shareWords(40, 100), "About 40%");
  assert.equal(shareWords(99, 100), "Nearly all");
  assert.equal(shareWords(0, 0), "None");
});

test("a model's release note names it MycoMap Vision and says what it learned from", () => {
  assert.equal(
    releaseNote({ backbone: "bioclip-2-ft-20261007-165400", embedded_photos: 152915,
                  trained_through: "2026-09-07" }),
    "MycoMap Vision · October 2026 model. Names learned from DNA-verified " +
    "records validated up to 7 September 2026. Compares your photos with 152,915 reference photos.");
  assert.equal(releaseNote({ backbone: "bioclip-2", embedded_photos: 10, trained_through: null }),
    "MycoMap Vision. Compares your photos with 10 reference photos.");
});

test("the newest week's count is said only when there is one", () => {
  assert.equal(recentWeekLine({ records: 286, through: "2026-10-06" }),
    "286 North American records were DNA-verified in the week to 6 October 2026.");
  assert.equal(recentWeekLine({ records: 1, through: "2026-10-06" }),
    "1 North American record was DNA-verified in the week to 6 October 2026.");
  assert.equal(recentWeekLine({ records: 0, through: "2026-10-06" }), null);
  assert.equal(recentWeekLine(null), null);
  assert.equal(recentWeekLine(undefined), null, "an older server sends no count");
});

test("the headline names the method its numbers come from: the served default", () => {
  const runs = [
    run(9, "new", "bioclip-2-ft-x", "nearest", 0.30),
    run(8, "new", "bioclip-2-ft-x", "nearest+mean+place", 0.41),
  ];
  const h = headline(runs, ["bioclip-2-ft-x"], "nearest+mean+place")!;
  assert.equal(h.ours.species_top1, 0.41);
  assert.match(headlineModel(h), /closest specimens \+ species average \+ place and date$/);
  assert.match(headlineModel(headline(runs, ["bioclip-2-ft-x"], "nearest")!), /, closest specimens?$/);
});
