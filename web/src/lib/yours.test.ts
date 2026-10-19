import { test } from "node:test";
import assert from "node:assert/strict";
import { answerSpecies, bestPerPhoto, othersHeading, yoursLine, yoursPanel } from "./yours";
import type { IdentifyResult, PhotoResult, Specimen } from "./api";
import type { Answer } from "./answer";

const spec = (id: string, extra: Partial<Specimen> = {}): Specimen => ({
  observation_id: id, species: "A x", genus: "A", family: "F", similarity: 0.5, matched_query_photo: 0,
  photo_url: null, photo_owner: null, photo_withheld: false, inat_url: null, species_url: null, ...extra,
});
const photo = (n: number, specimens: Specimen[]): PhotoResult =>
  ({ photo: n, ranks: { family: [], genus: [], species: [] }, overall_top: { family: null, genus: null, species: null }, specimens });

test("signed out (no `yours`): no panel at all, and today's heading", () => {
  assert.deepEqual(yoursPanel({}), { kind: "none" });
  assert.equal(othersHeading({}, "specimens"), "Closest DNA-verified specimens");
});

test("signed in with nothing connected: the one line, and today's heading", () => {
  const r = { yours: { linked: false, records: 0, specimens: [] } };
  assert.deepEqual(yoursPanel(r), { kind: "connect" });
  assert.equal(othersHeading(r, "specimens"), "Closest DNA-verified specimens");
});

test("connected but none of your records in the reference: nothing", () => {
  const r = { yours: { linked: true, records: 0, specimens: [] } };
  assert.deepEqual(yoursPanel(r), { kind: "none" });
  assert.equal(othersHeading(r, "specimens"), "Closest DNA-verified specimens");
});

test("your records: the panel, in the order sent, and list 2 is 'from other collectors'", () => {
  const mine = [spec("2", { similarity: 0.8 }), spec("1", { similarity: 0.6 })];
  const r = { yours: { linked: true, records: 37, specimens: mine } };
  assert.deepEqual(yoursPanel(r), { kind: "matches", specimens: mine, records: 37 });
  assert.equal(othersHeading(r, "look-alikes"), "Closest DNA-verified look-alikes from other collectors");
});

test("the line under the heading says how many of how many", () => {
  assert.equal(yoursLine(37, 6), "The 6 most alike of your 37 DNA-verified records in Vision's reference.");
  assert.equal(yoursLine(2, 2), "Your 2 DNA-verified records in Vision's reference, most alike first.");
  assert.equal(yoursLine(1, 1), "Your 1 DNA-verified record in Vision's reference, most alike first.");
  assert.equal(yoursLine(1240, 6), "The 6 most alike of your 1,240 DNA-verified records in Vision's reference.");
});

test("best match per photo: each photo's first specimen, hidden for a one-photo find", () => {
  const two: Pick<IdentifyResult, "per_photo"> = { per_photo: [photo(0, [spec("a"), spec("b")]), photo(1, [spec("c")])] };
  assert.deepEqual(bestPerPhoto(two).map((x) => [x.photo, x.specimen.observation_id]), [[0, "a"], [1, "c"]]);
  assert.deepEqual(bestPerPhoto({ per_photo: [photo(0, [spec("a")])] }), []);
});

test("best match per photo: a photo left out of the answer (no matches) has no row", () => {
  const r = { per_photo: [photo(0, [spec("a")]), photo(1, []), photo(2, [spec("c")])] };
  assert.deepEqual(bestPerPhoto(r).map((x) => x.photo), [0, 2]);
});

test("'same species as the answer' only against a settled species headline", () => {
  const named = (rank: "family" | "genus" | "species", name: string, lookAlike = false): Answer =>
    ({ headline: { rank, name, confidence: 0.9 }, lookAlike, above: [], inPlay: null } as unknown as Answer);
  assert.equal(answerSpecies(named("species", "A x")), "A x");
  assert.equal(answerSpecies(named("genus", "A")), null);
  assert.equal(answerSpecies(named("species", "A x", true)), null);
  assert.equal(answerSpecies(null), null);
});
