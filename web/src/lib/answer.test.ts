import { test } from "node:test";
import assert from "node:assert/strict";
import { answerFor, boxHints, coverageSummary, photoOnlyNote, placeCue, publicNote, usesPlace } from "./answer";
import type { Candidate, IdentifyResult, LikelySet, Rank } from "./api";

const c = (name: string, confidence: number, extra: Partial<Candidate> = {}): Candidate =>
  ({ name, confidence, score: confidence, reference_records: 10, ...extra });
const set = (coverage: number, ...names: [string, number][]): LikelySet => ({
  coverage, checked_coverage: null, capped: false,
  names: names.map(([name, confidence]) => ({ name, confidence, reference_records: 10 })),
});
const result = (ranks: Record<Rank, Candidate[]>, likely?: IdentifyResult["likely"]) => ({ ranks, likely });
const fungus = { notFungus: false, withheld: false };

// The find from the review: family and genus clear, species split.
const hebeloma = result(
  {
    family: [c("Hymenogastraceae", 0.88), c("Inocybaceae", 0.07)],
    genus: [c("Hebeloma", 0.84), c("Inocybe", 0.07)],
    species: [c("Hebeloma kelloggense", 0.3), c("Hebeloma sp. 'incarnatulum-velutipes'", 0.05), c("Inocybe praecox", 0.03)],
  },
  {
    family: set(0.9, ["Hymenogastraceae", 0.88]),
    genus: set(0.9, ["Hebeloma", 0.84], ["Inocybe", 0.07]),
    species: set(0.6, ["Hebeloma kelloggense", 0.3], ["Hebeloma sp. 'incarnatulum-velutipes'", 0.05]),
  },
);

test("the headline is the finest rank whose top name reaches 70%", () => {
  const a = answerFor(hebeloma, fungus)!;
  assert.equal(a.headline?.rank, "genus");
  assert.equal(a.headline?.name, "Hebeloma");
  assert.equal(a.headline?.confidence, 0.84);
  assert.deepEqual(a.above.map((n) => n.name), ["Hymenogastraceae"], "the family is named under it");
});

test("a species at 70% or more is the headline, with genus and family under it", () => {
  const r = result({
    family: [c("Amanitaceae", 0.97)], genus: [c("Amanita", 0.95)], species: [c("Amanita muscaria", 0.7)],
  });
  const a = answerFor(r, fungus)!;
  assert.equal(a.headline?.name, "Amanita muscaria");
  assert.deepEqual(a.above.map((n) => n.rank), ["genus", "family"]);
  assert.equal(a.inPlay, null, "nothing finer than species to list");
});

test("just under 70% is not settled", () => {
  const r = result({
    family: [c("Amanitaceae", 0.97)], genus: [c("Amanita", 0.95)], species: [c("Amanita muscaria", 0.699)],
  });
  assert.equal(answerFor(r, fungus)!.headline?.rank, "genus");
});

test("the species still in play come from the species likely list, at most three", () => {
  const a = answerFor(hebeloma, fungus)!;
  assert.equal(a.inPlay?.rank, "species");
  assert.deepEqual(a.inPlay?.names.map((n) => n.name),
    ["Hebeloma kelloggense", "Hebeloma sp. 'incarnatulum-velutipes'"]);
  const many = result(hebeloma.ranks, {
    ...hebeloma.likely,
    species: set(0.6, ["A a", 0.2], ["B b", 0.15], ["C c", 0.1], ["D d", 0.1], ["E e", 0.05]),
  });
  const m = answerFor(many, fungus)!.inPlay!;
  assert.equal(m.names.length, 3);
  assert.equal(m.more, 2);
});

test("without likely lists, names under 5% aren't listed as in play", () => {
  const r = result({
    family: [c("Hymenogastraceae", 0.9)], genus: [c("Hebeloma", 0.8)],
    species: [c("Hebeloma a", 0.3), c("Hebeloma b", 0.2), c("Hebeloma c", 0.04)],
  });
  assert.deepEqual(answerFor(r, fungus)!.inPlay?.names.map((n) => n.name), ["Hebeloma a", "Hebeloma b"]);
});

test("when no rank reaches 70%, there's no headline and the families in play are listed", () => {
  const r = result({
    family: [c("Russulaceae", 0.55), c("Albatrellaceae", 0.2)],
    genus: [c("Russula", 0.4)], species: [c("Russula emetica", 0.1)],
  });
  const a = answerFor(r, fungus)!;
  assert.equal(a.headline, null);
  assert.deepEqual(a.above, []);
  assert.equal(a.inPlay?.rank, "family");
  assert.deepEqual(a.inPlay?.names.map((n) => n.name), ["Russulaceae", "Albatrellaceae"]);
});

test("no answer box when the photos don't look like a fungus", () => {
  assert.equal(answerFor(hebeloma, { notFungus: true, withheld: true }), null);
});

test("when the chance is withheld, the box names the closest look-alike with no chance", () => {
  const a = answerFor(hebeloma, { notFungus: false, withheld: true })!;
  assert.equal(a.lookAlike, true);
  assert.equal(a.headline?.name, "Hebeloma kelloggense");
  assert.equal(a.headline?.confidence, null, "no % anywhere");
  assert.deepEqual(a.above, []);
  assert.equal(a.inPlay, null);
});

test("no names at all gives no answer box", () => {
  assert.equal(answerFor(result({ family: [], genus: [], species: [] }), fungus), null);
});

test("far-find cues read the place lane's fields, and say nothing without them", () => {
  assert.deepEqual(placeCue(undefined), []);
  assert.deepEqual(placeCue({}), []);
  assert.deepEqual(placeCue({ nearest_record_km: 20, far_from_known: false, in_season: true }), ["Known within about 20 km"]);
  assert.deepEqual(placeCue({ nearest_record_km: 0, far_from_known: false }), ["Known within about 10 km"],
    "distances come rounded to 10 km, so never closer than about 10");
  assert.deepEqual(placeCue({ nearest_record_km: 1830, far_from_known: true }),
    [`Far from known records, nearest about ${(1830).toLocaleString()} km (possible range extension)`]);
  assert.deepEqual(placeCue({ far_from_known: true }), ["Far from known records (possible range extension)"]);
  assert.deepEqual(placeCue({ in_season: false }), ["Outside its known season"]);
  assert.deepEqual(placeCue({ nearest_record_km: null, in_season: false, far_from_known: null }),
    ["Outside its known season"], "date only: no distance");
  assert.deepEqual(placeCue({ nearest_record_km: null, in_season: null, far_from_known: null }), [],
    "nulls say nothing");
});

test("the photos-alone name shows only when the place changed the headline", () => {
  const head = { rank: "species" as const, name: "Amanita muscaria", confidence: 0.8 };
  assert.equal(photoOnlyNote({ photo_only_top: { species: "Amanita persicina" } }, head),
    "From the photos alone: Amanita persicina");
  assert.equal(photoOnlyNote({ photo_only_top: { species: "Amanita muscaria" } }, head), null);
  assert.equal(photoOnlyNote({ photo_only_top: { genus: "Amanita" } }, head), null);
  assert.equal(photoOnlyNote({}, head), null);
});

test("internal comparison ids never reach the public note", () => {
  assert.equal(
    publicNote("Calibrated on 980 test records (comparison 20261008-012435-4ef7b0). Early and approximate."),
    "Calibrated on 980 test records. Early and approximate.");
  assert.equal(publicNote("Not calibrated yet."), "Not calibrated yet.");
});

test("coverage is said once for all ranks, grouping ranks that share it", () => {
  assert.equal(coverageSummary(hebeloma.likely),
    "In tests on the newest DNA-verified records, the right name was in these lists " +
    "about 9 times in 10 at family and genus, and about 6 times in 10 at species.");
  assert.equal(coverageSummary({ species: set(0.9, ["A a", 0.5]) }),
    "In tests on the newest DNA-verified records, the right name was in these lists about 9 times in 10 at species.");
  assert.equal(coverageSummary(undefined), null);
});

test("the range-and-season line is only for models that use place and date", () => {
  assert.equal(usesPlace(["nearest", "nearest+mean"]), false);
  assert.equal(usesPlace(["nearest", "hybrid+prior"]), true);
  assert.equal(usesPlace(["nearest+mean+place"]), true, "the place lane's served method");
});

test("the answer box keeps the advice but not a repeat of the species in play", () => {
  const settled = "The genus looks settled, but Hebeloma kelloggense and Hebeloma sp. 'x' are close. " +
    "Clear photos of the underside and stem base, or DNA sequencing, would separate them.";
  assert.deepEqual(boxHints([settled], true),
    ["Clear photos of the underside and stem base, or DNA sequencing, would separate them."]);
  assert.deepEqual(boxHints([settled], false), [settled], "kept whole when the box lists no species");
  const single = "The best species match, A b, rests on a single DNA-verified specimen, so treat it with caution.";
  assert.deepEqual(boxHints([single], true), [single]);
});
