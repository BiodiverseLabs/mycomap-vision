import { test } from "node:test";
import assert from "node:assert/strict";
import {
  alsoPossible, answerFor, boxHints, breakdownLabel, breakdownOpen, confusedAside, confusedNote, coverageSummary, listedNames,
  photoOnlyNote, placeCue, placeLines, placeWarns, publicNote, usesPlace,
} from "./answer";
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

// --- often confused with (docs/api.md §4) ---------------------------------------------

const confused = {
  names: [{ name: "Gymnopilus subspectabilis", finds: 8 }], called: 86, wrong: 29,
  note: "Often confused with G. subspectabilis · 9% of DNA-checked finds",
};

test("the confused-with note is the server's own line, and nothing without one", () => {
  assert.equal(confusedNote({ confused_with: confused }), "Often confused with G. subspectabilis · 9% of DNA-checked finds");
  assert.equal(confusedNote({}), null);
  assert.equal(confusedNote(null), null);
  assert.equal(confusedNote({ confused_with: { ...confused, note: " " } }), null);
  assert.equal(confusedAside({ confused_with: confused }), "often confused with G. subspectabilis · 9% of DNA-checked finds",
    "only the first letter is lowered: species names keep their capitals");
});

test("the answer box carries the confused-with field of its headline and the names in play", () => {
  const r = result(
    { family: [c("Hymenogastraceae", 0.95)], genus: [c("Gymnopilus", 0.9)],
      species: [c("Gymnopilus luteus", 0.4, { confused_with: confused }), c("Gymnopilus junonius", 0.3)] },
  );
  const a = answerFor(r, fungus)!;
  assert.equal(a.headline?.rank, "genus");
  assert.deepEqual(a.inPlay?.names.map((n) => confusedNote(n)), [confused.note, null]);
  const species = answerFor(result({ family: [c("Hymenogastraceae", 0.95)], genus: [c("Gymnopilus", 0.95)],
    species: [c("Gymnopilus luteus", 0.8, { confused_with: confused })] }), fungus)!;
  assert.equal(confusedNote(species.headline), confused.note);
  const lookAlike = answerFor(r, { notFungus: false, withheld: true })!;
  assert.equal(confusedNote(lookAlike.headline), confused.note, "kept when the chance is withheld");
});

// --- the full breakdown -----------------------------------------------------------------

const gym = [c("Gymnopilus luteus", 0.81), c("Gymnopilus subspectabilis", 0.11), c("Gymnopilus sapineus", 0.05),
  c("Gymnopilus penetrans", 0.02), c("Gymnopilus junonius", 0.004)];

test("one photo on its own lists the top name and any at 5% or more, at most three", () => {
  assert.deepEqual(listedNames(gym).map((n) => n.name), ["Gymnopilus luteus", "Gymnopilus subspectabilis", "Gymnopilus sapineus"]);
  assert.deepEqual(listedNames([c("A b", 0.3), c("A c", 0.04)]).map((n) => n.name), ["A b"], "under 5%: not listed");
  assert.deepEqual(listedNames([c("A b", 0.02)]).map((n) => n.name), ["A b"], "the top name always is");
  assert.deepEqual(listedNames([{ ...c("A b", 0), confidence: null }]), [], "nothing when the chance is withheld");
  const many = [c("A a", 0.3), c("A b", 0.2), c("A c", 0.2), c("A d", 0.1), c("A e", 0.08), c("A f", 0.06), c("A g", 0.06)];
  assert.equal(listedNames(many).length, 3, "at most three with a bar");
  assert.deepEqual(alsoPossible(many, listedNames(many)).map((n) => n.name), ["A d", "A e", "A f"], "and three more in words");
});

test("a name under 1% never shows in the breakdown, so nothing reads 0%", () => {
  assert.deepEqual(alsoPossible(gym, [{ name: "Gymnopilus luteus" }]).map((n) => n.name),
    ["Gymnopilus subspectabilis", "Gymnopilus sapineus", "Gymnopilus penetrans"]);
  assert.deepEqual(alsoPossible(gym, listedNames(gym)).map((n) => n.name), ["Gymnopilus penetrans"],
    "listed names aren't repeated; 0.4% is left out");
});

test("the breakdown is folded when there is an answer and open when there is none", () => {
  assert.equal(breakdownOpen(false, true, false), false);
  assert.equal(breakdownOpen(true, true, false), true);
  assert.equal(breakdownOpen(false, false, false), true, "without an answer the breakdown is the answer");
  assert.equal(breakdownOpen(true, false, true), false, "never when there are no ranks (not a fungus)");
});

test("the breakdown's button says which view it breaks down", () => {
  assert.equal(breakdownLabel(2, null), "Full breakdown: all 2 photos · every rank, with its chances");
  assert.equal(breakdownLabel(2, 0), "Full breakdown: photo 1 on its own · every rank, with its chances");
  assert.equal(breakdownLabel(1, null), "Full breakdown · every rank, with its chances");
});

test("names still in play show only a place cue that warns", () => {
  const near = { nearest_record_km: 20, in_season: true, far_from_known: false };
  const far = { nearest_record_km: 310, in_season: true, far_from_known: true };
  const late = { nearest_record_km: 20, in_season: false, far_from_known: false };
  assert.deepEqual(placeLines(near, true), []);
  assert.deepEqual(placeLines(near, false), placeCue(near), "the headline keeps every line");
  assert.deepEqual(placeLines(far, true), placeCue(far));
  assert.deepEqual(placeLines(late, true), placeCue(late));
  assert.equal(placeWarns(near), false);
  assert.equal(placeWarns(null), false);
});
