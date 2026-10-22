import { test } from "node:test";
import assert from "node:assert/strict";

import {
  chipTitle, KEPT_ALL_NOTE, keptAllNote, withoutKeptAllNote, includeOffers, includedAnyway, includedNow, leftOutByType, leftOutNotice, photoTypeChip,
  withIncluded, withoutIncluded,
} from "./photoTypes";
import type { PhotoResult, PhotoType } from "./api";

const tag = (type: string, over: Partial<PhotoType> = {}): PhotoType => ({
  type, p: 0.9, band: type === "specimen" ? null : "high",
  label: type === "specimen" ? null : ({ habitat: "Habitat", uv: "UV", micrograph: "Microscopy" } as Record<string, string>)[type] ?? type,
  left_out: false, included: false, ...over,
});
const photo = (i: number, t?: PhotoType | null): PhotoResult => ({
  photo: i, ranks: { species: [], genus: [], family: [] }, overall_top: { species: null, genus: null, family: null },
  specimens: [], ...(t === undefined ? {} : { photo_type: t }),
});

test("a photo that isn't an ordinary specimen photo gets its chip; an ordinary one gets none", () => {
  assert.equal(photoTypeChip(photo(0, tag("habitat"))), "Habitat");
  assert.equal(photoTypeChip(photo(0, tag("uv"))), "UV");
  assert.equal(photoTypeChip(photo(0, tag("specimen"))), null);
  assert.equal(photoTypeChip(photo(0, tag("uv", { band: "low", label: null }))), null, "too unsure to show");
  assert.equal(photoTypeChip(photo(0)), null, "a server without the tagger");
  assert.equal(photoTypeChip(photo(0, null)), null);
});

test("only a photo its type left out is offered Include anyway, with its chip", () => {
  const r = {
    per_photo: [
      photo(0, tag("specimen")),
      photo(1, tag("habitat", { left_out: true })),
      photo(2, tag("uv")),                                 // tagged only: still counts
      photo(3, tag("micrograph", { included: true })),     // already counted anyway
      photo(4),
    ],
  };
  assert.deepEqual(includeOffers(r), [{ photo: 1, label: "Habitat" }]);
  assert.equal(leftOutByType(r.per_photo[1]), true);
  assert.equal(leftOutByType(r.per_photo[2]), false);
  assert.equal(includedAnyway(r.per_photo[3]), true);
  assert.deepEqual(includeOffers({ per_photo: [photo(0), photo(1)] }), [], "no tagger: nothing offered");
});

test("Include anyway runs again with the photo added to what was already counted", () => {
  assert.deepEqual(withIncluded([], 1), [1]);
  assert.deepEqual(withIncluded([3], 1), [1, 3]);
  assert.deepEqual(withIncluded([1], 1), [1]);
  assert.deepEqual(withoutIncluded([1, 3], 1), [3]);
  assert.deepEqual(includedNow({ photo_types: { left_out: [], included: [2], kept_all: false, photos: [] } }), [2]);
  assert.deepEqual(includedNow({}), []);
});

test("a left-out photo picked on its own says why it was left out", () => {
  assert.equal(leftOutNotice(photo(1, tag("habitat", { left_out: true }))),
               "Photo 2 looks like a habitat photo, so it was left out of the answer.");
  assert.equal(leftOutNotice(photo(0, tag("slip", { left_out: true }))),
               "Photo 1 looks like a label or field slip, so it was left out of the answer.");
});

test("only non-specimen photos: the answer keeps its chance and says it is less reliable", () => {
  const all = { photo_types: { left_out: [], included: [], kept_all: true, photos: [] } };
  assert.equal(keptAllNote(all), "Only non-specimen photos: this answer is less reliable.");
  assert.equal(keptAllNote({ photo_types: { ...all.photo_types, kept_all: false } }), null);
  assert.equal(keptAllNote({}), null);
  assert.deepEqual(withoutKeptAllNote([KEPT_ALL_NOTE, "Add a photo of the underside."]),
                   ["Add a photo of the underside."], "shown once, on its own");
});

test("a phylogenetic tree or BLAST screen reads Phylo tree, with a tooltip saying what it is", () => {
  assert.equal(chipTitle("Phylo tree"), "Phylogenetic tree or BLAST result");
  assert.equal(chipTitle("Habitat"), undefined);
});
