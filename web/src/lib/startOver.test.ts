import { test } from "node:test";
import assert from "node:assert/strict";
import { canStartOver, startOver } from "./startOver";
import { fillFromPhotos } from "./photoPlaceDate";

const photo = (name: string) => new File(["x"], name, { type: "image/jpeg" });

test("start over leaves no photos and no place or date, typed or read from a photo", () => {
  const blank = startOver();
  assert.deepEqual(blank.files, []);
  assert.equal(blank.found.size, 0);
  assert.equal(blank.where.lat ?? "", "");
  assert.equal(blank.where.lng ?? "", "");
  assert.equal(blank.where.observedOn ?? "", "");
  assert.equal(blank.where.placeFromPhoto, undefined, "no 'Place from photo N' note survives");
  assert.equal(blank.where.dateFromPhoto, undefined, "no 'Date from photo N' note survives");
});

test("a cleared form stays clear: refilling from no photos adds no place or date", () => {
  const { where, files, found } = startOver();
  const refilled = fillFromPhotos(where, files.map((f) => found.get(f)));
  assert.equal(refilled.lat ?? "", "");
  assert.equal(refilled.observedOn ?? "", "");
  assert.equal(refilled.placeFromPhoto, undefined);
});

test("each start over is a fresh form, so the last find's photo reads can't leak in", () => {
  const a = startOver();
  a.found.set(photo("a.jpg"), { lat: 45, lng: -122, date: "2026-10-01" });
  assert.equal(startOver().found.size, 0);
});

test("Start over is offered once there's a photo, a place or date, an answer or an error", () => {
  const none = { files: [], where: {}, hasAnswer: false, hasError: false, identifying: false };
  assert.equal(canStartOver(none), false, "nothing to clear on a fresh page");
  assert.equal(canStartOver({ ...none, files: [photo("a.jpg")] }), true);
  assert.equal(canStartOver({ ...none, where: { observedOn: "2026-10-01" } }), true);
  assert.equal(canStartOver({ ...none, where: { lat: "45.1", lng: "" } }), true);
  assert.equal(canStartOver({ ...none, where: { lat: " ", lng: "" } }), false, "blank fields are nothing");
  assert.equal(canStartOver({ ...none, hasAnswer: true }), true);
  assert.equal(canStartOver({ ...none, hasError: true }), true);
});

test("Start over is never offered while an identification is on its way", () => {
  assert.equal(canStartOver({ files: [photo("a.jpg")], where: { observedOn: "2026-10-01" },
                               hasAnswer: false, hasError: false, identifying: true }), false);
});
