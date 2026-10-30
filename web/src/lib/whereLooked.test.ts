// "Where Vision looked": how a map is drawn, where the seen square sits, the match spots and
// what the panel says.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  cellColours, cropBox, darkness, firstPhoto, markers, ownLine, refusalLine, spotlightAlpha, squareOf,
  STRENGTH_DEFAULT, unseenStrips, workingLine,
} from "./whereLooked";
import type { LookedPhoto, PhotoResult } from "./api";

test("Spotlight keeps a photo's strongest 30% of patches clear and dims the weakest most, by rank", () => {
  const map = Array.from({ length: 100 }, (_, i) => i);     // 0..99
  const a = spotlightAlpha(map);
  const clear = a.filter((x) => x === 0).length;
  assert.ok(clear >= 30 && clear <= 31, `${clear} clear cells`);
  assert.equal(a[99], 0);
  assert.equal(Math.max(...a), a[0]);
  assert.ok(a[0] > 0.79 && a[0] <= 0.8);
  // By rank, not by size: a map ten times as strong dims exactly the same way.
  assert.deepEqual(spotlightAlpha(map.map((v) => v * 10)), a);
});

test("Spotlight is the one look: each cell is dimmed, never coloured, the strongest not at all", () => {
  const spot = cellColours([0, 255]);
  assert.deepEqual(spot[1], [18, 18, 18, 0]);                // the strongest cell: not dimmed
  assert.deepEqual(spot[0].slice(0, 3), [18, 18, 18]);
  assert.ok(spot[0][3] > 0.79);
  const map = Array.from({ length: 16 }, (_, i) => i * 17);
  assert.deepEqual(cellColours(map).map((c) => c[3]), spotlightAlpha(map));
});

const tall: LookedPhoto["crop"] = { x: 0, y: 0.125, w: 1, h: 0.75 };

test("the map sits on the centre square the model sees, and the unseen strips are marked", () => {
  assert.deepEqual(cropBox(tall), { left: "0%", top: "12.5%", width: "100%", height: "75%" });
  const strips = unseenStrips(tall);
  assert.deepEqual(strips.map((s) => s.side), ["top", "bottom"]);
  assert.equal(strips[0].style.height, "12.5%");
  assert.deepEqual(unseenStrips({ x: 0.25, y: 0, w: 0.5, h: 1 }).map((s) => s.side), ["left", "right"]);
  assert.deepEqual(unseenStrips({ x: 0, y: 0, w: 1, h: 1 }), []);
});

test("the slider starts where its old top was, and goes on darker to black", () => {
  assert.equal(STRENGTH_DEFAULT, 75);
  assert.equal(darkness(0), 0);
  assert.ok(Math.abs(darkness(STRENGTH_DEFAULT) - 0.8) < 1e-12, "the default dims as the old full strength did");
  assert.equal(darkness(100), 1);
  assert.ok(darkness(90) > 0.8 && darkness(90) < 1);
  assert.equal(darkness(140), 1);
  const map = Array.from({ length: 100 }, (_, i) => i);
  // The default looks exactly as the old full strength (cells at 0.8, no extra opacity).
  assert.deepEqual(cellColours(map).map((c) => c[3]), spotlightAlpha(map, 0.3, 0.8));
  const full = cellColours(map, 100).map((c) => c[3]);
  assert.equal(full[0], 1, "the weakest cell goes black at full strength");
  assert.equal(full.filter((a) => a === 0).length, cellColours(map).filter((c) => c[3] === 0).length,
               "strength darkens; it never changes which cells stay clear");
});

test("one strip of each pair says Vision doesn't see it, only where the words fit", () => {
  assert.deepEqual(unseenStrips(tall).map((s) => s.label), [true, false]);
  const wide = unseenStrips({ x: 0.125, y: 0, w: 0.75, h: 1 });
  assert.deepEqual(wide.map((s) => [s.side, s.label]), [["left", true], ["right", false]]);
  assert.deepEqual(unseenStrips({ x: 0.03, y: 0, w: 0.94, h: 1 }).map((s) => s.label), [false, false],
                   "a thin strip is striped but not labelled");
});

test("a photo is placed so that only its seen square fills the match square", () => {
  assert.deepEqual(squareOf(tall), { width: "100%", height: "133.33%", left: "0%", top: "-16.67%" });
});

test("match spots are numbered from 1 and placed in percent on both photos", () => {
  assert.deepEqual(markers([[0.25, 0.5, 0.75, 0.125]]),
    [{ n: 1, photo: { x: 25, y: 50 }, specimen: { x: 75, y: 12.5 } }]);
});

const per = (photo: number, position: number, confidence: number | null) =>
  ({ photo, overall_top: { species: { name: "Laetiporus sulphureus", position, confidence } } }) as unknown as PhotoResult;
const entry = (photo: number) => ({ photo }) as LookedPhoto;

test("the panel opens on the photo that on its own puts the answer highest", () => {
  const pp = [per(0, 1, 0.77), per(1, 2, 0.95), per(2, 1, 0.93)];
  assert.equal(firstPhoto([entry(0), entry(1), entry(2)], pp), 2);
  assert.equal(firstPhoto([entry(1)], pp), 1);
  assert.equal(firstPhoto([], pp), null);
});

test("the line under the controls says how the photo rates the name on its own", () => {
  assert.deepEqual(ownLine(4, [per(4, 1, 0.932)]),
    { lead: "On its own, photo 5 puts ", name: "Laetiporus sulphureus", tail: " first, at 93%." });
  assert.equal(ownLine(0, [per(0, 3, null)])!.tail, " at number 3.");
  assert.equal(ownLine(7, []), null);
});

test("while the maps are made the button says what is happening", () => {
  assert.equal(workingLine({ status: "working", photos: 5, done: 0, ahead: 2 }), "Waiting for a quiet moment on the server…");
  assert.equal(workingLine({ status: "working", photos: 5, done: 2, ahead: 0 }), "Looking at photo 3 of 5…");
  assert.equal(workingLine({ status: "working", photos: 1, done: 0, ahead: 0 }), "Looking at your photo…");
});

test("each refusal is worded for the person who asked", () => {
  assert.match(refusalLine(401, ""), /Sign in/);
  assert.equal(refusalLine(409, "this answer has no species to show the evidence for"),
               "There is nothing to show here: this answer has no species to show the evidence for.");
  assert.match(refusalLine(429, ""), /busy/);
  assert.match(refusalLine(503, ""), /could not be made/);
});
