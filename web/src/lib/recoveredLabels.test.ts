import { test } from "node:test";
import assert from "node:assert/strict";
import type { RecoveredCrop, RecoveredTile } from "./api";
import { boxOf, boxStyle, cropBackground, cropKey, cropShare, cropsUsed, dragBox, effectiveBox, flipCrop, minSide, pendingDecisions,
         startChoice, tileFacts } from "./recoveredLabels";

const crop = (n: number, box: RecoveredCrop["box"], decision: RecoveredCrop["decision"] = null): RecoveredCrop =>
  ({ n, box, area: 0.05, decision });

const tile = (id: number, crops: RecoveredCrop[], status: "open" | "decided" = "open"): RecoveredTile => ({
  item_id: id, photo_key: `inat:${id}`, record_key: String(id * 10), label: "A x", link: "https://x/obs",
  slip_p: 0.83, tranche: "high", width: 1000, height: 800, specimen_area: 0.04, crops, status,
  still_excluded: true, shown: "public", full: "https://x/large.jpg", license_code: "cc-by", owner_login: "o",
});

test("a box is drawn as percentages of the photo it sits on, kept inside it", () => {
  assert.deepEqual(boxStyle([100, 200, 300, 800], 1000, 800),
                   { left: "10.000%", top: "25.000%", width: "20.000%", height: "75.000%" });
  assert.equal(boxStyle([-50, 0, 10, 10], 1000, 800).left, "0.000%");
});

test("a crop is the original image scaled and placed by CSS, never a new file", () => {
  // A 200 px square box at (100, 100) on a 1000 x 800 photo: the image is 5 x 4 frames wide,
  // and the box's corner lands on the frame's corner.
  const css = cropBackground("https://x/large.jpg", [100, 100, 300, 300], 1000, 800);
  assert.equal(css.backgroundImage, 'url("https://x/large.jpg")');
  assert.equal(css.backgroundSize, "500.000% 400.000%");
  assert.equal(css.backgroundPosition, "12.500% 16.667%");
  // a box as wide as the photo and half as tall: centred in its square frame, the photo
  // shifted down by a quarter frame (125% of the 0.2 frame it is shorter than the frame)
  assert.equal(cropBackground("u", [0, 0, 1000, 500], 1000, 800).backgroundPosition, "0.000% 125.000%");
  assert.equal(cropShare([0, 0, 500, 400], 1000, 800), 25);
});

test("every crop starts as Include unless a saved decision says otherwise, and a click flips it", () => {
  const t = tile(1, [crop(1, [0, 0, 10, 10]), crop(2, [20, 20, 40, 40], "reject")], "decided");
  assert.equal(startChoice(t, t.crops[0]), "include");
  assert.equal(startChoice(t, t.crops[1]), "reject");
  const marks = flipCrop({}, t, 1);
  assert.equal(marks[cropKey(1, 1)], "reject");
  assert.equal(flipCrop(marks, t, 1)[cropKey(1, 1)], "include");
  assert.deepEqual(flipCrop(marks, t, 9), marks);                  // no such crop
});

test("an open tile is saved once seen, a decided one only when a crop changed", () => {
  const open = tile(1, [crop(1, [0, 0, 10, 10]), crop(2, [20, 20, 40, 40])]);
  const done = tile(2, [crop(1, [0, 0, 10, 10], "include")], "decided");
  assert.deepEqual(pendingDecisions([open, done], {}, new Set()), []);       // nothing seen
  assert.deepEqual(pendingDecisions([open, done], {}, new Set([1, 2])), [{ item_id: 1, include: [1, 2] }]);
  const marks = flipCrop(flipCrop({}, open, 2), done, 1);
  assert.deepEqual(pendingDecisions([open, done], marks, new Set()),
                   [{ item_id: 1, include: [1] }, { item_id: 2, include: [] }]);   // [] rejects all
});

test("the tile says its slip p, how big the find is and how many crops", () => {
  assert.equal(tileFacts(tile(1, [crop(1, [0, 0, 1, 1]), crop(2, [2, 2, 3, 3])])),
               "slip p 0.83 · specimen 4.0% of the photo · 2 crops");
  assert.equal(tileFacts({ slip_p: null, specimen_area: 0.25, crops: [crop(1, [0, 0, 1, 1])] }),
               "specimen 25% of the photo");
});

test("a box moves whole or by one corner and stays square: the region the model reads", () => {
  const b: RecoveredCrop["box"] = [100, 100, 300, 300];
  assert.deepEqual(dragBox(b, "move", 50, -20, 1000, 800), [150, 80, 350, 280]);
  assert.deepEqual(dragBox(b, "move", -500, 900, 1000, 800), [0, 600, 200, 800]);      // stops at the edge
  assert.deepEqual(dragBox(b, "nw", -60, -60, 1000, 800), [40, 40, 300, 300]);
  assert.deepEqual(dragBox(b, "se", 50, 10, 1000, 800), [100, 100, 350, 350]);         // the larger move wins
  assert.deepEqual(dragBox(b, "sw", 0, 80, 1000, 800), [20, 100, 300, 380]);           // grows from the fixed ne corner
  assert.deepEqual(dragBox(b, "se", 900, 900, 1000, 800), [100, 100, 800, 800]);       // stops at the bottom edge, square
  assert.deepEqual(dragBox(b, "nw", -500, -10, 1000, 800), [0, 0, 300, 300]);          // stops at the top-left edge
  assert.equal(minSide(1000, 800), 32);
  assert.deepEqual(dragBox(b, "ne", -500, 500, 1000, 800), [100, 268, 132, 300]);      // 32 px at least
  // keyboard: Shift+Alt+→ grows (dragBox "se" by the step on both axes), ← shrinks
  assert.deepEqual(dragBox(b, "se", 8 + 0, 8 + 0, 1000, 800), [100, 100, 308, 308]);
  assert.deepEqual(dragBox(b, "se", -8, -8, 1000, 800), [100, 100, 292, 292]);
  for (const h of ["move", "nw", "ne", "sw", "se"] as const) {
    for (const [dx, dy] of [[37, -213], [-999, 5], [400, 400], [-3, 650]]) {
      const [l, t, r, bb] = dragBox(b, h, dx, dy, 1000, 800);
      assert.equal(r - l, bb - t, `${h} ${dx},${dy} square`);
      assert.ok(l >= 0 && t >= 0 && r <= 1000 && bb <= 800, `${h} ${dx},${dy} inside`);
    }
  }
});

test("a detector's non-square box is shown as the central square the model reads", () => {
  assert.deepEqual(effectiveBox([0, 0, 300, 200]), [50, 0, 250, 200]);
  assert.deepEqual(effectiveBox([10, 20, 110, 320]), [10, 120, 110, 220]);
  assert.deepEqual(effectiveBox([5, 5, 55, 55]), [5, 5, 55, 55]);                      // a square is itself
  assert.deepEqual(dragBox([0, 0, 300, 200], "move", 0, 0, 1000, 800), [50, 0, 250, 200]);
  const t = tile(1, [crop(1, [0, 0, 300, 200])]);
  assert.deepEqual(boxOf(t, t.crops[0], {}), [50, 0, 250, 200]);
  assert.deepEqual(pendingDecisions([t], {}, new Set(), { [cropKey(1, 1)]: [50, 0, 250, 200] }), []);  // no edit
});

test("a moved box and a changed reason are saved with the page; a box only for a kept crop", () => {
  const t = { ...tile(1, [crop(1, [0, 0, 10, 10]), crop(2, [20, 20, 40, 40])]), reason: "slip" };
  const edits = { [cropKey(1, 2)]: [10, 10, 60, 60] as RecoveredCrop["box"] };
  assert.deepEqual(boxOf(t, t.crops[1], edits), [10, 10, 60, 60]);
  assert.deepEqual(pendingDecisions([t], {}, new Set(), edits),
                   [{ item_id: 1, include: [1, 2], boxes: { "2": [10, 10, 60, 60] } }]);
  assert.deepEqual(pendingDecisions([t], flipCrop({}, t, 2), new Set(), edits), [{ item_id: 1, include: [1] }]);
  assert.deepEqual(pendingDecisions([t], {}, new Set(), {}, { 1: "dried" }),
                   [{ item_id: 1, include: [1, 2], reason: "dried" }]);
  assert.deepEqual(pendingDecisions([t], {}, new Set(), {}, { 1: "slip" }), []);       // unchanged
  assert.equal(cropsUsed("slip"), true);
  assert.equal(cropsUsed("dried"), false);
  assert.equal(cropsUsed("keep"), false);
});
