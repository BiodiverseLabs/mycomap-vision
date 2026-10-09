import { test } from "node:test";
import assert from "node:assert/strict";

import { chanceWithheld, gateBanner, leftOut, photoGateLabel } from "./gate";
import type { FindGate, PhotoGate } from "./api";

const photo = (verdict: PhotoGate["verdict"]): PhotoGate => ({ verdict, p_fungus: 0.5, message: null });
const find = (verdict: FindGate["verdict"], message: string | null = null): FindGate => ({
  verdict, message, left_out: [], photos: [],
});

test("a find that isn't a fungus gets the server's message and no identification", () => {
  const b = gateBanner({ gate: find("not_fungus", "This doesn't look like a fungus.") });
  assert.equal(b?.verdict, "not_fungus");
  assert.equal(b?.title, "This doesn't look like a fungus.");
  assert.match(b!.body, /no name and no chance/);
});

test("a find the gate is unsure of says the names are look-alikes with no stated chance", () => {
  const b = gateBanner({ gate: find("unsure") });
  assert.equal(b?.title, "Not sure this is a fungus.");
  assert.match(b!.body, /look-alikes/);
  assert.equal(chanceWithheld({ gate: find("unsure") }), true);
});

test("a fungus, or a result from a server without the gate, gets no banner and keeps its chances", () => {
  assert.equal(gateBanner({ gate: find("fungus") }), null);
  assert.equal(gateBanner({ gate: null }), null);
  assert.equal(gateBanner({}), null);
  assert.equal(chanceWithheld({ gate: find("fungus") }), false);
  assert.equal(chanceWithheld({}), false);
});

test("the server's withheld flag hides the chance even without a gate verdict", () => {
  assert.equal(chanceWithheld({ confidence_withheld: true }), true);
});

test("a photo left out as not a fungus is labelled under its tile", () => {
  assert.equal(leftOut({ gate: photo("not_fungus") }), true);
  assert.equal(photoGateLabel({ gate: photo("not_fungus") }), "not a fungus");
  assert.equal(photoGateLabel({ gate: photo("unsure") }), "fungus?");
  assert.equal(photoGateLabel({ gate: photo("fungus") }), null);
  assert.equal(leftOut({}), false);
});
