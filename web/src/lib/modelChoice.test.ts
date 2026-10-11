import { test } from "node:test";
import assert from "node:assert/strict";
import { defaultChoice, methodFor, offeredBases, resolveMethod, switchesFor } from "./modelChoice";

const ALL = ["nearest", "species-mean", "linear", "hybrid", "linear+prior", "hybrid+prior",
             "nearest+prior", "species-mean+prior"];

test("each choice names one method", () => {
  assert.equal(methodFor({ base: "linear", weighNearest: true, usePlace: true }), "hybrid+prior");
  assert.equal(methodFor({ base: "linear", weighNearest: false, usePlace: false }), "linear");
  assert.equal(methodFor({ base: "nearest", weighNearest: true, usePlace: true }), "nearest+prior",
    "weighing the nearest specimens only applies to the trained classifier");
  assert.equal(methodFor({ base: "species-mean", weighNearest: false, usePlace: false }), "species-mean");
});

test("the default is the trained classifier with both add-ons, where the server offers it", () => {
  assert.deepEqual(defaultChoice(ALL), { base: "linear", weighNearest: true, usePlace: true });
  assert.equal(resolveMethod(defaultChoice(ALL)!, ALL), "hybrid+prior");
});

test("on a server that offers only nearest specimen, that is the default and nothing else shows", () => {
  const box = ["nearest"];
  assert.deepEqual(offeredBases(box), ["nearest"]);
  const c = defaultChoice(box)!;
  assert.equal(c.base, "nearest");
  assert.equal(resolveMethod(c, box), "nearest", "the place switch falls back to what is offered");
  assert.deepEqual(switchesFor("nearest", box), { weighNearest: false, usePlace: false });
});

test("the server's default method is the starting choice when it offers it", () => {
  const box = ["nearest", "nearest+mean"];
  const c = defaultChoice(box, "nearest+mean")!;
  assert.deepEqual(c, { base: "nearest+mean", weighNearest: false, usePlace: false });
  assert.equal(resolveMethod(c, box), "nearest+mean");
  assert.deepEqual(offeredBases(box), ["nearest+mean", "nearest"]);
  assert.deepEqual(switchesFor("nearest+mean", box), { weighNearest: false, usePlace: false });
  assert.equal(resolveMethod(defaultChoice(box, "nearest")!, box), "nearest");
  assert.equal(defaultChoice(["nearest"], "nearest+mean")!.base, "nearest", "not offered: first offered");
});

test("switches show only when both of their sides are offered", () => {
  assert.deepEqual(switchesFor("linear", ALL), { weighNearest: true, usePlace: true });
  assert.deepEqual(switchesFor("linear", ["hybrid", "hybrid+prior"]), { weighNearest: false, usePlace: true });
  assert.deepEqual(switchesFor("nearest", ["nearest", "nearest+prior"]), { weighNearest: false, usePlace: true });
});

test("a method never offered is never sent", () => {
  assert.equal(resolveMethod({ base: "linear", weighNearest: true, usePlace: true }, ["linear"]), "linear");
  assert.equal(resolveMethod({ base: "linear", weighNearest: false, usePlace: false }, ["hybrid+prior"]), "hybrid+prior");
  assert.equal(resolveMethod({ base: "species-mean", weighNearest: false, usePlace: false }, ["nearest"]), null);
  assert.equal(defaultChoice([]), null);
});

// The Vision box since 2026-10-11: MV_METHODS=nearest,nearest+mean,nearest+mean+place,
// MV_DEFAULT_METHOD=nearest+mean+place. The page sent plain nearest+mean, so no
// identification on the website used the place and date.
const BOX = ["nearest", "nearest+mean", "nearest+mean+place"];

test("the box's place model is what the website asks for by default", () => {
  const c = defaultChoice(BOX, "nearest+mean+place")!;
  assert.deepEqual(c, { base: "nearest+mean", weighNearest: false, usePlace: true });
  assert.equal(resolveMethod(c, BOX), "nearest+mean+place");
});

test("the place switch shows for nearest + species average when both variants are offered", () => {
  assert.deepEqual(switchesFor("nearest+mean", BOX), { weighNearest: false, usePlace: true });
  assert.equal(resolveMethod({ base: "nearest+mean", weighNearest: false, usePlace: false }, BOX), "nearest+mean");
  assert.deepEqual(offeredBases(BOX), ["nearest+mean", "nearest"]);
});

test("the place model alone is still reachable, and never asked for where it isn't offered", () => {
  assert.equal(resolveMethod({ base: "nearest+mean", weighNearest: false, usePlace: false }, ["nearest+mean+place"]),
               "nearest+mean+place");
  assert.equal(resolveMethod({ base: "nearest+mean", weighNearest: false, usePlace: true }, ["nearest", "nearest+mean"]),
               "nearest+mean");
  assert.equal(methodFor({ base: "nearest", weighNearest: false, usePlace: true }), "nearest+prior");
});
