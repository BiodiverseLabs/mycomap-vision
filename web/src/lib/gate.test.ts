import { test } from "node:test";
import assert from "node:assert/strict";

import { chanceWithheld, gateBanner, leftOut, pageHints, photoGateLabel, WHAT_NEXT, whatNext } from "./gate";
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

test("the page doesn't repeat the gate's verdict as a hint under its own banner", () => {
  const hints = ["This doesn't look like a fungus, so no identification is given.", "Add more photos."];
  assert.deepEqual(pageHints({ gate: find("not_fungus", "This doesn't look like a fungus."), hints }),
                   ["Add more photos."]);
  const left = ["Photo 2 doesn't look like a fungus, so it was left out of the answer."];
  assert.deepEqual(pageHints({ gate: find("fungus"), hints: left }), left);   // no banner: kept
});

test("the next steps compare with specimens only when there are specimens to compare", () => {
  assert.equal(whatNext([{ gate: find("fungus") }]), WHAT_NEXT.fungus);
  assert.equal(whatNext([{ gate: null }]), WHAT_NEXT.fungus, "no gate deployed");
  assert.equal(whatNext([]), WHAT_NEXT.fungus);
  assert.equal(whatNext([{ gate: find("unsure") }]), WHAT_NEXT.unsure);
  assert.equal(whatNext([{ gate: find("not_fungus") }]), WHAT_NEXT.not_fungus);
  // Two models side by side: the most hopeful result decides.
  assert.equal(whatNext([{ gate: find("not_fungus") }, { gate: find("unsure") }]), WHAT_NEXT.unsure);
  assert.equal(whatNext([{ gate: find("unsure") }, { gate: find("fungus") }]), WHAT_NEXT.fungus);
});

test("\"not a fungus\" never points at specimens above, and \"not sure\" calls them look-alikes", () => {
  const text = (k: keyof typeof WHAT_NEXT) => WHAT_NEXT[k].steps.join(" ");
  assert.doesNotMatch(text("not_fungus"), /(specimens|look-alikes) above|closest/);
  assert.match(text("not_fungus"), /start over/i);
  assert.doesNotMatch(text("unsure"), /closest specimens/);
  assert.match(text("unsure"), /look-alikes above/);
  assert.match(text("fungus"), /closest specimens above/);
  for (const k of ["fungus", "unsure", "not_fungus"] as const) {
    assert.equal(text(k).split("{sequencing}").length, 2, `${k}: one sequencing link`);
  }
});
