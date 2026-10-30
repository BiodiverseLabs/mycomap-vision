// An observation's photos that aren't Creative Commons, on the Identify page and a run's
// page (/r/<id>): the photo strip says so, and both photo sets link to the observation
// at its source in a new tab (Steve, 2026-10-30).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { HiddenPhotosLine } from "./HiddenPhotosLine";
import { ResultCard } from "../pages/Identify";
import { EXAMPLE_RUN } from "../lib/exampleRun.fixture";

const OBS = { source: "inat" as const, url: "https://www.inaturalist.org/observations/213312794" };
const RESULT = EXAMPLE_RUN.results[0];
const card = (urls: string[], observation?: typeof OBS) =>
  renderToStaticMarkup(createElement(ResultCard, { r: RESULT, urls, photos: urls.length, observation }));

test("the line links to the observation, opening in a new tab", () => {
  const h = renderToStaticMarkup(createElement(HiddenPhotosLine, { urls: ["", ""], observation: OBS, testId: "t" }));
  assert.match(h, /These photos aren&#x27;t Creative Commons\./);
  assert.match(h, /<a href="https:\/\/www\.inaturalist\.org\/observations\/213312794" target="_blank" rel="noreferrer"/);
  assert.match(h, /See them on iNaturalist/);
});

test("nothing is drawn when every photo shows", () => {
  assert.equal(renderToStaticMarkup(createElement(HiddenPhotosLine, { urls: ["a", "b"], observation: OBS, testId: "t" })), "");
});

test("an observation's photo strip marks each photo that isn't Creative Commons, and links the observation", () => {
  const h = card(["", "https://example.org/2.jpg", ""], OBS);
  assert.match(h, /data-testid="photo-not-cc-0"/);
  assert.doesNotMatch(h, /data-testid="photo-not-cc-1"/);
  assert.match(h, /data-testid="photo-not-cc-2"/);
  assert.match(h, /not CC/);
  assert.match(h, /data-testid="switcher-hidden-photos"/);
  assert.match(h, /href="https:\/\/www\.inaturalist\.org\/observations\/213312794" target="_blank"/);
});

test("uploaded photos (no observation) get no placeholder and no link", () => {
  const h = card(["", "", ""]);
  assert.doesNotMatch(h, /photo-not-cc-/);
  assert.doesNotMatch(h, /switcher-hidden-photos/);
});

test("a fully shown observation gets no link", () => {
  const h = card(["https://e/1.jpg", "https://e/2.jpg", "https://e/3.jpg"], OBS);
  assert.doesNotMatch(h, /photo-not-cc-/);
  assert.doesNotMatch(h, /switcher-hidden-photos/);
});
