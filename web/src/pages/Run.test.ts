// A run's page (/r/<id>) for an observation whose photos aren't Creative Commons: each
// placeholder and the line under the photos link to the observation, in a new tab
// (Steve, 2026-10-30).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { RunFacts } from "./Run";
import { EXAMPLE_RUN } from "../lib/exampleRun.fixture";
import type { RunView } from "../lib/api";

const URL_ = EXAMPLE_RUN.observation!.url!;
const withShown = (shown: (string | null)[]): RunView =>
  ({ ...EXAMPLE_RUN, observation: { ...EXAMPLE_RUN.observation!, photos_shown: shown } }) as RunView;
const count = (h: string, s: string) => h.split(s).length - 1;

test("photos that aren't Creative Commons are placeholders linking to the observation, with a line under them", () => {
  const h = renderToStaticMarkup(createElement(RunFacts, { r: withShown([null, "https://e/2.jpg", null]) }));
  assert.equal(count(h, 'data-testid="run-photo-missing"'), 2);
  assert.match(h, /<a href="[^"]+" target="_blank" rel="noreferrer" class="aspect-square[^"]*" title="Open on iNaturalist" data-testid="run-photo-missing"/);
  assert.match(h, /data-testid="run-hidden-photos"/);
  assert.match(h, /2 of these photos aren&#x27;t Creative Commons/);
  assert.match(h, /See them on iNaturalist/);
  assert.ok(count(h, `href="${URL_}" target="_blank"`) >= 3, "both placeholders and the line link the observation");
});

test("every photo shown: no placeholder and no line", () => {
  const h = renderToStaticMarkup(createElement(RunFacts, { r: withShown(["https://e/1.jpg", "https://e/2.jpg"]) }));
  assert.doesNotMatch(h, /run-photo-missing|run-hidden-photos/);
});
