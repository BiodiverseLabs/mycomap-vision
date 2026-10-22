import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { DepthCompanion } from "../components/DepthCompanion";
import { companionBars } from "./compareView";

// Dataset release v1's validation split, the served comparison (record set 36fd876aa871): the
// reference's species (reference_species.json) and the heatmap's finds (run 169).
const SPECIES = { answer_key: "species-rank-2026-10-18", records: 155648, species: 16809,
                  bands: { "1-4": 11131, "5-19": 3710, "20-99": 1781, "100+": 187 } };
const FINDS = [649, 1391, 2864, 5736, 2158];

test("the companion's two bars share out the reference's species and the test finds by band", () => {
  const bars = companionBars(SPECIES, FINDS);
  assert.deepEqual(bars.species!.map((s) => [s.band, s.share]),
                   [["1-4", 66], ["5-19", 22], ["20-99", 11], ["100+", 1]]);
  assert.deepEqual(bars.finds.map((s) => [s.band, s.share]),
                   [["0", 5], ["1-4", 11], ["5-19", 22], ["20-99", 45], ["100+", 17]]);
  assert.equal(companionBars(null, FINDS).species, null);
});

test("the chart renders both bars, with each band's share", () => {
  const html = renderToStaticMarkup(createElement(DepthCompanion, { species: SPECIES, finds: FINDS }));
  assert.match(html, /data-testid="companion-species"/);
  assert.match(html, /data-testid="companion-finds"/);
  assert.match(html, /Species in the reference set \(16,809\)/);
  assert.match(html, /Test finds \(12,798\)/);
  assert.match(html, /1-4: 66%/);
  assert.match(html, /1-4: 11%/);
  assert.match(html, /most finds are of common species/);
  // Without the reference's species there is nothing to set the finds against.
  assert.equal(renderToStaticMarkup(createElement(DepthCompanion, { species: null, finds: FINDS })), "");
});
