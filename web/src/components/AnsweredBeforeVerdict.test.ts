// Results' "Answered before the DNA verdict" card, rendered from its data alone.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnsweredBeforeVerdictCard } from "./AnsweredBeforeVerdict";
import { dayRange, leftOutLines } from "../lib/earlyAnswers";
import type { EarlyAnswersGroup } from "../lib/api";

const cell = (n: number, top1: number, top3: number, top5: number, top10: number | null, n_top10 = n) =>
  ({ n, top1, top3, top5, top10, n_top10 });

function group(over: Partial<EarlyAnswersGroup> = {}): EarlyAnswersGroup {
  return {
    backbone: "bioclip-2-336-ft-20261018-171420", method: "nearest+mean-records-balanced+place",
    context: "photos + place/date", answers: 40, scored: 29, awaiting_verdict: 6,
    asked_by: { "mycomap.org": 38, site: 2 },
    left_out: { answered_after_verdict: 3, in_reference: 0, twin_in_reference: 0,
                vision_assisted: 2, not_a_fungus_label: 0 },
    answered: { from: "2026-10-20", through: "2026-10-23" },
    validated: { from: "2026-10-20", through: "2026-10-23" },
    top_k: {
      "species strict": cell(27, 0.5556, 0.7037, 0.7778, 0.8889, 18),
      "species s.l.": cell(27, 0.5926, 0.7407, 0.8148, 0.8889, 18),
      "species complex (beta)": cell(27, 0.6296, 0.7778, 0.8519, 0.9444, 18),
      "genus strict": cell(29, 0.8621, 0.931, 0.9655, 1, 20),
      "genus s.l.": cell(29, 0.8966, 0.931, 0.9655, 1, 20),
    },
    species_by_true_species_reference_records: {
      "0": { n: 2, top1: 0, top5: 0 }, "1-4": { n: 5, top1: 0.2, top5: 0.6 },
      "5-19": { n: 8, top1: 0.5, top5: 0.875 }, "20-99": { n: 9, top1: 0.7778, top5: 0.8889 },
      "100+": { n: 3, top1: 1, top5: 1 },
    },
    inat_cv: { status: "not run" },
    ...over,
  };
}

const render = (groups: EarlyAnswersGroup[], sealedAfter = "2026-10-12T13:17:51+00:00") =>
  renderToStaticMarkup(createElement(AnsweredBeforeVerdictCard, { groups, sealedAfter }));

test("the card shows top 1/3/5/10 for species strict, s.l., complex and genus, with n", () => {
  const html = render([group()]);
  assert.match(html, /Answered before the DNA verdict/);
  for (const level of ["species strict", "species s.l.", "species complex (beta)", "genus strict", "genus s.l."]) {
    assert.ok(html.includes(level), level);
  }
  for (const k of [1, 3, 5, 10]) assert.ok(html.includes(`Top ${k}`), `top ${k}`);
  assert.ok(html.includes("55.6%") && html.includes("88.9%"), "species strict top 1 and top 10");
  assert.ok(html.includes("27 (18)"), "top 10's own n when some answers are five deep");
  assert.ok(html.includes("photos + place/date"));
});

test("the card says how many records were scored and over which dates", () => {
  const html = render([group()]);
  assert.match(html, /29 records scored/);
  assert.match(html, /answered 20 Oct – 23 Oct 2026/);
  assert.match(html, /; 6 awaiting their verdict/);
  assert.equal(dayRange({ from: "2026-10-21", through: "2026-10-21" }), "21 Oct 2026");
  assert.equal(dayRange({ from: null, through: null }), null);
});

test("the card counts records left out as validated with Vision's help, even when none", () => {
  assert.match(render([group()]), /Left out: validated with Vision&#x27;s help: 2; answered on or after the verdict: 3/);
  const none = group({ left_out: { answered_after_verdict: 0, in_reference: 0, twin_in_reference: 0,
                                   vision_assisted: 0, not_a_fungus_label: 0 } });
  assert.deepEqual(leftOutLines(none), ["validated with Vision's help: 0"]);
});

test("the card gives species by reference depth and says iNat CV was not run", () => {
  const html = render([group()]);
  for (const band of ["0", "1-4", "5-19", "20-99", "100+"]) assert.ok(html.includes(`>${band}<`), band);
  assert.match(html, /computer vision: not run on these records/);
});

test("the sealed test is a count only, and only when the server gives one (signed in)", () => {
  assert.ok(!render([group()]).includes("sealed test"), "signed out: no count at all");
  const html = render([group({ sealed: 4 }), group({ context: "photos only", sealed: 1 })]);
  assert.match(html, /their 5 answers are counted but not scored/);
  assert.match(html, /validated since 2026-10-12/);
});

test("with nothing kept yet the card says so", () => {
  assert.match(render([]), /None yet\./);
});
