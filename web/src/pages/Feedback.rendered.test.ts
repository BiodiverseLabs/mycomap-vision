// Human Feedback, rendered to markup from its loaded queries: the Photos | Taxonomy tabs a viewer
// meets first (Steve, 2026-11-20). Strings only: no DOM.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { FeedbackMe, FeedbackSummary } from "../lib/api";
import { FeedbackPage } from "./Feedback";

const fq = (type: string, section: string, open: number, title: string): FeedbackSummary => ({
  type, title, about: "", granularity: "record", model_flagged: false, section, blind: false, open, decided: 3,
});
const SUMMARY: FeedbackSummary[] = [
  fq("non_specimen_photo", "cleanup", 100, "Photos that show no specimen"),
  fq("name_retired", "cleanup", 5, "Names retired on MycoMap's synonym list"),
  fq("recovered_label", "cleanup", 20, "Recovered labels"),
  fq("photo_view", "views", 3, "Photo views"),
  fq("photos_unlike_label", "validation", 7, "Photos unlike their label"),
];
const ME: FeedbackMe = { user: { id: "1", name: "R" }, read_only: false, admin: false, projects: [] };
// An account below Contributor (Steve, 2026-11-20): the Photos tab only.
const MEMBER: FeedbackMe = { ...ME, taxonomy: false, all_photos: false };

function render(search = "", me: FeedbackMe = ME): string {
  (globalThis as { location?: unknown }).location = { search };
  const client = new QueryClient();
  client.setQueryData(["feedback", "me"], me);
  client.setQueryData(["feedback", "summary"], SUMMARY);
  client.setQueryData(["feedback", "catalogue"], []);
  return renderToStaticMarkup(createElement(QueryClientProvider, { client }, createElement(FeedbackPage)));
}

const selectedTab = (html: string) => /aria-selected="true"[^>]*data-testid="feedback-tab-(\w+)"/.exec(html)?.[1];

test("the page opens on the Photos tab, with only the three photo reviews and what each asks for", () => {
  const html = render();
  assert.equal(selectedTab(html), "photos");
  assert.ok(html.indexOf('data-testid="feedback-tab-photos"') < html.indexOf('data-testid="feedback-tab-taxonomy"'),
            "Photos is the first tab");
  const cards = [...html.matchAll(/data-testid="feedback-card-(\w+)"/g)].map((m) => m[1]);
  assert.deepEqual(cards, ["non_specimen_photo", "recovered_label", "photo_view"]);
  assert.match(html, /Tag what each photo shows/);
  assert.match(html, /Pick one of the reviews above\./);
  assert.match(html, /123 open/, "the Photos tab counts its open photos");
  assert.match(html, /12 open/, "the Taxonomy tab counts its open items");
});

test("a ?tab=taxonomy link opens Taxonomy with the name and validation queues by section, no photo review", () => {
  const html = render("?tab=taxonomy");
  assert.equal(selectedTab(html), "taxonomy");
  const cards = [...html.matchAll(/data-testid="feedback-card-(\w+)"/g)].map((m) => m[1]);
  assert.deepEqual(cards, ["name_retired", "photos_unlike_label"]);
  assert.match(html, /Data cleanup/);
  assert.match(html, /Data validation/);
});

test("an open photo review starts with its steps; a name queue has none", () => {
  const html = render("?queue=non_specimen_photo");
  assert.match(html, /data-testid="feedback-steps-non_specimen_photo"/);
  assert.match(html, /How to review Photos that show no specimen/);
  assert.match(html, /exclude it as Dried specimen/);
  assert.doesNotMatch(render("?queue=name_retired"), /feedback-steps-/);
});

test("the How to review box starts folded (Steve, 2026-11-20)", () => {
  const html = render("?queue=non_specimen_photo");
  const box = html.match(/<details[^>]*data-testid="feedback-steps-non_specimen_photo"[^>]*>/);
  assert.ok(box, "the steps box is there");
  assert.doesNotMatch(box[0], /\sopen(=|\s|>)/, "the steps box is not open on arrival");
});

test("an account below Contributor gets no Taxonomy tab, even from a Taxonomy link, and is told why", () => {
  for (const search of ["", "?tab=taxonomy", "?queue=name_retired"]) {
    const html = render(search, MEMBER);
    assert.doesNotMatch(html, /data-testid="feedback-tab-taxonomy"/, search);
    assert.equal(selectedTab(html), "photos", search);
    assert.doesNotMatch(html, /data-testid="feedback-card-name_retired"/, search);
    assert.match(html, /Taxonomy queues are for mycomap\.org Contributors, Lab Contributors, Project Leaders and admins/);
  }
  assert.match(render("", { ...ME, taxonomy: true }), /data-testid="feedback-tab-taxonomy"/);
});

test("a ?queue= link to a name queue opens it on the Taxonomy tab", () => {
  const html = render("?queue=name_retired");
  assert.equal(selectedTab(html), "taxonomy");
  assert.match(html, /data-testid="feedback-queue-section">Taxonomy · Data cleanup</);
});
