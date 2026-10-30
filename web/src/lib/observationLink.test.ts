import { test } from "node:test";
import assert from "node:assert/strict";
import { hiddenPhotosLine, observationFromQuery, parseObservationLink } from "./observationLink";

test("an iNaturalist observation link is read, from any site of the network", () => {
  for (const link of [
    "https://www.inaturalist.org/observations/385050809",
    "inaturalist.org/observations/385050809?tab=ids",
    "https://inaturalist.ca/observations/385050809",
    "https://www.naturalista.mx/observations/385050809#activity",
    "  https://www.inaturalist.org/observations/385050809/  ",
  ]) {
    assert.deepEqual(parseObservationLink(link), { source: "inat", id: "385050809" }, link);
  }
});

test("a Mushroom Observer link is read in each of its forms", () => {
  for (const link of [
    "https://mushroomobserver.org/123456",
    "mushroomobserver.org/obs/123456",
    "https://mushroomobserver.org/observations/123456?q=1",
    "https://mushroomobserver.org/observer/show_observation/123456",
  ]) {
    assert.deepEqual(parseObservationLink(link), { source: "mo", id: "123456" }, link);
  }
});

test("inat:<n> and mo:<n> are taken as shorthand", () => {
  assert.deepEqual(parseObservationLink("inat:42"), { source: "inat", id: "42" });
  assert.deepEqual(parseObservationLink("MO #7"), { source: "mo", id: "7" });
});

test("links that name no observation of either site are refused", () => {
  for (const link of [
    "", "385050809", "https://www.inaturalist.org/taxa/48715", "https://www.inaturalist.org/observations/abc",
    "https://www.inaturalist.org/observations/0", "https://mushroomobserver.org/names/123",
    "https://example.com/observations/5", "https://evil.example/inaturalist.org/observations/5",
    "https://inaturalist.org.evil.example/observations/5", "javascript:alert(1)",
    "https://www.inaturalist.org/observations/1234567890123",
  ]) {
    assert.equal(parseObservationLink(link), null, link);
  }
});

test("a link to the page with ?observation= names the observation to identify", () => {
  assert.deepEqual(observationFromQuery("?observation=inat:385050809"),
                   { text: "inat:385050809", ref: { source: "inat", id: "385050809" } });
  assert.deepEqual(observationFromQuery("?x=1&observation=mo%3A123")?.ref, { source: "mo", id: "123" });
  assert.deepEqual(
    observationFromQuery("?observation=" + encodeURIComponent("https://www.inaturalist.org/observations/42"))?.ref,
    { source: "inat", id: "42" });
  for (const search of ["", "?observation=", "?observation=12345", "?observation=gbif:1", "?other=inat:1"]) {
    assert.equal(observationFromQuery(search), null, search);
  }
});

const OBS = { source: "inat" as const, url: "https://www.inaturalist.org/observations/213312794" };

test("photos that aren't Creative Commons get a line linking to the observation at its source (Steve, 30 Oct)", () => {
  assert.deepEqual(hiddenPhotosLine(["", "", ""], OBS), {
    note: "These photos aren't Creative Commons", href: OBS.url, link: "See them on iNaturalist",
  });
  assert.deepEqual(hiddenPhotosLine(["https://a/1.jpg", "", "https://a/3.jpg"], OBS), {
    note: "1 of these photos isn't Creative Commons", href: OBS.url, link: "See it on iNaturalist",
  });
  assert.equal(hiddenPhotosLine(["https://a/1.jpg", "", ""], OBS)?.note, "2 of these photos aren't Creative Commons");
  assert.equal(hiddenPhotosLine([""], OBS)?.note, "This photo isn't Creative Commons");
  assert.equal(hiddenPhotosLine([""], { source: "mo", url: "https://mushroomobserver.org/1" })?.link,
               "See it on Mushroom Observer");
});

test("no line when every photo shows, or there is no observation to link to", () => {
  assert.equal(hiddenPhotosLine(["https://a/1.jpg", "https://a/2.jpg"], OBS), null);
  assert.equal(hiddenPhotosLine(["", ""], null), null);
  assert.equal(hiddenPhotosLine(["", ""], undefined), null);
  assert.equal(hiddenPhotosLine(["", ""], { source: "inat", url: null }), null);
});
