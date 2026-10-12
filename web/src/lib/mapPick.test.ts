import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { applyPick, DEFAULT_VIEW, fieldsFromPoint, PLACE_ZOOM, pointFromFields, startView, wrapLng } from "./mapPick";
import { placeNote, sourceWords } from "./photoPlace";

test("typed fields make a pin only when they make a real point", () => {
  assert.deepEqual(pointFromFields("44.5646", "-123.2620"), { lat: 44.5646, lng: -123.262 });
  assert.deepEqual(pointFromFields(" 0 ", " 0 "), { lat: 0, lng: 0 });
  assert.equal(pointFromFields("44.5", ""), null, "half typed");
  assert.equal(pointFromFields("abc", "-123"), null);
  assert.equal(pointFromFields("91", "0"), null);
  assert.equal(pointFromFields("0", "-180.5"), null);
  assert.equal(pointFromFields(undefined, undefined), null);
});

test("a pin gives the fields at the form's precision, round the world back to -180..180", () => {
  assert.deepEqual(fieldsFromPoint({ lat: 44.564512, lng: -123.262049 }), { lat: "44.5645", lng: "-123.2620" });
  assert.deepEqual(fieldsFromPoint({ lat: 10, lng: 236.7 }), { lat: "10.0000", lng: "-123.3000" });
  assert.equal(wrapLng(-200), 160);
  assert.equal(wrapLng(180), 180);
  assert.equal(wrapLng(540), 180);
  assert.deepEqual(fieldsFromPoint({ lat: 95, lng: 0 }).lat, "90.0000");
});

test("fields → pin → fields comes back the same", () => {
  const typed = { lat: "44.5645", lng: "-123.2620" };
  assert.deepEqual(fieldsFromPoint(pointFromFields(typed.lat, typed.lng)!), typed);
});

test("the map opens on the form's place, or on North America with no pin", () => {
  assert.deepEqual(startView({ lat: "44.5645", lng: "-123.2620" }),
                   { center: { lat: 44.5645, lng: -123.262 }, zoom: PLACE_ZOOM, pin: { lat: 44.5645, lng: -123.262 } });
  assert.deepEqual(startView({}), { ...DEFAULT_VIEW, pin: null });
  assert.deepEqual(startView({ lat: "44.5", lng: "" }), { ...DEFAULT_VIEW, pin: null });
});

test("using a picked place fills the fields and says it was picked on the map", () => {
  const before = { lat: "1", lng: "2", placeFromDevice: true, observedOn: "2026-10-04", dateFromPhoto: 1 };
  const after = applyPick(before, { lat: 44.56451, lng: -123.26204 });
  assert.deepEqual(after, { lat: "44.5645", lng: "-123.2620", placeFromMap: true, placeFromPhoto: undefined,
                            placeFromDevice: undefined, observedOn: "2026-10-04", dateFromPhoto: 1 });
  assert.equal(placeNote(after), "Place picked on the map. Date from photo 1.");
  assert.equal(sourceWords("entered", undefined, false, true), "picked on the map");
});

test("the map loads only when someone opens it", () => {
  const src = (f: string) => readFileSync(new URL(f, import.meta.url), "utf-8");
  for (const f of ["../pages/Identify.tsx", "../main.tsx", "../App.tsx"]) {
    const s = src(f);
    assert.doesNotMatch(s, /from ["'](leaflet|react-leaflet)["']/, `${f} imports the map library`);
    assert.doesNotMatch(s, /from ["']@\/components\/LocationPicker["']/, `${f} imports the picker eagerly`);
  }
  assert.match(src("../pages/Identify.tsx"), /lazy\(\(\) => import\("@\/components\/LocationPicker"\)\)/);
});
