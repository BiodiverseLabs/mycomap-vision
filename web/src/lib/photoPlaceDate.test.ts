import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { exifDate, fillFromPhotos, readPhotoPlaceDate, type PhotoPlaceDate } from "./photoPlaceDate";

// Tiny JPEGs written by PIL with known EXIF (see the commit that added them).
const photo = (name: string) =>
  new Blob([readFileSync(new URL(`./fixtures/${name}`, import.meta.url))], { type: "image/jpeg" });

test("a phone photo's GPS and original date are read", async () => {
  assert.deepEqual(await readPhotoPlaceDate(photo("gps-and-date.jpg")),
                   { lat: 42.28, lng: -83.74, date: "2026-09-27" });
});

test("south and east hemispheres keep their signs", async () => {
  const pd = await readPhotoPlaceDate(photo("southern.jpg"));
  assert.ok(Math.abs(pd.lat! - -33.8667) < 1e-4);
  assert.equal(pd.lng, 151.2);
  assert.equal(pd.date, null);
});

test("the plain DateTime is used when there is no original date", async () => {
  assert.deepEqual(await readPhotoPlaceDate(photo("date-only.jpg")),
                   { lat: null, lng: null, date: "2025-10-03" });
});

test("a 0,0 GPS block is an unset one, not a place", async () => {
  assert.deepEqual(await readPhotoPlaceDate(photo("zero-gps.jpg")), { lat: null, lng: null, date: null });
});

test("a photo without EXIF, or a file that isn't an image, gives nothing and doesn't throw", async () => {
  const none = { lat: null, lng: null, date: null };
  assert.deepEqual(await readPhotoPlaceDate(photo("no-exif.jpg")), none);
  assert.deepEqual(await readPhotoPlaceDate(new Blob(["not an image"])), none);
});

test("EXIF dates that aren't real days are ignored", () => {
  assert.equal(exifDate("2026:09:27 14:05:00"), "2026-09-27");
  assert.equal(exifDate("0000:00:00 00:00:00"), null);
  assert.equal(exifDate("2026:02:30 10:00:00"), null);
  assert.equal(exifDate("    :  :     :  :  "), null);
  assert.equal(exifDate(undefined), null);
});

const A: PhotoPlaceDate = { lat: 42.28, lng: -83.74, date: "2026-09-27" };
const DATE_ONLY: PhotoPlaceDate = { lat: null, lng: null, date: "2025-10-03" };
const NOTHING: PhotoPlaceDate = { lat: null, lng: null, date: null };

test("a 1970-01-01 EXIF date is a camera clock never set, not a date", async () => {
  assert.equal(exifDate("1970:01:01 00:00:00"), null);
  assert.equal(exifDate("1970:01:02 00:00:00"), "1970-01-02");
  // The photo's place is still read; its date is not, so it can't fill the form:
  // the next photo's real date does.
  const unset = await readPhotoPlaceDate(photo("unset-clock.jpg"));
  assert.deepEqual(unset, { lat: 42.28, lng: -83.74, date: null });
  assert.deepEqual(fillFromPhotos({}, [unset, DATE_ONLY]),
                   { lat: "42.2800", lng: "-83.7400", placeFromPhoto: 1,
                     observedOn: "2025-10-03", dateFromPhoto: 2 });
});

test("empty fields fill from the first photo that has each", () => {
  assert.deepEqual(fillFromPhotos({}, [NOTHING, DATE_ONLY, A]),
                   { lat: "42.2800", lng: "-83.7400", placeFromPhoto: 3,
                     observedOn: "2025-10-03", dateFromPhoto: 2 });
});

test("what the person typed is never overwritten", () => {
  const typed = { lat: "40.1", lng: "-80.2", observedOn: "2026-01-02" };
  assert.deepEqual(fillFromPhotos(typed, [A]), typed);
});

test("a filled place or date follows the photos: replaced, then cleared when its photo goes", () => {
  const filled = fillFromPhotos({}, [A]);
  assert.deepEqual(fillFromPhotos(filled, [DATE_ONLY]),
                   { lat: "", lng: "", placeFromPhoto: undefined,
                     observedOn: "2025-10-03", dateFromPhoto: 1 });
  assert.deepEqual(fillFromPhotos(filled, []),
                   { lat: "", lng: "", placeFromPhoto: undefined,
                     observedOn: "", dateFromPhoto: undefined });
});

test("a typed date stays while the place still fills from a photo", () => {
  assert.deepEqual(fillFromPhotos({ observedOn: "2026-01-02" }, [A]),
                   { observedOn: "2026-01-02", lat: "42.2800", lng: "-83.7400", placeFromPhoto: 1 });
});
