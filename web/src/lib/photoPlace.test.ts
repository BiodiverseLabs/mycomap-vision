import { test } from "node:test";
import assert from "node:assert/strict";
import { locateError, NO_PLACE_TIP, noPhotoPlace, placeNote, platformOf, sourceWords } from "./photoPlace";
import type { PhotoPlaceDate } from "./photoPlaceDate";

const IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1";
const IPAD_DESKTOP = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
const ANDROID = "Mozilla/5.0 (Linux; Android 15; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36";
const WINDOWS = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36";

test("phones are told apart from computers, iPads in desktop mode included", () => {
  assert.equal(platformOf(IPHONE, 5), "ios");
  assert.equal(platformOf(IPAD_DESKTOP, 5), "ios");
  assert.equal(platformOf(IPAD_DESKTOP, 0), "other", "a Mac");
  assert.equal(platformOf(ANDROID, 5), "android");
  assert.equal(platformOf(WINDOWS, 0), "other");
});

const STRIPPED: PhotoPlaceDate = { lat: null, lng: null, date: "2026-10-04" };   // what the pickers hand over
const WITH_GPS: PhotoPlaceDate = { lat: 44.0476, lng: -123.087, date: "2026-10-04" };

test("a phone whose photos carry no place is offered its current location, with its own tip", () => {
  assert.deepEqual(noPhotoPlace({ found: [STRIPPED, STRIPPED], where: {}, platform: "ios" }),
                   { tip: NO_PLACE_TIP.ios });
  assert.match(NO_PLACE_TIP.ios, /Options/);
  assert.deepEqual(noPhotoPlace({ found: [STRIPPED], where: { lat: "", lng: " " }, platform: "android" }),
                   { tip: NO_PLACE_TIP.android });
});

test("the offer waits, steps aside, or never shows when it isn't needed", () => {
  assert.equal(noPhotoPlace({ found: [], where: {}, platform: "ios" }), null, "no photos yet");
  assert.equal(noPhotoPlace({ found: [STRIPPED, undefined], where: {}, platform: "ios" }), null, "still reading");
  assert.equal(noPhotoPlace({ found: [STRIPPED, WITH_GPS], where: {}, platform: "ios" }), null, "a photo has a place");
  assert.equal(noPhotoPlace({ found: [STRIPPED], where: { lat: "44.5", lng: "-123.2" }, platform: "ios" }), null,
               "a place is filled in");
  assert.equal(noPhotoPlace({ found: [STRIPPED], where: {}, platform: "other" }), null,
               "on a computer the current location is rarely where the find was");
});

test("the note under the fields says where the place and the date came from", () => {
  assert.equal(placeNote({ placeFromPhoto: 2, dateFromPhoto: 2 }), "Place and date from photo 2.");
  assert.equal(placeNote({ placeFromDevice: true, dateFromPhoto: 1 }),
               "Place from your current location. Date from photo 1.");
  assert.equal(placeNote({ placeFromPhoto: 1 }), "Place from photo 1.");
  assert.equal(placeNote({ lat: "44.5", lng: "-123.2" }), null, "typed: nothing to say");
});

test("the line above the answer names the place's source", () => {
  assert.equal(sourceWords("photo"), "from your photo");
  assert.equal(sourceWords("entered", 3), "from photo 3");
  assert.equal(sourceWords("entered", undefined, true), "your current location");
  assert.equal(sourceWords("entered"), "as entered");
});

test("a failed location request says what to do", () => {
  assert.match(locateError(1), /blocked/);
  assert.match(locateError(3), /Try again/);
});
