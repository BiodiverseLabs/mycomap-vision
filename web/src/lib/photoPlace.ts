/**
 * The place of a find on a phone (Steve, 2026-10-11). Phones usually hand a web page their
 * photos without the location: iOS Safari keeps it only when Options → Location is on in
 * the photo picker (iOS 17+, library photos only; never for a photo taken in the picker),
 * and Android's photo picker removes it. The date survives. So when no photo carries a
 * place, a phone is offered its current location, and the page says where each value came
 * from.
 */
import type { Where } from "@/lib/api";
import type { PhotoPlaceDate } from "@/lib/photoPlaceDate";

export type Platform = "ios" | "android" | "other";

/** iPadOS asks for desktop pages with a Mac user agent; its touch points give it away. */
export function platformOf(userAgent: string, maxTouchPoints = 0): Platform {
  if (/iPhone|iPad|iPod/.test(userAgent)) return "ios";
  if (/Macintosh/.test(userAgent) && maxTouchPoints > 1) return "ios";
  if (/Android/.test(userAgent)) return "android";
  return "other";
}

export const NO_PLACE_TIP: Record<Exclude<Platform, "other">, string> = {
  ios: "On iPhone, tap Options when you choose photos and turn on Location; the place then comes from the photos.",
  android: "Android takes the location out of photos shared with websites, so it can't come from the photos.",
};

/**
 * The offer of the phone's current location, or null. Only on a phone, once every photo
 * has been read and none carries a place, and while no place is filled in.
 */
export function noPhotoPlace(s: {
  found: (PhotoPlaceDate | undefined)[];
  where: Where;
  platform: Platform;
}): { tip: string } | null {
  if (s.platform === "other" || !s.found.length) return null;
  if (s.found.some((f) => f === undefined)) return null;                  // still reading
  if (s.found.some((f) => f!.lat != null && f!.lng != null)) return null;  // a photo has one
  if (s.where.lat?.trim() || s.where.lng?.trim()) return null;            // already filled in
  return { tip: NO_PLACE_TIP[s.platform] };
}

/** Under the fields: where the place and the date shown came from. */
export function placeNote(w: Where): string | null {
  const place = w.placeFromPhoto ? `photo ${w.placeFromPhoto}` : w.placeFromDevice ? "your current location" : null;
  const date = w.dateFromPhoto ? `photo ${w.dateFromPhoto}` : null;
  if (place && place === date) return `Place and date from ${place}.`;
  return [place && `Place from ${place}.`, date && `Date from ${date}.`].filter(Boolean).join(" ") || null;
}

/** In the line above the answer: where the place or date the server used came from. */
export function sourceWords(from: "entered" | "photo" | null, sentFromPhoto?: number,
                            sentFromDevice?: boolean): string {
  if (from === "photo") return "from your photo";        // the server read it from a photo itself
  if (sentFromPhoto) return `from photo ${sentFromPhoto}`;
  if (sentFromDevice) return "your current location";
  return "as entered";
}

/** A failed "Use my current location", in words (GeolocationPositionError codes). */
export function locateError(code: number): string {
  return code === 1
    ? "Location is blocked for this site. Allow it in your browser's settings, or type the coordinates."
    : "Couldn't get your location. Try again, or type the coordinates.";
}
