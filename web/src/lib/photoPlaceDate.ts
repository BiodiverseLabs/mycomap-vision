/**
 * Place and date from a photo's EXIF, read in the browser as soon as a photo is added so the
 * "Where and when" fields fill in on screen. Same rules as the server's exif.py: GPS out of
 * range or exactly 0,0 counts as none; the date is DateTimeOriginal, else DateTime.
 */
import type { Where } from "@/lib/api";

export interface PhotoPlaceDate {
  lat: number | null;
  lng: number | null;
  date: string | null; // YYYY-MM-DD
}

const NONE: PhotoPlaceDate = { lat: null, lng: null, date: null };

/** Reads one photo; anything missing, odd or unreadable comes back as null. */
export async function readPhotoPlaceDate(photo: Blob): Promise<PhotoPlaceDate> {
  let tags: Record<string, unknown> | undefined;
  try {
    const { default: exifr } = await import("exifr"); // loaded only once someone adds a photo
    tags = await exifr.parse(await photo.arrayBuffer(), {
      gps: true,
      reviveValues: false,
      pick: ["DateTimeOriginal", "ModifyDate", "GPSLatitude", "GPSLatitudeRef", "GPSLongitude",
             "GPSLongitudeRef", "latitude", "longitude"],
    });
  } catch {
    return NONE;
  }
  if (!tags) return NONE;
  let lat = typeof tags.latitude === "number" ? tags.latitude : null;
  let lng = typeof tags.longitude === "number" ? tags.longitude : null;
  if (lat === null || lng === null || !Number.isFinite(lat) || !Number.isFinite(lng) ||
      Math.abs(lat) > 90 || Math.abs(lng) > 180 || (lat === 0 && lng === 0)) {
    lat = lng = null;
  }
  return { lat, lng, date: exifDate(tags.DateTimeOriginal) ?? exifDate(tags.ModifyDate) };
}

/**
 * "2026:09:27 14:05:00" (EXIF local time) -> "2026-09-27"; anything else -> null.
 * 1970-01-01 is a camera whose clock was never set, not a date: null, like the server's dates.py.
 */
export function exifDate(raw: unknown): string | null {
  const m = typeof raw === "string" ? /^\s*(\d{4}):(\d{2}):(\d{2})/.exec(raw) : null;
  if (!m) return null;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const day = new Date(Date.UTC(y, mo - 1, d));
  if (y < 1900 || day.getUTCMonth() !== mo - 1 || day.getUTCDate() !== d) return null;
  if (y === 1970 && mo === 1 && d === 1) return null;
  return `${m[1]}-${m[2]}-${m[3]}`;
}

/**
 * Fills the place and the date from the first photo that has each, like the server does.
 * A field the person typed is never touched. A field filled from a photo is refilled when the
 * photos change, and cleared when no photo left carries it.
 * `found` is in photo order; undefined means that photo hasn't been read yet.
 */
export function fillFromPhotos(where: Where, found: (PhotoPlaceDate | undefined)[]): Where {
  const next: Where = { ...where };
  const placeIsOurs = (!where.lat?.trim() && !where.lng?.trim()) || where.placeFromPhoto != null;
  const dateIsOurs = !where.observedOn || where.dateFromPhoto != null;
  if (placeIsOurs) {
    const i = found.findIndex((f) => f?.lat != null && f.lng != null);
    const f = found[i];
    if (f && f.lat != null && f.lng != null) {
      Object.assign(next, { lat: f.lat.toFixed(4), lng: f.lng.toFixed(4), placeFromPhoto: i + 1 });
    } else {
      Object.assign(next, { lat: "", lng: "", placeFromPhoto: undefined });
    }
  }
  if (dateIsOurs) {
    const i = found.findIndex((f) => f?.date);
    const f = found[i];
    Object.assign(next, f?.date
      ? { observedOn: f.date, dateFromPhoto: i + 1 }
      : { observedOn: "", dateFromPhoto: undefined });
  }
  return next;
}
