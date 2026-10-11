/**
 * Picking the place of a find on a map (Steve, 2026-10-11): for people who'd rather not
 * share their current location. The pin and the latitude/longitude fields stay in step
 * both ways; these are the rules, the map itself is components/LocationPicker.tsx.
 */
import type { Where } from "@/lib/api";

export interface Point {
  lat: number;
  lng: number;
}

/** The map opens here when the form has no place yet: North America, where most finds are. */
export const DEFAULT_VIEW = { center: { lat: 42, lng: -98 }, zoom: 4 } as const;
/** How close the map opens on a place it already has. */
export const PLACE_ZOOM = 13;

/** A point from typed fields, or null while they don't make one (empty, partial, out of range). */
export function pointFromFields(lat: string | undefined, lng: string | undefined): Point | null {
  if (!lat?.trim() || !lng?.trim()) return null;
  const a = Number(lat.trim()), o = Number(lng.trim());
  if (!Number.isFinite(a) || !Number.isFinite(o) || Math.abs(a) > 90 || Math.abs(o) > 180) return null;
  return { lat: a, lng: o };
}

/** A map can be panned round the world more than once: bring a longitude back to -180..180. */
export function wrapLng(lng: number): number {
  const w = ((((lng + 180) % 360) + 360) % 360) - 180;
  return w === -180 && lng > 0 ? 180 : w;
}

/** The fields for a pin, at the form's precision (4 decimals, about 10 m). */
export function fieldsFromPoint(p: Point): { lat: string; lng: string } {
  const lat = Math.max(-90, Math.min(90, p.lat));
  return { lat: lat.toFixed(4), lng: wrapLng(p.lng).toFixed(4) };
}

/** Where the map opens: on the form's place with its pin, else the default view, no pin. */
export function startView(w: Where): { center: Point; zoom: number; pin: Point | null } {
  const pin = pointFromFields(w.lat, w.lng);
  return pin ? { center: pin, zoom: PLACE_ZOOM, pin } : { ...DEFAULT_VIEW, pin: null };
}

/** The form after "Use this place": the pin's place, marked as picked on the map. */
export function applyPick(w: Where, pin: Point): Where {
  return { ...w, ...fieldsFromPoint(pin), placeFromMap: true, placeFromPhoto: undefined,
           placeFromDevice: undefined };
}
