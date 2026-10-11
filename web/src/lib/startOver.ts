/**
 * "Start over" on the Identify page: one find's photos, place and date, and the answer go
 * together, so they clear together. The model choice is a preference and stays.
 */
import type { Where } from "@/lib/api";
import type { PhotoPlaceDate } from "@/lib/photoPlaceDate";

/** The part of the Identify form that belongs to one find. */
export interface FindForm {
  files: File[];
  where: Where;
  /** Place and date read from each photo, so the fields can fill in. */
  found: Map<File, PhotoPlaceDate>;
}

/** A blank form for the next find: no photos, no place or date (typed or read from a photo). */
export function startOver(): FindForm {
  return { files: [], where: {}, found: new Map() };
}

/**
 * Whether Start over is offered: only when there is something to clear, and never while an
 * identification is on its way (its answer would land on the blank form).
 */
export function canStartOver(s: {
  files: File[];
  where: Where;
  hasAnswer: boolean;
  hasError: boolean;
  identifying: boolean;
}): boolean {
  if (s.identifying) return false;
  const typed = !!(s.where.lat?.trim() || s.where.lng?.trim() || s.where.observedOn);
  return s.files.length > 0 || typed || s.hasAnswer || s.hasError;
}
