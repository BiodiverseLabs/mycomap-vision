import { useEffect, useState } from "react";
import { AlertTriangle, ImageOff, MapPin } from "lucide-react";
import type { PairPhoto, PairPlace } from "@/lib/api";
import { tileImage } from "@/lib/photoReview";
import { distanceText, placeLine, type PairSide } from "@/lib/twinCompare";
import { BODY, Ext, INK, PhotoLightbox, Placeholder, sciLike } from "./ReviewPhotos";

// A record pair side by side (One find, two places; One find posted to both): each copy's
// photos large enough to compare at a glance, the thumbnails beneath switching the large one,
// a click opening the viewer. MO on the left, iNat on the right, on a phone too: the point is
// to see both at once.

export function PairCompare({ sides, distanceKm, likelyWrong, places, placesLoading = false }: {
  sides: PairSide[]; distanceKm?: unknown; likelyWrong?: unknown;
  /** Where each copy is, as MO and iNat list it (asked apart from the item). */
  places?: Record<string, PairPlace>; placesLoading?: boolean;
}) {
  const far = distanceText(distanceKm);
  const wrong = likelyWrong == null ? "" : String(likelyWrong).trim();
  return (
    <div className="min-w-0 space-y-3" data-testid="feedback-pair">
      {(far || wrong) && (
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
          {far && (
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 font-semibold text-amber-900 dark:bg-amber-950 dark:text-amber-100">
              <MapPin className="h-3.5 w-3.5" aria-hidden /> {far}
            </span>
          )}
          {wrong && <span className="text-muted-foreground">Likely wrong: <span className={BODY}>{wrong}</span></span>}
        </p>
      )}
      <div className="grid grid-cols-2 gap-3">
        {sides.map((s) => <PairColumn key={s.key} side={s} place={places?.[s.key]} placeLoading={placesLoading} />)}
      </div>
    </div>
  );
}

function PairColumn({ side, place, placeLoading }: { side: PairSide; place?: PairPlace; placeLoading: boolean }) {
  const listed = placeLine(place, side.site, placeLoading);
  const [at, setAt] = useState(0);
  const [viewing, setViewing] = useState<number | null>(null);
  const photos = side.photos;
  const record = { record_key: side.key, link: side.link ?? "" };
  const stepBy = (d: number) => setViewing((v) => (v == null ? v : Math.min(photos.length - 1, Math.max(0, v + d))));
  useEffect(() => {
    if (viewing == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setViewing(null);
      else if (e.key === "ArrowRight") stepBy(1);
      else if (e.key === "ArrowLeft") stepBy(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [viewing != null]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <section className={`min-w-0 space-y-2 rounded-md border p-2 sm:p-3 ${
      side.suspect ? "border-amber-400 dark:border-amber-600" : "border-[#A87146]/20 dark:border-border"}`}
             aria-label={side.label} data-testid={`feedback-pair-${side.site}`}>
      <div className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5">
        <span className={`text-sm font-semibold ${INK}`}>
          {side.site === "MO" ? "Mushroom Observer" : "iNaturalist"}
        </span>
        {side.link ? <Ext href={side.link}>{side.label}</Ext> : <span className="text-xs text-muted-foreground">{side.label}</span>}
      </div>
      <p className="flex min-w-0 items-start gap-1 text-sm" data-testid={`feedback-pair-place-${side.site}`}>
        <MapPin className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
        <span className="sr-only">Listed on {side.site} at </span>
        <span className={`min-w-0 break-words ${listed.muted ? "text-muted-foreground" : `font-medium ${INK}`}`}>{listed.text}</span>
      </p>
      {photos.length > 0 ? (
        <>
          <MainPhoto photo={photos[Math.min(at, photos.length - 1)]} record={record}
                     onOpen={() => setViewing(Math.min(at, photos.length - 1))} />
          {photos.length > 1 && (
            <ul className="flex flex-wrap gap-1.5" aria-label={`${side.label} photos`}>
              {photos.map((p, i) => (
                <li key={p.photo_key}>
                  <Thumb photo={p} on={i === at} onPick={() => setAt(i)} label={`${side.label} photo ${i + 1}`} />
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <div className="flex aspect-[4/3] w-full flex-col items-center justify-center gap-1 rounded bg-[#f3efe9] p-2 text-center text-xs text-muted-foreground dark:bg-muted">
          <ImageOff className="h-5 w-5" aria-hidden />
          Vision holds no photos of this record.
          {side.link && <Ext href={side.link}>See them on {side.site}</Ext>}
        </div>
      )}
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 gap-y-0.5 text-xs">
        <dt className="text-muted-foreground">Label now</dt>
        <dd className={`min-w-0 break-words ${BODY}`}>
          {side.nameNow ? <span className={sciLike(side.nameNow) ? "sci" : ""}>{side.nameNow}</span> : "–"}
        </dd>
        <dt className="text-muted-foreground">Placed by</dt>
        <dd className={`min-w-0 break-words ${BODY}`}>{side.place || "–"}</dd>
      </dl>
      {side.suspect && (
        <p className="inline-flex items-center gap-1 text-xs font-medium text-amber-800 dark:text-amber-200">
          <AlertTriangle className="h-3.5 w-3.5" aria-hidden /> This copy's place looks wrong
        </p>
      )}
      {viewing != null && (
        <PhotoLightbox record={record} photos={photos} index={viewing} choices={{}}
                       onStep={stepBy}
                       onClose={() => setViewing(null)} hint="← / → photos · Esc close" />
      )}
    </section>
  );
}

function MainPhoto({ photo, record, onOpen }: { photo: PairPhoto; record: { record_key: string; link: string }; onOpen: () => void }) {
  const [failed, setFailed] = useState<string | null>(null);
  const src = failed === photo.photo_key ? null : tileImage(photo, "thumb");
  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded bg-[#f3efe9] dark:bg-muted">
      {src ? (
        <button type="button" onClick={onOpen} className="block h-full w-full" aria-label="Open the photo large">
          <img key={photo.photo_key} src={src} alt={`Photo ${photo.position ?? ""}`} referrerPolicy="no-referrer"
               decoding="async" draggable={false} onError={() => setFailed(photo.photo_key)}
               className="h-full w-full object-contain" />
        </button>
      ) : (
        <Placeholder photo={photo} record={record} failed={failed === photo.photo_key} />
      )}
      {photo.shown === "private" && (
        <span className="pointer-events-none absolute right-1 top-1 rounded bg-[#4a3728]/85 px-1.5 py-0.5 text-[10px] font-medium text-white">
          reviewers only · not public
        </span>
      )}
    </div>
  );
}

function Thumb({ photo, on, onPick, label }: { photo: PairPhoto; on: boolean; onPick: () => void; label: string }) {
  const [failed, setFailed] = useState(false);
  const src = failed ? null : tileImage(photo, "thumb");
  return (
    <button type="button" onClick={onPick} aria-pressed={on} aria-label={label}
            className={`block h-12 w-12 overflow-hidden rounded border-2 sm:h-14 sm:w-14 ${
              on ? "border-myco-green" : "border-transparent hover:border-myco-green/40"}`}>
      {src ? (
        <img src={src} alt="" loading="lazy" decoding="async" referrerPolicy="no-referrer" draggable={false}
             onError={() => setFailed(true)} className="h-full w-full object-cover" />
      ) : (
        <span className="flex h-full w-full items-center justify-center bg-[#f3efe9] text-[10px] text-muted-foreground dark:bg-muted">
          {photo.position ?? "?"}
        </span>
      )}
    </button>
  );
}
