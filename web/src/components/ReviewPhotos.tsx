import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUpRight, Check, ChevronLeft, ChevronRight, Lock, X } from "lucide-react";
import type { FeedbackStatus, ReviewPhoto } from "@/lib/api";
import { recordLabel } from "@/lib/feedback";
import { openOnLabel, placeholderText, QUICK_REASONS, quickReasons, tileImage, type Choice,
         type Choices } from "@/lib/photoReview";

/** The quick reason buttons beneath Include / Exclude in an expanded photo (Steve, 2026-10-21):
 *  one click excludes the photo for that reason. Only the reasons the server lists are shown;
 *  `current` (the photo's reason while excluded) is drawn selected. Keys 1–5 follow QUICK_REASONS. */
export function QuickReasons({ reasons, current, onPick, disabled = false }: {
  reasons: { code: string; label: string }[]; current: string | null; onPick: (code: string) => void; disabled?: boolean;
}) {
  const shown = quickReasons(reasons);
  if (!shown.length) return null;
  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5" role="group" aria-label="Exclude as"
         data-testid="quick-reasons">
      {shown.map((q) => {
        const on = current === q.code;
        const key = QUICK_REASONS.findIndex((r) => r.code === q.code) + 1;
        return (
          <button key={q.code} type="button" aria-pressed={on} disabled={disabled} title={`Exclude: ${q.title} (${key})`}
                  onClick={(e) => { e.stopPropagation(); onPick(q.code); }}
                  className={`inline-flex min-h-10 items-center gap-1.5 rounded-md border px-3 text-sm font-medium disabled:opacity-40 ${
                    on ? "border-red-500 bg-red-600 text-white" : "border-white/40 bg-white/10 text-white hover:bg-white/20"}`}>
            <span className={`text-[11px] tabular-nums ${on ? "text-white/80" : "text-white/60"}`} aria-hidden>{key}</span>
            {q.label}
          </button>
        );
      })}
    </div>
  );
}

// The pieces the Human Feedback review pages share (the photo review and the label review):
// a record's photo tiles, the placeholder, Keep / Exclude buttons, the photo viewer, the key
// help and the Open / Decided toggle.
//
// Photo rule (lib/photoReview.ts tileImage): a "hidden" photo never gets an image URL, and an
// admin's "private" photo is only Vision's own same-origin copy, shown by a plain <img>: never
// linked for download, prefetched or copied. Only the photo being viewed is loaded large.

export const CARD = "rounded-lg border border-[#A87146]/20 bg-white shadow-sm dark:border-border dark:bg-card";
export const INK = "text-[#4a3728] dark:text-foreground";
export const BODY = "text-[#5c4a3a] dark:text-foreground/90";
export const LINK = "text-myco-green underline-offset-2 hover:underline";
export const FIELD = "min-h-10 min-w-0 rounded-md border border-[#A87146]/30 bg-white px-2 py-1.5 text-sm "
  + "dark:border-border dark:bg-background focus:outline-none focus:ring-2 focus:ring-myco-green/40";
export const BTN = "inline-flex min-h-10 items-center justify-center gap-1 rounded-md border border-[#A87146]/30 bg-white "
  + "px-3 py-1.5 text-sm font-medium hover:bg-myco-green/5 disabled:pointer-events-none disabled:opacity-40 "
  + "dark:border-border dark:bg-card";
export const NEW_TAB = { target: "_blank", rel: "noopener" } as const;

export const sciLike = (s: string | null | undefined) => !!s && /^[A-Z][a-z]+(\s|$)/.test(s.trim());

/** What a tile needs of a photo, and of its record. */
export type TilePhoto = Pick<ReviewPhoto, "photo_key" | "position" | "shown" | "thumb" | "full" | "hidden_reason">;
export type TileRecord = { record_key: string; link: string };

export function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} {...NEW_TAB} className={`inline-flex items-center gap-0.5 ${LINK}`}
       onClick={(e) => e.stopPropagation()}>
      {children} <ArrowUpRight className="h-3 w-3 shrink-0" aria-hidden />
    </a>
  );
}

export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** The Open / Decided toggle. */
export function StatusToggle({ status, onChange }: { status: FeedbackStatus; onChange: (s: FeedbackStatus) => void }) {
  const seg = (on: boolean) => `min-h-10 px-3 py-1.5 text-sm font-medium ${
    on ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`;
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
         role="group" aria-label="Status">
      <button type="button" className={seg(status === "open")} aria-pressed={status === "open"}
              onClick={() => onChange("open")}>Open</button>
      <button type="button" className={`${seg(status === "decided")} border-l border-[#A87146]/30 dark:border-border`}
              aria-pressed={status === "decided"} onClick={() => onChange("decided")}>Decided</button>
    </div>
  );
}

export function KeyHelp({ rows }: { rows: [string, string][] }) {
  return (
    <dl className={`${CARD} grid max-w-md grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 p-3 text-sm`}
        aria-label="Keyboard keys">
      {rows.map(([k, what]) => (
        <div key={k} className="contents">
          <dt><kbd className="rounded border border-[#A87146]/30 bg-[#faf9f7] px-1.5 py-0.5 font-mono text-xs dark:border-border dark:bg-muted">{k}</kbd></dt>
          <dd className={BODY}>{what}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Placeholder({ photo, record, failed = false, large = false }: {
  photo: TilePhoto; record: TileRecord; failed?: boolean; large?: boolean;
}) {
  return (
    <div className={`flex h-full w-full flex-col items-center justify-center gap-1 bg-[#f3efe9] p-2 text-center dark:bg-muted ${
      large ? "text-base" : "text-xs"}`}>
      <Lock className={large ? "h-6 w-6 text-muted-foreground" : "h-4 w-4 text-muted-foreground"} aria-hidden />
      <span className={`font-semibold ${INK}`}>Photo {photo.position ?? "?"}</span>
      <span className="text-muted-foreground">{placeholderText(photo, failed)}</span>
      <Ext href={record.link}>{openOnLabel(record.record_key)}</Ext>
    </div>
  );
}

/** Keep / Exclude. `suggested`: the Exclude is only a suggestion (the photo review's "suggest"
 *  mode), drawn lighter and outlined; clicking it confirms it, Keep turns it into a keep. */
export function ChoiceButtons({ choice, onChoose, disabled = false, big = false, suggested = false }: {
  choice: Choice | null; onChoose: (c: Choice) => void; disabled?: boolean; big?: boolean; suggested?: boolean;
}) {
  const base = `inline-flex ${big ? "min-h-11 px-5" : "min-h-10 px-2"} items-center justify-center gap-1 rounded-md border text-sm font-medium disabled:opacity-40`;
  const off = "border-[#A87146]/30 bg-white text-gray-700 hover:bg-myco-green/5 dark:border-border dark:bg-card dark:text-foreground/80";
  const excluded = suggested
    ? "border-dashed border-red-500 bg-red-50 text-red-700 hover:bg-red-100 dark:border-red-400 dark:bg-red-950/40 dark:text-red-200"
    : "border-red-600 bg-red-600 text-white";
  return (
    <div className={big ? "flex gap-2" : "grid grid-cols-2 gap-1"}>
      <button type="button" aria-pressed={choice === "keep"} disabled={disabled} onClick={() => onChoose("keep")}
              className={`${base} ${choice === "keep" ? "border-myco-green bg-myco-green text-white" : off}`}>
        <Check className="h-4 w-4" aria-hidden /> Keep
      </button>
      <button type="button" aria-pressed={choice === "exclude"} disabled={disabled} onClick={() => onChoose("exclude")}
              title={choice === "exclude" && suggested ? "Suggested: click to confirm" : undefined}
              aria-label={choice === "exclude" && suggested ? "Exclude (suggested; click to confirm)" : undefined}
              className={`${base} ${choice === "exclude" ? excluded : off}`}>
        <X className="h-4 w-4" aria-hidden /> Exclude
      </button>
    </div>
  );
}

const BADGE: Record<Choice, string> = { keep: "Keep", exclude: "Exclude" };

/** One photo as a square tile: the image (or the placeholder, also when it fails to load), its
 *  position, the private badge and the choice made. `children` go under it (labels, buttons).
 *  `suggested`: the Exclude is only a suggestion, badged "Exclude (suggested)" in an outlined
 *  style (the caller draws the dashed outline). */
export function PhotoTile({ photo, record, choice, focused, outline = "border-transparent", className = "sm:w-44",
                            badge = BADGE, suggested = false, onOpen, onFocus, children }: {
  photo: TilePhoto; record: TileRecord; choice: Choice | null; focused: boolean; outline?: string;
  className?: string; badge?: Record<Choice, string>; suggested?: boolean; onOpen: () => void; onFocus: () => void;
  children?: ReactNode;
}) {
  const asSuggestion = suggested && choice === "exclude";
  // A thumb that fails to load (e.g. a private copy Vision does not hold) becomes the placeholder.
  const [failed, setFailed] = useState(false);
  const src = failed ? null : tileImage(photo, "thumb");
  return (
    <li className={`min-w-0 space-y-1 ${className}`} onMouseDown={onFocus}>
      <div className={`relative aspect-square w-full overflow-hidden rounded-md border-4 ${outline} ${
        focused ? "ring-2 ring-myco-green ring-offset-2 dark:ring-offset-background" : ""}`}>
        {src ? (
          <button type="button" onClick={onOpen} className="block h-full w-full"
                  aria-label={`Open photo ${photo.position ?? ""} large`}>
            <img src={src} alt={`Photo ${photo.position ?? ""} of ${recordLabel(record.record_key)}`}
                 loading="lazy" decoding="async" referrerPolicy="no-referrer" draggable={false}
                 onError={() => setFailed(true)}
                 className={`h-full w-full object-cover ${
                   asSuggestion ? "opacity-75" : choice === "exclude" ? "opacity-50" : ""}`} />
          </button>
        ) : (
          <div className="h-full w-full cursor-pointer" onClick={onOpen}>
            <Placeholder photo={photo} record={record} failed={failed} />
          </div>
        )}
        {src && (
          <span className="pointer-events-none absolute left-1 top-1 rounded bg-black/60 px-1.5 text-[11px] font-medium text-white">
            {photo.position ?? "?"}
          </span>
        )}
        {photo.shown === "private" && (
          <span className="pointer-events-none absolute right-1 top-1 rounded bg-[#4a3728]/85 px-1.5 py-0.5 text-[10px] font-medium text-white">
            reviewers only · not public
          </span>
        )}
        {choice && (
          <span className={`pointer-events-none absolute bottom-1 right-1 rounded px-1.5 py-0.5 text-[11px] font-semibold ${
            asSuggestion ? "border border-dashed border-red-600 bg-white/90 text-red-700 dark:bg-black/70 dark:text-red-200"
              : choice === "keep" ? "bg-myco-green text-white" : "bg-red-600 text-white"}`}>
            {asSuggestion ? `${badge.exclude} (suggested)` : badge[choice]}
          </span>
        )}
      </div>
      {children}
    </li>
  );
}

/** The photo viewer: one photo large, ←/→ between the record's photos, Keep / Exclude (none
 *  when `onChoose` is left out: a decided card), Esc or a click outside to close. `suggested`
 *  says which exclusions are only suggestions; `hint` replaces the key line. */
export function PhotoLightbox<P extends TilePhoto>({ record, photos, index, choices, onStep, onChoose, onClose, extra,
                                                    below, suggested, hint }: {
  record: TileRecord; photos: P[]; index: number; choices: Choices; onStep: (d: 1 | -1) => void;
  onChoose?: (c: Choice) => void; onClose: () => void; extra?: (photo: P) => ReactNode;
  /** Drawn directly beneath Keep / Exclude (the quick reasons). */
  below?: (photo: P) => ReactNode;
  suggested?: (photo: P) => boolean; hint?: string;
}) {
  const photo = photos[index];
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  useEffect(() => { ref.current?.focus(); }, []);
  if (!photo) return null;
  // Only the photo being viewed is loaded: nothing is fetched ahead.
  const broken = !!failed[photo.photo_key];
  const src = broken ? null : tileImage(photo, "full");
  const choice = choices[photo.photo_key] ?? null;
  return (
    <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true"
         aria-label={`Photo ${photo.position ?? ""} of ${recordLabel(record.record_key)}`}
         className="fixed inset-0 z-[60] flex flex-col bg-black/85 p-3 outline-none sm:p-6" onClick={onClose}>
      <div className="flex min-w-0 items-center justify-between gap-3 text-sm text-white" onClick={(e) => e.stopPropagation()}>
        <span className="min-w-0 break-words">
          {recordLabel(record.record_key)} · photo {index + 1} of {photos.length}
          {photo.shown === "private" && <span className="ml-2 rounded bg-white/15 px-1.5 py-0.5 text-[11px]">reviewers only · not public</span>}
        </span>
        <button type="button" onClick={onClose} aria-label="Close"
                className="inline-flex min-h-10 min-w-10 items-center justify-center rounded-md hover:bg-white/10">
          <X className="h-5 w-5" aria-hidden />
        </button>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center gap-2 py-3">
        <button type="button" onClick={(e) => { e.stopPropagation(); onStep(-1); }} disabled={index === 0}
                aria-label="Previous photo"
                className="inline-flex min-h-10 min-w-10 shrink-0 items-center justify-center rounded-md text-white hover:bg-white/10 disabled:opacity-30">
          <ChevronLeft className="h-6 w-6" aria-hidden />
        </button>
        <div className="flex min-h-0 min-w-0 flex-1 items-center justify-center self-stretch" onClick={(e) => e.stopPropagation()}>
          {src ? (
            <img key={photo.photo_key} src={src} alt={`Photo ${photo.position ?? ""}`} referrerPolicy="no-referrer"
                 draggable={false} className="max-h-full max-w-full object-contain"
                 onError={() => setFailed((f) => ({ ...f, [photo.photo_key]: true }))} />
          ) : (
            <div className="aspect-square w-full max-w-sm overflow-hidden rounded-md">
              <Placeholder photo={photo} record={record} failed={broken} large />
            </div>
          )}
        </div>
        <button type="button" onClick={(e) => { e.stopPropagation(); onStep(1); }} disabled={index >= photos.length - 1}
                aria-label="Next photo"
                className="inline-flex min-h-10 min-w-10 shrink-0 items-center justify-center rounded-md text-white hover:bg-white/10 disabled:opacity-30">
          <ChevronRight className="h-6 w-6" aria-hidden />
        </button>
      </div>
      <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-2" onClick={(e) => e.stopPropagation()}>
        {extra?.(photo)}
        {onChoose && <ChoiceButtons choice={choice} onChoose={onChoose} big suggested={!!suggested?.(photo)} />}
        {below?.(photo)}
        <p className="text-center text-xs text-white/70">
          {hint ?? (onChoose ? "← / → photos · K keep · X exclude · Esc close" : "← / → photos · Esc close")}
        </p>
      </div>
    </div>
  );
}
