import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, Check, ChevronLeft, ChevronRight, Lock, Plus, Save, X } from "lucide-react";
import { BODY, BTN, Ext, FIELD, INK, Placeholder, StatusToggle } from "@/components/ReviewPhotos";
import { ApiError, feedbackApi, type FeedbackStatus, type FeedbackSummary, type PhotoViewTile,
         type PhotoViewsPage } from "@/lib/api";
import { recordLabel } from "@/lib/feedback";
import { tileImage } from "@/lib/photoReview";
import { batchFor, chips, choiceOf, LEAVE_PROMPT, markSaved, needsSave, saveLine, setNoSpecimen, toggleView,
         VIEWS_PAGE_SIZE, type ViewChoices } from "@/lib/photoViews";

// Photo views (Steve, 2026-10-21): what each photo shows, as tags to confirm or correct. A grid
// of the tab's photos; each tile shows the tagger's suggestions switched on; tapping a tag
// toggles it, "+" opens the photo large with every view to add, "No specimen" sends the photo
// to the exclusion review. Nothing is written until Save (the tile's, or Save page for every
// tile still to save). Leaving or paging with unsaved tiles asks first. Rules: lib/photoViews.ts.

type Views = PhotoViewsPage["views"];
const PRIMARY = "inline-flex min-h-10 items-center justify-center gap-1 rounded-md border border-myco-green bg-myco-green "
  + "px-3 py-1.5 text-sm font-medium text-white hover:bg-myco-green/90 disabled:pointer-events-none disabled:opacity-40";


export function PhotoViews({ summary }: { summary: FeedbackSummary }) {
  const qc = useQueryClient();
  const [tab, setTab] = useState("all");
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [text, setText] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const [choices, setChoices] = useState<ViewChoices>({});
  const [saved, setSaved] = useState<ViewChoices>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const page = useQuery({
    queryKey: ["feedback", "photo-views", tab, status, q, offset],
    queryFn: () => feedbackApi.photoViews({ tab, status, q, offset, limit: VIEWS_PAGE_SIZE }),
    staleTime: Infinity, refetchOnWindowFocus: false,
  });
  const views: Views = page.data?.views ?? [];
  const tiles = page.data?.tiles ?? [];
  const pending = useMemo(() => batchFor(tiles, choices, saved, views), [tiles, choices, saved, views]);

  // Leaving the page with unsaved tiles asks first.
  useEffect(() => {
    if (!pending.length) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = LEAVE_PROMPT; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [pending.length]);

  const leave = useCallback((go: () => void) => {
    if (pending.length && !window.confirm(LEAVE_PROMPT)) return;
    setChoices({});
    setSaved({});
    setOpen(null);
    go();
  }, [pending.length]);

  const save = useCallback(async (ids?: number[]) => {
    const body = batchFor(tiles, choices, saved, views, ids);
    if (!body.length) return;
    setSaving(true);
    setError(null);
    try {
      await feedbackApi.photoViewsBatch(body);
      setSaved((s) => markSaved(s, body));
      void qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Saving failed. Nothing was written; try again.");
    } finally {
      setSaving(false);
    }
  }, [tiles, choices, saved, views, qc]);

  const tabs = page.data?.tabs.filter((t) => t.code === "all" || t.open + t.decided > 0) ?? [];
  const lb = open == null ? null : tiles.find((t) => t.item_id === open) ?? null;
  const lbIndex = lb ? tiles.indexOf(lb) : -1;
  const total = page.data?.total ?? 0;

  return (
    <section className="min-w-0 space-y-4" aria-label={summary.title}>
      <div>
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{summary.about}</p>
      </div>
      <div className="flex min-w-0 gap-1 overflow-x-auto border-b border-[#A87146]/20 pb-1 dark:border-border"
           role="tablist" aria-label="View">
        {tabs.map((t) => {
          const on = t.code === tab;
          return (
            <button key={t.code} type="button" role="tab" aria-selected={on}
                    onClick={() => leave(() => { setTab(t.code); setOffset(0); })}
                    className={`min-h-10 shrink-0 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium ${
                      on ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`}>
              {t.title}
              <span className="ml-1 tabular-nums text-muted-foreground">
                ({(status === "open" ? t.open : t.decided).toLocaleString("en-US")})
              </span>
            </button>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <StatusToggle status={status} onChange={(s) => leave(() => { setStatus(s); setOffset(0); })} />
        <form className="flex min-w-0 gap-1" onSubmit={(e) => { e.preventDefault(); leave(() => { setQ(text); setOffset(0); }); }}>
          <input className={`${FIELD} w-44`} value={text} onChange={(e) => setText(e.target.value)}
                 placeholder="Name or record" aria-label="Search by name or record" />
        </form>
        <span className="ml-auto flex items-center gap-2">
          <span className="text-sm text-muted-foreground" data-testid="photo-views-pending">{saveLine(pending.length)}</span>
          <button type="button" className={BTN} disabled={!pending.length || saving}
                  onClick={() => { setChoices({}); }}>Discard</button>
          <button type="button" disabled={!pending.length || saving} onClick={() => void save()}
                  className={PRIMARY} data-testid="photo-views-save-page">
            <Save className="h-4 w-4" aria-hidden /> Save page ({pending.length})
          </button>
        </span>
      </div>
      {error && <p className="text-sm text-red-700 dark:text-red-400" role="alert">{error}</p>}
      {page.isPending && <p className="text-sm text-muted-foreground">Loading the photos…</p>}
      {page.data && !tiles.length && <p className={`text-sm ${BODY}`}>No photos here.</p>}
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {tiles.map((t) => (
          <ViewTile key={t.item_id} tile={t} views={views} choice={choiceOf(choices, t, views)}
                    dirty={needsSave(t, choices, saved, views)} changed={t.item_id in choices}
                    saved={!!saved[t.item_id]} saving={saving}
                    onToggle={(code) => setChoices((c) => toggleView(c, t, code, views))}
                    onNoSpecimen={(on) => setChoices((c) => setNoSpecimen(c, t, on, views))}
                    onOpen={() => setOpen(t.item_id)} onSave={() => void save([t.item_id])} />
        ))}
      </ul>
      {total > VIEWS_PAGE_SIZE && (
        <div className="flex items-center justify-center gap-2">
          <button type="button" className={BTN} disabled={offset === 0}
                  onClick={() => leave(() => setOffset(Math.max(0, offset - VIEWS_PAGE_SIZE)))}>
            <ChevronLeft className="h-4 w-4" aria-hidden /> Previous
          </button>
          <span className="text-sm text-muted-foreground tabular-nums">
            {offset + 1}–{Math.min(offset + VIEWS_PAGE_SIZE, total)} of {total.toLocaleString("en-US")}
          </span>
          <button type="button" className={BTN} disabled={offset + VIEWS_PAGE_SIZE >= total}
                  onClick={() => leave(() => setOffset(offset + VIEWS_PAGE_SIZE))}>
            Next <ChevronRight className="h-4 w-4" aria-hidden />
          </button>
        </div>
      )}
      {lb && (
        <ViewLightbox tile={lb} views={views} choice={choiceOf(choices, lb, views)}
                      index={lbIndex} count={tiles.length} saving={saving}
                      dirty={needsSave(lb, choices, saved, views)}
                      onToggle={(code) => setChoices((c) => toggleView(c, lb, code, views))}
                      onNoSpecimen={(on) => setChoices((c) => setNoSpecimen(c, lb, on, views))}
                      onSave={() => void save([lb.item_id])}
                      onStep={(d) => { const n = tiles[lbIndex + d]; if (n) setOpen(n.item_id); }}
                      onClose={() => setOpen(null)} />
      )}
    </section>
  );
}

function Chip({ label, on, suggested, onClick, disabled }: {
  label: string; on: boolean; suggested: boolean; onClick: () => void; disabled?: boolean;
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} aria-pressed={on}
            title={suggested ? "Suggested by the tagger" : "Added by you"}
            className={`inline-flex min-h-7 items-center gap-0.5 rounded-full border px-2 text-[11px] font-medium ${
              on ? "border-myco-green bg-myco-green text-white"
                : `border-[#A87146]/40 bg-white text-gray-600 dark:bg-card dark:text-foreground/70 ${
                  suggested ? "line-through decoration-gray-400" : ""}`
            } disabled:opacity-40`}>
      {on && <Check className="h-3 w-3" aria-hidden />}{label}
    </button>
  );
}

function TileImage({ tile, large = false }: { tile: PhotoViewTile; large?: boolean }) {
  const [failed, setFailed] = useState(false);
  const src = failed ? null : tileImage(tile, large ? "full" : "thumb");
  const photo = { ...tile, position: tile.position == null ? null : tile.position + 1 };
  return src ? (
    <img src={src} alt={`${tile.label || "Photo"}, ${recordLabel(tile.record_key)}`} loading="lazy" decoding="async"
         referrerPolicy="no-referrer" draggable={false} onError={() => setFailed(true)}
         className={large ? "max-h-[70vh] w-auto max-w-full object-contain" : "h-full w-full object-cover"} />
  ) : <Placeholder photo={photo} record={tile} failed={failed} large={large} />;
}

function ViewTile({ tile, views, choice, dirty, changed, saved, saving, onToggle, onNoSpecimen, onOpen, onSave }: {
  tile: PhotoViewTile; views: Views; choice: ReturnType<typeof choiceOf>; dirty: boolean; changed: boolean; saved: boolean;
  saving: boolean; onToggle: (code: string) => void; onNoSpecimen: (on: boolean) => void; onOpen: () => void;
  onSave: () => void;
}) {
  const shown = chips(tile, choice, views);
  return (
    <li className="min-w-0 space-y-1" data-testid={`photo-views-tile-${tile.item_id}`}>
      <div className={`relative aspect-square w-full overflow-hidden rounded-md border-2 ${
        choice.noSpecimen ? "border-red-500/70" : dirty && changed ? "border-amber-400" : dirty ? "border-transparent" : "border-myco-green/60"}`}>
        <button type="button" onClick={onOpen} className="block h-full w-full" aria-label="Open full screen">
          <TileImage tile={tile} />
        </button>
        {tile.shown === "private" && (
          <span className="pointer-events-none absolute right-1 top-1 rounded bg-[#4a3728]/85 p-0.5 text-white"
                title="reviewers only · not public"><Lock className="h-3 w-3" aria-label="reviewers only, not public" /></span>
        )}
        <span className={`pointer-events-none absolute bottom-1 left-1 rounded px-1.5 py-0.5 text-[10px] font-semibold text-white ${
          dirty && changed ? "bg-amber-500" : dirty ? "bg-black/55" : "bg-myco-green"}`}>
          {dirty && changed ? "changed · unsaved" : dirty ? "to confirm" : "saved"}
        </span>
      </div>
      {choice.noSpecimen ? (
        <p className="text-[11px] font-medium text-red-700 dark:text-red-400">No specimen: goes to the exclusion review</p>
      ) : (
        <div className="flex flex-wrap gap-1">
          {shown.map((c) => <Chip key={c.code} label={c.label} on={c.on} suggested={c.suggested} onClick={() => onToggle(c.code)} />)}
          <button type="button" onClick={onOpen} title="Add another view"
                  className="inline-flex min-h-7 items-center rounded-full border border-dashed border-[#A87146]/50 px-2 text-[11px] text-gray-600 dark:text-foreground/70">
            <Plus className="h-3 w-3" aria-hidden /> view
          </button>
        </div>
      )}
      <div className="flex items-center gap-1">
        <button type="button" disabled={!dirty || saving} onClick={onSave}
                className="inline-flex min-h-7 items-center gap-0.5 rounded border border-myco-green px-2 text-[11px] font-semibold text-myco-green disabled:opacity-40">
          <Save className="h-3 w-3" aria-hidden /> Save
        </button>
        <button type="button" onClick={() => onNoSpecimen(!choice.noSpecimen)}
                className="inline-flex min-h-7 items-center gap-0.5 rounded border border-[#A87146]/40 px-2 text-[11px] text-gray-700 dark:text-foreground/80">
          <Ban className="h-3 w-3" aria-hidden /> {choice.noSpecimen ? "Undo" : "No specimen"}
        </button>
      </div>
      <p className={`truncate text-xs italic ${INK}`} title={tile.label}>{tile.label || "(no name)"}</p>
      <p className="truncate text-xs"><Ext href={tile.link}>{recordLabel(tile.record_key)}</Ext></p>
    </li>
  );
}

function ViewLightbox({ tile, views, choice, index, count, saving, dirty, onToggle, onNoSpecimen, onSave, onStep, onClose }: {
  tile: PhotoViewTile; views: Views; choice: ReturnType<typeof choiceOf>; index: number; count: number; saving: boolean;
  dirty: boolean; onToggle: (code: string) => void; onNoSpecimen: (on: boolean) => void; onSave: () => void;
  onStep: (d: number) => void; onClose: () => void;
}) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") onStep(1);
      else if (e.key === "ArrowLeft") onStep(-1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose, onStep]);
  const sug = new Set(tile.suggested.map((s) => s.view));
  const on = new Set(choice.views);
  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/90 p-4" role="dialog" aria-modal="true"
         aria-label={`Photo views: ${recordLabel(tile.record_key)}`}>
      <div className="flex items-center justify-between text-white">
        <span className="text-sm">{index + 1} / {count} · <span className="italic">{tile.label}</span> · {recordLabel(tile.record_key)}</span>
        <button type="button" onClick={onClose} aria-label="Close" className="rounded p-2 hover:bg-white/10"><X className="h-5 w-5" /></button>
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center gap-2">
        <button type="button" onClick={() => onStep(-1)} disabled={index <= 0} aria-label="Previous photo"
                className="rounded p-2 text-white hover:bg-white/10 disabled:opacity-30"><ChevronLeft className="h-6 w-6" /></button>
        <div className="flex min-h-0 max-w-full items-center justify-center"><TileImage tile={tile} large /></div>
        <button type="button" onClick={() => onStep(1)} disabled={index >= count - 1} aria-label="Next photo"
                className="rounded p-2 text-white hover:bg-white/10 disabled:opacity-30"><ChevronRight className="h-6 w-6" /></button>
      </div>
      <div className="mx-auto mt-3 flex max-w-4xl flex-wrap items-center justify-center gap-1.5">
        {views.map((v) => (
          <Chip key={v.code} label={v.label} on={on.has(v.code)} suggested={sug.has(v.code)}
                disabled={choice.noSpecimen} onClick={() => onToggle(v.code)} />
        ))}
      </div>
      <div className="mt-3 flex items-center justify-center gap-2">
        <button type="button" onClick={() => onNoSpecimen(!choice.noSpecimen)}
                className={`${BTN} ${choice.noSpecimen ? "border-red-500 bg-red-600 text-white" : ""}`}>
          <Ban className="h-4 w-4" aria-hidden /> {choice.noSpecimen ? "No specimen (undo)" : "No specimen"}
        </button>
        <button type="button" disabled={!dirty || saving} onClick={onSave}
                className={PRIMARY}>
          <Save className="h-4 w-4" aria-hidden /> Save this photo
        </button>
      </div>
    </div>
  );
}
