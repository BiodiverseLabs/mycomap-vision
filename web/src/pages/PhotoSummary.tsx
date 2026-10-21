import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronLeft, ChevronRight, Lock, RotateCcw, X } from "lucide-react";
import { BODY, BTN, Ext, FIELD, INK, Placeholder } from "@/components/ReviewPhotos";
import { ApiError, feedbackApi, type FeedbackStatus, type FlaggedPage, type FlaggedTile, type PhotoReviewSort,
         type PhotoReviewTab } from "@/lib/api";
import { pageText, recordLabel } from "@/lib/feedback";
import { tileImage, type Choice } from "@/lib/photoReview";
import { AUTOSAVE_MS, batchBody, beginSend, decidedText, dueDecisions, gridColumns, gridKey, initialMarks, markLoaded,
         markSeen, markTouched, mergeMarks, moveFocus, nextOffset, overlay, reasonLabel, retry, SEEN_MS, SEEN_RATIO,
         sendDone, sendFailed, setDecision, setReason, STATUS_TEXT, SUMMARY_PAGE_SIZE, tileStatus, toggle,
         visitLine, visitTotals, type GridAction, type Mark, type Marks, type SaveState, type SaveStates,
         type TileSize } from "@/lib/photoSummary";

// The photo review's Summary view (Steve, 2026-10-21): the flagged photos of the tab as a grid,
// one tile each, to scan and leave the few that belong. Each open tile starts as Exclude with its
// suggested reason; its Include / Exclude flipper is the saved decision. A flip (or a reason
// picked full screen) saves at once; the default Exclude is saved for the tiles actually shown
// (in view a second and loaded), every couple of seconds. The rules: lib/photoSummary.ts.
//
// Photos follow the record view's rule (lib/photoReview.ts tileImage): a hidden photo is a
// placeholder with the record's link; a private one only Vision's own same-origin copy.

type Reasons = FlaggedPage["reasons"];

export function PhotoSummary({ tab, status, sort, q, help, onHelp, tileSize, onTotal }: {
  tab: PhotoReviewTab; status: FeedbackStatus; sort: PhotoReviewSort; q: string; help: boolean;
  onHelp: () => void; tileSize: TileSize; onTotal: (n: number | null) => void;
}) {
  const qc = useQueryClient();
  const [offset, setOffset] = useState(0);
  // Bumped on every page change, so each page is read afresh (saved tiles drop out of Open)
  // while the page shown never reloads under the reviewer.
  const [gen, setGen] = useState(0);
  // The reviewer's marks and the saving of each tile this visit, by item id. Kept in refs too:
  // the autosave reads the latest without waiting for a render.
  const editsRef = useRef<Marks>({});
  const [edits, setEditsState] = useState<Marks>({});
  const statesRef = useRef<SaveStates>({});
  const [states, setStatesState] = useState<SaveStates>({});
  const tilesRef = useRef<Map<number, FlaggedTile>>(new Map());
  const updateStates = useCallback((fn: (s: SaveStates) => SaveStates) => {
    statesRef.current = fn(statesRef.current);
    setStatesState(statesRef.current);
  }, []);
  const updateEdits = useCallback((fn: (m: Marks) => Marks) => {
    editsRef.current = fn(editsRef.current);
    setEditsState(editsRef.current);
  }, []);

  const list = useQuery({
    queryKey: ["feedback", "photo-summary", tab, status, sort, q, offset, gen],
    queryFn: () => feedbackApi.photoReviewFlagged({ tab, status, sort, q, offset, limit: SUMMARY_PAGE_SIZE }),
    placeholderData: (prev) => prev,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
    gcTime: 0,
  });
  const data = list.data;
  const fresh = data && !list.isPlaceholderData ? data : null;
  useEffect(() => {
    if (!fresh) return;
    for (const t of fresh.tiles) tilesRef.current.set(t.item_id, t);
    onTotal(fresh.total);
  }, [fresh, onTotal]);
  useEffect(() => () => onTotal(null), [onTotal]);

  // --- saving ---------------------------------------------------------------------------------
  const flushRef = useRef<(o?: { keepalive?: boolean }) => void>(() => {});
  flushRef.current = (o = {}) => {
    const tiles = [...tilesRef.current.values()];
    const marks = mergeMarks(initialMarks(tiles), editsRef.current);
    const due = dueDecisions(tiles, marks, statesRef.current);
    const body = batchBody(due);
    if (!body) return;
    updateStates((s) => beginSend(s, due));
    feedbackApi.photoReviewBatch(body, o).then(
      () => {
        updateStates((s) => sendDone(s, due));
        // The queue's card counts are stale now; they are read again on the next page change.
        qc.invalidateQueries({ queryKey: ["feedback", "summary"], refetchType: "none" });
        flushRef.current();                    // a flip made while this was in flight
      },
      (e) => updateStates((s) => sendFailed(s, due, e instanceof ApiError || e instanceof Error ? e.message : "Not saved.")),
    );
  };
  useEffect(() => {
    const timer = window.setInterval(() => flushRef.current(), AUTOSAVE_MS);
    const onHidden = () => { if (document.visibilityState === "hidden") flushRef.current({ keepalive: true }); };
    const onLeave = () => flushRef.current({ keepalive: true });
    document.addEventListener("visibilitychange", onHidden);
    window.addEventListener("pagehide", onLeave);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onHidden);
      window.removeEventListener("pagehide", onLeave);
      flushRef.current({ keepalive: true });
      qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
    };
  }, [qc]);

  const tiles = fresh?.tiles ?? [];
  const marks = useMemo(() => mergeMarks(initialMarks(tiles), edits), [tiles, edits]);

  const flip = useCallback((id: number, choice?: Choice) => {
    const t = tilesRef.current.get(id);
    if (!t) return;
    const now = mergeMarks(initialMarks([t]), editsRef.current);
    const next = choice ? setDecision(now, id, choice) : toggle(now, id);
    if (next === now) return;
    updateEdits((m) => ({ ...m, [id]: next[id] }));
    updateStates((s) => markTouched(s, id));
    flushRef.current();
  }, [updateEdits, updateStates]);
  const pickReason = useCallback((id: number, reason: string) => {
    const t = tilesRef.current.get(id);
    if (!t) return;
    const now = mergeMarks(initialMarks([t]), editsRef.current);
    const next = setReason(now, id, reason);
    if (next === now) return;
    updateEdits((m) => ({ ...m, [id]: next[id] }));
    updateStates((s) => markTouched(s, id));
    flushRef.current();
  }, [updateEdits, updateStates]);
  const retryTile = useCallback((id: number) => { updateStates((s) => retry(s, id)); flushRef.current(); }, [updateStates]);
  const retryAll = () => {
    updateStates((s) => Object.keys(s).reduce((acc, k) => retry(acc, Number(k)), s));
    flushRef.current();
  };
  const seen = useCallback((id: number) => updateStates((s) => markSeen(s, id)), [updateStates]);
  const loaded = useCallback((id: number) => updateStates((s) => markLoaded(s, id)), [updateStates]);

  // A filter change starts at the first page; the page shown is saved first.
  const filters = useRef({ tab, status, sort, q });
  useEffect(() => {
    const f = filters.current;
    if (f.tab === tab && f.status === status && f.sort === sort && f.q === q) return;
    filters.current = { tab, status, sort, q };
    flushRef.current();
    setOffset(0);
    setGen((g) => g + 1);
  }, [tab, status, sort, q]);

  const go = (next: number) => {
    flushRef.current();
    setOffset(Math.max(0, next));
    setGen((g) => g + 1);
    qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
  };

  const totals = visitTotals(states);
  const failed = Object.values(states).filter((s) => s.failed).length;
  const savedOnPage = tiles.filter((t) => t.status === "open" && (states[t.item_id]?.saved || states[t.item_id]?.sending)).length;
  const total = fresh?.total ?? 0;
  return (
    <div className="min-w-0 space-y-3">
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1 text-sm" aria-live="polite">
        <span className={`font-medium ${INK}`} data-testid="photo-summary-visit">{visitLine(totals)}</span>
        {failed > 0 && (
          <span className="inline-flex items-center gap-2 text-red-700 dark:text-red-300">
            {failed} not saved
            <button type="button" className={BTN} onClick={retryAll}>
              <RotateCcw className="h-3.5 w-3.5" aria-hidden /> Retry
            </button>
          </span>
        )}
        {list.isFetching && <span className="text-xs text-muted-foreground">Loading…</span>}
      </div>
      {status === "open" && (
        <p className="text-xs text-muted-foreground">
          Every photo starts as Exclude and is saved that way once you have seen it. Flip the few that
          belong to Include; each flip saves at once.
        </p>
      )}
      {list.error && (
        <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The photos could not be loaded."}</p>
      )}
      {data && list.isPlaceholderData && <p className="text-sm text-muted-foreground">Loading the photos…</p>}
      {fresh && !tiles.length && (
        <p className="text-sm text-muted-foreground">
          {q ? "No record or name matches that search." : status === "open" ? "Nothing open here." : "Nothing decided here yet."}
        </p>
      )}
      {fresh && tiles.length > 0 && (
        <Grid key={`${tab}|${status}|${sort}|${q}|${offset}|${gen}`} tiles={tiles} reasons={fresh.reasons}
              marks={marks} states={states} tileSize={tileSize} help={help} onHelp={onHelp}
              onFlip={flip} onReason={pickReason} onRetry={retryTile} onSeen={seen} onLoaded={loaded} />
      )}
      {total > 0 && fresh && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground">{pageText(offset, tiles.length, total)}</span>
          <span className="flex gap-2">
            <button type="button" className={BTN} disabled={offset === 0}
                    onClick={() => go(offset - SUMMARY_PAGE_SIZE)}>Previous</button>
            <button type="button" className={BTN} disabled={offset + tiles.length >= total}
                    onClick={() => go(nextOffset(status, offset, SUMMARY_PAGE_SIZE, savedOnPage))}>Next</button>
          </span>
        </div>
      )}
    </div>
  );
}

/** The columns the grid is drawn with now (for ↑ / ↓). */
function columnsOf(el: HTMLElement | null): number {
  if (!el || typeof getComputedStyle === "undefined") return 1;
  return getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length || 1;
}

function Grid({ tiles, reasons, marks, states, tileSize, help, onHelp, onFlip, onReason, onRetry, onSeen, onLoaded }: {
  tiles: FlaggedTile[]; reasons: Reasons; marks: Marks; states: SaveStates; tileSize: TileSize; help: boolean;
  onHelp: () => void; onFlip: (id: number, c?: Choice) => void; onReason: (id: number, r: string) => void;
  onRetry: (id: number) => void; onSeen: (id: number) => void; onLoaded: (id: number) => void;
}) {
  const [focus, setFocus] = useState(0);
  const [lightbox, setLightbox] = useState<number | null>(null);
  const gridRef = useRef<HTMLUListElement>(null);
  const tileRefs = useRef<(HTMLLIElement | null)[]>([]);
  const onSeenRef = useRef(onSeen);
  onSeenRef.current = onSeen;

  // A tile is seen when SEEN_RATIO of it has been in view for SEEN_MS.
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const timers = new Map<Element, number>();
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const el = e.target as HTMLElement;
        if (e.isIntersecting && e.intersectionRatio >= SEEN_RATIO) {
          if (timers.has(el)) continue;
          timers.set(el, window.setTimeout(() => {
            timers.delete(el);
            io.unobserve(el);
            onSeenRef.current(Number(el.dataset.itemId));
          }, SEEN_MS));
        } else {
          const t = timers.get(el);
          if (t != null) { window.clearTimeout(t); timers.delete(el); }
        }
      }
    }, { threshold: [0, SEEN_RATIO, 1] });
    for (const el of tileRefs.current) if (el) io.observe(el);
    return () => {
      io.disconnect();
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [tiles, tileSize]);

  const focusTile = (i: number) => {
    setFocus(i);
    const el = tileRefs.current[i];
    if (el) {
      el.focus({ preventScroll: true });
      el.scrollIntoView({ block: "nearest" });
    }
  };

  const run = (a: GridAction) => {
    const n = tiles.length;
    switch (a.type) {
      case "move": focusTile(moveFocus(focus, n, a.delta)); return;
      case "open": setLightbox(focus); return;
      case "choose": {
        const t = tiles[lightbox ?? focus];
        if (t) onFlip(t.item_id, a.choice);
        return;
      }
      case "step": {
        if (lightbox == null) return;
        const i = moveFocus(lightbox, n, a.delta);
        setLightbox(i);
        setFocus(i);
        return;
      }
      case "help": onHelp(); return;
      case "close": {
        const i = lightbox ?? focus;
        setLightbox(null);
        focusTile(i);
        return;
      }
    }
  };
  const runRef = useRef(run);
  runRef.current = run;
  const stateRef = useRef({ lightbox: lightbox != null, help });
  stateRef.current = { lightbox: lightbox != null, help };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target instanceof HTMLElement ? e.target : null;
      const typing = !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      const onControl = !!t && /^(BUTTON|A)$/.test(t.tagName);
      const a = gridKey(e, { typing, onControl, columns: columnsOf(gridRef.current), ...stateRef.current });
      if (!a) return;
      e.preventDefault();
      runRef.current(a);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const lb = lightbox != null ? tiles[lightbox] : null;
  return (
    <>
      <ul ref={gridRef} className="grid gap-2" style={{ gridTemplateColumns: gridColumns(tileSize) }}
          aria-label="Flagged photos" data-testid="photo-summary-grid">
        {tiles.map((t, i) => (
          <SummaryTile key={t.item_id} tile={t} reasons={reasons} mark={marks[t.item_id]} state={states[t.item_id]}
                       focused={focus === i} tileRef={(el) => { tileRefs.current[i] = el; }}
                       onFocus={() => setFocus(i)} onOpen={() => { setFocus(i); setLightbox(i); }}
                       onFlip={() => onFlip(t.item_id)} onRetry={() => onRetry(t.item_id)}
                       onLoaded={() => onLoaded(t.item_id)} />
        ))}
      </ul>
      {lb && lightbox != null && (
        <SummaryLightbox tile={lb} index={lightbox} count={tiles.length} reasons={reasons} mark={marks[lb.item_id]}
                         state={states[lb.item_id]}
                         onStep={(d) => run({ type: "step", delta: d })}
                         onChoose={(c) => onFlip(lb.item_id, c)} onReason={(r) => onReason(lb.item_id, r)}
                         onRetry={() => onRetry(lb.item_id)}
                         onShown={() => { onLoaded(lb.item_id); onSeen(lb.item_id); }}
                         onClose={() => run({ type: "close" })} />
      )}
    </>
  );
}

/** The record view's placeholder, numbering the photo from 1 (the Summary's positions count from 0). */
const shownPhoto = (t: FlaggedTile) => ({ ...t, position: t.position == null ? null : t.position + 1 });

function StatusMark({ mark, state, onRetry, dark = false }: {
  mark: Mark | undefined; state: SaveState | undefined; onRetry: () => void; dark?: boolean;
}) {
  const s = tileStatus(mark, state);
  if (s === "decided" || s === "unsaved") return null;
  if (s === "failed") {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-red-600 px-1.5 py-0.5 text-[11px] font-semibold text-white"
            title={state?.failed}>
        not saved
        <button type="button" onClick={(e) => { e.stopPropagation(); onRetry(); }} aria-label="Retry saving"
                className="pointer-events-auto rounded px-0.5 hover:bg-white/20">
          <RotateCcw className="h-3 w-3" aria-hidden />
        </button>
      </span>
    );
  }
  const tone = s === "saved" ? (dark ? "bg-white/20 text-white" : "bg-white/90 text-myco-green dark:bg-black/70")
    : s === "needs-reason" ? "bg-amber-500 text-white"
      : dark ? "bg-white/20 text-white" : "bg-white/90 text-muted-foreground dark:bg-black/70";
  return <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${tone}`}>{STATUS_TEXT[s]}</span>;
}

function SummaryTile({ tile, reasons, mark, state, focused, tileRef, onFocus, onOpen, onFlip, onRetry, onLoaded }: {
  tile: FlaggedTile; reasons: Reasons; mark: Mark | undefined; state: SaveState | undefined; focused: boolean;
  tileRef: (el: HTMLLIElement | null) => void; onFocus: () => void; onOpen: () => void; onFlip: () => void;
  onRetry: () => void; onLoaded: () => void;
}) {
  // A thumb that fails to load becomes the placeholder (and never saves by itself).
  const [failed, setFailed] = useState(false);
  const src = failed ? null : tileImage(tile, "thumb");
  const keep = mark?.decision === "keep";
  const exclude = mark?.decision === "exclude";
  const readOnly = !!mark?.readOnly;
  const reason = exclude ? reasonLabel(mark?.reason, reasons) : null;
  const outline = keep ? "border-myco-green" : exclude ? "border-red-500/70 dark:border-red-400/70" : "border-transparent";
  return (
    <li ref={tileRef} data-item-id={tile.item_id} tabIndex={focused ? 0 : -1} onFocus={onFocus} onMouseDown={onFocus}
        className="min-w-0 space-y-0.5 outline-none" data-testid={`photo-summary-tile-${tile.item_id}`}>
      <div className={`relative aspect-square w-full overflow-hidden rounded-md ${keep ? "border-4" : "border-2"} ${outline} ${
        focused ? "ring-2 ring-myco-green ring-offset-2 dark:ring-offset-background" : ""}`}>
        {src ? (
          <button type="button" onClick={onOpen} className="block h-full w-full" tabIndex={-1}
                  aria-label={`Open ${recordLabel(tile.record_key)} full screen`}>
            <img src={src} alt={`${tile.label || "Photo"}, ${recordLabel(tile.record_key)}`} loading="lazy" decoding="async"
                 referrerPolicy="no-referrer" draggable={false} onLoad={onLoaded} onError={() => setFailed(true)}
                 className="h-full w-full object-cover" />
          </button>
        ) : (
          <div className="h-full w-full cursor-pointer" onClick={onOpen}>
            <Placeholder photo={shownPhoto(tile)} record={tile} failed={failed} />
          </div>
        )}
        {exclude && (
          <span className={`pointer-events-none absolute left-1 top-1 max-w-[calc(100%-0.5rem)] truncate rounded px-1.5 py-0.5 text-[10px] font-medium ${
            reason ? "bg-black/65 text-white" : "bg-amber-500 text-white"}`} title={reason ?? "Pick a reason full screen"}>
            {reason ?? "pick a reason"}
          </span>
        )}
        {tile.shown === "private" && (
          <span className="pointer-events-none absolute right-1 top-1 rounded bg-[#4a3728]/85 p-0.5 text-white"
                title="reviewers only · not public">
            <Lock className="h-3 w-3" aria-label="reviewers only, not public" />
          </span>
        )}
        <span className="pointer-events-none absolute bottom-1 left-1">
          <StatusMark mark={mark} state={state} onRetry={onRetry} />
        </span>
        {mark && (readOnly ? (
          <span className={`pointer-events-none absolute bottom-1 right-1 rounded px-1.5 py-0.5 text-[11px] font-semibold text-white ${
            keep ? "bg-myco-green" : "bg-red-600"}`}>{keep ? "Included" : "Excluded"}</span>
        ) : (
          <button type="button" onClick={(e) => { e.stopPropagation(); onFlip(); }} tabIndex={-1}
                  aria-label={keep ? "Included: click to exclude" : "Excluded: click to include"}
                  title={keep ? "Included: click to exclude (X)" : "Excluded: click to include (K)"}
                  className={`absolute bottom-1 right-1 inline-flex min-h-7 items-center gap-0.5 rounded px-1.5 text-[11px] font-semibold text-white shadow ${
                    keep ? "bg-myco-green hover:bg-myco-green/90" : "bg-red-600 hover:bg-red-700"}`}>
            {keep ? <><Check className="h-3 w-3" aria-hidden /> Include</> : <><X className="h-3 w-3" aria-hidden /> Exclude</>}
          </button>
        ))}
      </div>
      <p className={`truncate text-xs italic ${INK}`} title={tile.label}>{tile.label || "(no name)"}</p>
      <p className="truncate text-xs"><Ext href={tile.link}>{recordLabel(tile.record_key)}</Ext></p>
    </li>
  );
}

function SummaryLightbox({ tile, index, count, reasons, mark, state, onStep, onChoose, onReason, onRetry, onShown, onClose }: {
  tile: FlaggedTile; index: number; count: number; reasons: Reasons; mark: Mark | undefined;
  state: SaveState | undefined; onStep: (d: 1 | -1) => void; onChoose: (c: Choice) => void;
  onReason: (r: string) => void; onRetry: () => void; onShown: () => void; onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState<Record<number, boolean>>({});
  useEffect(() => { ref.current?.focus(); }, []);
  // Only the photo being viewed is loaded large: nothing is fetched ahead.
  const broken = !!failed[tile.item_id];
  const src = broken ? null : tileImage(tile, "full");
  const o = overlay(tile, reasons);
  const keep = mark?.decision === "keep";
  const readOnly = !mark || !!mark.readOnly;
  const big = "inline-flex min-h-12 items-center justify-center gap-1 rounded-md border px-6 text-base font-semibold";
  const known = !mark?.reason || reasons.some((r) => r.code === mark.reason);
  return (
    <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`${o.name}, ${o.link.text}`}
         className="fixed inset-0 z-[60] flex flex-col bg-black/90 p-2 outline-none sm:p-4" onClick={onClose}>
      <div className="flex min-w-0 items-center justify-between gap-3 text-sm text-white" onClick={(e) => e.stopPropagation()}>
        <span className="min-w-0 truncate">{index + 1} of {count} on this page</span>
        <button type="button" onClick={onClose} aria-label="Close"
                className="inline-flex min-h-10 min-w-10 items-center justify-center rounded-md hover:bg-white/10">
          <X className="h-5 w-5" aria-hidden />
        </button>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center gap-1 py-2">
        <button type="button" onClick={(e) => { e.stopPropagation(); onStep(-1); }} disabled={index === 0}
                aria-label="Previous photo"
                className="inline-flex min-h-10 min-w-10 shrink-0 items-center justify-center rounded-md text-white hover:bg-white/10 disabled:opacity-30">
          <ChevronLeft className="h-7 w-7" aria-hidden />
        </button>
        <div className="relative flex min-h-0 min-w-0 flex-1 items-center justify-center self-stretch"
             onClick={(e) => e.stopPropagation()}>
          {src ? (
            <img key={tile.item_id} src={src} alt={`${o.name}, ${o.link.text}`} referrerPolicy="no-referrer"
                 draggable={false} className="max-h-full max-w-full object-contain" onLoad={onShown}
                 onError={() => setFailed((f) => ({ ...f, [tile.item_id]: true }))} />
          ) : (
            <div className="aspect-square w-full max-w-sm overflow-hidden rounded-md">
              <Placeholder photo={shownPhoto(tile)} record={tile} failed={broken} large />
            </div>
          )}
          <div className="absolute inset-x-0 bottom-0 space-y-0.5 bg-black/65 p-3 text-sm text-white sm:inset-x-auto sm:left-0 sm:max-w-md sm:rounded-tr-md"
               data-testid="photo-summary-overlay">
            <p className="flex flex-wrap items-baseline gap-x-3">
              <span className="text-base font-semibold italic">{o.name}</span>
              <a href={o.link.href} target="_blank" rel="noopener" className="text-white underline underline-offset-2">
                {o.link.text}
              </a>
            </p>
            {o.lines.map((l) => <p key={l} className="break-words text-white/85">{l}</p>)}
          </div>
        </div>
        <button type="button" onClick={(e) => { e.stopPropagation(); onStep(1); }} disabled={index >= count - 1}
                aria-label="Next photo"
                className="inline-flex min-h-10 min-w-10 shrink-0 items-center justify-center rounded-md text-white hover:bg-white/10 disabled:opacity-30">
          <ChevronRight className="h-7 w-7" aria-hidden />
        </button>
      </div>
      <div className="mx-auto flex w-full max-w-2xl flex-col items-center gap-2 pt-1" onClick={(e) => e.stopPropagation()}>
        {readOnly ? (
          <p className="text-sm text-white">{decidedText(tile, reasons) ?? "Decided"}</p>
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button type="button" onClick={() => onChoose("keep")} aria-pressed={keep}
                      className={`${big} ${keep ? "border-myco-green bg-myco-green text-white"
                        : "border-white/40 bg-white/10 text-white hover:bg-white/20"}`}>
                <Check className="h-5 w-5" aria-hidden /> Include (K)
              </button>
              <button type="button" onClick={() => onChoose("exclude")} aria-pressed={!keep}
                      className={`${big} ${!keep ? "border-red-600 bg-red-600 text-white"
                        : "border-white/40 bg-white/10 text-white hover:bg-white/20"}`}>
                <X className="h-5 w-5" aria-hidden /> Exclude (X)
              </button>
              <StatusMark mark={mark} state={state} onRetry={onRetry} dark />
            </div>
            {!keep && (
              <select value={mark?.reason ?? ""} aria-label="Why this photo is left out"
                      onChange={(e) => onReason(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === "Escape") {
                          e.preventDefault();
                          e.stopPropagation();
                          e.currentTarget.blur();
                        }
                      }}
                      className={`${FIELD} w-full max-w-xs text-[#4a3728] dark:text-foreground ${
                        mark?.reason ? "" : "border-red-500 dark:border-red-500"}`}>
                {!mark?.reason && <option value="" disabled>Pick a reason…</option>}
                {reasons.map((r) => <option key={r.code} value={r.code}>{r.label}</option>)}
                {!known && mark?.reason && <option value={mark.reason}>{mark.reason}</option>}
              </select>
            )}
            {state?.failed && <p className="text-center text-xs text-red-300" role="alert">{state.failed}</p>}
          </>
        )}
        <p className="text-center text-xs text-white/70">
          {readOnly ? "← / → photos · Esc close" : "← / → photos · K include · X exclude (each saves) · Esc close"}
        </p>
      </div>
    </div>
  );
}
