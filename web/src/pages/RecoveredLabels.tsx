import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Lock, Move, RotateCcw, X } from "lucide-react";
import { BODY, BTN, Ext, FIELD, INK } from "@/components/ReviewPhotos";
import { SaveBar } from "@/components/ReviewSave";
import { ReadOnlyContext } from "@/components/ReadOnlyReview";
import { ApiError, feedbackApi, type FeedbackStatus, type FeedbackSummary, type RecoveredTile } from "@/lib/api";
import { pageText, recordLabel } from "@/lib/feedback";
import { tileImage } from "@/lib/photoReview";
import { type Box, boxOf, type BoxEdits, boxStyle, cropBackground, cropKey, cropShare, cropsUsed, dragBox, effectiveBox, flipCrop,
         type Handle, KEEP_PHOTO, pendingDecisions, RECOVERED_SORTS, type ReasonEdits, startChoice, tileFacts, TRANCHES,
         type CropMarks } from "@/lib/recoveredLabels";

// Recovered labels (Steve, 2026-10-26): photos a person excluded as field slips, where the
// specimen detector found a specimen. Each tile is the ORIGINAL photo with the proposed crop
// boxes drawn on it and, beside it, each box's part of the same image (CSS, no new file). Every
// crop starts as Include: it goes back into Vision unless a person rejects it. Smallest specimen
// first, where false finds (a printed logo, a finger) sit. The rules: lib/recoveredLabels.ts.
// Steve, 2026-10-27: a reviewer may change the photo's exclusion reason on the tile (only a
// field slip gives crops), and click the photo to move or resize its crop boxes.

const PAGE = 24;

export function RecoveredLabels({ summary }: { summary: FeedbackSummary }) {
  const readOnly = useContext(ReadOnlyContext);
  const qc = useQueryClient();
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [tranche, setTranche] = useState("all");
  const [sort, setSort] = useState("riskiest");
  const [offset, setOffset] = useState(0);
  const [marks, setMarks] = useState<CropMarks>({});
  const [edits, setEdits] = useState<BoxEdits>({});
  const [reasons, setReasons] = useState<ReasonEdits>({});
  const [editing, setEditing] = useState<number | null>(null);
  const [seen, setSeen] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const list = useQuery({
    queryKey: ["feedback", "recovered", status, tranche, sort, offset],
    queryFn: () => feedbackApi.recoveredLabels({ status, tranche, sort, offset, limit: PAGE }),
    placeholderData: (prev) => prev,
    refetchOnWindowFocus: false,
  });
  const data = list.data;
  const tiles = data?.tiles ?? [];
  const discard = () => { setMarks({}); setEdits({}); setReasons({}); };
  useEffect(() => { discard(); setSeen(new Set()); }, [status, tranche, sort, offset]);
  const pending = useMemo(() => (readOnly ? [] : pendingDecisions(tiles, marks, seen, edits, reasons)),
                          [tiles, marks, seen, edits, reasons, readOnly]);
  const reasonList = data?.reasons ?? [];
  const editTile = tiles.find((t) => t.item_id === editing);
  const save = () => {
    if (!pending.length) return;
    setBusy(true);
    setError(null);
    feedbackApi.recoveredLabelsBatch(pending).then(
      () => { setBusy(false); discard(); qc.invalidateQueries({ queryKey: ["feedback"] }); },
      (e) => { setBusy(false); setError(e instanceof ApiError || e instanceof Error ? e.message : "Not saved."); });
  };
  const count = (code: string) => {
    const t = data?.tranches.find((x) => x.code === code);
    if (code === "all") {
      const n = (data?.tranches ?? []).reduce((a, x) => a + (status === "open" ? x.open : x.decided), 0);
      return data ? ` (${n.toLocaleString("en-US")})` : "";
    }
    return t ? ` (${(status === "open" ? t.open : t.decided).toLocaleString("en-US")})` : "";
  };
  return (
    <section className="min-w-0 space-y-3" data-testid="recovered-labels">
      <div className="min-w-0">
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{summary.about}</p>
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border" role="group"
             aria-label="Open or decided">
          {(["open", "decided"] as FeedbackStatus[]).map((s, i) => (
            <button key={s} type="button" aria-pressed={status === s} onClick={() => { setStatus(s); setOffset(0); }}
                    className={`min-h-10 px-3 py-1.5 text-sm font-medium ${i ? "border-l border-[#A87146]/30 dark:border-border" : ""} ${
                      status === s ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`}>
              {s === "open" ? "Open" : "Decided"}
            </button>
          ))}
        </div>
        <label className="flex min-w-0 items-center gap-2 text-sm"
               title="The photo-type tagger's slip p when the photo was put up for review">
          <span className="text-muted-foreground">Tranche</span>
          <select className={FIELD} value={tranche} onChange={(e) => { setTranche(e.target.value); setOffset(0); }}
                  data-testid="recovered-tranche">
            {TRANCHES.map((t) => <option key={t.code} value={t.code}>{t.label}{count(t.code)}</option>)}
          </select>
        </label>
        <label className="flex min-w-0 items-center gap-2 text-sm">
          <span className="text-muted-foreground">Sort</span>
          <select className={FIELD} value={sort} onChange={(e) => { setSort(e.target.value); setOffset(0); }}>
            {RECOVERED_SORTS.map((s) => <option key={s.code} value={s.code}>{s.label}</option>)}
          </select>
        </label>
        {list.isFetching && <span className="text-xs text-muted-foreground">Loading…</span>}
      </div>
      {!readOnly && status === "open" && (
        <SaveBar count={pending.length} busy={busy} onSave={save} onDiscard={discard} sticky
                 note="A photo you have seen counts as reviewed; its crops are saved as shown." />
      )}
      {status === "open" && (
        <p className="text-xs text-muted-foreground">
          {readOnly
            ? "Each outlined box is a crop of the original photo that goes back into Vision unless a reviewer rejects it."
            : "Every crop starts as Include and goes back into Vision. Reject a crop that is not a fungus (a printed logo, " +
              "a hand, a stick) or that is still mostly label. Click a photo to move or resize its crops. The photo " +
              "itself stays excluded either way; only a photo excluded as a field slip gives crops."}
        </p>
      )}
      {error && <p className="text-sm text-red-700 dark:text-red-300">{error}</p>}
      {list.error && <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The crops could not be loaded."}</p>}
      {data && !tiles.length && <p className="text-sm text-muted-foreground">{status === "open" ? "Nothing open here." : "Nothing decided here yet."}</p>}
      <ul className="grid items-start gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(100%, 22rem), 1fr))" }}
          aria-label="Proposed crops" data-testid="recovered-grid">
        {tiles.map((t) => (
          <Tile key={t.item_id} tile={t} marks={marks} edits={edits} readOnly={readOnly} reasons={reasonList}
                reason={reasons[t.item_id] ?? t.reason ?? "slip"}
                onSeen={() => setSeen((s) => (s.has(t.item_id) ? s : new Set(s).add(t.item_id)))}
                onFlip={(n) => setMarks((m) => flipCrop(m, t, n))}
                onReason={(r) => setReasons((x) => ({ ...x, [t.item_id]: r }))}
                onEdit={() => setEditing(t.item_id)} />
        ))}
      </ul>
      {editTile && !readOnly && (
        <CropEditor tile={editTile} marks={marks} edits={edits} onClose={() => setEditing(null)}
                    onDone={(boxes, keep) => {
                      setEdits((e) => ({ ...e, ...boxes }));
                      setMarks((m) => ({ ...m, ...keep }));
                      setSeen((s) => new Set(s).add(editTile.item_id));
                      setEditing(null);
                    }} />
      )}
      {data && data.total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground">{pageText(offset, tiles.length, data.total)}</span>
          <span className="flex gap-2">
            <button type="button" className={BTN} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
            <button type="button" className={BTN} disabled={offset + tiles.length >= data.total}
                    onClick={() => setOffset(offset + PAGE)}>Next</button>
          </span>
        </div>
      )}
    </section>
  );
}

function Tile({ tile, marks, edits, readOnly, reasons, reason, onSeen, onFlip, onReason, onEdit }: {
  tile: RecoveredTile; marks: CropMarks; edits: BoxEdits; readOnly: boolean; reasons: { code: string; label: string }[];
  reason: string; onSeen: () => void; onFlip: (n: number) => void; onReason: (r: string) => void; onEdit: () => void;
}) {
  const ref = useRef<HTMLLIElement>(null);
  const src = tileImage(tile, "full");
  // Seen: half the tile in view for a second (the Summary view's rule).
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    let timer: number | undefined;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && e.intersectionRatio >= 0.5) timer = window.setTimeout(() => { onSeen(); io.disconnect(); }, 1000);
      else if (timer) window.clearTimeout(timer);
    }, { threshold: [0, 0.5, 1] });
    io.observe(el);
    return () => { io.disconnect(); if (timer) window.clearTimeout(timer); };
  }, [onSeen]);
  const choice = (n: number) => {
    const c = tile.crops.find((x) => x.n === n)!;
    return marks[cropKey(tile.item_id, n)] ?? startChoice(tile, c);
  };
  const box = (c: RecoveredTile["crops"][number]) => boxOf(tile, c, edits);
  const moved = tile.crops.some((c) => box(c).some((v, i) => v !== effectiveBox(c.box)[i]));
  const used = cropsUsed(reason);
  const changed = tile.crops.some((c) => choice(c.n) !== startChoice(tile, c)) || moved || reason !== (tile.reason ?? "slip");
  return (
    <li ref={ref} className={`min-w-0 rounded-lg border bg-white p-2 dark:bg-card ${
          changed ? "border-amber-400 border-dashed" : "border-[#A87146]/20 dark:border-border"}`}
        data-testid={`recovered-tile-${tile.item_id}`}>
      <div className="flex min-w-0 gap-2">
        <div className={`group relative w-[58%] shrink-0 self-start overflow-hidden rounded-md bg-black/5 ${
               !readOnly && src ? "cursor-pointer" : ""}`} style={{ aspectRatio: `${tile.width} / ${tile.height}` }}
             onClick={!readOnly && src ? onEdit : undefined} role={!readOnly && src ? "button" : undefined}
             tabIndex={!readOnly && src ? 0 : undefined} aria-label={!readOnly && src ? "Move or resize the crops" : undefined}
             onKeyDown={(e) => { if (!readOnly && src && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onEdit(); } }}>
          {src ? <img src={src} alt={`${tile.label}, ${recordLabel(tile.record_key)} (excluded as a field slip)`} loading="lazy"
                      referrerPolicy="no-referrer" className="h-full w-full object-cover opacity-80" draggable={false} />
               : <div className="flex h-full items-center justify-center p-2 text-center text-xs text-muted-foreground">{tile.hidden_reason ?? "Photo not shown"}</div>}
          {tile.crops.map((c) => {
            const on = choice(c.n) === "include" && used;
            return (
              <span key={c.n} className={`pointer-events-none absolute rounded-sm border-2 ${on ? "border-amber-400" : "border-red-500 border-dashed"}`}
                    style={boxStyle(box(c), tile.width, tile.height)}>
                {tile.crops.length > 1 && <span className="absolute -left-0.5 -top-0.5 rounded-br bg-amber-400 px-1 text-[10px] font-bold text-black">{c.n}</span>}
              </span>
            );
          })}
          {tile.shown === "private" && (
            <span className="pointer-events-none absolute right-1 top-1 rounded bg-[#4a3728]/85 p-0.5 text-white" title="reviewers only · not public">
              <Lock className="h-3 w-3" aria-label="reviewers only, not public" />
            </span>
          )}
          <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-black/65 px-1.5 py-0.5 text-[10px] font-medium text-white">
            {tile.still_excluded === false ? "no longer excluded: crops not used"
              : reason === KEEP_PHOTO ? "kept: crops not used"
              : `excluded: ${(reasons.find((r) => r.code === reason)?.label ?? "field slip").toLowerCase()}`}
          </span>
          {!readOnly && src && (
            <span className="pointer-events-none absolute right-1 bottom-1 hidden items-center gap-1 rounded bg-black/65 px-1.5 py-0.5 text-[10px] font-medium text-white group-hover:inline-flex">
              <Move className="h-3 w-3" aria-hidden /> adjust
            </span>
          )}
        </div>
        <div className={`grid min-w-0 flex-1 content-start gap-2 ${tile.crops.length > 2 ? "grid-cols-2" : "grid-cols-1"}`}>
          {tile.crops.map((c) => {
            const on = choice(c.n) === "include" && used;
            const edited = box(c).some((v, i) => v !== effectiveBox(c.box)[i]);
            return (
              <div key={c.n} className="min-w-0">
                {/* Reviewers see the box's part of the original; the public view keeps to the outline. */}
                {src && !readOnly ? (
                  <div className={`relative aspect-square w-full rounded-md border-2 bg-black/5 ${on ? "border-myco-green" : "border-red-500/70 opacity-50"}`}
                       style={cropBackground(src, box(c), tile.width, tile.height)} role="img"
                       aria-label={`Crop ${c.n} of ${tile.crops.length}`}>
                    {edited && <span className="absolute left-1 top-1 rounded bg-amber-400 px-1 text-[10px] font-semibold text-black">edited</span>}
                  </div>
                ) : null}
                <div className="mt-1 flex items-center justify-between gap-1">
                  <span className="text-[11px] text-muted-foreground">
                    {tile.crops.length > 1 ? `${c.n}: ` : ""}{cropShare(box(c), tile.width, tile.height)}% of photo
                  </span>
                  {readOnly ? (
                    <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold text-white ${on ? "bg-myco-green" : "bg-red-600"}`}>
                      {on ? "Included" : "Rejected"}
                    </span>
                  ) : (
                    <button type="button" onClick={() => onFlip(c.n)} disabled={!used}
                            title={used ? undefined : "Only a photo excluded as a field slip gives crops"}
                            aria-label={on ? `Crop ${c.n} included: click to reject` : `Crop ${c.n} rejected: click to include`}
                            className={`inline-flex min-h-7 items-center gap-0.5 rounded px-1.5 text-[11px] font-semibold text-white shadow disabled:opacity-50 ${
                              on ? "bg-myco-green hover:bg-myco-green/90" : "bg-red-600 hover:bg-red-700"}`}>
                      {on ? <><Check className="h-3 w-3" aria-hidden /> Include</> : <><X className="h-3 w-3" aria-hidden /> Reject</>}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <p className={`mt-1 truncate text-xs italic ${INK}`} title={tile.label}>{tile.label || "(no name)"}</p>
      <p className="flex min-w-0 flex-wrap justify-between gap-x-2 text-xs">
        <Ext href={tile.link}>{recordLabel(tile.record_key)}</Ext>
        <span className="text-muted-foreground">{tileFacts(tile)}</span>
      </p>
      {!readOnly && reasons.length > 0 && (
        <label className="mt-1.5 flex min-w-0 items-center gap-2 text-xs">
          <span className="shrink-0 text-muted-foreground">Excluded as</span>
          <select className={`${FIELD} !min-h-8 flex-1 !py-1 !text-xs`} value={reason} onChange={(e) => onReason(e.target.value)}
                  aria-label="Why this photo is left out" data-testid={`recovered-reason-${tile.item_id}`}>
            {reasons.map((r) => <option key={r.code} value={r.code}>{r.label}</option>)}
            <option value={KEEP_PHOTO}>Not excluded: keep the whole photo</option>
          </select>
        </label>
      )}
      {!readOnly && !used && (
        <p className="mt-1 text-[11px] text-amber-800 dark:text-amber-300">
          Its crops are not used: only a photo excluded as a field slip gives crops.
        </p>
      )}
    </li>
  );
}

// --- the crop editor: the original photo large, each box moved or resized by dragging ------------

const HANDLES: Exclude<Handle, "move">[] = ["nw", "ne", "sw", "se"];
const HANDLE_POS: Record<Exclude<Handle, "move">, string> = {
  nw: "-left-2 -top-2 cursor-nwse-resize", ne: "-right-2 -top-2 cursor-nesw-resize",
  sw: "-left-2 -bottom-2 cursor-nesw-resize", se: "-right-2 -bottom-2 cursor-nwse-resize",
};

/** One photo's crops on the original, large: drag a box to move it, a corner to resize it;
 *  arrow keys move the chosen box (Shift: 10x), Shift+Alt+arrows resize it. Every box stays
 *  square, the region the model reads (dragBox, effectiveBox). Done keeps the boxes (an edited
 *  crop is included), Esc or Cancel leaves them as they were. */
function CropEditor({ tile, marks, edits, onDone, onClose }: {
  tile: RecoveredTile; marks: CropMarks; edits: BoxEdits;
  onDone: (boxes: BoxEdits, keep: CropMarks) => void; onClose: () => void;
}) {
  const src = tileImage(tile, "full");
  const [boxes, setBoxes] = useState<Record<number, Box>>(
    () => Object.fromEntries(tile.crops.map((c) => [c.n, boxOf(tile, c, edits)])));
  const [active, setActive] = useState(tile.crops[0]?.n ?? 1);
  const frame = useRef<HTMLDivElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const drag = useRef<{ n: number; handle: Handle; x: number; y: number; start: Box } | null>(null);
  useEffect(() => { root.current?.focus(); }, []);
  const W = tile.width, H = tile.height;
  const scale = () => { const r = frame.current?.getBoundingClientRect(); return r && r.width ? W / r.width : 1; };
  const down = (n: number, handle: Handle) => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    frame.current?.setPointerCapture(e.pointerId);
    setActive(n);
    drag.current = { n, handle, x: e.clientX, y: e.clientY, start: boxes[n] };
  };
  const moveTo = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const k = scale();
    setBoxes((b) => ({ ...b, [d.n]: dragBox(d.start, d.handle, (e.clientX - d.x) * k, (e.clientY - d.y) * k, W, H) }));
  };
  const up = () => { drag.current = null; };
  const done = () => {
    const out: BoxEdits = {}, keep: CropMarks = {};
    for (const c of tile.crops) {
      const b = boxes[c.n];
      out[cropKey(tile.item_id, c.n)] = b;
      if (b.some((v, i) => v !== effectiveBox(c.box)[i])) keep[cropKey(tile.item_id, c.n)] = "include";
    }
    onDone(out, keep);
  };
  const key = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
    if (e.key === "Enter") { e.preventDefault(); done(); return; }
    if (/^[1-9]$/.test(e.key) && tile.crops.some((c) => c.n === +e.key)) { setActive(+e.key); return; }
    const dir: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!dir[e.key]) return;
    e.preventDefault();
    const step = Math.max(1, Math.round(0.01 * Math.min(W, H))) * (e.shiftKey && !e.altKey ? 10 : 1);
    const [dx, dy] = dir[e.key].map((v) => v * step);
    // Shift+Alt: → or ↓ grows the square, ← or ↑ shrinks it, its top-left corner fixed.
    setBoxes((b) => ({ ...b, [active]: e.shiftKey && e.altKey
      ? dragBox(b[active], "se", dx + dy, dx + dy, W, H)
      : dragBox(b[active], "move", dx, dy, W, H) }));
  };
  const reset = () => setBoxes((b) => {
    const c = tile.crops.find((x) => x.n === active)!;
    return { ...b, [active]: effectiveBox(c.proposed ?? c.box) };
  });
  return (
    <div ref={root} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Adjust the crops of ${recordLabel(tile.record_key)}`}
         className="fixed inset-0 z-[60] flex flex-col bg-black/85 p-3 outline-none sm:p-6" onKeyDown={key} onClick={onClose}
         data-testid="crop-editor">
      <div className="flex min-w-0 items-center justify-between gap-3 text-sm text-white" onClick={(e) => e.stopPropagation()}>
        <span className="min-w-0 break-words">
          <span className="italic">{tile.label}</span> · {recordLabel(tile.record_key)} · drag a box to move it, a corner to resize it ·
          boxes stay square: the model reads a square, and a box stops at the photo's edge
        </span>
        <button type="button" onClick={onClose} aria-label="Cancel"
                className="inline-flex min-h-10 min-w-10 items-center justify-center rounded-md hover:bg-white/10">
          <X className="h-5 w-5" aria-hidden />
        </button>
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center gap-4 py-3" onClick={(e) => e.stopPropagation()}>
        <div ref={frame} className="relative max-h-full touch-none select-none" onPointerMove={moveTo} onPointerUp={up}
             onPointerCancel={up}
             style={{ aspectRatio: `${W} / ${H}`, height: "min(100%, 78vh)", maxWidth: "min(100%, 70vw)" }}>
          <img src={src ?? undefined} alt="" referrerPolicy="no-referrer" draggable={false} className="h-full w-full object-contain" />
          {tile.crops.map((c) => {
            const on = active === c.n;
            const off = (marks[cropKey(tile.item_id, c.n)] ?? startChoice(tile, c)) === "reject";
            return (
              <div key={c.n} onPointerDown={down(c.n, "move")}
                   className={`absolute cursor-move border-2 ${on ? "border-amber-300 bg-amber-300/10" : off ? "border-dashed border-red-500" : "border-amber-400/80"}`}
                   style={boxStyle(boxes[c.n], W, H)} aria-label={`Crop ${c.n}`}>
                {tile.crops.length > 1 && <span className="absolute -left-0.5 -top-0.5 rounded-br bg-amber-400 px-1 text-[11px] font-bold text-black">{c.n}</span>}
                {on && HANDLES.map((h) => (
                  <span key={h} onPointerDown={down(c.n, h)}
                        className={`absolute h-4 w-4 rounded-sm border-2 border-black bg-amber-300 ${HANDLE_POS[h]}`} />
                ))}
              </div>
            );
          })}
        </div>
        <div className="hidden w-48 shrink-0 flex-col gap-2 md:flex">
          {tile.crops.map((c) => (
            <button key={c.n} type="button" onClick={() => setActive(c.n)}
                    className={`relative aspect-square w-full rounded-md border-2 bg-black/30 ${active === c.n ? "border-amber-300" : "border-white/20"}`}
                    style={src ? cropBackground(src, boxes[c.n], W, H) : undefined} aria-label={`Crop ${c.n}: choose`}>
              {tile.crops.length > 1 && <span className="absolute left-1 top-1 rounded bg-amber-400 px-1 text-[10px] font-bold text-black">{c.n}</span>}
            </button>
          ))}
        </div>
      </div>
      <div className="mx-auto flex w-full max-w-2xl flex-wrap items-center justify-center gap-2" onClick={(e) => e.stopPropagation()}>
        <button type="button" className={BTN} onClick={reset}><RotateCcw className="h-4 w-4" aria-hidden /> Detector's box</button>
        <button type="button" className={BTN} onClick={onClose}>Cancel</button>
        <button type="button" onClick={done}
                className="inline-flex min-h-10 items-center gap-1 rounded-md bg-myco-green px-4 text-sm font-semibold text-white hover:bg-myco-green/90">
          <Check className="h-4 w-4" aria-hidden /> Done
        </button>
        <p className="w-full text-center text-xs text-white/70">
          drag to move · corners to resize (stays square) · arrows move (Shift ×10) · Shift+Alt+→/↓ grow, ←/↑ shrink
          {tile.crops.length > 1 ? ` · 1–${tile.crops.length} choose a box` : ""} · Enter done · Esc cancel
        </p>
      </div>
    </div>
  );
}
