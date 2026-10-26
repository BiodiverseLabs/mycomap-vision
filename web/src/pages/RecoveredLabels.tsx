import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Lock, X } from "lucide-react";
import { BODY, BTN, Ext, FIELD, INK } from "@/components/ReviewPhotos";
import { SaveBar } from "@/components/ReviewSave";
import { ReadOnlyContext } from "@/components/ReadOnlyReview";
import { ApiError, feedbackApi, type FeedbackStatus, type FeedbackSummary, type RecoveredTile } from "@/lib/api";
import { pageText, recordLabel } from "@/lib/feedback";
import { tileImage } from "@/lib/photoReview";
import { boxStyle, cropBackground, cropKey, cropShare, flipCrop, pendingDecisions, RECOVERED_SORTS, startChoice,
         tileFacts, TRANCHES, type CropMarks } from "@/lib/recoveredLabels";

// Recovered labels (Steve, 2026-10-26): photos a person excluded as field slips, where the
// specimen detector found a specimen. Each tile is the ORIGINAL photo with the proposed crop
// boxes drawn on it and, beside it, each box's part of the same image (CSS, no new file). Every
// crop starts as Include: it goes back into Vision unless a person rejects it. Smallest specimen
// first, where false finds (a printed logo, a finger) sit. The rules: lib/recoveredLabels.ts.

const PAGE = 24;

export function RecoveredLabels({ summary }: { summary: FeedbackSummary }) {
  const readOnly = useContext(ReadOnlyContext);
  const qc = useQueryClient();
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [tranche, setTranche] = useState("all");
  const [sort, setSort] = useState("riskiest");
  const [offset, setOffset] = useState(0);
  const [marks, setMarks] = useState<CropMarks>({});
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
  useEffect(() => { setMarks({}); setSeen(new Set()); }, [status, tranche, sort, offset]);
  const pending = useMemo(() => (readOnly ? [] : pendingDecisions(tiles, marks, seen)), [tiles, marks, seen, readOnly]);
  const save = () => {
    if (!pending.length) return;
    setBusy(true);
    setError(null);
    feedbackApi.recoveredLabelsBatch(pending).then(
      () => { setBusy(false); qc.invalidateQueries({ queryKey: ["feedback"] }); },
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
        <SaveBar count={pending.length} busy={busy} onSave={save} onDiscard={() => setMarks({})} sticky
                 note="A photo you have seen counts as reviewed; its crops are saved as shown." />
      )}
      {status === "open" && (
        <p className="text-xs text-muted-foreground">
          {readOnly
            ? "Each outlined box is a crop of the original photo that goes back into Vision unless a reviewer rejects it."
            : "Every crop starts as Include and goes back into Vision. Reject a crop that is not a fungus (a printed logo, " +
              "a hand, a stick) or that is still mostly label. The photo itself stays excluded either way."}
        </p>
      )}
      {error && <p className="text-sm text-red-700 dark:text-red-300">{error}</p>}
      {list.error && <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The crops could not be loaded."}</p>}
      {data && !tiles.length && <p className="text-sm text-muted-foreground">{status === "open" ? "Nothing open here." : "Nothing decided here yet."}</p>}
      <ul className="grid items-start gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(100%, 22rem), 1fr))" }}
          aria-label="Proposed crops" data-testid="recovered-grid">
        {tiles.map((t) => (
          <Tile key={t.item_id} tile={t} marks={marks} readOnly={readOnly}
                onSeen={() => setSeen((s) => (s.has(t.item_id) ? s : new Set(s).add(t.item_id)))}
                onFlip={(n) => setMarks((m) => flipCrop(m, t, n))} />
        ))}
      </ul>
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

function Tile({ tile, marks, readOnly, onSeen, onFlip }: {
  tile: RecoveredTile; marks: CropMarks; readOnly: boolean; onSeen: () => void; onFlip: (n: number) => void;
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
  const changed = tile.crops.some((c) => choice(c.n) !== startChoice(tile, c));
  return (
    <li ref={ref} className={`min-w-0 rounded-lg border bg-white p-2 dark:bg-card ${
          changed ? "border-amber-400 border-dashed" : "border-[#A87146]/20 dark:border-border"}`}
        data-testid={`recovered-tile-${tile.item_id}`}>
      <div className="flex min-w-0 gap-2">
        <div className="relative w-[58%] shrink-0 self-start overflow-hidden rounded-md bg-black/5" style={{ aspectRatio: `${tile.width} / ${tile.height}` }}>
          {src ? <img src={src} alt={`${tile.label}, ${recordLabel(tile.record_key)} (excluded as a field slip)`} loading="lazy"
                      referrerPolicy="no-referrer" className="h-full w-full object-cover opacity-80" draggable={false} />
               : <div className="flex h-full items-center justify-center p-2 text-center text-xs text-muted-foreground">{tile.hidden_reason ?? "Photo not shown"}</div>}
          {tile.crops.map((c) => {
            const on = choice(c.n) === "include";
            return (
              <span key={c.n} className={`pointer-events-none absolute rounded-sm border-2 ${on ? "border-amber-400" : "border-red-500 border-dashed"}`}
                    style={boxStyle(c.box, tile.width, tile.height)}>
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
            {tile.still_excluded === false ? "no longer excluded: crops not used" : "excluded: field slip"}
          </span>
        </div>
        <div className={`grid min-w-0 flex-1 content-start gap-2 ${tile.crops.length > 2 ? "grid-cols-2" : "grid-cols-1"}`}>
          {tile.crops.map((c) => {
            const on = choice(c.n) === "include";
            return (
              <div key={c.n} className="min-w-0">
                {/* Reviewers see the box's part of the original; the public view keeps to the outline. */}
                {src && !readOnly ? (
                  <div className={`relative aspect-square w-full rounded-md border-2 bg-black/5 ${on ? "border-myco-green" : "border-red-500/70 opacity-50"}`}
                       style={cropBackground(src, c.box, tile.width, tile.height)} role="img"
                       aria-label={`Crop ${c.n} of ${tile.crops.length}`} />
                ) : null}
                <div className="mt-1 flex items-center justify-between gap-1">
                  <span className="text-[11px] text-muted-foreground">
                    {tile.crops.length > 1 ? `${c.n}: ` : ""}{cropShare(c.box, tile.width, tile.height)}% of photo
                  </span>
                  {readOnly ? (
                    <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold text-white ${on ? "bg-myco-green" : "bg-red-600"}`}>
                      {on ? "Included" : "Rejected"}
                    </span>
                  ) : (
                    <button type="button" onClick={() => onFlip(c.n)}
                            aria-label={on ? `Crop ${c.n} included: click to reject` : `Crop ${c.n} rejected: click to include`}
                            className={`inline-flex min-h-7 items-center gap-0.5 rounded px-1.5 text-[11px] font-semibold text-white shadow ${
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
    </li>
  );
}
