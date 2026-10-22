import { createContext, useContext, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { feedbackApi, signInUrl, type FeedbackMe, type FeedbackStatus, type FeedbackSummary,
         type PhotoBandFilter, type PhotoReviewSort, type PhotoReviewTab } from "@/lib/api";
import { BODY, FIELD, StatusToggle } from "@/components/ReviewPhotos";
import { photoLines, readOnlyNote } from "@/lib/feedbackReadOnly";

// Human Feedback for a read-only viewer (lib/feedbackReadOnly.ts): the same queues, photos and
// outcomes, with no control that writes. The editing views (the Summary's flips and Auto-save,
// the record view's choices, Photo views' tags, the label review's outcomes) are not shown to
// them; these plain grids are.

export const ReadOnlyContext = createContext(false);
export const useReadOnly = () => useContext(ReadOnlyContext);

const INK = "text-[#4a3728] dark:text-foreground";
const BTN = "rounded-md border border-[#A87146]/30 bg-white px-3 py-1 text-sm hover:bg-myco-green/5 disabled:opacity-40 dark:border-border dark:bg-card";

export function ReadOnlyNote({ me }: { me: FeedbackMe | undefined }) {
  const note = readOnlyNote(me);
  if (!note) return null;
  return (
    <p className="rounded-md border border-[#A87146]/20 bg-[#faf9f7] px-3 py-2 text-sm dark:border-border dark:bg-muted/30"
       data-testid="feedback-read-only">
      {note.text}{" "}
      {note.signIn && <a href={signInUrl()} className="text-myco-green underline">Sign in</a>}
    </p>
  );
}

interface GridPhoto {
  key: string;
  shown: "public" | "private" | "hidden";
  thumb?: string;
  full?: string;
  hidden_reason?: string;
  title?: string;
  link?: string;
  lines: string[];
}

/** Photos with what is known of each; a photo the viewer may not see is a placeholder. */
export function ReadOnlyGrid({ photos }: { photos: GridPhoto[] }) {
  return (
    <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5" data-testid="feedback-read-only-grid">
      {photos.map((p) => (
        <li key={p.key} className="min-w-0 space-y-1 text-xs">
          {p.shown === "public" && p.thumb ? (
            <a href={p.full ?? p.thumb} target="_blank" rel="noreferrer">
              <img src={p.thumb} alt={p.title ?? p.key} loading="lazy"
                   className="aspect-square w-full rounded-md bg-muted object-cover" />
            </a>
          ) : (
            <div className="flex aspect-square w-full items-center justify-center rounded-md bg-muted p-2 text-center text-muted-foreground">
              Not shown{p.hidden_reason ? `: ${p.hidden_reason}` : ""}
            </div>
          )}
          {p.title && (
            <p className={`truncate font-medium ${INK}`} title={p.title}>
              {p.link ? <a href={p.link} target="_blank" rel="noreferrer" className="underline">{p.title}</a> : p.title}
            </p>
          )}
          {p.lines.map((l) => <p key={l} className="break-words text-muted-foreground">{l}</p>)}
        </li>
      ))}
    </ul>
  );
}

function Pager({ offset, size, total, onPage }: { offset: number; size: number; total: number; onPage: (o: number) => void }) {
  if (total <= size) return null;
  return (
    <div className="flex items-center gap-3 text-sm">
      <button type="button" className={BTN} disabled={offset === 0} onClick={() => onPage(Math.max(0, offset - size))}>Previous</button>
      <span className="text-muted-foreground">{offset + 1}–{Math.min(total, offset + size)} of {total.toLocaleString("en-US")}</span>
      <button type="button" className={BTN} disabled={offset + size >= total} onClick={() => onPage(offset + size)}>Next</button>
    </div>
  );
}

const reasonOf = (reasons: { code: string; label: string }[] | undefined) =>
  (code: string) => reasons?.find((r) => r.code === code)?.label ?? code;

/** The photo review read-only: Summary (one tile per flagged photo) or Full (by record). */
export function ReadOnlyPhotoReview({ view, tab, status, sort, q, band, pageSize }: {
  view: "summary" | "full"; tab: PhotoReviewTab; status: FeedbackStatus; sort: PhotoReviewSort; q: string;
  band: PhotoBandFilter; pageSize: number;
}) {
  const [offset, setOffset] = useState(0);
  const size = view === "summary" ? pageSize : 10;
  const flagged = useQuery({
    queryKey: ["feedback", "read-only", "flagged", tab, status, sort, q, band, offset, size],
    queryFn: () => feedbackApi.photoReviewFlagged({ tab, status, sort, q, band, offset, limit: size }),
    enabled: view === "summary",
  });
  const records = useQuery({
    queryKey: ["feedback", "read-only", "records", tab, status, sort, q, band, offset],
    queryFn: () => feedbackApi.photoReview({ tab, status, sort, q, band, offset, limit: size }),
    enabled: view === "full",
  });
  if (view === "summary") {
    const d = flagged.data;
    if (!d) return <p className="text-sm text-muted-foreground">{flagged.error ? "The photos could not be loaded." : "Loading…"}</p>;
    if (!d.tiles.length) return <p className="text-sm text-muted-foreground">No photos here.</p>;
    const why = reasonOf(d.reasons);
    return (
      <div className="space-y-3">
        <ReadOnlyGrid photos={d.tiles.map((t) => ({
          key: t.photo_key, shown: t.shown, thumb: t.thumb, full: t.full, hidden_reason: t.hidden_reason,
          title: t.label || t.record_key, link: t.link,
          lines: [...(t.shown === "public" && t.owner_login ? [`Photo: ${t.owner_login}`] : []),
                  ...photoLines(t, why)],
        }))} />
        <Pager offset={offset} size={size} total={d.total} onPage={setOffset} />
      </div>
    );
  }
  const d = records.data;
  if (!d) return <p className="text-sm text-muted-foreground">{records.error ? "The records could not be loaded." : "Loading…"}</p>;
  if (!d.records.length) return <p className="text-sm text-muted-foreground">No records here.</p>;
  const why = reasonOf(d.reasons);
  return (
    <div className="space-y-6">
      {d.records.map((r) => (
        <section key={r.record_key} className="space-y-2">
          <h3 className={`text-sm font-semibold ${INK}`}>
            <a href={r.link} target="_blank" rel="noreferrer" className="underline">{r.label || r.record_key}</a>
            <span className="ml-2 font-normal text-muted-foreground">{r.record_key}</span>
          </h3>
          <ReadOnlyGrid photos={r.photos.map((p) => ({
            key: p.photo_key, shown: p.shown, thumb: p.thumb, full: p.full, hidden_reason: p.hidden_reason,
            lines: p.flagged ? photoLines(p, why) : ["Not flagged"],
          }))} />
        </section>
      ))}
      <Pager offset={offset} size={size} total={d.total} onPage={setOffset} />
    </div>
  );
}

/** Photo views read-only: each photo with the tagger's suggestions and the confirmed views. */
export function ReadOnlyPhotoViews({ tab, status, q }: { tab: string; status: FeedbackStatus; q: string }) {
  const [offset, setOffset] = useState(0);
  const size = 60;
  const page = useQuery({
    queryKey: ["feedback", "read-only", "photo-views", tab, status, q, offset],
    queryFn: () => feedbackApi.photoViews({ tab, status, q, offset, limit: size }),
  });
  const d = page.data;
  if (!d) return <p className="text-sm text-muted-foreground">{page.error ? "The photos could not be loaded." : "Loading…"}</p>;
  if (!d.tiles.length) return <p className="text-sm text-muted-foreground">No photos here.</p>;
  return (
    <div className="space-y-3">
      <ReadOnlyGrid photos={d.tiles.map((t) => ({
        key: t.photo_key, shown: t.shown, thumb: t.thumb, full: t.full, hidden_reason: t.hidden_reason,
        title: t.label || t.record_key, link: t.link, lines: photoLines(t),
      }))} />
      <Pager offset={offset} size={size} total={d.total} onPage={setOffset} />
    </div>
  );
}

/** The label review read-only: each record and its photos; an open card stays blind. */
export function ReadOnlyLabelReview({ status, q }: { status: FeedbackStatus; q: string }) {
  const [offset, setOffset] = useState(0);
  const size = 10;
  const page = useQuery({
    queryKey: ["feedback", "read-only", "label-review", status, q, offset],
    queryFn: () => feedbackApi.labelReview({ status, q, offset, limit: size }),
  });
  const d = page.data;
  if (!d) return <p className="text-sm text-muted-foreground">{page.error ? "The records could not be loaded." : "Loading…"}</p>;
  if (!d.records.length) return <p className="text-sm text-muted-foreground">No records here.</p>;
  const outcome = (code: string) => d.outcomes.find((o) => o.code === code)?.label ?? code;
  return (
    <div className="space-y-6">
      {d.records.map((c) => (
        <section key={c.item_id} className="space-y-2">
          <h3 className={`text-sm font-semibold ${INK}`}>
            <a href={c.link} target="_blank" rel="noreferrer" className="underline">{c.label || c.record_key}</a>
            <span className="ml-2 font-normal text-muted-foreground">
              {c.status === "decided" && c.decision
                ? `Decided: ${outcome(c.decision.decision)}${c.decision.value ? `, ${c.decision.value}` : ""}`
                : "Open"}
            </span>
          </h3>
          <ReadOnlyGrid photos={c.photos.map((p) => ({
            key: p.photo_key, shown: p.shown, thumb: p.thumb, full: p.full, hidden_reason: p.hidden_reason,
            lines: p.excluded ? ["Left out of Vision"] : [],
          }))} />
        </section>
      ))}
      <Pager offset={offset} size={size} total={d.total} onPage={setOffset} />
    </div>
  );
}

/** A queue's heading, status and search for the read-only sections below. */
function Shell({ summary, status, onStatus, onSearch, children }: {
  summary: FeedbackSummary; status: FeedbackStatus; onStatus: (s: FeedbackStatus) => void;
  onSearch: (q: string) => void; children: React.ReactNode;
}) {
  const [text, setText] = useState("");
  return (
    <section className="min-w-0 space-y-4" aria-label={summary.title}>
      <div>
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{summary.about}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <StatusToggle status={status} onChange={onStatus} />
        <form className="flex min-w-0 gap-1" onSubmit={(e) => { e.preventDefault(); onSearch(text.trim()); }}>
          <input type="search" className={`${FIELD} w-44`} value={text} onChange={(e) => setText(e.target.value)}
                 placeholder="Record or name" aria-label="Search records or names" />
        </form>
      </div>
      {children}
    </section>
  );
}

/** Photo views for a read-only viewer: its tabs, and each photo's suggested and confirmed views. */
export function ReadOnlyPhotoViewsSection({ summary }: { summary: FeedbackSummary }) {
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [q, setQ] = useState("");
  const [tab, setTab] = useState("all");
  const tabs = useQuery({ queryKey: ["feedback", "read-only", "photo-views", "tabs", status],
                          queryFn: () => feedbackApi.photoViews({ tab: "all", status, limit: 1 }) });
  return (
    <Shell summary={summary} status={status} onStatus={setStatus} onSearch={setQ}>
      <label className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">View</span>
        <select className={FIELD} value={tab} onChange={(e) => setTab(e.target.value)}>
          {(tabs.data?.tabs ?? [{ code: "all", title: "All", open: 0, decided: 0 }]).map((t) => (
            <option key={t.code} value={t.code}>{t.title}</option>
          ))}
        </select>
      </label>
      <ReadOnlyPhotoViews key={`${tab}|${status}|${q}`} tab={tab} status={status} q={q} />
    </Shell>
  );
}

/** The label review for a read-only viewer: records and photos; an open card stays blind. */
export function ReadOnlyLabelReviewSection({ summary }: { summary: FeedbackSummary }) {
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [q, setQ] = useState("");
  return (
    <Shell summary={summary} status={status} onStatus={setStatus} onSearch={setQ}>
      <ReadOnlyLabelReview key={`${status}|${q}`} status={status} q={q} />
    </Shell>
  );
}
