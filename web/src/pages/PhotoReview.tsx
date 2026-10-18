import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, Check, ChevronLeft, ChevronRight, Keyboard, Lock, X } from "lucide-react";
import { ApiError, feedbackApi, type FeedbackStatus, type FeedbackSummary, type PhotoReviewDecisionBody,
         type PhotoReviewPage, type PhotoReviewSort, type PhotoReviewTab, type ReviewPhoto,
         type ReviewRecord } from "@/lib/api";
import { pageText, recordLabel } from "@/lib/feedback";
import { bulkExcludeFlagged, bulkKeepAll, chosenCount, defaultReason, flagLabel, initialChoices, KEY_HELP,
         keyAction, MIN_FLAGGED, nextPageOffset, nextUnsaved, offsetWhenPageDone, openOnLabel, outlineOf,
         photoLabels, placeholderText, REVIEW_PAGE_SIZE, saveBody, savedSummary, SORTS, sortedPhotos, step, tileImage,
         toggleChoice, type Choice, type Choices, type KeyAction } from "@/lib/photoReview";

// The Human Feedback photo review (Steve, 2026-10-18): the "Photos that show no specimen" queue
// a record at a time. Every photo of the record is on its card, the flagged ones outlined with
// why; Keep / Exclude per photo, then save the record. Keys: see KEY_HELP.
//
// Photos an admin sees privately (all rights reserved) come from Vision's own same-origin
// route and are shown by a plain <img>: never linked for download, prefetched or copied.

const CARD = "rounded-lg border border-[#A87146]/20 bg-white shadow-sm dark:border-border dark:bg-card";
const INK = "text-[#4a3728] dark:text-foreground";
const BODY = "text-[#5c4a3a] dark:text-foreground/90";
const LINK = "text-myco-green underline-offset-2 hover:underline";
const FIELD = "min-h-10 min-w-0 rounded-md border border-[#A87146]/30 bg-white px-2 py-1.5 text-sm "
  + "dark:border-border dark:bg-background focus:outline-none focus:ring-2 focus:ring-myco-green/40";
const BTN = "inline-flex min-h-10 items-center justify-center gap-1 rounded-md border border-[#A87146]/30 bg-white "
  + "px-3 py-1.5 text-sm font-medium hover:bg-myco-green/5 disabled:pointer-events-none disabled:opacity-40 "
  + "dark:border-border dark:bg-card";
const NEW_TAB = { target: "_blank", rel: "noopener" } as const;

const sciLike = (s: string | null | undefined) => !!s && /^[A-Z][a-z]+(\s|$)/.test(s.trim());

function Ext({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} {...NEW_TAB} className={`inline-flex items-center gap-0.5 ${LINK}`}
       onClick={(e) => e.stopPropagation()}>
      {children} <ArrowUpRight className="h-3 w-3 shrink-0" aria-hidden />
    </a>
  );
}

function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function PhotoReview({ summary }: { summary: FeedbackSummary }) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<PhotoReviewTab>("all");
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [sort, setSort] = useState<PhotoReviewSort>("p");
  const [minFlagged, setMinFlagged] = useState<number>(1);
  const [text, setText] = useState("");
  const q = useDebounced(text.trim());
  const [offset, setOffset] = useState(0);
  // Bumped to load the page afresh (after a page is done), so saved records drop out.
  const [gen, setGen] = useState(0);
  const [help, setHelp] = useState(false);
  useEffect(() => { setOffset(0); }, [tab, status, sort, minFlagged, q]);
  const list = useQuery({
    queryKey: ["feedback", "photo-review", tab, status, sort, minFlagged, q, offset, gen],
    queryFn: () => feedbackApi.photoReview({ tab, status, sort, min_flagged: minFlagged, q, offset,
                                             limit: REVIEW_PAGE_SIZE }),
    placeholderData: (prev) => prev,
    // A reload under the reviewer would move the cards they are working through.
    refetchOnWindowFocus: false,
  });
  const data = list.data;
  const total = data?.total ?? 0;
  const reload = (nextOffset: number) => {
    setOffset(nextOffset);
    setGen((g) => g + 1);
    qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
  };
  const seg = (on: boolean) => `min-h-10 px-3 py-1.5 text-sm font-medium ${
    on ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`;
  return (
    <section className="min-w-0 space-y-4" aria-label={summary.title}>
      <div>
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{summary.about}</p>
      </div>
      <Tabs data={data} tab={tab} status={status} onPick={setTab} />
      <div className="flex flex-wrap items-center gap-3">
        <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
             role="group" aria-label="Status">
          <button type="button" className={seg(status === "open")} aria-pressed={status === "open"}
                  onClick={() => setStatus("open")}>Open</button>
          <button type="button" className={`${seg(status === "decided")} border-l border-[#A87146]/30 dark:border-border`}
                  aria-pressed={status === "decided"} onClick={() => setStatus("decided")}>Decided</button>
        </div>
        <label className="flex min-w-0 items-center gap-2 text-sm">
          <span className="text-muted-foreground">Sort</span>
          <select className={FIELD} value={sort} onChange={(e) => setSort(e.target.value as PhotoReviewSort)}>
            {SORTS.map((s) => <option key={s.code} value={s.code}>{s.label}</option>)}
          </select>
        </label>
        <label className="flex min-w-0 items-center gap-2 text-sm">
          <span className="text-muted-foreground">Show</span>
          <select className={FIELD} value={minFlagged} onChange={(e) => setMinFlagged(Number(e.target.value))}>
            {MIN_FLAGGED.map((n) => <option key={n} value={n}>{n}+ flagged photo{n === 1 ? "" : "s"}</option>)}
          </select>
        </label>
        <input type="search" value={text} onChange={(e) => setText(e.target.value)}
               placeholder="Record or name" aria-label="Search records or names"
               className={`${FIELD} w-full flex-1 sm:max-w-xs`} data-testid="photo-review-search" />
        <button type="button" className={BTN} aria-pressed={help} onClick={() => setHelp((h) => !h)}>
          <Keyboard className="h-4 w-4" aria-hidden /> Keys
        </button>
        {list.isFetching && <span className="text-xs text-muted-foreground">Loading…</span>}
      </div>
      {help && <KeyHelp />}
      {list.error && (
        <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The photo review could not be loaded."}</p>
      )}
      {data && list.isPlaceholderData && <p className="text-sm text-muted-foreground">Loading the next records…</p>}
      {data && !list.isPlaceholderData && (
        <Board key={`${tab}|${status}|${sort}|${minFlagged}|${q}|${offset}|${gen}`} data={data} tab={tab}
               status={status} searching={!!q} offset={offset} help={help} onHelp={() => setHelp((h) => !h)}
               onPageDone={() => reload(offsetWhenPageDone(status, offset, REVIEW_PAGE_SIZE, total))}
               onPage={reload} />
      )}
    </section>
  );
}

function Tabs({ data, tab, status, onPick }: {
  data?: PhotoReviewPage; tab: PhotoReviewTab; status: FeedbackStatus; onPick: (t: PhotoReviewTab) => void;
}) {
  const tabs: { code: PhotoReviewTab; title: string; n: number | null }[] = [
    // A record can be in several tabs, so All's count is only known when it is the one shown.
    { code: "all", title: "All", n: tab === "all" && data ? data.total : null },
    ...(data?.tabs ?? []).map((t) => ({ code: t.code, title: t.title, n: status === "open" ? t.open : t.decided })),
  ];
  return (
    <div className="flex min-w-0 gap-1 overflow-x-auto border-b border-[#A87146]/20 pb-1 dark:border-border"
         role="tablist" aria-label="Kind of flag">
      {tabs.map((t) => {
        const on = t.code === tab;
        return (
          <button key={t.code} type="button" role="tab" aria-selected={on} onClick={() => onPick(t.code)}
                  className={`min-h-10 shrink-0 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium ${
                    on ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`}>
            {t.title}
            {t.n != null && <span className="ml-1 tabular-nums text-muted-foreground">({t.n.toLocaleString("en-US")})</span>}
          </button>
        );
      })}
    </div>
  );
}

function KeyHelp() {
  return (
    <dl className={`${CARD} grid max-w-md grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 p-3 text-sm`}
        aria-label="Keyboard keys">
      {KEY_HELP.map(([k, what]) => (
        <div key={k} className="contents">
          <dt><kbd className="rounded border border-[#A87146]/30 bg-[#faf9f7] px-1.5 py-0.5 font-mono text-xs dark:border-border dark:bg-muted">{k}</kbd></dt>
          <dd className={BODY}>{what}</dd>
        </div>
      ))}
    </dl>
  );
}

type Focus = { rec: number; photo: number };

/** One page of cards, with the choices made on it. Mounted afresh for each page load. */
function Board({ data, tab, status, searching, offset, help, onHelp, onPageDone, onPage }: {
  data: PhotoReviewPage; tab: PhotoReviewTab; status: FeedbackStatus; searching: boolean; offset: number; help: boolean;
  onHelp: () => void; onPageDone: () => void; onPage: (offset: number) => void;
}) {
  const qc = useQueryClient();
  const records = data.records;
  const keys = useMemo(() => records.map((r) => r.record_key), [records]);
  const photosOf = useMemo(() => records.map(sortedPhotos), [records]);
  const reasonCodes = useMemo(() => data.reasons.map((r) => r.code), [data.reasons]);
  const [choices, setChoices] = useState<Record<string, Choices>>(
    () => Object.fromEntries(records.map((r) => [r.record_key, initialChoices(r)])));
  const startReason = reasonCodes.includes(defaultReason(tab)) ? defaultReason(tab) : (reasonCodes[0] ?? "other");
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState<Record<string, string>>({});
  const savedRef = useRef<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [focus, setFocus] = useState<Focus>({ rec: 0, photo: 0 });
  const [lightbox, setLightbox] = useState<Focus | null>(null);
  const cardRefs = useRef<(HTMLElement | null)[]>([]);
  const moveTo = useRef<number | null>(null);

  // When keys or a save move focus to another card, bring it into view and focus it.
  useEffect(() => {
    if (moveTo.current == null) return;
    const el = cardRefs.current[moveTo.current];
    moveTo.current = null;
    if (el) {
      el.focus({ preventScroll: true });
      el.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  }, [focus.rec, saved]);

  const goTo = (rec: number, photo = 0) => {
    moveTo.current = rec;
    setFocus({ rec, photo });
  };

  const reasonOf = (key: string) => reasons[key] ?? startReason;
  const choicesOf = (key: string) => choices[key] ?? {};
  const setFor = (key: string, next: Choices) => {
    setChoices((c) => ({ ...c, [key]: next }));
    setErrors((e) => ({ ...e, [key]: "" }));
  };

  const finish = (key: string, index: number, text: string) => {
    savedRef.current = { ...savedRef.current, [key]: text };
    setSaved(savedRef.current);
    const next = nextUnsaved(keys, new Set(Object.keys(savedRef.current)), index);
    if (next < 0) onPageDone();
    else goTo(next);
  };

  const decide = useMutation({
    mutationFn: (v: { key: string; index: number; body: PhotoReviewDecisionBody }) =>
      feedbackApi.photoReviewDecide(v.key, v.body),
    onSuccess: (_res, v) => {
      setPending((p) => ({ ...p, [v.key]: false }));
      qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
      finish(v.key, v.index, savedSummary(v.body));
    },
    onError: (e, v) => {
      setPending((p) => ({ ...p, [v.key]: false }));
      if (e instanceof ApiError && e.status === 404) {
        finish(v.key, v.index, "No longer in the review (decided or removed elsewhere)");
        return;
      }
      setErrors((x) => ({ ...x, [v.key]: e instanceof Error ? e.message : "The record could not be saved." }));
    },
  });

  const save = (index: number) => {
    const rec = records[index];
    if (!rec || saved[rec.record_key] || pending[rec.record_key]) return;
    const b = saveBody(rec, choicesOf(rec.record_key), reasonOf(rec.record_key), notes[rec.record_key], reasonCodes);
    if ("error" in b) {
      setErrors((x) => ({ ...x, [rec.record_key]: b.error }));
      return;
    }
    setPending((p) => ({ ...p, [rec.record_key]: true }));
    decide.mutate({ key: rec.record_key, index, body: b.body });
  };

  const choose = (index: number, photoKey: string, choice: Choice) => {
    const key = keys[index];
    if (!key || saved[key] || pending[key]) return;
    setFor(key, toggleChoice(choicesOf(key), photoKey, choice));
  };

  const run = (a: KeyAction) => {
    const at = lightbox ?? focus;
    const rec = records[at.rec];
    if (!rec) return;
    const photos = photosOf[at.rec];
    switch (a.type) {
      case "record": {
        let i = focus.rec + a.step;
        while (i >= 0 && i < keys.length && saved[keys[i]]) i += a.step;
        if (i >= 0 && i < keys.length) goTo(i);
        return;
      }
      case "photo": {
        const p = step(at.photo, photos.length, a.step);
        if (lightbox) setLightbox({ rec: at.rec, photo: p });
        setFocus({ rec: at.rec, photo: p });
        return;
      }
      case "choose": {
        const ph = photos[at.photo];
        if (ph) choose(at.rec, ph.photo_key, a.choice);
        return;
      }
      case "keepAll":
        if (!saved[rec.record_key]) setFor(rec.record_key, bulkKeepAll(rec));
        return;
      case "excludeFlagged":
        if (!saved[rec.record_key]) setFor(rec.record_key, bulkExcludeFlagged(rec, tab, choicesOf(rec.record_key)));
        return;
      case "save":
        save(at.rec);
        return;
      case "help":
        onHelp();
        return;
      case "close":
        setLightbox(null);
        cardRefs.current[at.rec]?.focus({ preventScroll: true });
        return;
    }
  };

  // The keys, read on the window; the latest handler is kept in a ref.
  const runRef = useRef(run);
  runRef.current = run;
  const stateRef = useRef({ lightbox: !!lightbox, help });
  stateRef.current = { lightbox: !!lightbox, help };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target instanceof HTMLElement ? e.target : null;
      const typing = !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      const onControl = !!t && /^(BUTTON|A)$/.test(t.tagName);
      const a = keyAction(e, { typing, onControl, ...stateRef.current });
      if (!a) return;
      e.preventDefault();
      runRef.current(a);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const savedHere = keys.filter((k) => saved[k]).length;
  const total = data.total;
  const lbRecord = lightbox ? records[lightbox.rec] : null;
  return (
    <>
      {!records.length && (
        <p className="text-sm text-muted-foreground">
          {searching ? "No record or name matches that search."
            : status === "open" ? "Nothing open here." : "Nothing decided here yet."}
        </p>
      )}
      <ul className="space-y-4">
        {records.map((rec, i) => {
          const key = rec.record_key;
          if (saved[key]) {
            return (
              <li key={key} ref={(el) => { cardRefs.current[i] = el; }} tabIndex={-1}
                  className={`${CARD} flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 p-3 text-sm outline-none ${
                    focus.rec === i ? "ring-2 ring-myco-green" : ""}`}>
                <Check className="h-4 w-4 shrink-0 text-myco-green" aria-hidden />
                <span className={`font-medium ${INK}`}>{recordLabel(key)}</span>
                <span className={BODY}>{saved[key]}</span>
              </li>
            );
          }
          return (
            <RecordCard key={key} record={rec} photos={photosOf[i]} tab={tab} reasons={data.reasons}
                        choices={choicesOf(key)} reason={reasonOf(key)} note={notes[key] ?? ""}
                        error={errors[key] || null} saving={!!pending[key]}
                        focused={focus.rec === i} focusPhoto={focus.rec === i ? focus.photo : -1}
                        cardRef={(el) => { cardRefs.current[i] = el; }}
                        onFocusCard={() => setFocus((f) => (f.rec === i ? f : { rec: i, photo: 0 }))}
                        onFocusPhoto={(p) => setFocus({ rec: i, photo: p })}
                        onOpen={(p) => { setFocus({ rec: i, photo: p }); setLightbox({ rec: i, photo: p }); }}
                        onChoose={(photoKey, c) => choose(i, photoKey, c)}
                        onKeepAll={() => setFor(key, bulkKeepAll(rec))}
                        onExcludeFlagged={() => setFor(key, bulkExcludeFlagged(rec, tab, choicesOf(key)))}
                        onReason={(r) => setReasons((x) => ({ ...x, [key]: r }))}
                        onNote={(n) => setNotes((x) => ({ ...x, [key]: n }))}
                        onSave={() => save(i)} />
          );
        })}
      </ul>
      {total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground">{pageText(offset, records.length, total)}</span>
          <span className="flex gap-2">
            <button type="button" className={BTN} disabled={offset === 0}
                    onClick={() => onPage(Math.max(0, offset - REVIEW_PAGE_SIZE))}>Previous</button>
            <button type="button" className={BTN} disabled={offset + REVIEW_PAGE_SIZE >= total}
                    onClick={() => onPage(nextPageOffset(status, offset, REVIEW_PAGE_SIZE, savedHere))}>Next</button>
          </span>
        </div>
      )}
      {lightbox && lbRecord && (
        <Lightbox record={lbRecord} photos={photosOf[lightbox.rec]} index={lightbox.photo} tab={tab}
                  choices={choicesOf(lbRecord.record_key)}
                  onStep={(d) => run({ type: "photo", step: d })}
                  onChoose={(c) => run({ type: "choose", choice: c })}
                  onClose={() => run({ type: "close" })} />
      )}
    </>
  );
}

function RecordCard(p: {
  record: ReviewRecord; photos: ReviewPhoto[]; tab: PhotoReviewTab; reasons: { code: string; label: string }[];
  choices: Choices; reason: string; note: string; error: string | null; saving: boolean;
  focused: boolean; focusPhoto: number; cardRef: (el: HTMLElement | null) => void;
  onFocusCard: () => void; onFocusPhoto: (i: number) => void; onOpen: (i: number) => void;
  onChoose: (photoKey: string, c: Choice) => void; onKeepAll: () => void; onExcludeFlagged: () => void;
  onReason: (r: string) => void; onNote: (n: string) => void; onSave: () => void;
}) {
  const { record, photos, tab } = p;
  const chosen = chosenCount(record, p.choices);
  const anyExcluded = photos.some((ph) => p.choices[ph.photo_key] === "exclude");
  return (
    <li ref={p.cardRef} tabIndex={-1} onFocus={p.onFocusCard} onMouseDown={p.onFocusCard}
        className={`${CARD} min-w-0 scroll-mt-24 space-y-3 p-3 outline-none ${
          p.focused ? "ring-2 ring-myco-green border-myco-green/50" : ""}`}
        aria-label={`${recordLabel(record.record_key)} ${record.label}`}
        data-testid={`photo-review-${record.record_key}`}>
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className={`font-semibold ${INK}`}>{recordLabel(record.record_key)}</h3>
        {record.label && <span className={`break-words ${BODY} ${sciLike(record.label) ? "sci" : ""}`}>{record.label}</span>}
        <span className="text-sm"><Ext href={record.link}>{openOnLabel(record.record_key)}</Ext></span>
        <span className="text-xs text-muted-foreground">
          {record.flagged_in_tab} flagged here · strongest {record.strongest.toFixed(2)}
        </span>
      </div>
      <ul className="grid grid-cols-2 gap-3 sm:flex sm:flex-wrap">
        {photos.map((ph, i) => (
          <Tile key={ph.photo_key} photo={ph} tab={tab} record={record} choice={p.choices[ph.photo_key] ?? null}
                focused={p.focusPhoto === i} disabled={p.saving}
                onOpen={() => p.onOpen(i)} onFocus={() => p.onFocusPhoto(i)}
                onChoose={(c) => p.onChoose(ph.photo_key, c)} />
        ))}
      </ul>
      <div className="flex min-w-0 flex-wrap items-end gap-2">
        <button type="button" className={BTN} onClick={p.onKeepAll} disabled={p.saving}>Keep all</button>
        <button type="button" className={BTN} onClick={p.onExcludeFlagged} disabled={p.saving}>Exclude all flagged</button>
        <label className="flex min-w-0 flex-col text-xs">
          <span className="text-muted-foreground">Reason for excluded photos</span>
          <select className={`${FIELD} ${anyExcluded ? "border-red-300 dark:border-red-800" : ""}`} value={p.reason}
                  onChange={(e) => p.onReason(e.target.value)} disabled={p.saving}>
            {p.reasons.map((r) => <option key={r.code} value={r.code}>{r.label}</option>)}
          </select>
        </label>
        <label className="flex min-w-0 flex-1 basis-40 flex-col text-xs">
          <span className="text-muted-foreground">Note (optional)</span>
          <input className={FIELD} value={p.note} onChange={(e) => p.onNote(e.target.value)} disabled={p.saving} />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={p.onSave} disabled={chosen === 0 || p.saving}
                className="inline-flex min-h-10 items-center rounded-md bg-myco-green px-4 py-1.5 text-sm font-medium text-white hover:bg-myco-green/90 disabled:opacity-50">
          {p.saving ? "Saving…" : "Save this record"}
        </button>
        <span className="text-xs text-muted-foreground">
          {chosen} of {photos.length} photo{photos.length === 1 ? "" : "s"} chosen
        </span>
        {p.error && <span className="text-sm text-red-700 dark:text-red-300" role="alert">{p.error}</span>}
      </div>
    </li>
  );
}

const OUTLINE = {
  strong: "border-amber-500 dark:border-amber-400",
  light: "border-dashed border-amber-300 dark:border-amber-700",
  none: "border-transparent",
} as const;

function Placeholder({ photo, record, failed = false, large = false }: {
  photo: ReviewPhoto; record: ReviewRecord; failed?: boolean; large?: boolean;
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

function ChoiceButtons({ choice, onChoose, disabled = false, big = false }: {
  choice: Choice | null; onChoose: (c: Choice) => void; disabled?: boolean; big?: boolean;
}) {
  const base = `inline-flex ${big ? "min-h-11 px-5" : "min-h-10 px-2"} items-center justify-center gap-1 rounded-md border text-sm font-medium disabled:opacity-40`;
  const off = "border-[#A87146]/30 bg-white text-gray-700 hover:bg-myco-green/5 dark:border-border dark:bg-card dark:text-foreground/80";
  return (
    <div className={big ? "flex gap-2" : "grid grid-cols-2 gap-1"}>
      <button type="button" aria-pressed={choice === "keep"} disabled={disabled} onClick={() => onChoose("keep")}
              className={`${base} ${choice === "keep" ? "border-myco-green bg-myco-green text-white" : off}`}>
        <Check className="h-4 w-4" aria-hidden /> Keep
      </button>
      <button type="button" aria-pressed={choice === "exclude"} disabled={disabled} onClick={() => onChoose("exclude")}
              className={`${base} ${choice === "exclude" ? "border-red-600 bg-red-600 text-white" : off}`}>
        <X className="h-4 w-4" aria-hidden /> Exclude
      </button>
    </div>
  );
}

function FlagLabels({ photo, tab, dark = false }: { photo: ReviewPhoto; tab: PhotoReviewTab; dark?: boolean }) {
  const { strong, light } = photoLabels(photo, tab);
  if (!strong.length && !light.length) return null;
  return (
    <div className="flex flex-wrap gap-1 text-[11px] leading-tight">
      {strong.map((l) => (
        <span key={`s${l}`} className="break-words rounded bg-amber-100 px-1.5 py-0.5 font-medium text-amber-900 dark:bg-amber-950 dark:text-amber-200">{l}</span>
      ))}
      {light.map((l) => (
        <span key={`l${l}`} className={`break-words rounded border border-dashed px-1.5 py-0.5 ${
          dark ? "border-white/40 text-white/80" : "border-amber-300 text-muted-foreground dark:border-amber-700"}`}>{l}</span>
      ))}
    </div>
  );
}

function decidedLine(photo: ReviewPhoto): string | null {
  if (photo.status !== "decided" || !photo.decision) return null;
  const what = photo.decision.decision === "exclude_photo" ? "left out" : photo.decision.decision === "keep_photo" ? "kept" : photo.decision.decision;
  return `Decided: ${what}${photo.decision.reason ? ` (${photo.decision.reason})` : ""}${photo.decision.by ? ` by ${photo.decision.by}` : ""}`;
}

function Tile({ photo, tab, record, choice, focused, disabled, onOpen, onFocus, onChoose }: {
  photo: ReviewPhoto; tab: PhotoReviewTab; record: ReviewRecord; choice: Choice | null; focused: boolean;
  disabled: boolean; onOpen: () => void; onFocus: () => void; onChoose: (c: Choice) => void;
}) {
  // A thumb that fails to load (e.g. a private copy Vision does not hold) becomes the placeholder.
  const [failed, setFailed] = useState(false);
  const src = failed ? null : tileImage(photo, "thumb");
  const decided = decidedLine(photo);
  return (
    <li className="min-w-0 space-y-1 sm:w-44" onMouseDown={onFocus}>
      <div className={`relative aspect-square w-full overflow-hidden rounded-md border-4 ${OUTLINE[outlineOf(photo)]} ${
        focused ? "ring-2 ring-myco-green ring-offset-2 dark:ring-offset-background" : ""}`}>
        {src ? (
          <button type="button" onClick={onOpen} className="block h-full w-full"
                  aria-label={`Open photo ${photo.position ?? ""} large`}>
            <img src={src} alt={`Photo ${photo.position ?? ""} of ${recordLabel(record.record_key)}`}
                 loading="lazy" decoding="async" referrerPolicy="no-referrer" draggable={false}
                 onError={() => setFailed(true)}
                 className={`h-full w-full object-cover ${choice === "exclude" ? "opacity-50" : ""}`} />
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
            admin only · not public
          </span>
        )}
        {choice && (
          <span className={`pointer-events-none absolute bottom-1 right-1 rounded px-1.5 py-0.5 text-[11px] font-semibold text-white ${
            choice === "keep" ? "bg-myco-green" : "bg-red-600"}`}>
            {choice === "keep" ? "Keep" : "Exclude"}
          </span>
        )}
      </div>
      <FlagLabels photo={photo} tab={tab} />
      {decided && <p className="break-words text-[11px] text-muted-foreground">{decided}</p>}
      <ChoiceButtons choice={choice} onChoose={onChoose} disabled={disabled} />
    </li>
  );
}

function Lightbox({ record, photos, index, tab, choices, onStep, onChoose, onClose }: {
  record: ReviewRecord; photos: ReviewPhoto[]; index: number; tab: PhotoReviewTab; choices: Choices;
  onStep: (d: 1 | -1) => void; onChoose: (c: Choice) => void; onClose: () => void;
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
          {photo.shown === "private" && <span className="ml-2 rounded bg-white/15 px-1.5 py-0.5 text-[11px]">admin only · not public</span>}
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
        <FlagLabels photo={photo} tab={tab} dark />
        <ChoiceButtons choice={choice} onChoose={onChoose} big />
        <p className="text-center text-xs text-white/70">← / → photos · K keep · X exclude · Esc close</p>
      </div>
    </div>
  );
}
