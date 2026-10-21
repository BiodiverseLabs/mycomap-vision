import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Keyboard } from "lucide-react";
import { BODY, BTN, CARD, ChoiceButtons, Ext, FIELD, INK, KeyHelp, PhotoLightbox, PhotoTile, QuickReasons, sciLike,
         StatusToggle, useDebounced } from "@/components/ReviewPhotos";
import { ApiError, feedbackApi, type FeedbackStatus, type FeedbackSummary, type PhotoReviewDecisionBody,
         type PhotoBandFilter, type PhotoReviewPage, type PhotoReviewSort, type PhotoReviewTab, type ReviewPhoto,
         type ReviewRecord } from "@/lib/api";
import { pageText, recordLabel } from "@/lib/feedback";
import { bulkExcludeFlagged, bulkKeepAll, chooseDecision, chosenCount, cycleReason, decisionsOf, initialChoices,
         KEY_HELP, keyAction, MIN_FLAGGED, missingReason, nextPageOffset, nextUnsaved, offsetWhenPageDone,
         openOnLabel, outlineOf, photoLabels, photoName, QUICK_REASONS, quickChoice, quickReasons, readPreselectMode,
         REVIEW_PAGE_SIZE, saveBody, savedLine,
         setReason, SORTS, BAND_FILTERS, bandCount, sortedPhotos, step, suggestedCount, withMode, writePreselectMode, type Choice,
         type KeyAction, type PhotoChoice, type PhotoChoices, type PickedReasons,
         type PreselectMode } from "@/lib/photoReview";
import { readTileSize, readView, SUMMARY_KEY_HELP, SUMMARY_TABS, TILE_SIZES, writeTileSize, writeView,
         type ReviewView, type TileSize } from "@/lib/photoSummary";
import { autosaveOnText, changedPhotos, pageSaves, readAutosave, writeAutosave } from "@/lib/reviewSave";
import { AutosaveToggle, SaveBar, useLeaveGuard } from "@/components/ReviewSave";
import { PhotoSummary } from "./PhotoSummary";

// The Human Feedback photo review (Steve, 2026-10-18): the "Photos that show no specimen" queue
// a record at a time. Every photo of the record is on its card, the flagged ones outlined with
// why; Keep / Exclude per photo, each left-out photo with its own reason (starting at its flag's
// suggested_reason), then save the record. Keys: see KEY_HELP.
//
// "Suggest Exclude for flagged photos" (a per-viewer setting, off by default) starts every open
// photo flagged in the tab as a suggested Exclude, drawn dashed and lighter so it reads as a
// suggestion, not a choice; Save sends suggestions too.
//
// Photos an admin sees privately (all rights reserved) come from Vision's own same-origin
// route and are shown by a plain <img>: never linked for download, prefetched or copied. The
// tiles, the viewer and the controls are shared with the label review (components/ReviewPhotos).
//
// Summary | Full (Steve, 2026-10-21): Summary (the default, remembered per viewer) shows the
// tab's flagged photos alone as a grid that saves as the reviewer goes (PhotoSummary); Full is
// the record-by-record view below. The tabs, sort and search apply to both; "N+ flagged" and
// "Suggest Exclude" to Full only, the tile size to Summary only. Full's records are not read
// while Summary is shown.
//
// Auto-save (Steve, 2026-10-21; lib/reviewSave.ts), at the top, off by default and remembered per
// viewer. On: both views save as they always have. Off: nothing is written until a Save: the
// changes are marked unsaved, "Save page (N changes)" writes them all, and paging, a tab, a filter,
// the view or the queue with changes unsaved asks first (then drops them).

export function PhotoReview({ summary }: { summary: FeedbackSummary }) {
  const qc = useQueryClient();
  const [tab, setTabNow] = useState<PhotoReviewTab>("all");
  const [status, setStatusNow] = useState<FeedbackStatus>("open");
  const [sort, setSortNow] = useState<PhotoReviewSort>("p");
  const [minFlagged, setMinFlaggedNow] = useState<number>(1);
  const [band, setBandNow] = useState<PhotoBandFilter>("all");
  const [text, setTextNow] = useState("");
  const q = useDebounced(text.trim());
  const [autosave, setAutosave] = useState(() => readAutosave(browserStorage));
  const [unsaved, setUnsaved] = useState(0);
  const canLeave = useLeaveGuard(unsaved, "photo-review");
  // Bumped when the reviewer chose to leave changes unsaved: the view shown drops them.
  const [discard, setDiscard] = useState(0);
  /** Runs `fn` (a tab, filter or view change) once the reviewer agrees to leave what is unsaved. */
  const leaving = <T,>(fn: (v: T) => void) => (v: T) => {
    if (!canLeave()) return;
    if (unsaved > 0) {
      setDiscard((d) => d + 1);
      setUnsaved(0);
    }
    fn(v);
  };
  const setTab = leaving(setTabNow);
  const setStatus = leaving(setStatusNow);
  const setSort = leaving(setSortNow);
  const setMinFlagged = leaving(setMinFlaggedNow);
  const setBand = leaving(setBandNow);
  const setText = leaving(setTextNow);
  const pickAutosave = (on: boolean) => {
    if (on && unsaved > 0 && view === "summary" && !window.confirm(autosaveOnText(unsaved))) return;
    setAutosave(on);
    writeAutosave(browserStorage, on);
  };
  const [offset, setOffset] = useState(0);
  // Bumped to load the page afresh (after a page is done), so saved records drop out.
  const [gen, setGen] = useState(0);
  const [help, setHelp] = useState(false);
  const [mode, setMode] = useState<PreselectMode>(() => readPreselectMode(browserStorage));
  const [view, setView] = useState<ReviewView>(() => readView(browserStorage));
  const full = view === "full";
  const pickView = leaving((v: ReviewView) => {
    setView(v);
    writeView(browserStorage, v);
  });
  const [tileSize, setTileSize] = useState<TileSize>(() => readTileSize(browserStorage));
  const pickTileSize = (t: TileSize) => {
    setTileSize(t);
    writeTileSize(browserStorage, t);
  };
  const [summaryTotal, setSummaryTotal] = useState<number | null>(null);
  // The reasons this reviewer picked, by photo: kept across tabs and pages (Board remounts).
  const [picked, setPicked] = useState<PickedReasons>({});
  const pickMode = (on: boolean) => {
    const next: PreselectMode = on ? "suggest" : "none";
    setMode(next);
    writePreselectMode(browserStorage, next);
  };
  useEffect(() => { setOffset(0); }, [tab, status, sort, minFlagged, q, band]);
  const list = useQuery({
    queryKey: ["feedback", "photo-review", tab, status, sort, minFlagged, q, band, offset, gen],
    queryFn: () => feedbackApi.photoReview({ tab, status, sort, min_flagged: minFlagged, q, band, offset,
                                             limit: REVIEW_PAGE_SIZE }),
    placeholderData: (prev) => prev,
    // A reload under the reviewer would move the cards they are working through.
    refetchOnWindowFocus: false,
    enabled: full,
  });
  const data = list.data;
  const total = data?.total ?? 0;
  const reload = (nextOffset: number) => {
    setOffset(nextOffset);
    setGen((g) => g + 1);
    qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
  };
  return (
    <section className="min-w-0 space-y-4" aria-label={summary.title}>
      <div>
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{summary.about}</p>
      </div>
      <Tabs data={full ? data : undefined} photos={full ? null : summaryTotal} tab={tab} status={status}
            onPick={setTab} />
      <div className="flex flex-wrap items-center gap-3">
        <AutosaveToggle on={autosave} onChange={pickAutosave}
                        title={autosave
                          ? "On: Summary saves each photo once you have seen it and each flip at once. Click to save only when you press Save."
                          : "Off: nothing is saved until you press Save (a photo's, a record's, or Save page). Click to save as you go."} />
        <ViewToggle view={view} onChange={pickView} />
        <StatusToggle status={status} onChange={setStatus} />
        <label className="flex min-w-0 items-center gap-2 text-sm">
          <span className="text-muted-foreground">Sort</span>
          <select className={FIELD} value={sort} onChange={(e) => setSort(e.target.value as PhotoReviewSort)}>
            {SORTS.map((s) => <option key={s.code} value={s.code}>{s.label}</option>)}
          </select>
        </label>
        <label className="flex min-w-0 items-center gap-2 text-sm"
               title="Tagger v2's confidence: High is checked at 85% or better by eye, Medium 50-85%, Low below. Low photos start as Include.">
          <span className="text-muted-foreground">Confidence</span>
          <select className={FIELD} value={band} onChange={(e) => setBand(e.target.value as PhotoBandFilter)}
                  data-testid="photo-review-band">
            {BAND_FILTERS.map((b) => (
              <option key={b.code} value={b.code}>{b.label}{bandCount(data, tab, b.code)}</option>
            ))}
          </select>
        </label>
        {full && <label className="flex min-w-0 items-center gap-2 text-sm">
          <span className="text-muted-foreground">Show</span>
          <select className={FIELD} value={minFlagged} onChange={(e) => setMinFlagged(Number(e.target.value))}>
            {MIN_FLAGGED.map((n) => <option key={n} value={n}>{n}+ flagged photo{n === 1 ? "" : "s"}</option>)}
          </select>
        </label>}
        <input type="search" value={text} onChange={(e) => setText(e.target.value)}
               placeholder="Record or name" aria-label="Search records or names"
               className={`${FIELD} w-full min-w-[12rem] basis-full sm:basis-auto sm:flex-1 sm:max-w-xs`} data-testid="photo-review-search" />
        {full && <label className="flex min-h-10 min-w-0 items-center gap-2 text-sm"
               title="Open flagged photos in this tab start as Exclude with their suggested reason, shown dashed until you confirm or keep them. Save sends them too.">
          <input type="checkbox" className="h-4 w-4 accent-myco-green" checked={mode === "suggest"}
                 onChange={(e) => pickMode(e.target.checked)} data-testid="photo-review-suggest" />
          <span className={BODY}>Suggest Exclude for flagged photos</span>
        </label>}
        {!full && <SizeToggle size={tileSize} onChange={pickTileSize} />}
        <button type="button" className={BTN} aria-pressed={help} onClick={() => setHelp((h) => !h)}>
          <Keyboard className="h-4 w-4" aria-hidden /> Keys
        </button>
        {full && list.isFetching && <span className="text-xs text-muted-foreground">Loading…</span>}
      </div>
      {help && <KeyHelp rows={full ? KEY_HELP : SUMMARY_KEY_HELP} />}
      {!full && (
        <PhotoSummary tab={tab} status={status} sort={sort} q={q} band={band} help={help} onHelp={() => setHelp((h) => !h)}
                      tileSize={tileSize} onTotal={setSummaryTotal} autosave={autosave} onPending={setUnsaved}
                      canLeave={canLeave} discard={discard} />
      )}
      {full && list.error && (
        <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The photo review could not be loaded."}</p>
      )}
      {full && data && list.isPlaceholderData && <p className="text-sm text-muted-foreground">Loading the next records…</p>}
      {full && data && !list.isPlaceholderData && (
        <Board key={`${tab}|${status}|${sort}|${minFlagged}|${q}|${band}|${offset}|${gen}`} data={data} tab={tab}
               status={status} searching={!!q} offset={offset} help={help} onHelp={() => setHelp((h) => !h)}
               mode={mode} picked={picked} onPick={(photoKey, r) => setPicked((x) => ({ ...x, [photoKey]: r }))}
               onPageDone={() => reload(offsetWhenPageDone(status, offset, REVIEW_PAGE_SIZE, total))}
               onPage={leaving(reload)} autosave={autosave} onPending={setUnsaved} discard={discard} />
      )}
    </section>
  );
}

/** The viewer's localStorage; reading `window.localStorage` itself can throw (blocked site data). */
const browserStorage = () => (typeof window === "undefined" ? null : window.localStorage);

/** The id of a photo's reason select (`lb`: the one in the photo viewer), so X can focus it. */
const reasonSelectId = (recordKey: string, photoKey: string, lb = false) =>
  `${lb ? "lb-" : ""}reason-${recordKey}-${photoKey}`.replace(/[^A-Za-z0-9_-]/g, "_");

/** Summary | Full, left of Open / Decided. */
function ViewToggle({ view, onChange }: { view: ReviewView; onChange: (v: ReviewView) => void }) {
  const seg = (on: boolean) => `min-h-10 px-3 py-1.5 text-sm font-medium ${
    on ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`;
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
         role="group" aria-label="View" data-testid="photo-review-view">
      <button type="button" className={seg(view === "summary")} aria-pressed={view === "summary"}
              title="The flagged photos as a grid, saved as you go" onClick={() => onChange("summary")}>Summary</button>
      <button type="button" className={`${seg(view === "full")} border-l border-[#A87146]/30 dark:border-border`}
              aria-pressed={view === "full"} title="Record by record, every photo"
              onClick={() => onChange("full")}>Full</button>
    </div>
  );
}

/** The Summary's tile size: S / M / L. */
function SizeToggle({ size, onChange }: { size: TileSize; onChange: (s: TileSize) => void }) {
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
         role="group" aria-label="Tile size">
      {TILE_SIZES.map((t, i) => (
        <button key={t.code} type="button" aria-pressed={size === t.code} title={`${t.label} tiles`}
                aria-label={`${t.label} tiles`} onClick={() => onChange(t.code)}
                className={`min-h-10 min-w-10 px-2 py-1.5 text-sm font-medium ${i ? "border-l border-[#A87146]/30 dark:border-border" : ""} ${
                  size === t.code ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`}>
          {t.code.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

/** The tabs. Full: from its records' response, with record counts. Summary: by name, and the
 *  tab shown with its count of photos (`photos`). */
function Tabs({ data, photos, tab, status, onPick }: {
  data?: PhotoReviewPage; photos: number | null; tab: PhotoReviewTab; status: FeedbackStatus;
  onPick: (t: PhotoReviewTab) => void;
}) {
  const shownCount = (code: PhotoReviewTab) => (photos != null && code === tab ? photos : null);
  const tabs: { code: PhotoReviewTab; title: string; n: number | null }[] = [
    // A record can be in several tabs, so All's count is only known when it is the one shown.
    { code: "all", title: "All", n: tab === "all" && data ? data.total : shownCount("all") },
    ...(data?.tabs
      ? data.tabs.map((t) => ({ code: t.code, title: t.title, n: status === "open" ? t.open : t.decided }))
      : SUMMARY_TABS.map((t) => ({ code: t.code, title: t.title, n: shownCount(t.code) }))),
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

type Focus = { rec: number; photo: number };

/** One page of cards, with the choices made on it. Mounted afresh for each page load. */
function Board({ data, tab, status, searching, offset, help, onHelp, onPageDone, onPage, mode, picked, onPick, autosave,
                 onPending, discard }: {
  data: PhotoReviewPage; tab: PhotoReviewTab; status: FeedbackStatus; searching: boolean; offset: number; help: boolean;
  onHelp: () => void; onPageDone: () => void; onPage: (offset: number) => void;
  mode: PreselectMode; picked: PickedReasons; onPick: (photoKey: string, reason: string) => void;
  /** Off: the changed photos are marked unsaved, with "Save page" and a guard on leaving. */
  autosave: boolean; onPending: (n: number) => void; discard: number;
}) {
  const qc = useQueryClient();
  const records = data.records;
  const keys = useMemo(() => records.map((r) => r.record_key), [records]);
  const photosOf = useMemo(() => records.map(sortedPhotos), [records]);
  const reasonCodes = useMemo(() => data.reasons.map((r) => r.code), [data.reasons]);
  const [choices, setChoices] = useState<Record<string, PhotoChoices>>(
    () => Object.fromEntries(records.map((r) => [r.record_key, initialChoices(r, tab, mode, picked)])));
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState<Record<string, string>>({});
  const savedRef = useRef<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [focus, setFocus] = useState<Focus>({ rec: 0, photo: 0 });
  const [lightbox, setLightbox] = useState<Focus | null>(null);
  const cardRefs = useRef<(HTMLElement | null)[]>([]);
  const moveTo = useRef<number | null>(null);
  // A reason select to focus once it is drawn (X on a photo with no reason to start at).
  const focusSelect = useRef<string | null>(null);
  const pickedRef = useRef(picked);
  pickedRef.current = picked;

  // Switching the mode adds or drops the suggestions; the reviewer's own choices stay.
  const modeRef = useRef(mode);
  useEffect(() => {
    if (modeRef.current === mode) return;
    modeRef.current = mode;
    setChoices((c) => Object.fromEntries(records.map((r) => [
      r.record_key, withMode(r, tab, mode, c[r.record_key] ?? {}, pickedRef.current)])));
  }, [mode, records, tab]);

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

  useEffect(() => {
    if (!focusSelect.current) return;
    const el = document.getElementById(focusSelect.current);
    focusSelect.current = null;
    el?.focus();
  }, [choices]);

  const goTo = (rec: number, photo = 0) => {
    moveTo.current = rec;
    setFocus({ rec, photo });
  };

  const choicesOf = (key: string): PhotoChoices => choices[key] ?? {};
  const setFor = (key: string, next: PhotoChoices) => {
    setChoices((c) => ({ ...c, [key]: next }));
    setErrors((e) => ({ ...e, [key]: "" }));
  };
  const editable = (key: string | undefined): key is string => !!key && !saved[key] && !pending[key];

  const finish = (key: string, index: number, text: string) => {
    savedRef.current = { ...savedRef.current, [key]: text };
    setSaved(savedRef.current);
    const next = nextUnsaved(keys, new Set(Object.keys(savedRef.current)), index);
    if (next < 0) onPageDone();
    else goTo(next);
  };

  const decide = useMutation({
    mutationFn: (v: { key: string; index: number; body: PhotoReviewDecisionBody; line: string }) =>
      feedbackApi.photoReviewDecide(v.key, v.body),
    onSuccess: (_res, v) => {
      setPending((p) => ({ ...p, [v.key]: false }));
      qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
      finish(v.key, v.index, v.line);
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
    if (!rec || !editable(rec.record_key)) return;
    const mine = choicesOf(rec.record_key);
    const b = saveBody(rec, mine, notes[rec.record_key], reasonCodes);
    if ("error" in b) {
      setErrors((x) => ({ ...x, [rec.record_key]: b.error }));
      return;
    }
    setPending((p) => ({ ...p, [rec.record_key]: true }));
    decide.mutate({ key: rec.record_key, index, body: b.body, line: savedLine(b.body, mine) });
  };

  // Auto-save off: what Save page writes (each changed record through its own route), and Discard.
  const plan = useMemo(() => pageSaves(records, choices, { saved, busy: pending, notes, reasonCodes }),
                       [records, choices, saved, pending, notes, reasonCodes]);
  const unsaved = autosave ? 0 : plan.changes;
  useEffect(() => { onPending(unsaved); }, [unsaved, onPending]);
  useEffect(() => () => onPending(0), [onPending]);
  const savePage = () => {
    if (plan.blocked.length) setErrors((x) => ({ ...x, ...Object.fromEntries(plan.blocked.map((b) => [b.key, b.error])) }));
    for (const s of plan.saves) {
      setPending((p) => ({ ...p, [s.key]: true }));
      decide.mutate({ key: s.key, index: s.index, body: s.body, line: savedLine(s.body, choicesOf(s.key)) });
    }
  };
  // Discard drops every unsaved change, the suggested Excludes too (they come back on the next page).
  const discardPage = () => {
    setChoices((c) => ({ ...c, ...Object.fromEntries(records.filter((r) => !saved[r.record_key] && !pending[r.record_key])
      .map((r) => [r.record_key, initialChoices(r, tab, "none")])) }));
    setNotes({});
    setErrors({});
  };
  const discardSeen = useRef(discard);
  useEffect(() => {
    if (discardSeen.current === discard) return;
    discardSeen.current = discard;
    discardPage();
  });

  /** Keep or Exclude on one photo. `byKey`: from the keyboard, so an exclusion with no reason
   *  to start at puts focus on its reason select. */
  const choose = (index: number, photo: ReviewPhoto, choice: Choice, byKey = false) => {
    const key = keys[index];
    if (!editable(key)) return;
    const next = chooseDecision(choicesOf(key), photo, choice, picked);
    const c = next[photo.photo_key];
    if (byKey && c?.decision === "exclude" && !c.reason) {
      focusSelect.current = reasonSelectId(key, photo.photo_key, !!lightbox);
    }
    setFor(key, next);
  };

  const pickReason = (index: number, photoKey: string, reason: string) => {
    const key = keys[index];
    if (!editable(key) || !reason) return;
    setFor(key, setReason(choicesOf(key), photoKey, reason));
    onPick(photoKey, reason);
  };

  /** A quick reason button (or key 1–5) on a photo: left out for that reason. */
  const quick = (index: number, photoKey: string, reason: string) => {
    const key = keys[index];
    if (!editable(key) || !reason) return;
    setFor(key, quickChoice(choicesOf(key), photoKey, reason));
    onPick(photoKey, reason);
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
        if (ph) choose(at.rec, ph, a.choice, true);
        return;
      }
      case "reason": {
        const ph = photos[at.photo];
        if (!ph || !editable(rec.record_key)) return;
        const r = cycleReason(choicesOf(rec.record_key), ph.photo_key, reasonCodes)[ph.photo_key]?.reason;
        if (r) pickReason(at.rec, ph.photo_key, r);
        return;
      }
      case "quick": {
        const ph = photos[at.photo];
        const code = QUICK_REASONS[a.index]?.code;
        if (ph && lightbox && code && quickReasons(data.reasons).some((q) => q.code === code)) quick(at.rec, ph.photo_key, code);
        return;
      }
      case "keepAll":
        if (editable(rec.record_key)) setFor(rec.record_key, bulkKeepAll(rec, choicesOf(rec.record_key)));
        return;
      case "excludeFlagged":
        if (editable(rec.record_key)) {
          setFor(rec.record_key, bulkExcludeFlagged(rec, tab, choicesOf(rec.record_key), picked));
        }
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
  const busy = Object.values(pending).some(Boolean);
  const bar = (sticky: boolean) => (
    <SaveBar count={unsaved} busy={busy && unsaved === 0} onSave={savePage} onDiscard={discardPage} sticky={sticky}
             note={sticky ? "Each record also has its own Save." : undefined} />
  );
  return (
    <>
      {!autosave && records.length > 0 && bar(true)}
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
                        choices={choicesOf(key)} note={notes[key] ?? ""}
                        error={errors[key] || null} saving={!!pending[key]}
                        focused={focus.rec === i} focusPhoto={focus.rec === i ? focus.photo : -1}
                        cardRef={(el) => { cardRefs.current[i] = el; }}
                        onFocusCard={() => setFocus((f) => (f.rec === i ? f : { rec: i, photo: 0 }))}
                        onFocusPhoto={(p) => setFocus({ rec: i, photo: p })}
                        onOpen={(p) => { setFocus({ rec: i, photo: p }); setLightbox({ rec: i, photo: p }); }}
                        onChoose={(ph, c) => choose(i, ph, c)}
                        onReason={(photoKey, r) => pickReason(i, photoKey, r)}
                        onKeepAll={() => setFor(key, bulkKeepAll(rec, choicesOf(key)))}
                        onExcludeFlagged={() => setFor(key, bulkExcludeFlagged(rec, tab, choicesOf(key), picked))}
                        onNote={(n) => setNotes((x) => ({ ...x, [key]: n }))}
                        onSave={() => save(i)} manual={!autosave} />
          );
        })}
      </ul>
      {!autosave && records.length > 0 && bar(false)}
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
                  reasons={data.reasons} choices={choicesOf(lbRecord.record_key)}
                  disabled={!editable(lbRecord.record_key)}
                  onStep={(d) => run({ type: "photo", step: d })}
                  onChoose={(c) => run({ type: "choose", choice: c })}
                  onReason={(photoKey, r) => pickReason(lightbox.rec, photoKey, r)}
                  onQuick={(photoKey, r) => quick(lightbox.rec, photoKey, r)}
                  onClose={() => run({ type: "close" })} />
      )}
    </>
  );
}

function RecordCard(p: {
  record: ReviewRecord; photos: ReviewPhoto[]; tab: PhotoReviewTab; reasons: { code: string; label: string }[];
  choices: PhotoChoices; note: string; error: string | null; saving: boolean;
  focused: boolean; focusPhoto: number; cardRef: (el: HTMLElement | null) => void;
  onFocusCard: () => void; onFocusPhoto: (i: number) => void; onOpen: (i: number) => void;
  onChoose: (photo: ReviewPhoto, c: Choice) => void; onReason: (photoKey: string, r: string) => void;
  onKeepAll: () => void; onExcludeFlagged: () => void; onNote: (n: string) => void; onSave: () => void;
  /** Auto-save off: each changed photo is marked unsaved. */
  manual: boolean;
}) {
  const { record, photos, tab } = p;
  const chosen = chosenCount(record, p.choices);
  const suggested = suggestedCount(record, p.choices);
  const missing = missingReason(record, p.choices);
  const changed = useMemo(() => new Set(p.manual ? changedPhotos(record, p.choices).map((ph) => ph.photo_key) : []),
                          [p.manual, record, p.choices]);
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
          <Tile key={ph.photo_key} photo={ph} tab={tab} record={record} reasons={p.reasons} unsaved={changed.has(ph.photo_key)}
                choice={p.choices[ph.photo_key] ?? null} focused={p.focusPhoto === i} disabled={p.saving}
                onOpen={() => p.onOpen(i)} onFocus={() => p.onFocusPhoto(i)}
                onChoose={(c) => p.onChoose(ph, c)} onReason={(r) => p.onReason(ph.photo_key, r)} />
        ))}
      </ul>
      <div className="flex min-w-0 flex-wrap items-end gap-2">
        <button type="button" className={BTN} onClick={p.onKeepAll} disabled={p.saving}>Keep all</button>
        <button type="button" className={BTN} onClick={p.onExcludeFlagged} disabled={p.saving}>Exclude all flagged</button>
        <label className="flex min-w-0 flex-1 basis-40 flex-col text-xs">
          <span className="text-muted-foreground">Note (optional)</span>
          <input className={FIELD} value={p.note} onChange={(e) => p.onNote(e.target.value)} disabled={p.saving} />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={p.onSave} disabled={chosen === 0 || !!missing || p.saving}
                className="inline-flex min-h-10 items-center rounded-md bg-myco-green px-4 py-1.5 text-sm font-medium text-white hover:bg-myco-green/90 disabled:opacity-50">
          {p.saving ? "Saving…" : "Save this record"}
        </button>
        <span className="text-xs text-muted-foreground">
          {chosen} of {photos.length} photo{photos.length === 1 ? "" : "s"} chosen
          {suggested > 0 && ` · ${suggested} suggested (dashed): saved as shown unless you change them`}
        </span>
        {changed.size > 0 && (
          <span className="rounded bg-amber-500 px-1.5 py-0.5 text-[11px] font-semibold text-white">
            {changed.size} unsaved
          </span>
        )}
        {missing && (
          <span className="text-sm text-red-700 dark:text-red-300">
            Pick a reason for {photoName(record, missing.photo_key)} to save.
          </span>
        )}
        {p.error && <span className="text-sm text-red-700 dark:text-red-300" role="alert">{p.error}</span>}
      </div>
    </li>
  );
}

const OUTLINE = {
  suggested: "border-dashed border-red-500 dark:border-red-400",
  strong: "border-amber-500 dark:border-amber-400",
  light: "border-dashed border-amber-300 dark:border-amber-700",
  none: "border-transparent",
} as const;

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

/** A left-out photo's own reason. It starts at the photo's suggested reason (none for an
 *  unflagged photo: "Pick a reason"). Enter or Esc hands the keys back to the page. */
function ReasonSelect({ id, choice, reasons, label, disabled, onReason }: {
  id: string; choice: PhotoChoice; reasons: { code: string; label: string }[]; label: string; disabled: boolean;
  onReason: (r: string) => void;
}) {
  const known = !choice.reason || reasons.some((r) => r.code === choice.reason);
  return (
    <select id={id} value={choice.reason ?? ""} disabled={disabled} aria-label={label}
            onChange={(e) => onReason(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                e.currentTarget.blur();
              }
            }}
            className={`${FIELD} w-full text-xs text-[#4a3728] dark:text-foreground ${
              !choice.reason ? "border-red-500 dark:border-red-500"
                : choice.suggested ? "border-dashed border-red-400 dark:border-red-500" : ""}`}>
      {!choice.reason && <option value="" disabled>Pick a reason…</option>}
      {reasons.map((r) => <option key={r.code} value={r.code}>{r.label}</option>)}
      {!known && choice.reason && <option value={choice.reason}>{choice.reason}</option>}
    </select>
  );
}

function Tile({ photo, tab, record, reasons, choice, focused, disabled, unsaved, onOpen, onFocus, onChoose, onReason }: {
  photo: ReviewPhoto; tab: PhotoReviewTab; record: ReviewRecord; reasons: { code: string; label: string }[];
  choice: PhotoChoice | null; focused: boolean; disabled: boolean; unsaved: boolean; onOpen: () => void;
  onFocus: () => void; onChoose: (c: Choice) => void; onReason: (r: string) => void;
}) {
  const decided = decidedLine(photo);
  const suggested = !!choice?.suggested && choice.decision === "exclude";
  return (
    <PhotoTile photo={photo} record={record} choice={choice?.decision ?? null} focused={focused} suggested={suggested}
               outline={suggested ? OUTLINE.suggested : OUTLINE[outlineOf(photo)]} onOpen={onOpen} onFocus={onFocus}>
      {unsaved && (
        <span className="self-start rounded bg-amber-500 px-1.5 py-0.5 text-[11px] font-semibold text-white"
              data-testid="photo-review-unsaved">unsaved</span>
      )}
      <FlagLabels photo={photo} tab={tab} />
      {decided && <p className="break-words text-[11px] text-muted-foreground">{decided}</p>}
      <ChoiceButtons choice={choice?.decision ?? null} onChoose={onChoose} disabled={disabled} suggested={suggested} />
      {choice?.decision === "exclude" && (
        <ReasonSelect id={reasonSelectId(record.record_key, photo.photo_key)} choice={choice} reasons={reasons}
                      label={`Why ${photoName(record, photo.photo_key)} is left out`} disabled={disabled}
                      onReason={onReason} />
      )}
    </PhotoTile>
  );
}

function Lightbox({ record, photos, index, tab, reasons, choices, disabled, onStep, onChoose, onReason, onQuick, onClose }: {
  record: ReviewRecord; photos: ReviewPhoto[]; index: number; tab: PhotoReviewTab;
  reasons: { code: string; label: string }[]; choices: PhotoChoices; disabled: boolean;
  onStep: (d: 1 | -1) => void; onChoose: (c: Choice) => void; onReason: (photoKey: string, r: string) => void;
  onQuick: (photoKey: string, r: string) => void; onClose: () => void;
}) {
  const plain = useMemo(() => decisionsOf(choices), [choices]);
  return (
    <PhotoLightbox record={record} photos={photos} index={index} choices={plain} onStep={onStep}
                   onChoose={onChoose} onClose={onClose}
                   suggested={(photo) => !!choices[photo.photo_key]?.suggested}
                   hint="← / → photos · K keep · X exclude · R reason · 1–5 quick reason · Esc close"
                   below={(photo) => {
                     const c = choices[photo.photo_key];
                     return <QuickReasons reasons={reasons} disabled={disabled}
                                          current={c?.decision === "exclude" && !c.suggested ? c.reason : null}
                                          onPick={(r) => onQuick(photo.photo_key, r)} />;
                   }}
                   extra={(photo) => {
                     const c = choices[photo.photo_key];
                     return (
                       <>
                         <FlagLabels photo={photo} tab={tab} dark />
                         {c?.decision === "exclude" && (
                           <div className="w-full max-w-xs">
                             <ReasonSelect id={reasonSelectId(record.record_key, photo.photo_key, true)} choice={c}
                                           reasons={reasons} disabled={disabled}
                                           label={`Why ${photoName(record, photo.photo_key)} is left out`}
                                           onReason={(r) => onReason(photo.photo_key, r)} />
                           </div>
                         )}
                       </>
                     );
                   }} />
  );
}
