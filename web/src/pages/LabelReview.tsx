import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Keyboard } from "lucide-react";
import { BODY, BTN, CARD, ChoiceButtons, Ext, FIELD, INK, KeyHelp, PhotoLightbox, PhotoTile, StatusToggle,
         useDebounced } from "@/components/ReviewPhotos";
import { RevealAnswer } from "@/components/RevealAnswer";
import { ApiError, feedbackApi, type FeedbackStatus, type FeedbackSummary, type LabelReviewCard,
         type LabelReviewDecisionBody, type LabelReviewOutcome, type LabelReviewPage,
         type LabelReviewPhoto } from "@/lib/api";
import { pageText, recordLabel, whenText } from "@/lib/feedback";
import { decidedSummary, DEFAULT_REASON, LABEL_KEY_HELP, LABEL_PAGE_SIZE, LABEL_REVIEW_ABOUT, labelKeyAction,
         maskedCard, OUTCOME_BUTTONS, outcomeHint, outcomeStates, saveBody, savedLine, sortedPhotos,
         visionSaid, type LabelKeyAction } from "@/lib/labelReview";
import { nextPageOffset, nextUnsaved, offsetWhenPageDone, openOnLabel, step, toggleChoice, type Choice,
         type Choices } from "@/lib/photoReview";

// The Human Feedback label review (Steve, 2026-10-18): the BLIND "Photos unlike the label"
// queue a record at a time. Each card shows the record, its label and every photo; Keep /
// Exclude per photo, then one outcome for the record. Keys: see LABEL_KEY_HELP.
//
// Blind: every card goes through maskedCard before it is drawn, so an open card shows nothing
// about Vision or why the record is here, and nothing that differs between cards. The order is
// the server's. The one control that names Vision is "Show Vision's answer" under the photos: the
// same button on every open card, which asks before showing this person Vision's top 5 (and the
// decision then stays out of the blind check). A decided card shows the stored outcome and, under
// "What Vision said", the rule and evidence. Tiles, the viewer and the controls are the photo
// review's (ReviewPhotos).

// The three outcomes look alike: none is the "expected" answer.
const OUTCOME_BTN = `${BTN} w-full sm:w-auto`;
const LEFT_OUT_BADGE: Record<Choice, string> = { keep: "Kept", exclude: "Left out" };

export function LabelReview({ summary }: { summary: FeedbackSummary }) {
  const qc = useQueryClient();
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [text, setText] = useState("");
  const q = useDebounced(text.trim());
  const [offset, setOffset] = useState(0);
  // Bumped to load the page afresh (after a page is done), so saved records drop out.
  const [gen, setGen] = useState(0);
  const [help, setHelp] = useState(false);
  useEffect(() => { setOffset(0); }, [status, q]);
  const list = useQuery({
    queryKey: ["feedback", "label-review", status, q, offset, gen],
    queryFn: () => feedbackApi.labelReview({ status, q, offset, limit: LABEL_PAGE_SIZE }),
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
  return (
    <section className="min-w-0 space-y-4" aria-label={summary.title}>
      <div>
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{LABEL_REVIEW_ABOUT}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <StatusToggle status={status} onChange={setStatus} />
        <input type="search" value={text} onChange={(e) => setText(e.target.value)}
               placeholder="Record or name" aria-label="Search records or names"
               className={`${FIELD} w-full min-w-[12rem] basis-full sm:basis-auto sm:flex-1 sm:max-w-xs`}
               data-testid="label-review-search" />
        <button type="button" className={BTN} aria-pressed={help} onClick={() => setHelp((h) => !h)}>
          <Keyboard className="h-4 w-4" aria-hidden /> Keys
        </button>
        {list.isFetching && <span className="text-xs text-muted-foreground">Loading…</span>}
      </div>
      {help && <KeyHelp rows={LABEL_KEY_HELP} />}
      {list.error && (
        <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The label review could not be loaded."}</p>
      )}
      {data && list.isPlaceholderData && <p className="text-sm text-muted-foreground">Loading the next records…</p>}
      {data && !list.isPlaceholderData && (
        <Board key={`${status}|${q}|${offset}|${gen}`} data={data} status={status} searching={!!q} offset={offset}
               help={help} onHelp={() => setHelp((h) => !h)}
               onPageDone={() => reload(offsetWhenPageDone(status, offset, LABEL_PAGE_SIZE, total))}
               onPage={reload} />
      )}
    </section>
  );
}

type Focus = { rec: number; photo: number };

/** One page of cards, with the choices made on it. Mounted afresh for each page load. */
function Board({ data, status, searching, offset, help, onHelp, onPageDone, onPage }: {
  data: LabelReviewPage; status: FeedbackStatus; searching: boolean; offset: number; help: boolean;
  onHelp: () => void; onPageDone: () => void; onPage: (offset: number) => void;
}) {
  const qc = useQueryClient();
  // Every card is drawn from what the blind rule lets it show, never from the reply itself.
  const cards = useMemo(() => data.records.map(maskedCard), [data.records]);
  const keys = useMemo(() => cards.map((c) => String(c.item_id)), [cards]);
  const photosOf = useMemo(() => cards.map((c) => sortedPhotos(c.photos)), [cards]);
  const reasonCodes = useMemo(() => data.reasons.map((r) => r.code), [data.reasons]);
  const startReason = reasonCodes.includes(DEFAULT_REASON) ? DEFAULT_REASON : (reasonCodes[0] ?? DEFAULT_REASON);
  const decidedView = status === "decided";
  const [choices, setChoices] = useState<Record<string, Choices>>({});
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [names, setNames] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState<Record<string, string>>({});
  const savedRef = useRef<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [focus, setFocus] = useState<Focus>({ rec: 0, photo: 0 });
  const [lightbox, setLightbox] = useState<Focus | null>(null);
  const cardRefs = useRef<(HTMLElement | null)[]>([]);
  const renameRefs = useRef<(HTMLInputElement | null)[]>([]);
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

  const inputOf = (key: string) => ({ reason: reasons[key] ?? startReason, value: names[key] ?? "", note: notes[key] ?? "" });
  const choicesOf = (key: string): Choices => {
    if (!decidedView) return choices[key] ?? {};
    // A decided card shows its left-out photos; nothing else is chosen.
    const card = cards[keys.indexOf(key)];
    return Object.fromEntries((card?.photos ?? []).filter((p) => p.excluded).map((p) => [p.photo_key, "exclude"]));
  };
  const clearError = (key: string) => setErrors((e) => ({ ...e, [key]: "" }));

  const finish = (key: string, index: number, line: string) => {
    savedRef.current = { ...savedRef.current, [key]: line };
    setSaved(savedRef.current);
    const next = nextUnsaved(keys, new Set(Object.keys(savedRef.current)), index);
    if (next < 0) onPageDone();
    else goTo(next);
  };

  const decide = useMutation({
    mutationFn: (v: { key: string; index: number; itemId: number; body: LabelReviewDecisionBody }) =>
      feedbackApi.labelReviewDecide(v.itemId, v.body),
    onSuccess: (_res, v) => {
      setPending((p) => ({ ...p, [v.key]: false }));
      qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
      finish(v.key, v.index, savedLine(v.body, data.reasons));
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

  const save = (index: number, outcome: LabelReviewOutcome) => {
    const card = cards[index];
    const key = keys[index];
    if (decidedView || !card || saved[key] || pending[key]) return;
    const b = saveBody(card, choicesOf(key), outcome, inputOf(key), reasonCodes);
    if ("error" in b) {
      setErrors((x) => ({ ...x, [key]: b.error }));
      return;
    }
    setPending((p) => ({ ...p, [key]: true }));
    decide.mutate({ key, index, itemId: card.item_id, body: b.body });
  };

  const choose = (index: number, photoKey: string, choice: Choice) => {
    const key = keys[index];
    if (decidedView || !key || saved[key] || pending[key]) return;
    setChoices((c) => ({ ...c, [key]: toggleChoice(c[key] ?? {}, photoKey, choice) }));
    clearError(key);
  };

  const run = (a: LabelKeyAction) => {
    const at = lightbox ?? focus;
    const card = cards[at.rec];
    if (!card) return;
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
      case "outcome":
        // Only when the outcome is allowed; otherwise the line under the buttons says why.
        if (outcomeStates(card, choicesOf(keys[at.rec]), inputOf(keys[at.rec]), reasonCodes)[a.outcome].enabled) {
          save(at.rec, a.outcome);
        }
        return;
      case "rename":
        renameRefs.current[at.rec]?.focus();
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
      const a = labelKeyAction(e, { typing, ...stateRef.current });
      if (!a) return;
      e.preventDefault();
      runRef.current(a);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const savedHere = keys.filter((k) => saved[k]).length;
  const total = data.total;
  const lbCard = lightbox ? cards[lightbox.rec] : null;
  return (
    <>
      {!cards.length && (
        <p className="text-sm text-muted-foreground">
          {searching ? "No record or name matches that search."
            : status === "open" ? "Nothing open here." : "Nothing decided here yet."}
        </p>
      )}
      <ul className="space-y-4">
        {cards.map((card, i) => {
          const key = keys[i];
          const common = {
            card, photos: photosOf[i], focused: focus.rec === i, focusPhoto: focus.rec === i ? focus.photo : -1,
            cardRef: (el: HTMLElement | null) => { cardRefs.current[i] = el; },
            onFocusCard: () => setFocus((f) => (f.rec === i ? f : { rec: i, photo: 0 })),
            onFocusPhoto: (p: number) => setFocus({ rec: i, photo: p }),
            onOpen: (p: number) => { setFocus({ rec: i, photo: p }); setLightbox({ rec: i, photo: p }); },
          };
          if (saved[key]) {
            return (
              <li key={key} ref={common.cardRef} tabIndex={-1}
                  className={`${CARD} flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 p-3 text-sm outline-none ${
                    focus.rec === i ? "ring-2 ring-myco-green" : ""}`}>
                <Check className="h-4 w-4 shrink-0 text-myco-green" aria-hidden />
                <span className={`font-medium ${INK}`}>{recordLabel(card.record_key)}</span>
                <span className={`min-w-0 break-words ${BODY}`}>{saved[key]}</span>
              </li>
            );
          }
          if (decidedView) {
            return <DecidedCard key={key} {...common} reasons={data.reasons} choices={choicesOf(key)} />;
          }
          const input = inputOf(key);
          return (
            <OpenCard key={key} {...common} reasons={data.reasons} reasonCodes={reasonCodes}
                      choices={choicesOf(key)} reason={input.reason} name={input.value} note={input.note}
                      error={errors[key] || null} saving={!!pending[key]}
                      renameRef={(el) => { renameRefs.current[i] = el; }}
                      onChoose={(photoKey, c) => choose(i, photoKey, c)}
                      onReason={(r) => { setReasons((x) => ({ ...x, [key]: r })); clearError(key); }}
                      onName={(n) => { setNames((x) => ({ ...x, [key]: n })); clearError(key); }}
                      onNote={(n) => setNotes((x) => ({ ...x, [key]: n }))}
                      onSave={(o) => save(i, o)} />
          );
        })}
      </ul>
      {total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground">{pageText(offset, cards.length, total)}</span>
          <span className="flex gap-2">
            <button type="button" className={BTN} disabled={offset === 0}
                    onClick={() => onPage(Math.max(0, offset - LABEL_PAGE_SIZE))}>Previous</button>
            <button type="button" className={BTN} disabled={offset + LABEL_PAGE_SIZE >= total}
                    onClick={() => onPage(nextPageOffset(status, offset, LABEL_PAGE_SIZE, savedHere))}>Next</button>
          </span>
        </div>
      )}
      {lightbox && lbCard && (
        <PhotoLightbox record={lbCard} photos={photosOf[lightbox.rec]} index={lightbox.photo}
                       choices={choicesOf(keys[lightbox.rec])}
                       onStep={(d) => run({ type: "photo", step: d })}
                       onChoose={decidedView ? undefined : (c) => run({ type: "choose", choice: c })}
                       onClose={() => run({ type: "close" })} />
      )}
    </>
  );
}

type CardProps = {
  card: LabelReviewCard; photos: LabelReviewPhoto[]; focused: boolean; focusPhoto: number;
  cardRef: (el: HTMLElement | null) => void; onFocusCard: () => void; onFocusPhoto: (i: number) => void;
  onOpen: (i: number) => void;
};

/** The record, its link and its label: the same on every card. */
function CardHead({ card }: { card: LabelReviewCard }) {
  return (
    <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
      <h3 className={`font-semibold ${INK}`}>{recordLabel(card.record_key)}</h3>
      {card.label && <span className={`sci min-w-0 break-words ${BODY}`}>{card.label}</span>}
      <span className="text-sm"><Ext href={card.link}>{openOnLabel(card.record_key)}</Ext></span>
    </div>
  );
}

function cardShell(p: CardProps) {
  return {
    ref: p.cardRef, tabIndex: -1, onFocus: p.onFocusCard, onMouseDown: p.onFocusCard,
    className: `${CARD} min-w-0 scroll-mt-24 space-y-3 p-3 outline-none ${
      p.focused ? "ring-2 ring-myco-green border-myco-green/50" : ""}`,
    "aria-label": `${recordLabel(p.card.record_key)} ${p.card.label}`,
    "data-testid": `label-review-${p.card.record_key}`,
  };
}

function OpenCard(p: CardProps & {
  reasons: { code: string; label: string }[]; reasonCodes: string[]; choices: Choices; reason: string; name: string;
  note: string; error: string | null; saving: boolean; renameRef: (el: HTMLInputElement | null) => void;
  onChoose: (photoKey: string, c: Choice) => void; onReason: (r: string) => void; onName: (n: string) => void;
  onNote: (n: string) => void; onSave: (o: LabelReviewOutcome) => void;
}) {
  const { card, photos } = p;
  const states = outcomeStates(card, p.choices, { reason: p.reason, value: p.name }, p.reasonCodes);
  const hint = outcomeHint(states);
  const off = (o: LabelReviewOutcome) => !states[o].enabled || p.saving;
  return (
    <li {...cardShell(p)}>
      <CardHead card={card} />
      <ul className="grid grid-cols-2 gap-3 sm:flex sm:flex-wrap">
        {photos.map((ph, i) => (
          <PhotoTile key={ph.photo_key} photo={ph} record={card} choice={p.choices[ph.photo_key] ?? null}
                     focused={p.focusPhoto === i} className="sm:w-56"
                     onOpen={() => p.onOpen(i)} onFocus={() => p.onFocusPhoto(i)}>
            <ChoiceButtons choice={p.choices[ph.photo_key] ?? null} disabled={p.saving}
                           onChoose={(c) => p.onChoose(ph.photo_key, c)} />
          </PhotoTile>
        ))}
      </ul>
      {!photos.length && <p className="text-sm text-muted-foreground">This record has no photos to show here.</p>}
      <RevealAnswer itemId={card.item_id} revealed={card.revealed} />
      <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-end">
        <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-end">
          <label className="flex min-w-0 flex-col text-xs">
            <span className="text-muted-foreground">Why they're left out</span>
            <select className={FIELD} value={p.reason} onChange={(e) => p.onReason(e.target.value)} disabled={p.saving}>
              {p.reasons.map((r) => <option key={r.code} value={r.code}>{r.label}</option>)}
            </select>
          </label>
          <button type="button" className={OUTCOME_BTN} disabled={off("exclude_photos")}
                  onClick={() => p.onSave("exclude_photos")}>
            {OUTCOME_BUTTONS.exclude_photos}
          </button>
        </div>
        <button type="button" className={OUTCOME_BTN} disabled={off("hard_example")}
                onClick={() => p.onSave("hard_example")}>
          {OUTCOME_BUTTONS.hard_example}
        </button>
        <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-end">
          <button type="button" className={OUTCOME_BTN} disabled={off("rename")} onClick={() => p.onSave("rename")}>
            {OUTCOME_BUTTONS.rename}
          </button>
          <input ref={p.renameRef} className={`${FIELD} sci w-full sm:w-56`} value={p.name} disabled={p.saving}
                 placeholder="Genus species" aria-label={`New name for ${recordLabel(card.record_key)}`}
                 onChange={(e) => p.onName(e.target.value)}
                 onKeyDown={(e) => { if (e.key === "Enter" && states.rename.enabled) { e.preventDefault(); p.onSave("rename"); } }} />
        </div>
      </div>
      <label className="flex min-w-0 flex-col text-xs sm:max-w-xl">
        <span className="text-muted-foreground">Note (optional)</span>
        <input className={FIELD} value={p.note} onChange={(e) => p.onNote(e.target.value)} disabled={p.saving} />
      </label>
      <div className="flex min-w-0 flex-wrap items-center gap-3 text-xs">
        {p.saving ? <span className="text-muted-foreground">Saving…</span>
          : hint && <span className="text-muted-foreground">{hint}</span>}
        {p.error && <span className="text-sm text-red-700 dark:text-red-300" role="alert">{p.error}</span>}
      </div>
    </li>
  );
}

function DecidedCard(p: CardProps & { reasons: { code: string; label: string }[]; choices: Choices }) {
  const { card, photos } = p;
  const done = decidedSummary(card, p.reasons);
  const said = visionSaid(card);
  return (
    <li {...cardShell(p)}>
      <CardHead card={card} />
      <ul className="grid grid-cols-2 gap-3 sm:flex sm:flex-wrap">
        {photos.map((ph, i) => (
          <PhotoTile key={ph.photo_key} photo={ph} record={card} choice={p.choices[ph.photo_key] ?? null}
                     focused={p.focusPhoto === i} className="sm:w-56" badge={LEFT_OUT_BADGE}
                     onOpen={() => p.onOpen(i)} onFocus={() => p.onFocusPhoto(i)} />
        ))}
      </ul>
      {done ? (
        <div className={`space-y-1 text-sm ${BODY}`}>
          <p>
            <span className={`font-medium ${INK}`}>Decided: </span>{done.words}
            {done.by && <> by {done.by}</>}
            {done.at != null && done.at !== "" && <span className="text-muted-foreground"> · {whenText(done.at)}</span>}
          </p>
          {done.note && <p className="break-words">Note: {done.note}</p>}
        </div>
      ) : <p className="text-sm text-muted-foreground">No decision on file.</p>}
      {said && (said.rule || said.rows.length > 0) && (
        <div className="space-y-1 rounded-md bg-[#faf9f7] p-3 text-sm dark:bg-muted/40">
          <h4 className={`font-semibold ${INK}`}>What Vision said</h4>
          {said.rule && <p className={`break-words ${BODY}`}>{said.rule}</p>}
          {said.rows.length > 0 && (
            <dl className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-[auto_minmax(0,1fr)]">
              {said.rows.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className={`min-w-0 break-words ${BODY}`}>{v}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      )}
    </li>
  );
}
