import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, ChevronDown, Download, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/Layout";
import { RevealAnswer } from "@/components/RevealAnswer";
import { ReadOnlyContext, ReadOnlyNote, useReadOnly } from "@/components/ReadOnlyReview";
import { ApiError, feedbackApi, feedbackChangesUrl, signInUrl, type FeedbackItem, type FeedbackItemDetail,
         type FeedbackMe, type FeedbackOption, type FeedbackStatus, type FeedbackSummary, type FeedbackType,
         type LiveRecord, type NameContext, type OrgChip, type OrgPreview, type OrgRequest, type RefreshedRecord,
         type Top5, type WritingList } from "@/lib/api";
import { FEEDBACK_TABS, openTab, QUEUE_BLURBS, REVIEW_STEPS, tabOf, tabOpen, tabParam, tabSections, tabsFor, type FeedbackTab } from "@/lib/feedback";
import { appliesWords, blastAnchor, decisionBody, decisionLabel, evidenceRows, feedbackAccess,
         historyOrgText, inatLinks, isNameItem, itemBadges, linkedItem, NO_PROPOSAL_TEXT, noProposal, openQueue,
         orgChipClass, PAGE_SIZE, pageText, prefillValue, roleText, alreadyDoneText, decideOptions,
         FORMAT_PANEL_TEXT, formatRow, hasLists, listCount, listReasonText, seededNote, WRITING_LISTS,
         queueParam, readLastQueue, reaskModel, recordLabel, refreshVerdict, shownRule, stepLook, stepText, storedVerdict, takesPhotoKeys,
         statCardLines, valueBox, whenText, writeLastQueue, yes, type SectionGroup,
         type Verdict } from "@/lib/feedback";
import { optionGroups, pairSides } from "@/lib/twinCompare";
import { PairCompare } from "@/components/PairCompare";
import { LABEL_REVIEW_TYPE } from "@/lib/labelReview";
import { PHOTO_REVIEW_TYPE } from "@/lib/photoReview";
import { confirmLeave, reviewUnsaved } from "@/lib/reviewSave";
import { LabelReview } from "./LabelReview";
import { PhotoReview } from "./PhotoReview";
import { PhotoViews } from "./PhotoViews";
import { RecoveredLabels } from "./RecoveredLabels";
import { RECOVERED_TYPE } from "@/lib/recoveredLabels";
import { PHOTO_VIEW_TYPE } from "@/lib/photoViews";

// Human Feedback (Steve, 2026-10-18): a review queue of places where Vision and the reference
// data disagree (Photos for any signed-in mycomap.org account, Taxonomy for Contributors, Lab
// Contributors, Project Leaders and admins; Steve, 2026-11-20), laid out like
// mycomap.org's Validator dashboards: stat cards above the list, in two sections (data cleanup,
// then data validation; groupBySection). Each item opens the records, their sequences and BLAST
// results in new tabs, and can be re-checked once the records are updated.

const CARD = "rounded-lg border border-[#A87146]/20 bg-white shadow-sm dark:border-border dark:bg-card";
const INK = "text-[#4a3728] dark:text-foreground";
const BODY = "text-[#5c4a3a] dark:text-foreground/90";
const LINK = "text-myco-green underline-offset-2 hover:underline";
const INPUT = "w-full min-w-0 rounded-md border border-[#A87146]/30 bg-white px-2 py-1.5 text-sm "
  + "dark:border-border dark:bg-background focus:outline-none focus:ring-2 focus:ring-myco-green/40";

const NEW_TAB = { target: "_blank", rel: "noopener" } as const;

function Ext({ href, children, className = "" }: { href: string; children: React.ReactNode; className?: string }) {
  return (
    <a href={href} {...NEW_TAB} className={`inline-flex items-center gap-0.5 ${LINK} ${className}`}>
      {children} <ArrowUpRight className="h-3 w-3 shrink-0" aria-hidden />
    </a>
  );
}

function AccessNote({ error }: { error: unknown }) {
  const why = feedbackAccess(error);
  if (!why) return null;
  if (why === "signin") {
    return (
      <p className={BODY}>
        <a href={signInUrl()} className="text-myco-green underline">Sign in</a> with your
        mycomap.org account to tag photos.
      </p>
    );
  }
  if (why === "forbidden") {
    return (
      <p className={BODY} data-testid="feedback-forbidden">
        Your account can't open this page.
      </p>
    );
  }
  return <p className={BODY}>Human Feedback could not be loaded.</p>;
}

export function FeedbackPage() {
  // Read-only viewers (Steve, 2026-10-22: anyone who is not a reviewer) get no control that writes.
  const me = useQuery({ queryKey: ["feedback", "me"], queryFn: feedbackApi.me, retry: false });
  return (
    <ReadOnlyContext.Provider value={!!me.data?.read_only}>
      <FeedbackBody />
    </ReadOnlyContext.Provider>
  );
}

function FeedbackBody() {
  const me = useQuery({ queryKey: ["feedback", "me"], queryFn: feedbackApi.me, retry: false });
  const ok = !!me.data;
  const summary = useQuery({ queryKey: ["feedback", "summary"], queryFn: feedbackApi.summary, enabled: ok });
  const catalogue = useQuery({ queryKey: ["feedback", "catalogue"], queryFn: feedbackApi.catalogue,
                               enabled: ok, staleTime: Infinity });
  // No queue opens by itself (Steve, 2026-10-21: the first load did too much): the one picked,
  // a ?queue= or ?item= link, or the one this viewer last opened; otherwise none (openQueue).
  const [picked, setPicked] = useState<string | null>(null);
  const [linkedId] = useState(() => linkedItem(location.search));
  const [param] = useState(() => queueParam(location.search));
  const [stored] = useState(() => readLastQueue(browserStorage));
  const linked = useQuery({ queryKey: ["feedback", "item", linkedId], enabled: ok && linkedId != null,
                            queryFn: () => feedbackApi.item(linkedId!), retry: false });
  // Photos | Taxonomy (Steve, 2026-11-20): the page opens on Photos unless a link says otherwise.
  const [pickedTab, setPickedTab] = useState<FeedbackTab | null>(null);
  const [tabLink] = useState(() => tabParam(location.search));
  const types = summary.data ?? [];
  // Taxonomy is for Contributors and up (Steve, 2026-11-20): nobody else gets its tab or a link to it.
  const tabs = tabsFor(me.data);
  const tab = openTab({ picked: pickedTab, param: tabLink, tabs,
                        linkQueue: (param && types.some((t) => t.type === param) ? param : null)
                                   ?? linked.data?.item_type ?? null });
  const inTab = types.filter((t) => tabOf(t.type) === tab);
  const sections = tabSections(types, tab);
  const selected = openQueue({ picked, param, stored, types: inTab.map((t) => t.type),
                               linkedPending: linkedId != null && linked.isPending,
                               linkedType: linked.data?.item_type ?? null });
  const sel = types.find((t) => t.type === selected);
  const selSection = sections.find((g) => g.types.some((t) => t.type === selected));
  const catalogueType = catalogue.data?.find((t) => t.type === selected);
  // The photo review with Auto-save off may hold unsaved changes: ask before closing it.
  const mayLeave = () => confirmLeave(reviewUnsaved.total(), (m) => window.confirm(m));
  const pick = (type: string) => {
    if (type !== selected && !mayLeave()) return;
    setPicked(type);
    writeLastQueue(browserStorage, type);
  };
  const pickTab = (next: FeedbackTab) => {
    if (next === tab || (selected && !mayLeave())) return;
    setPickedTab(next);
    // The address says which tab is open, so it can be shared or bookmarked.
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    url.searchParams.delete("queue");
    url.searchParams.delete("item");
    window.history.replaceState(null, "", url.toString());
  };
  const tabInfo = FEEDBACK_TABS.find((t) => t.code === tab)!;
  return (
    <>
      <PageHeader title="Human Feedback">
        Help check the records Vision learns from. Photos: say what each photo shows and which
        ones belong in Vision. Taxonomy: tidy names and check whether a record's label is right.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl space-y-6">
        {me.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
        <AccessNote error={me.error} />
        <ReadOnlyNote me={me.data} />
        {me.data && (
          <>
            {me.data.admin && <AdminDownloads />}
            <MainTabs tabs={tabs} tab={tab} onPick={pickTab} counts={summary.data ? types : null} />
            <div role="tabpanel" id={`feedback-tabpanel-${tab}`} aria-labelledby={`feedback-tab-${tab}`}
                 className="min-w-0 space-y-6" data-testid={`feedback-tabpanel-${tab}`}>
              <p className={`max-w-3xl text-sm ${BODY}`}>{tabInfo.intro}</p>
              {summary.isPending && <p className="text-sm text-muted-foreground">Loading the queues…</p>}
              {summary.error && <p className={BODY}>The queues could not be loaded.</p>}
              {summary.data && !inTab.length && (
                <p className="text-sm text-muted-foreground">Nothing to review here right now.</p>
              )}
              {sections.map((g) => (
                <SectionCards key={g.code} group={g} selected={selected} onPick={pick} heading={tab !== "photos"} />
              ))}
              {summary.data && inTab.length > 0 && !sel && !(linkedId != null && linked.isPending) && (
                <p className="text-sm text-muted-foreground" data-testid="feedback-pick-queue">
                  {tab === "photos" ? "Pick one of the reviews above." : "Pick a queue above."}
                </p>
              )}
            </div>
            {sel && (
              <div className="min-w-0 space-y-1 border-t border-[#A87146]/20 pt-6 dark:border-border">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                   data-testid="feedback-queue-section">
                  {tabInfo.label}{selSection && tab !== "photos" ? ` · ${selSection.title}` : ""}
                </p>
                {REVIEW_STEPS[sel.type] && <ReviewSteps type={sel.type} title={sel.title} />}
                {sel.type === PHOTO_REVIEW_TYPE
                  ? <PhotoReview key={sel.type} summary={sel} />
                  : sel.type === PHOTO_VIEW_TYPE
                  ? <PhotoViews key={sel.type} summary={sel} />
                  : sel.type === RECOVERED_TYPE
                  ? <RecoveredLabels key={sel.type} summary={sel} />
                  : sel.type === LABEL_REVIEW_TYPE
                    ? <LabelReview key={sel.type} summary={sel} />
                    : <TypeQueue key={sel.type} summary={sel} type={catalogueType} me={me.data}
                                 linked={linked.data?.item_type === sel.type ? stripDetail(linked.data) : undefined} />}
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

/** The viewer's localStorage; reading `window.localStorage` itself can throw (blocked site data). */
const browserStorage = () => (typeof window === "undefined" ? null : window.localStorage);

function AdminDownloads() {
  return (
    <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
      <span>Change lists (admins):</span>
      <a href={feedbackChangesUrl("org")} download className={`inline-flex items-center gap-1 ${LINK}`}>
        <Download className="h-3.5 w-3.5" aria-hidden /> Download change list (mycomap.org)
      </a>
      <a href={feedbackChangesUrl("vision")} download className={`inline-flex items-center gap-1 ${LINK}`}>
        <Download className="h-3.5 w-3.5" aria-hidden /> Download change list (Vision)
      </a>
    </p>
  );
}

/** The basic steps at the top of a photo review, folded away by default (Steve, 2026-11-20), opened with a click. */
export function ReviewSteps({ type, title }: { type: string; title: string }) {
  const s = REVIEW_STEPS[type];
  return (
    <details className="mb-4 rounded-lg border border-myco-green/30 bg-myco-green/5 p-4 dark:bg-myco-green/10"
             data-testid={`feedback-steps-${type}`}>
      <summary className={`cursor-pointer font-semibold ${INK}`} title={`How to review ${title}`}>How to use</summary>
      <p className={`mt-2 max-w-3xl text-sm ${BODY}`}>{s.lead}</p>
      <ol className={`mt-2 max-w-3xl list-decimal space-y-1 pl-5 text-sm ${BODY}`}>
        {s.steps.map((step) => <li key={step}>{step}</li>)}
      </ol>
      {typeof s.note === "string" && (
        <p className={`mt-3 max-w-3xl border-l-2 border-myco-green/60 pl-3 text-sm ${BODY}`}>{s.note}</p>
      )}
      {s.note && typeof s.note !== "string" && (
        <div className={`mt-3 max-w-3xl border-l-2 border-myco-green/60 pl-3 text-sm ${BODY}`}
             data-testid={`feedback-note-${type}`}>
          <p className="font-semibold">{s.note.title}</p>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {s.note.rules.map((r) => <li key={r}>{r}</li>)}
          </ul>
        </div>
      )}
    </details>
  );
}

/** Photos | Taxonomy, with each tab's open count once the summary is in. */
function MainTabs({ tabs, tab, onPick, counts }: {
  tabs: FeedbackTab[]; tab: FeedbackTab; onPick: (t: FeedbackTab) => void; counts: FeedbackSummary[] | null;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const shown = FEEDBACK_TABS.filter((t) => tabs.includes(t.code));
  // Arrow keys move between the tabs (the usual tablist keys).
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const j = (i + step + shown.length) % shown.length;
    onPick(shown[j].code);
    refs.current[j]?.focus();
  };
  return (
    <div role="tablist" aria-label="Kind of review"
         className="flex gap-1 border-b border-[#A87146]/25 dark:border-border" data-testid="feedback-tabs">
      {shown.map((t, i) => {
        const on = t.code === tab;
        return (
          <button key={t.code} ref={(el) => { refs.current[i] = el; }} type="button" role="tab"
                  id={`feedback-tab-${t.code}`} aria-selected={on} aria-controls={`feedback-tabpanel-${t.code}`}
                  tabIndex={on ? 0 : -1} onClick={() => onPick(t.code)} onKeyDown={(e) => onKey(e, i)}
                  className={`-mb-px inline-flex items-baseline gap-2 border-b-2 px-4 py-2.5 font-display text-lg font-semibold transition-colors ${
                    on ? `border-myco-green ${INK}` : "border-transparent text-muted-foreground hover:text-[#4a3728] dark:hover:text-foreground"}`}
                  data-testid={`feedback-tab-${t.code}`}>
            {t.label}
            {counts && (
              <span className="whitespace-nowrap text-sm font-normal tabular-nums text-muted-foreground">
                {tabOpen(counts, t.code).toLocaleString("en-US")} open
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** One section of the page: its name, open count and intro, then its queues' stat cards. The
 *  Photos tab is a single group whose heading is the tab itself. */
function SectionCards({ group, selected, onPick, heading = true }: {
  group: SectionGroup<FeedbackSummary>; selected: string | null; onPick: (t: string) => void; heading?: boolean;
}) {
  const id = `feedback-section-${group.code}`;
  return (
    <section className="min-w-0 space-y-3" aria-labelledby={heading ? id : undefined}
             aria-label={heading ? undefined : group.title} data-testid={id}>
      {heading && (
        <div className="min-w-0">
          <h2 id={id} className={`flex flex-wrap items-baseline gap-x-3 font-display text-xl font-semibold ${INK}`}>
            {group.title}
            <span className="text-sm font-normal text-muted-foreground tabular-nums">
              {group.open.toLocaleString("en-US")} open
            </span>
          </h2>
          {group.intro && <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{group.intro}</p>}
        </div>
      )}
      <StatCards types={group.types} selected={selected} onPick={onPick} label={group.title}
                 wide={!heading} />
    </section>
  );
}

function StatCards({ types, selected, onPick, label, wide = false }: {
  types: FeedbackSummary[]; selected: string | null; onPick: (t: string) => void; label: string; wide?: boolean;
}) {
  if (!types.length) return null;
  return (
    <div className={`grid grid-cols-1 gap-3 ${wide ? "md:grid-cols-3" : "min-[420px]:grid-cols-2 md:grid-cols-3 xl:grid-cols-5"}`}
         role="group" aria-label={`${label} queues`}>
      {types.map((t) => {
        const on = t.type === selected;
        return (
          <button key={t.type} type="button" onClick={() => onPick(t.type)} aria-pressed={on}
                  className={`${CARD} min-w-0 p-4 text-left transition-colors hover:border-myco-green/50 ${
                    on ? "ring-2 ring-myco-green border-myco-green/50" : ""}`}
                  data-testid={`feedback-card-${t.type}`}>
            <span className={wide ? `block text-base font-semibold leading-snug ${INK}`
                                  : "block text-sm font-medium leading-snug text-muted-foreground"}>{t.title}</span>
            {wide && QUEUE_BLURBS[t.type] && (
              <span className={`mt-1 block text-sm leading-snug ${BODY}`}>{QUEUE_BLURBS[t.type]}</span>
            )}
            {(() => {
              const line = statCardLines(t);
              return (<>
                <span className={`mt-2 block text-2xl font-bold tabular-nums ${INK}`}>
                  {line.big}
                  {line.unit && <span className="ml-1 text-sm font-medium text-muted-foreground">{line.unit}</span>}
                </span>
                <span className="block text-xs text-muted-foreground">{line.sub}</span>
              </>);
            })()}
            {wide && (
              <span className="mt-3 block text-sm font-medium text-myco-green">
                {on ? "Open below ↓" : "Start reviewing →"}
              </span>
            )}
          </button>
        );
      })}
    </div>
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

function TypeQueue({ summary, type, me, linked }: {
  summary: FeedbackSummary; type?: FeedbackType; me: FeedbackMe; linked?: FeedbackItem;
}) {
  const [status, setStatus] = useState<FeedbackStatus>("open");
  const [text, setText] = useState("");
  const q = useDebounced(text.trim());
  const [offset, setOffset] = useState(0);
  const [openId, setOpenId] = useState<number | null>(linked?.id ?? null);
  const [showLinked, setShowLinked] = useState(!!linked);
  // "One name, several writings" shows two lists; a linked item opens in its own.
  const [part, setPart] = useState<WritingList | null>(
    hasLists(summary.type) ? (linked?.list ?? "format") : null);
  const shown = useRef({ status, q, part });    // the linked item stays open until these change
  useEffect(() => {
    if (shown.current.status === status && shown.current.q === q && shown.current.part === part) return;
    shown.current = { status, q, part };
    setOffset(0); setOpenId(null); setShowLinked(false);
  }, [status, q, part]);
  const list = useQuery({
    queryKey: ["feedback", "items", summary.type, status, q, offset, part],
    queryFn: () => feedbackApi.items({ type: summary.type, status, q, offset, limit: PAGE_SIZE, ...(part ? { list: part } : {}) }),
    placeholderData: (prev) => prev,
  });
  const total = list.data?.total ?? 0;
  const items = list.data?.items ?? [];
  const seg = (on: boolean) => `px-3 py-1.5 text-sm font-medium ${
    on ? "bg-myco-green/10 text-myco-green" : "text-gray-700 hover:bg-myco-green/5 dark:text-foreground/80"}`;
  return (
    <section className="space-y-4" aria-label={summary.title}>
      <div>
        <h2 className={`font-display text-xl font-semibold ${INK}`}>{summary.title}</h2>
        <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{summary.about}</p>
      </div>
      {part && (
        <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
             role="group" aria-label="List" data-testid="feedback-lists">
          {WRITING_LISTS.map((l, i) => (
            <button key={l.code} type="button" aria-pressed={part === l.code} onClick={() => setPart(l.code)}
                    className={`${seg(part === l.code)} ${i > 0 ? "border-l border-[#A87146]/30 dark:border-border" : ""}`}>
              {l.label} ({listCount(summary, l.code, "open").toLocaleString("en-US")})
            </button>
          ))}
        </div>
      )}
      {part === "format" && (
        <div className="space-y-1 rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-3 text-sm dark:border-border dark:bg-muted/30"
             data-testid="feedback-format-panel">
          <p className={BODY}>{FORMAT_PANEL_TEXT}</p>
          {list.data?.names_to_fix_url && <Ext href={list.data.names_to_fix_url}>Open Name Update on mycomap.org</Ext>}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
             role="group" aria-label="Status">
          <button type="button" className={seg(status === "open")} aria-pressed={status === "open"}
                  onClick={() => setStatus("open")}>Open ({listCount(summary, part, "open").toLocaleString("en-US")})</button>
          <button type="button" className={`${seg(status === "decided")} border-l border-[#A87146]/30 dark:border-border`}
                  aria-pressed={status === "decided"} onClick={() => setStatus("decided")}>
            Decided ({listCount(summary, part, "decided").toLocaleString("en-US")})
          </button>
        </div>
        <input type="search" value={text} onChange={(e) => setText(e.target.value)}
               placeholder="Search names or records" aria-label="Search"
               className={`${INPUT} sm:max-w-xs flex-1`} data-testid="feedback-search" />
        {list.isFetching && <span className="text-xs text-muted-foreground">Loading…</span>}
      </div>
      {list.error && <p className={BODY}>{list.error instanceof ApiError ? list.error.message : "The list could not be loaded."}</p>}
      {list.data && !items.length && (
        <p className="text-sm text-muted-foreground">
          {q ? "Nothing matches that search." : status === "open" ? "Nothing open here." : "Nothing decided yet."}
        </p>
      )}
      {showLinked && linked && part !== "format" && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Linked from mycomap.org's Name Update:</p>
          <ul>
            <ItemRow item={linked} type={type} me={me} open={openId === linked.id}
                     onToggle={() => setOpenId(openId === linked.id ? null : linked.id)} />
          </ul>
        </div>
      )}
      {part === "format" ? (
        <ul className="divide-y divide-[#A87146]/15 rounded-lg border border-[#A87146]/20 dark:divide-border dark:border-border"
            data-testid="feedback-format-list">
          {items.map((it) => <FormatRow key={it.id} item={it} />)}
        </ul>
      ) : (
        <ul className="space-y-2">
          {items.filter((it) => !(showLinked && it.id === linked?.id)).map((it) => (
            <ItemRow key={it.id} item={it} type={type} me={me} open={openId === it.id}
                     onToggle={() => setOpenId(openId === it.id ? null : it.id)} />
          ))}
        </ul>
      )}
      {total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="text-muted-foreground">{pageText(offset, items.length, total)}</span>
          <span className="flex gap-2">
            <button type="button" className={PAGER} disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</button>
            <button type="button" className={PAGER} disabled={offset + PAGE_SIZE >= total}
                    onClick={() => setOffset(offset + PAGE_SIZE)}>Next</button>
          </span>
        </div>
      )}
    </section>
  );
}

const PAGER = "rounded-md border border-[#A87146]/30 bg-white px-3 py-1 text-sm hover:bg-myco-green/5 "
  + "disabled:opacity-40 disabled:pointer-events-none dark:border-border dark:bg-card";

const sciLike = (s: string | null | undefined) => !!s && /^[A-Z][a-z]+(\s|$)/.test(s.trim());

function Name({ s }: { s: string | null | undefined }) {
  if (!s) return <span className="text-muted-foreground">–</span>;
  return <span className={sciLike(s) ? "sci" : ""}>{s}</span>;
}

/** A "Format only" item, read only: each old writing and the one it becomes, with the records
 *  Vision holds under each now. Decided on mycomap.org's Names to fix, never here. */
function FormatRow({ item }: { item: FeedbackItem }) {
  const row = formatRow(item);
  const n = (c: number | null) => c == null ? null
    : <span className="tabular-nums text-muted-foreground"> ({c.toLocaleString("en-US")})</span>;
  return (
    <li className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5 px-3 py-2 text-sm">
      <span className={`min-w-0 break-words ${BODY}`}>
        {row.old.map((o, i) => <span key={o.name}>{i > 0 && ", "}<Name s={o.name} />{n(o.count)}</span>)}
      </span>
      <span className="text-muted-foreground">→</span>
      <span className={`min-w-0 break-words font-medium ${INK}`}><Name s={row.keep.name} />{n(row.keep.count)}</span>
      {item.status === "decided" && (
        <span className="rounded-full bg-myco-green/10 px-2 py-0.5 text-[11px] font-medium text-myco-green">done</span>
      )}
    </li>
  );
}

function ItemRow({ item, type, me, open, onToggle }: {
  item: FeedbackItem; type?: FeedbackType; me: FeedbackMe; open: boolean; onToggle: () => void;
}) {
  const rule = shownRule(item);
  return (
    <li className={`${CARD} min-w-0 ${open ? "border-myco-green/50" : ""}`}>
      <button type="button" onClick={onToggle} aria-expanded={open}
              className="flex w-full min-w-0 items-start gap-3 p-3 text-left hover:bg-myco-green/5 rounded-lg">
        <span className="min-w-0 flex-1">
          <span className={`block break-words font-semibold ${INK}`}><Name s={item.subject} /></span>
          <span className={`mt-0.5 block break-words text-sm ${BODY}`}>
            <Name s={item.current} />
            {item.proposed ? <> <span className="text-muted-foreground">→</span> <Name s={item.proposed} /></>
              : noProposal(item) ? <> <span className="text-muted-foreground">→ {NO_PROPOSAL_TEXT}</span></> : null}
          </span>
          <span className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>{item.records.toLocaleString("en-US")} record{item.records === 1 ? "" : "s"}</span>
            {itemBadges(item).map((b) => (
              <span key={b} className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
                {b}
              </span>
            ))}
            {item.status === "decided" && (
              <span className="rounded-full bg-myco-green/10 px-2 py-0.5 text-[11px] font-medium text-myco-green">decided</span>
            )}
            {item.status === "decided" && item.org_chip && <OrgChipBadge chip={item.org_chip} />}
            {rule && <span className="break-words">{rule}</span>}
          </span>
        </span>
        <ChevronDown className={`mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </button>
      {open && <ItemDetail id={item.id} type={type} me={me} />}
    </li>
  );
}

/** After a decision, reopen or refresh: keep the detail's live records, update the row where it
 *  is (it stays in view until the list is reloaded), and recount the cards. */
function useItemUpdate(id: number) {
  const qc = useQueryClient();
  return (next: FeedbackItemDetail) => {
    qc.setQueryData<FeedbackItemDetail>(["feedback", "item", id], (old) => ({ ...next, live: next.live ?? old?.live }));
    qc.setQueriesData<{ total: number; items: FeedbackItem[] }>({ queryKey: ["feedback", "items"] }, (old) =>
      old && { ...old, items: old.items.map((x) => (x.id === id ? { ...x, ...stripDetail(next) } : x)) });
    qc.invalidateQueries({ queryKey: ["feedback", "items"], refetchType: "none" });
    qc.invalidateQueries({ queryKey: ["feedback", "summary"] });
  };
}

function stripDetail(d: FeedbackItemDetail): FeedbackItem {
  const { decisions: _d, refreshes: _r, live: _l, org_request: org, ...rest } = d;
  return { ...rest, org_chip: org?.chip ?? null };
}

function OrgChipBadge({ chip }: { chip: OrgChip }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${orgChipClass(chip.tone)}`}
          data-testid="feedback-org-chip">
      {chip.text}
    </span>
  );
}

function ItemDetail({ id, type, me }: { id: number; type?: FeedbackType; me: FeedbackMe }) {
  const qc = useQueryClient();
  const readOnly = useReadOnly();
  const q = useQuery({ queryKey: ["feedback", "item", id], queryFn: () => feedbackApi.item(id) });
  const item = q.data;
  const sides = item ? pairSides(item) : [];
  // A pair's places as MO and iNat list them: asked apart, so the photos show first.
  const places = useQuery({ queryKey: ["feedback", "places", id], queryFn: () => feedbackApi.places(id),
                            enabled: sides.length > 0, staleTime: 10 * 60_000, retry: false });
  // A name item's names on iNat and MycoMap's synonym list: asked apart, so the item opens first.
  const names = useQuery({ queryKey: ["feedback", "names", id], queryFn: () => feedbackApi.names(id),
                           enabled: !!item && isNameItem(item), staleTime: 10 * 60_000, retry: false });
  const chip = item?.org_request?.chip ?? null;
  useEffect(() => {          // opening the item read .org: its row's chip follows
    if (!chip) return;
    qc.setQueriesData<{ total: number; items: FeedbackItem[] }>({ queryKey: ["feedback", "items"] }, (old) =>
      old && { ...old, items: old.items.map((x) => (x.id === id ? { ...x, org_chip: chip } : x)) });
  }, [qc, id, chip?.text, chip?.tone]); // eslint-disable-line react-hooks/exhaustive-deps
  if (q.isPending) return <p className="border-t px-3 py-3 text-sm text-muted-foreground">Loading…</p>;
  if (q.error || !item) {
    return <p className={`border-t px-3 py-3 text-sm ${BODY}`}>This item could not be loaded.</p>;
  }
  const facts = evidenceRows(item);
  return (
    <div className="space-y-5 border-t border-[#A87146]/15 p-3 dark:border-border" data-testid={`feedback-item-${id}`}>
      {sides.length > 0 ? (<>
        {/* A record pair: both copies' photos side by side, then the decision under them. */}
        <PairCompare sides={sides} distanceKm={item.evidence?.distance_km} likelyWrong={item.evidence?.likely_wrong_copy}
                     places={places.data?.places} placesLoading={places.isPending} />
        {readOnly ? <Outcome item={item} /> : <Decide item={item} type={type} wide />}
      </>) : (
      <div className="grid gap-5 lg:grid-cols-2">
        <div className="min-w-0 space-y-3">
          {facts.length > 0 && (
            <dl className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-x-3 gap-y-1 text-sm">
              {facts.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted-foreground break-words">{k}</dt>
                  <dd className={`min-w-0 break-words ${BODY}`}>{v}</dd>
                </div>
              ))}
            </dl>
          )}
          {(item.list === "look" && listReasonText(item.list_reason) || seededNote(item)) && (
            <div className="space-y-0.5 text-sm text-muted-foreground" data-testid="feedback-list-reason">
              {item.list === "look" && listReasonText(item.list_reason) && <p>{listReasonText(item.list_reason)}</p>}
              {seededNote(item) && <p>{seededNote(item)}</p>}
            </div>
          )}
          {isNameItem(item) && <NameContextPanel data={names.data} loading={names.isPending} failed={!!names.error} />}
          {item.links.length > 0 && (
            <p className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
              <span className="text-muted-foreground">Public pages:</span>
              {item.links.map((href, i) => (
                <Ext key={href} href={href}>{item.record_keys[i] ? recordLabel(item.record_keys[i]) : `Link ${i + 1}`}</Ext>
              ))}
            </p>
          )}
        </div>
        {readOnly ? <Outcome item={item} /> : <Decide item={item} type={type} />}
      </div>
      )}
      <LiveRecords item={item} reloading={q.isFetching} onReload={() => q.refetch()} me={me} />
      {readOnly ? (item.masked && (
        <p className="text-xs text-muted-foreground">Vision's answer stays hidden until a reviewer decides.</p>
      )) : item.masked ? (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            Judge the label from the photos and the record. Refresh appears after your decision.
          </p>
          <RevealAnswer itemId={item.id} revealed={item.revealed}
                        onRevealed={(r) => qc.setQueryData<FeedbackItemDetail>(["feedback", "item", id],
                                                                               (old) => old && { ...old, revealed: r })} />
        </div>
      ) : item.record_keys.length > 0 && <Refresh item={item} />}
      {item.org_request && (
        <OrgRequestBlock item={item} org={item.org_request} checking={q.isFetching} onCheck={() => q.refetch()} />
      )}
      <History item={item} type={type} />
    </div>
  );
}

/** Each name the item is about, how iNaturalist holds it (linked), and MycoMap's synonym list
 *  around it (Steve, 2026-10-21): enough to see why the name was listed and what to do. */
function NameContextPanel({ data, loading, failed }: { data: NameContext | undefined; loading: boolean; failed: boolean }) {
  if (loading) return <p className="text-sm text-muted-foreground">Looking the names up on iNaturalist…</p>;
  if (failed || !data) return <p className="text-sm text-muted-foreground">The names could not be looked up just now.</p>;
  const rows = data.synonyms.rows;
  return (
    <div className="space-y-3" data-testid="feedback-name-context">
      <div className="space-y-2">
        <h3 className={`text-sm font-semibold ${INK}`}>The names on iNaturalist</h3>
        <ul className="space-y-2">
          {data.names.map((n) => (
            <li key={n.name} className="min-w-0 rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-2 text-sm dark:border-border dark:bg-muted/40">
              <p className="flex flex-wrap items-baseline gap-x-2">
                <span className={`font-medium ${INK}`}><Name s={n.name} /></span>
                <span className="text-xs text-muted-foreground">{roleText(n.roles)}</span>
              </p>
              <p className={`text-xs ${BODY}`}>{n.about}</p>
              <p className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
                {inatLinks(n).map((l) => (
                  <Ext key={l.href + l.text} href={l.href} className={l.strong ? "font-medium" : ""}>{l.text}</Ext>
                ))}
              </p>
            </li>
          ))}
        </ul>
      </div>
      <div className="space-y-1">
        <h3 className={`text-sm font-semibold ${INK}`}>MycoMap's synonym list</h3>
        {data.synonyms.status === "unavailable" ? (
          <p className="text-xs text-muted-foreground">mycomap.org did not answer for its synonym list.</p>
        ) : rows.length === 0 ? (
          <p className="text-xs text-muted-foreground">No synonyms of these names on MycoMap's list.</p>
        ) : (
          <ul className={`space-y-0.5 text-sm ${BODY}`}>
            {rows.map((r) => (
              <li key={`${r.retired}→${r.accepted}`} className="break-words">
                <Name s={r.retired} /> <span className="text-muted-foreground">retired →</span> <Name s={r.accepted} />
              </li>
            ))}
            {data.synonyms.more > 0 && <li className="text-xs text-muted-foreground">and {data.synonyms.more} more</li>}
          </ul>
        )}
      </div>
    </div>
  );
}

function LiveRecords({ item, reloading, onReload, me }: {
  item: FeedbackItemDetail; reloading: boolean; onReload: () => void; me: FeedbackMe;
}) {
  const live = item.live ?? [];
  if (!live.length) return null;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className={`text-sm font-semibold ${INK}`}>On mycomap.org now</h3>
        <button type="button" onClick={onReload} disabled={reloading}
                className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-myco-green disabled:opacity-50">
          <RefreshCw className={`h-3 w-3 ${reloading ? "animate-spin" : ""}`} aria-hidden />
          {reloading ? "Reloading…" : "Reload from mycomap.org"}
        </button>
      </div>
      {live.map((r) => <LiveCard key={r.record_key} r={r} me={me} />)}
    </div>
  );
}

function LiveCard({ r, me }: { r: LiveRecord; me: FeedbackMe }) {
  if (r.unavailable) {
    return (
      <div className="rounded-md border border-dashed border-[#A87146]/40 p-3 text-sm dark:border-border">
        <span className={`font-medium ${INK}`}>{recordLabel(r.record_key)}</span>{" "}
        <span className="text-muted-foreground">mycomap.org did not answer for this record.</span>
      </div>
    );
  }
  const seqs = r.sequences ?? [];
  const projects = r.projects ?? [];
  const validate = r.links?.validate ?? [];
  // One Validate Records link per project, in the same order, when the counts match.
  const paired = validate.length > 0 && validate.length === projects.length;
  return (
    <div className="min-w-0 rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-3 text-sm dark:border-border dark:bg-muted/40">
      <p className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className={`font-medium ${INK}`}>{recordLabel(r.record_key)}</span>
        <span className="text-muted-foreground">label now:</span>
        <span className={`font-semibold ${INK}`}><Name s={r.name} /></span>
        {r.source && <span className="text-xs text-muted-foreground">({r.source})</span>}
      </p>
      {(projects.length > 0 || validate.length > 0 || r.links?.name_update) && (
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
          {projects.map((p, i) => (
            <span key={String(p.id)} className={BODY}>
              {p.name ?? `Project ${p.id}`}
              {p.status && <span className="text-muted-foreground"> · {p.status}</span>}
              {paired && <> · <Ext href={validate[i]}>Validate Records</Ext></>}
            </span>
          ))}
          {!paired && validate.map((href) => <Ext key={href} href={href}>Validate Records</Ext>)}
          {r.links?.name_update && <Ext href={r.links.name_update}>Name Update</Ext>}
        </div>
      )}
      {seqs.length > 0 ? (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[34rem] text-left text-xs">
            <thead className="text-muted-foreground">
              <tr className="border-b border-[#A87146]/15 dark:border-border">
                <th className="py-1 pr-3 font-medium">Sequence</th>
                <th className="py-1 pr-3 font-medium">Name</th>
                <th className="py-1 pr-3 font-medium text-right">Length</th>
                <th className="py-1 pr-3 font-medium">BLAST</th>
                <th className="py-1 font-medium">Saved</th>
              </tr>
            </thead>
            <tbody>
              {seqs.map((s) => {
                const blast = blastAnchor(s);
                return (
                  <tr key={String(s.id)} className="border-b border-[#A87146]/10 last:border-0 dark:border-border/60">
                    <td className="py-1.5 pr-3 align-top whitespace-nowrap">
                      {s.url ? <Ext href={s.url}>{String(s.id)}</Ext> : String(s.id)}
                      {yes(s.trash) && (
                        <span className="ml-1 rounded bg-red-50 px-1 py-0.5 text-[10px] font-medium text-red-700 dark:bg-red-950 dark:text-red-200">trash</span>
                      )}
                    </td>
                    <td className={`py-1.5 pr-3 align-top break-words ${BODY}`}><Name s={s.name} /></td>
                    <td className="py-1.5 pr-3 align-top text-right tabular-nums">{s.length ?? "–"}</td>
                    <td className="py-1.5 pr-3 align-top whitespace-nowrap">
                      {blast ? (
                        <a href={blast.href} target={blast.target} rel={blast.rel}
                           className={`inline-flex items-center gap-0.5 ${LINK}`}>
                          {blast.label} <ArrowUpRight className="h-3 w-3" aria-hidden />
                        </a>
                      ) : <span className="text-muted-foreground">–</span>}
                    </td>
                    <td className="py-1.5 align-top whitespace-nowrap text-muted-foreground">
                      {whenText(s.blast?.saved_at) || "–"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!me.staff && seqs.some((s) => s.blast_link) && (
            <p className="mt-1 text-[11px] text-muted-foreground">BLAST links open the public results page.</p>
          )}
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">No sequences on this record.</p>
      )}
    </div>
  );
}

/** Where an item stands, for a read-only viewer: open, or decided (the outcome is in History). */
function Outcome({ item }: { item: FeedbackItemDetail }) {
  return (
    <div className="min-w-0 space-y-1" data-testid="feedback-outcome">
      <h3 className={`text-sm font-semibold ${INK}`}>{item.status === "decided" ? "Decided" : "Open"}</h3>
      <p className="text-sm text-muted-foreground">
        {item.status === "decided" ? "The decision is below." : "Not decided yet."}
      </p>
    </div>
  );
}

function Decide({ item, type, wide = false }: { item: FeedbackItemDetail; type?: FeedbackType; wide?: boolean }) {
  const update = useItemUpdate(item.id);
  const options = decideOptions(item, type?.options ?? []);
  const doneLine = alreadyDoneText(item.already_done);
  const [choice, setChoice] = useState<FeedbackOption | null>(null);
  const [value, setValue] = useState("");
  const [note, setNote] = useState("");
  const [photoKeys, setPhotoKeys] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const box = choice ? valueBox(choice) : null;
  const typed = useDebounced(value.trim(), 400);
  const preview = useQuery({
    queryKey: ["feedback", "org-preview", item.id, choice?.code, typed],
    queryFn: () => feedbackApi.orgPreview(item.id, { decision: choice!.code, ...(typed ? { value: typed } : {}) }),
    enabled: !!choice?.name_update, retry: false,
  });
  const decide = useMutation({
    mutationFn: (o: FeedbackOption) => {
      const b = decisionBody(o, { value, note, photoKeys });
      if ("error" in b) throw new Error(b.error);
      return feedbackApi.decide(item.id, b.body);
    },
    onSuccess: (next) => { update(next); setChoice(null); setNote(""); setPhotoKeys(""); setProblem(null); },
    onError: (e) => setProblem(e instanceof Error ? e.message : "The decision could not be saved."),
  });
  const reopen = useMutation({
    mutationFn: () => feedbackApi.reopen(item.id),
    onSuccess: update,
    onError: (e) => setProblem(e instanceof Error ? e.message : "The item could not be reopened."),
  });
  const pick = (o: FeedbackOption) => {
    setChoice(o);
    setValue(prefillValue(o, item));
    setProblem(null);
  };
  if (item.status === "decided") {
    const waiting = ["searching", "waiting", "refused"].includes(item.org_request?.status ?? "");
    return (
      <div className="min-w-0 space-y-2">
        <h3 className={`text-sm font-semibold ${INK}`}>Decided</h3>
        <p className="text-sm text-muted-foreground">The decision and who made it are below.</p>
        <button type="button" onClick={() => reopen.mutate()} disabled={reopen.isPending} className={PAGER}>
          {reopen.isPending ? "Reopening…" : "Reopen"}
        </button>
        {waiting && (
          <p className="text-xs text-muted-foreground">
            Reopening while mycomap.org is still waiting withdraws the update there.
          </p>
        )}
        {problem && <p className="text-sm text-red-700 dark:text-red-300">{problem}</p>}
      </div>
    );
  }
  return (
    <div className="min-w-0 space-y-3">
      <h3 className={`text-sm font-semibold ${INK}`}>Decide</h3>
      {!options.length && <p className="text-sm text-muted-foreground">Loading the choices…</p>}
      {noProposal(item) && options.length > 0 && (
        <p className="text-xs text-muted-foreground">
          iNaturalist names no current taxon for this name, so there is no name of theirs to use.
        </p>
      )}
      {doneLine && options.length > 0 && (
        <p className="rounded-md border border-myco-green/30 bg-myco-green/5 p-2 text-sm text-myco-green"
           data-testid="feedback-already-done">{doneLine}</p>
      )}
      {/* Options of one group sit under its heading (e.g. Duplicate observation: Use iNat,
          Use MO); `wide` (a record pair, decided under the photos) lays the blocks in a row. */}
      <div className={wide ? "grid gap-3 sm:grid-cols-[2fr_1fr]" : "flex flex-col gap-2"}>
        {optionGroups(options).map((g) => (
          <div key={g.group ?? g.options[0].code} className="min-w-0 space-y-1.5"
               role={g.group ? "group" : undefined} aria-label={g.group ?? undefined}>
            {g.group && <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{g.group}</p>}
            <div className={g.options.length > 1 && g.group ? "grid grid-cols-2 gap-2" : "flex flex-col gap-2"}>
              {g.options.map((o) => {
                const on = choice?.code === o.code;
                return (
                  <button key={o.code} type="button" onClick={() => pick(o)} aria-pressed={on}
                          aria-label={o.short ? o.label : undefined}
                          disabled={!!o.off} title={o.off ?? undefined}
                          className={`min-w-0 rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                            o.off ? "cursor-not-allowed border-[#A87146]/20 bg-white opacity-50 dark:border-border dark:bg-card"
                              : on ? "border-myco-green bg-myco-green/10" : "border-[#A87146]/30 bg-white hover:bg-myco-green/5 dark:border-border dark:bg-card"}`}>
                    <span className={`block break-words font-medium ${INK}`}>{o.short ?? o.label}</span>
                    <span className="block break-words text-xs text-muted-foreground">{o.off ? "Nothing left to rename" : o.hint ?? appliesWords(o.applies)}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {choice && (
        <form className="space-y-2" onSubmit={(e) => { e.preventDefault(); decide.mutate(choice); }}>
          {box && (
            <label className="block text-sm">
              <span className="text-muted-foreground">{box.label}</span>
              <input className={INPUT} value={value} onChange={(e) => setValue(e.target.value)} required={box.required} />
            </label>
          )}
          {choice.name_update && <OrgPreviewNote preview={preview.data} error={preview.error} />}
          {takesPhotoKeys(choice) && (
            <label className="block text-sm">
              <span className="text-muted-foreground">Photos to leave out (comma-separated photo keys)</span>
              <input className={INPUT} value={photoKeys} onChange={(e) => setPhotoKeys(e.target.value)} />
            </label>
          )}
          <label className="block text-sm">
            <span className="text-muted-foreground">Note (optional)</span>
            <textarea className={`${INPUT} min-h-[3.5rem]`} value={note} onChange={(e) => setNote(e.target.value)} />
          </label>
          <div className="flex flex-wrap items-center gap-3">
            <button type="submit" disabled={decide.isPending}
                    className="rounded-md bg-myco-green px-4 py-1.5 text-sm font-medium text-white hover:bg-myco-green/90 disabled:opacity-50">
              {decide.isPending ? "Saving…" : "Save decision"}
            </button>
            <span className="text-xs text-muted-foreground">
              {choice.name_update ? "Opens a Name Update on mycomap.org." : `Takes effect ${appliesWords(choice.applies)}.`}
            </span>
          </div>
        </form>
      )}
      {problem && <p className="text-sm text-red-700 dark:text-red-300">{problem}</p>}
    </div>
  );
}

const TONE: Record<Verdict["tone"], string> = {
  disagrees: "text-amber-800 dark:text-amber-200",
  agrees: "text-myco-green",
  unknown: "text-muted-foreground",
};

function Top5Line({ top5 }: { top5: Top5 | undefined }) {
  if (!top5?.length) return null;
  return (
    <ol className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
      {top5.slice(0, 5).map(([name, score], i) => (
        <li key={`${name}-${i}`} className={BODY}>
          <span className="text-muted-foreground">{i + 1}.</span> <Name s={name} />{" "}
          <span className="tabular-nums text-muted-foreground">{score.toFixed(2)}</span>
        </li>
      ))}
    </ol>
  );
}

function Refresh({ item }: { item: FeedbackItemDetail }) {
  const update = useItemUpdate(item.id);
  const [found, setFound] = useState<RefreshedRecord[] | null>(null);
  const run = useMutation({
    mutationFn: () => feedbackApi.refresh(item.id),
    onSuccess: (res) => { update(res.item); setFound(res.refreshed); },
  });
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={() => run.mutate()} disabled={run.isPending}
                className={`${PAGER} inline-flex items-center gap-1`}>
          <RefreshCw className={`h-3.5 w-3.5 ${run.isPending ? "animate-spin" : ""}`} aria-hidden />
          {run.isPending ? "Re-checking…" : "Refresh"}
        </button>
        <span className="text-xs text-muted-foreground">
          Re-reads the records from mycomap.org and asks Vision again.
        </span>
      </div>
      {run.error && <p className="text-sm text-red-700 dark:text-red-300">
        {run.error instanceof Error ? run.error.message : "The refresh failed."}
      </p>}
      {found?.map((r) => {
        const v = refreshVerdict(r);
        return (
          <div key={r.record_key} className="rounded-md border border-[#A87146]/20 p-2 text-sm dark:border-border">
            <p className="flex flex-wrap items-baseline gap-x-2">
              <span className={`font-medium ${INK}`}>{recordLabel(r.record_key)}</span>
              <span className="text-muted-foreground">name now:</span> <Name s={r.name_now} />
            </p>
            <p className={`font-medium ${TONE[v.tone]}`}>{v.text}</p>
            {r.vision.available && reaskModel(r.vision) && (
              <p className="text-xs text-muted-foreground" data-testid="text-reask-model">{reaskModel(r.vision)}</p>
            )}
            {r.vision.available && <Top5Line top5={r.vision.top5} />}
          </div>
        );
      })}
    </div>
  );
}

function History({ item, type }: { item: FeedbackItemDetail; type?: FeedbackType }) {
  const decisions = item.decisions ?? [];
  const refreshes = item.refreshes ?? [];
  if (!decisions.length && !refreshes.length) return null;
  return (
    <div className="space-y-2">
      <h3 className={`text-sm font-semibold ${INK}`}>History</h3>
      <ul className="space-y-1.5 text-sm">
        {decisions.map((d) => (
          <li key={`d${d.id}`} className="min-w-0 break-words">
            <span className="text-xs text-muted-foreground">{whenText(d.at)}</span>{" "}
            <span className={`font-medium ${INK}`}>{d.by_name ?? d.by_user}</span>{" "}
            <span className={BODY}>{decisionLabel(d.decision, type?.options)}</span>
            {d.value && <>: <Name s={d.value} /></>}
            <span className="text-xs text-muted-foreground"> ({appliesWords(d.applies)}{
              d.id === decisions[decisions.length - 1]?.id && item.org_request
                ? historyOrgText(item.org_request) : yes(d.applied) ? ", applied" : ""})</span>
            {d.photo_keys?.length > 0 && <span className="text-xs text-muted-foreground"> · photos {d.photo_keys.join(", ")}</span>}
            {d.note && <span className="block text-xs text-muted-foreground">“{d.note}”</span>}
          </li>
        ))}
        {refreshes.map((r, i) => {
          const v = storedVerdict(r.still_disagrees);
          return (
            <li key={`r${i}`} className="min-w-0 break-words">
              <span className="text-xs text-muted-foreground">{whenText(r.at)}</span>{" "}
              <span className={BODY}>Refreshed: name <Name s={r.name_now} /></span>{" "}
              <span className={`text-xs font-medium ${TONE[v.tone]}`}>{v.text}</span>
              <Top5Line top5={r.top5} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function OrgPreviewNote({ preview, error }: { preview: OrgPreview | undefined; error: unknown }) {
  if (error) {
    return <p className="text-sm text-red-700 dark:text-red-300">
      {error instanceof ApiError ? error.message : "This decision cannot make a Name Update as it stands."}
    </p>;
  }
  if (!preview || !preview.opens) return null;
  const n = (name: string) => preview.records[name];
  return (
    <div className="space-y-1 rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-2 text-sm dark:border-border dark:bg-muted/30"
         data-testid="feedback-org-preview">
      <p className={BODY}>
        mycomap.org will be asked to rename every record of{" "}
        {preview.old_names.map((o, i) => (
          <span key={o}>{i > 0 && ", "}<Name s={o} />{n(o) != null && <span className="text-muted-foreground"> ({n(o)})</span>}</span>
        ))}{" "}
        to <Name s={preview.new_name} />{n(preview.new_name) != null && (
          <span className="text-muted-foreground"> ({n(preview.new_name)})</span>)}, not only the ones flagged here.
        A person with Name Update rights there runs it.
      </p>
      {preview.warnings.map((w) => (
        <p key={w} className="text-xs font-medium text-amber-800 dark:text-amber-200">{w}</p>
      ))}
    </div>
  );
}

function OrgRequestBlock({ item, org, checking, onCheck }: {
  item: FeedbackItemDetail; org: OrgRequest; checking: boolean; onCheck: () => void;
}) {
  const update = useItemUpdate(item.id);
  const send = useMutation({ mutationFn: () => feedbackApi.sendAgain(item.id), onSuccess: update });
  const readOnly = useReadOnly();
  return (
    <div className="space-y-2" data-testid="feedback-org-request">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className={`text-sm font-semibold ${INK}`}>On mycomap.org</h3>
        <button type="button" onClick={onCheck} disabled={checking}
                className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-myco-green disabled:opacity-50">
          <RefreshCw className={`h-3 w-3 ${checking ? "animate-spin" : ""}`} aria-hidden />
          {checking ? "Checking…" : "Check now"}
        </button>
      </div>
      <div className="rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-3 text-sm dark:border-border dark:bg-muted/30">
        <p className="flex flex-wrap items-baseline gap-x-2">
          <span className={`font-medium ${INK}`}>
            {org.request_id ? `Name Update request ${org.request_id}` : "Name Update"}
          </span>
          {org.url && <Ext href={org.url}>Open in Name Update</Ext>}
        </p>
        <ol className="mt-2 grid list-none gap-1 p-0 text-sm sm:grid-cols-4">
          {org.steps.map((st) => {
            const look = stepLook(st.state);
            return (
              <li key={st.step} className={`${BODY} ${look.textClass}`}>
                <span className={look.markClass} aria-hidden>{look.mark}</span> {stepText(st, org)}
              </li>
            );
          })}
        </ol>
        {org.declined_reason && (
          <p className={`mt-2 ${BODY}`}>
            Declined{org.declined_by ? ` by ${org.declined_by}` : ""}: “{org.declined_reason}”
          </p>
        )}
        {org.note && <p className="mt-2 text-amber-800 dark:text-amber-200">{org.note}</p>}
        {org.can_send && !readOnly && (
          <button type="button" onClick={() => send.mutate()} disabled={send.isPending} className={`${PAGER} mt-2`}>
            {send.isPending ? "Sending…" : "Send again"}
          </button>
        )}
        {send.error && <p className="mt-1 text-sm text-red-700 dark:text-red-300">
          {send.error instanceof Error ? send.error.message : "It could not be sent."}</p>}
        <p className="mt-2 text-xs text-muted-foreground">
          Every record of {org.old_names.map((o, i) => <span key={o}>{i > 0 && ", "}<Name s={o} /></span>)} is in
          the update, not only the ones flagged here. A person with Name Update rights on mycomap.org runs it.
          This item counts as applied when mycomap.org reports it done; Vision then relabels these records
          straight away, without waiting for the nightly.
        </p>
      </div>
    </div>
  );
}
