import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, ChevronDown, Download, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/Layout";
import { RevealAnswer } from "@/components/RevealAnswer";
import { ApiError, feedbackApi, feedbackChangesUrl, signInUrl, type FeedbackItem, type FeedbackItemDetail,
         type FeedbackMe, type FeedbackOption, type FeedbackStatus, type FeedbackSummary, type FeedbackType,
         type LiveRecord, type OrgChip, type OrgPreview, type OrgRequest, type RefreshedRecord,
         type Top5 } from "@/lib/api";
import { appliesWords, blastAnchor, decisionBody, decisionLabel, evidenceRows, feedbackAccess, groupBySection,
         historyOrgText, itemBadges, linkedItem, openQueue, orgChipClass, PAGE_SIZE, pageText, prefillValue,
         queueParam, readLastQueue, recordLabel, refreshVerdict, shownRule, stepLook, stepText, storedVerdict, takesPhotoKeys,
         statCardLines, valueBox, whenText, writeLastQueue, yes, type SectionGroup,
         type Verdict } from "@/lib/feedback";
import { LABEL_REVIEW_TYPE } from "@/lib/labelReview";
import { PHOTO_REVIEW_TYPE } from "@/lib/photoReview";
import { LabelReview } from "./LabelReview";
import { PhotoReview } from "./PhotoReview";

// Human Feedback (Steve, 2026-10-18): a review queue of places where Vision and the reference
// data disagree, for mycomap.org admins, project leaders and contributors, laid out like
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
        Human Feedback is for mycomap.org admins, project leaders and contributors.{" "}
        <a href={signInUrl()} className="text-myco-green underline">Sign in</a> to review.
      </p>
    );
  }
  if (why === "forbidden") {
    return (
      <p className={BODY} data-testid="feedback-forbidden">
        This page is for mycomap.org admins, project leaders and contributors. Your account
        isn't one of those yet.
      </p>
    );
  }
  return <p className={BODY}>Human Feedback could not be loaded.</p>;
}

export function FeedbackPage() {
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
  const pick = (type: string) => {
    setPicked(type);
    writeLastQueue(browserStorage, type);
  };
  const types = summary.data ?? [];
  const sections = groupBySection(types);
  const selected = openQueue({ picked, param, stored, types: types.map((t) => t.type),
                               linkedPending: linkedId != null && linked.isPending,
                               linkedType: linked.data?.item_type ?? null });
  const sel = types.find((t) => t.type === selected);
  const selSection = sections.find((g) => g.types.some((t) => t.type === selected));
  const catalogueType = catalogue.data?.find((t) => t.type === selected);
  return (
    <>
      <PageHeader title="Human Feedback">
        Two kinds of work on the records Vision learns from: data cleanup, fixing what a rule
        found, and data validation, checking whether a record's label is right.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-6xl space-y-6">
        {me.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
        <AccessNote error={me.error} />
        {me.data && (
          <>
            {me.data.admin && <AdminDownloads />}
            {summary.isPending && <p className="text-sm text-muted-foreground">Loading the queues…</p>}
            {summary.error && <p className={BODY}>The queues could not be loaded.</p>}
            {sections.map((g) => (
              <SectionCards key={g.code} group={g} selected={selected} onPick={pick} />
            ))}
            {summary.data && types.length > 0 && !sel && !(linkedId != null && linked.isPending) && (
              <p className="text-sm text-muted-foreground" data-testid="feedback-pick-queue">Pick a queue above.</p>
            )}
            {sel && (
              <div className="min-w-0 space-y-1 border-t border-[#A87146]/20 pt-6 dark:border-border">
                {selSection && (
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                     data-testid="feedback-queue-section">{selSection.title}</p>
                )}
                {sel.type === PHOTO_REVIEW_TYPE
                  ? <PhotoReview key={sel.type} summary={sel} />
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

/** One section of the page: its name, open count and intro, then its queues' stat cards. */
function SectionCards({ group, selected, onPick }: {
  group: SectionGroup<FeedbackSummary>; selected: string | null; onPick: (t: string) => void;
}) {
  const id = `feedback-section-${group.code}`;
  return (
    <section className="min-w-0 space-y-3" aria-labelledby={id} data-testid={id}>
      <div className="min-w-0">
        <h2 id={id} className={`flex flex-wrap items-baseline gap-x-3 font-display text-xl font-semibold ${INK}`}>
          {group.title}
          <span className="text-sm font-normal text-muted-foreground tabular-nums">
            {group.open.toLocaleString("en-US")} open
          </span>
        </h2>
        {group.intro && <p className={`mt-1 max-w-3xl text-sm ${BODY}`}>{group.intro}</p>}
      </div>
      <StatCards types={group.types} selected={selected} onPick={onPick} label={group.title} />
    </section>
  );
}

function StatCards({ types, selected, onPick, label }: {
  types: FeedbackSummary[]; selected: string | null; onPick: (t: string) => void; label: string;
}) {
  if (!types.length) return null;
  return (
    <div className="grid grid-cols-1 gap-3 min-[420px]:grid-cols-2 md:grid-cols-3 xl:grid-cols-5"
         role="group" aria-label={`${label} queues`}>
      {types.map((t) => {
        const on = t.type === selected;
        return (
          <button key={t.type} type="button" onClick={() => onPick(t.type)} aria-pressed={on}
                  className={`${CARD} min-w-0 p-4 text-left transition-colors hover:border-myco-green/50 ${
                    on ? "ring-2 ring-myco-green border-myco-green/50" : ""}`}
                  data-testid={`feedback-card-${t.type}`}>
            <span className="block text-sm font-medium leading-snug text-muted-foreground">{t.title}</span>
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
  const shown = useRef({ status, q });          // the linked item stays open until these change
  useEffect(() => {
    if (shown.current.status === status && shown.current.q === q) return;
    shown.current = { status, q };
    setOffset(0); setOpenId(null); setShowLinked(false);
  }, [status, q]);
  const list = useQuery({
    queryKey: ["feedback", "items", summary.type, status, q, offset],
    queryFn: () => feedbackApi.items({ type: summary.type, status, q, offset, limit: PAGE_SIZE }),
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
      <div className="flex flex-wrap items-center gap-3">
        <div className="inline-flex overflow-hidden rounded-md border border-[#A87146]/30 dark:border-border"
             role="group" aria-label="Status">
          <button type="button" className={seg(status === "open")} aria-pressed={status === "open"}
                  onClick={() => setStatus("open")}>Open ({summary.open.toLocaleString("en-US")})</button>
          <button type="button" className={`${seg(status === "decided")} border-l border-[#A87146]/30 dark:border-border`}
                  aria-pressed={status === "decided"} onClick={() => setStatus("decided")}>
            Decided ({summary.decided.toLocaleString("en-US")})
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
      {showLinked && linked && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Linked from mycomap.org's Name Update:</p>
          <ul>
            <ItemRow item={linked} type={type} me={me} open={openId === linked.id}
                     onToggle={() => setOpenId(openId === linked.id ? null : linked.id)} />
          </ul>
        </div>
      )}
      <ul className="space-y-2">
        {items.filter((it) => !(showLinked && it.id === linked?.id)).map((it) => (
          <ItemRow key={it.id} item={it} type={type} me={me} open={openId === it.id}
                   onToggle={() => setOpenId(openId === it.id ? null : it.id)} />
        ))}
      </ul>
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
            {item.proposed ? <> <span className="text-muted-foreground">→</span> <Name s={item.proposed} /></> : null}
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
  const q = useQuery({ queryKey: ["feedback", "item", id], queryFn: () => feedbackApi.item(id) });
  const item = q.data;
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
          {item.links.length > 0 && (
            <p className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
              <span className="text-muted-foreground">Public pages:</span>
              {item.links.map((href, i) => (
                <Ext key={href} href={href}>{item.record_keys[i] ? recordLabel(item.record_keys[i]) : `Link ${i + 1}`}</Ext>
              ))}
            </p>
          )}
        </div>
        <Decide item={item} type={type} />
      </div>
      <LiveRecords item={item} reloading={q.isFetching} onReload={() => q.refetch()} me={me} />
      {item.masked ? (
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

function Decide({ item, type }: { item: FeedbackItemDetail; type?: FeedbackType }) {
  const update = useItemUpdate(item.id);
  const options = type?.options ?? [];
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
      <div className="flex flex-col gap-2">
        {options.map((o) => {
          const on = choice?.code === o.code;
          return (
            <button key={o.code} type="button" onClick={() => pick(o)} aria-pressed={on}
                    className={`rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                      on ? "border-myco-green bg-myco-green/10" : "border-[#A87146]/30 bg-white hover:bg-myco-green/5 dark:border-border dark:bg-card"}`}>
              <span className={`block font-medium ${INK}`}>{o.label}</span>
              <span className="block text-xs text-muted-foreground">{appliesWords(o.applies)}</span>
            </button>
          );
        })}
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
        {org.can_send && (
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
