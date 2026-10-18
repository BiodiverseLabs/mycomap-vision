import { lazy, Suspense, useEffect, useMemo, useRef, useState, type Dispatch, type ReactNode,
         type RefObject, type SetStateAction } from "react";
import { useQuery, useMutation, type UseQueryResult } from "@tanstack/react-query";
import { ArrowUpRight, CalendarDays, Camera, Check, ChevronRight, Circle, CircleHelp, Dna, ImagePlus,
         Lightbulb, Link2, Loader2, LocateFixed, Lock, Map as MapIcon, MapPin, Microscope, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/Layout";
import {
  api,
  ApiError,
  signInUrl,
  modelKey,
  num,
  pct,
  RANKS,
  type Candidate,
  type ContextUsed,
  type IdentifyResult,
  type LikelySet,
  type LinkedObservation,
  type ModelsInfo,
  type PhotoResult,
  type Where,
  type Rank,
  type Specimen,
} from "@/lib/api";
import { chanceWithheld, gateBanner, leftOut, pageHints, photoGateLabel, whatNext, type GateBanner } from "@/lib/gate";
import { fillFromPhotos, readPhotoPlaceDate, type PhotoPlaceDate } from "@/lib/photoPlaceDate";
import { locateError, noPhotoPlace, placeNote, platformOf, sourceWords } from "@/lib/photoPlace";
import { applyPick } from "@/lib/mapPick";
import { observationFromQuery, parseObservationLink, sourceName, type ObservationRef } from "@/lib/observationLink";

// The map (Leaflet) loads only when someone opens it: the Identify page stays light.
const LocationPicker = lazy(() => import("@/components/LocationPicker"));
import {
  identifyGate, LIKELY_COPY, PHOTO_CHECKLIST, SAFETY_LINE, SEQUENCING_URL,
} from "@/lib/publicView";
import { capitalize, publicMethodLabel, publicModelName } from "@/lib/publicNames";
import { SafetyNote } from "@/components/SafetyNote";
import { canStartOver, startOver } from "@/lib/startOver";
import {
  alsoPossible, answerFor, boxHints, breakdownLabel, breakdownOpen, confusedNote, coverageSummary, listedNames,
  photoOnlyNote, placeLines, placeWarns, publicNote, usesPlace, type Answer, type Named,
} from "@/lib/answer";
import { shouldScrollTo, showPinnedBar, waitingSteps, type Step } from "@/lib/phoneFlow";
import {
  BASE_LABEL, MAX_COMPARED, defaultChoice, offeredBases, resolveMethod, switchesFor, type Base, type Choice,
} from "@/lib/modelChoice";

const MAX_PHOTOS = 10;
const RANK_LABEL: Record<Rank, string> = { family: "Family", genus: "Genus", species: "Species" };

export function IdentifyPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [where, setWhere] = useState<Where>({});
  const [found, setFound] = useState<Map<File, PhotoPlaceDate>>(new Map());
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, staleTime: 60_000 });
  const gate = identifyGate(me.data, me.isError);

  // Read each new photo's place and date, then fill the fields once every photo is read.
  // Phones hand photos over without their location; they get offered the current one.
  const platform = useMemo(() => platformOf(navigator.userAgent, navigator.maxTouchPoints), []);
  const reading = useRef(new Set<File>());
  useEffect(() => {
    files.filter((f) => !found.has(f) && !reading.current.has(f)).forEach((f) => {
      reading.current.add(f);
      readPhotoPlaceDate(f).then((pd) => setFound((m) => new Map(m).set(f, pd)));
    });
  }, [files, found]);
  useEffect(() => {
    if (files.every((f) => found.has(f))) setWhere((w) => fillFromPhotos(w, files.map((f) => found.get(f))));
  }, [files, found]);

  // The photos go with the request, so results keep pointing at the photos they scored
  // even if the picker changes afterwards.
  // A busy server answers "wait": the photos wait in line and the button says so.
  // The upload's progress (0 to 1) drives the waiting card's first step.
  const [waiting, setWaiting] = useState<number | null>(null);
  const [sentPart, setSentPart] = useState<number | null>(null);
  const run = useMutation({
    mutationFn: (sent: { where: Where; files: File[] }) =>
      api.identify(sent.files, chosen, sent.where, (_attempt, seconds) => setWaiting(seconds), setSentPart),
    onMutate: () => setSentPart(null),
    onSettled: () => {
      setWaiting(null);
      setSentPart(null);
    },
  });
  const sentUrls = useObjectUrls(run.variables?.files ?? NO_FILES);

  // Identify from a link: an iNat or MO observation's own photos, place and date, read
  // by Vision. Whichever was started last, photos or link, is the answer shown.
  // A link here with ?observation= (mycomap.org's "Open in Vision") fills the box.
  const fromQuery = useMemo(() => observationFromQuery(window.location.search), []);
  const [link, setLink] = useState(fromQuery?.text ?? "");
  const [mode, setMode] = useState<"photos" | "link">("photos");
  const byLink = useMutation({
    mutationFn: (ref: ObservationRef) =>
      api.identifyObservation(ref.source, ref.id, chosen, (_attempt, seconds) => setWaiting(seconds)),
    onSettled: () => setWaiting(null),
  });
  const viaLink = mode === "link";
  const answer = viaLink ? byLink.data : run.data;
  const pending = viaLink ? byLink.isPending : run.isPending;
  const failed = viaLink ? byLink.error : run.error;
  const shownUrls = viaLink ? (byLink.data?.observation.photos_shown.map((u) => u ?? "") ?? []) : sentUrls;
  const shownPhotos = viaLink ? byLink.data?.observation.photos_used : run.variables?.files.length;
  const submittedAt = viaLink ? byLink.submittedAt : run.submittedAt;
  const startPhotos = () => { setMode("photos"); byLink.reset(); run.mutate({ where, files }); };
  const startLink = (ref: ObservationRef) => { setMode("link"); run.reset(); byLink.mutate(ref); };
  // ...and identifies it once, when the person is signed in and a model is chosen.
  const autoStarted = useRef(false);
  useEffect(() => {
    if (!fromQuery || autoStarted.current || gate !== "ready" || !chosen.length) return;
    autoStarted.current = true;
    startLink(fromQuery.ref);
  }, [fromQuery, gate, chosen.length]);

  // Start over: this find's photos, place and date, answer and error go; the model stays.
  // Clearing the photos and the request releases their object URLs (useObjectUrls).
  const pickZone = useRef<HTMLDivElement>(null);
  const clearable = canStartOver({
    files, where, hasAnswer: !!answer, hasError: !!failed, identifying: pending,
  }) || !!link.trim();
  const restart = () => {
    const blank = startOver();
    reading.current = new Set();
    setFiles(blank.files);
    setFound(blank.found);
    setWhere(blank.where);
    setWaiting(null);
    setSentPart(null);
    run.reset();
    byLink.reset();
    setLink("");
    setMode("photos");
    window.scrollTo({ top: 0, behavior: "smooth" });
    pickZone.current?.focus({ preventScroll: true });
  };

  // On a phone the waiting card, then the answer, appear below the form: bring each into
  // view (phoneFlow.shouldScrollTo has the rule).
  const results = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!pending && !answer) return;
    const el = results.current?.querySelector<HTMLElement>(
      answer ? "[data-answer-box], [data-result-card]" : "[data-waiting-card]");
    bringIntoView(el ?? null);
  }, [pending, answer]);

  // The pinned Identify bar (phones) hides while the page's own button is on screen.
  const inlineButton = useRef<HTMLDivElement>(null);
  const inlineVisible = useOnScreen(inlineButton);
  const pinned = showPinnedBar({
    hasPhotos: files.length > 0, hasAnswer: !!answer, signInNeeded: gate === "sign-in",
    inlineButtonVisible: inlineVisible,
  });
  const identify = (
    <IdentifyButton count={files.length} pending={run.isPending} waiting={waiting != null && !viaLink}
                    disabled={!files.length || !chosen.length || pending || gate === "checking"}
                    onClick={startPhotos} />
  );
  const referencePhotos = models.data?.backbones.find((b) => chosen[0]?.startsWith(`${b.backbone}/`))
    ?.embedded_photos ?? null;

  return (
    <>
      <PageHeader title="Identify a fungus">
        From all your photos of one find, compared with DNA-sequenced specimens, not community
        votes. Free, from the nonprofit MycoMap.
      </PageHeader>
      <div className={`container mx-auto px-4 sm:px-6 lg:px-8 pt-8 lg:pb-8 grid gap-8 lg:grid-cols-[360px_1fr] ${
        pinned ? "pb-28" : "pb-8"}`}>
        <div className="space-y-6">
          {gate === "sign-in" && <SignInFirst />}
          <PhotoPicker files={files} setFiles={setFiles} zone={pickZone}
                       onStartOver={restart} canStartOver={clearable} />
          <LinkBox value={link} setValue={setLink} pending={byLink.isPending} waiting={viaLink ? waiting : null}
                   disabled={!chosen.length || pending || gate !== "ready"} signIn={gate === "sign-in"}
                   onRun={startLink} />
          <PhotoChecklist hasPhotos={files.length > 0} />
          <WhereWhen where={where} setWhere={setWhere}
                     noPlace={noPhotoPlace({ found: files.map((f) => found.get(f)), where, platform })} />
          <ModelPicker models={models} onChange={setChosen} />
          {gate === "sign-in" ? (
            <Button asChild size="lg" className="w-full" data-testid="button-signin-to-identify">
              <a href={signInUrl()}>Sign in to identify</a>
            </Button>
          ) : (
            <div ref={inlineButton}>{identify}</div>
          )}
          {failed && (
            <p className="text-sm text-destructive">
              {(failed as Error).message}
              {failed instanceof ApiError && failed.status === 401 && (
                <> <a className="underline font-medium" href={signInUrl()}>Sign in</a></>
              )}
            </p>
          )}
        </div>

        <div className="min-w-0 scroll-mt-20" ref={results}>
          {!answer && !pending && <EmptyState />}
          {pending && viaLink && byLink.variables && (
            <LinkWaitingCard observation={byLink.variables} waitSeconds={waiting} />
          )}
          {pending && !viaLink && (
            <WaitingCard urls={sentUrls} steps={waitingSteps({
              photos: run.variables?.files.length ?? files.length, sent: sentPart, waitSeconds: waiting,
              referencePhotos,
            })} />
          )}
          {viaLink && byLink.data && <LinkedObservationLine o={byLink.data.observation} cached={!!byLink.data.cached} />}
          {answer && (
            <div className="mb-4 flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
              <ContextLine used={answer.context_used} sent={viaLink ? undefined : run.variables?.where}
                           placeModel={usesPlace(answer.results.map((r) => r.model.method))} />
              <StartOverButton onClick={restart} disabled={!clearable} testId="button-start-over-top" />
            </div>
          )}
          {answer && (
            <div
              className={`grid gap-6 ${answer.results.length > 1 ? "xl:grid-cols-2" : ""}`}
            >
              {answer.results.map((r) => (
                <ResultCard key={`${modelKey(r.model.backbone, r.model.method)}@${submittedAt}`}
                            r={r} urls={shownUrls} photos={shownPhotos ?? r.photos} />
              ))}
            </div>
          )}
          {answer && <WhatNext results={answer.results} />}
          {answer && (
            <div className="mt-6 flex justify-center">
              <StartOverButton onClick={restart} disabled={!clearable} testId="button-start-over-bottom">
                Start over with another find
              </StartOverButton>
            </div>
          )}
        </div>
      </div>
      {pinned && (
        <div className="lg:hidden fixed inset-x-0 bottom-0 z-40 flex items-center gap-2 border-t border-[#A87146]/15 bg-white/95 px-4 pt-2.5 pb-[calc(0.625rem+env(safe-area-inset-bottom))] shadow-[0_-4px_16px_rgba(74,55,40,0.12)] backdrop-blur"
             data-testid="bar-identify-pinned">
          <div className="flex-1">{identify}</div>
          <Button type="button" variant="outline" size="icon" className="h-11 w-11 shrink-0" onClick={restart}
                  disabled={!clearable} aria-label="Start over" title="Start over"
                  data-testid="button-start-over-pinned">
            <RotateCcw />
          </Button>
        </div>
      )}
    </>
  );
}

/** Identify from a link: one iNat or MO observation, its photos, place and date read by Vision. */
function LinkBox({ value, setValue, pending, waiting, disabled, signIn, onRun }: {
  value: string;
  setValue: (v: string) => void;
  pending: boolean;
  waiting: number | null;
  disabled: boolean;
  signIn: boolean;
  onRun: (ref: ObservationRef) => void;
}) {
  const parsed = parseObservationLink(value);
  const typed = value.trim().length > 0;
  return (
    <div className="space-y-2 rounded-lg border border-[#A87146]/20 bg-white p-4" data-testid="box-link">
      <label htmlFor="observation-link" className="flex items-center gap-1.5 text-sm font-medium text-[#4a3728]">
        <Link2 className="h-4 w-4" aria-hidden /> Or identify from an iNaturalist or Mushroom Observer link
      </label>
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (parsed && !disabled) onRun(parsed); }}>
        <input id="observation-link" type="text" inputMode="url" autoComplete="off" spellCheck={false}
               className="h-9 min-w-0 flex-1 rounded-md border border-input bg-background px-2 text-sm"
               placeholder="inaturalist.org/observations/…" value={value}
               onChange={(e) => setValue(e.target.value)} data-testid="input-link" />
        <Button type="submit" disabled={!parsed || disabled} data-testid="button-identify-link">
          {pending ? <Loader2 className="animate-spin" aria-hidden /> : null}
          {pending ? (waiting != null ? "Waiting…" : "Reading…") : "Identify"}
        </Button>
      </form>
      <p className="text-xs text-muted-foreground" aria-live="polite">
        {signIn ? "Sign in first to identify from a link."
          : typed && !parsed ? "Paste a link to one observation, e.g. inaturalist.org/observations/123 or mushroomobserver.org/123."
            : parsed ? `${sourceName(parsed.source)} observation ${parsed.id}: Vision reads its photos, place and date.`
              : "Vision reads the observation's photos, place and date itself."}
      </p>
    </div>
  );
}

function LinkWaitingCard({ observation, waitSeconds }: { observation: ObservationRef; waitSeconds: number | null }) {
  return (
    <Card className="overflow-hidden scroll-mt-20" data-waiting-card data-testid="card-waiting-link">
      <CardHeader className="bg-[#f8f5f0] border-b border-[#A87146]/10 py-3">
        <CardTitle className="flex items-center gap-2 text-base text-[#4a3728]">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          Reading {sourceName(observation.source)} observation {observation.id}…
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-2 px-4 sm:px-6 text-sm text-muted-foreground">
        <p>
          {waitSeconds != null
            ? `Vision is busy; trying again in ${waitSeconds} s.`
            : "Vision fetches the observation's photos, then compares them with DNA-sequenced specimens."}
        </p>
        <p className="text-xs">
          Usually 10 to 30 seconds{observation.source === "mo" ? "; Mushroom Observer asks for 5 seconds between photos" : ""}.
        </p>
      </CardContent>
    </Card>
  );
}

function LinkedObservationLine({ o, cached }: { o: LinkedObservation; cached: boolean }) {
  const hidden = o.photos_shown.filter((u) => !u).length;
  return (
    <div className="mb-3 rounded-md border border-[#A87146]/20 bg-[#faf9f7] px-3 py-2 text-sm text-[#4a3728]"
         data-testid="linked-observation">
      {o.url
        ? <a href={o.url} target="_blank" rel="noreferrer" className="font-medium underline">{sourceName(o.source)} {o.id}</a>
        : <span className="font-medium">{sourceName(o.source)} {o.id}</span>}
      {o.name_at_source && <> · called <span className="sci">{o.name_at_source}</span> there</>}
      {` · ${o.photos_used} of ${o.photos} photo${o.photos === 1 ? "" : "s"} read`}
      {hidden > 0 && (
        <span className="block text-xs text-muted-foreground">
          {hidden === o.photos_shown.length ? "Its photos aren't" : `${hidden} of its photos aren't`} Creative Commons,
          so they are numbered here rather than shown.
        </span>
      )}
      {cached && (
        <span className="block text-xs text-muted-foreground" data-testid="linked-cached">
          Answered before and kept: Vision's model and reference set haven't changed since.
        </span>
      )}
      {o.excluded_reference && (
        <span className="block text-xs text-amber-700">
          This observation is one of Vision's DNA-checked reference records; it was left out of its own answer.
        </span>
      )}
    </div>
  );
}

/** Scrolls an element that just appeared into view when phoneFlow.shouldScrollTo says so. */
function bringIntoView(el: HTMLElement | null) {
  if (!el) return;
  const header = document.querySelector("header")?.getBoundingClientRect().bottom ?? 64;
  if (!shouldScrollTo({ viewportWidth: window.innerWidth, viewportHeight: window.innerHeight,
                        top: el.getBoundingClientRect().top, header })) return;
  const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  el.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
}

/** True while at least half of the element is on screen, below the sticky header. */
function useOnScreen(ref: RefObject<HTMLElement | null>): boolean {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    // The sticky site header (64 px) covers the top of the screen.
    const io = new IntersectionObserver(([e]) => setOn(e.isIntersecting),
                                        { threshold: 0.5, rootMargin: "-64px 0px 0px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  return on;
}

function IdentifyButton({ count, pending, waiting, disabled, onClick }: {
  count: number;
  pending: boolean;
  waiting: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <Button size="lg" className="w-full" disabled={disabled} onClick={onClick}>
      {pending ? <Loader2 className="animate-spin" /> : <Microscope />}
      {pending
        ? waiting ? "Busy: waiting in line…" : "Identifying…"
        : `Identify from ${count || "your"} photo${count === 1 ? "" : "s"}`}
    </Button>
  );
}

/** Where the answer will appear, while the photos are sent and compared. */
function WaitingCard({ urls, steps }: { urls: string[]; steps: Step[] }) {
  const icon = (st: Step["state"]) =>
    st === "done" ? <Check className="h-4 w-4 text-myco-green" aria-hidden />
      : st === "active" ? <Loader2 className="h-4 w-4 animate-spin text-myco-brown" aria-hidden />
        : <Circle className="h-3.5 w-3.5 text-muted-foreground/50" aria-hidden />;
  return (
    <Card className="overflow-hidden scroll-mt-20" data-waiting-card data-testid="card-waiting">
      <CardHeader className="bg-[#f8f5f0] border-b border-[#A87146]/10 py-3">
        <CardTitle className="flex items-center gap-2 text-base text-[#4a3728]">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          Identifying your {urls.length === 1 ? "photo" : `${urls.length} photos`}…
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-4 px-4 sm:px-6">
        <div className="flex flex-wrap gap-2">
          {urls.map((u, i) => (
            <img key={u} src={u} alt={`Your photo ${i + 1}`} className="h-14 w-14 rounded-md border object-cover" />
          ))}
        </div>
        <ol className="space-y-2 text-sm" aria-live="polite">
          {steps.map((st) => (
            <li key={st.key} className="flex items-start gap-2" data-testid={`step-${st.key}`} data-state={st.state}>
              <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">{icon(st.state)}</span>
              <span className="min-w-0">
                <span className={st.state === "active" ? "font-semibold text-[#4a3728]"
                  : st.state === "done" ? "text-[#5c4a3a]" : "text-muted-foreground"}>
                  {st.label}
                </span>
                {st.detail && <span className="ml-2 text-muted-foreground tabular-nums">{st.detail}</span>}
              </span>
            </li>
          ))}
        </ol>
        <p className="text-xs text-muted-foreground">Usually 5 to 10 seconds.</p>
        <div className="space-y-2" aria-hidden>
          {["w-1/3 h-2", "w-2/3 h-3.5", "w-full h-2", "w-1/2 h-2", "w-5/6 h-3.5", "w-full h-2"].map((c, i) => (
            <div key={i} className={`${c} rounded-full bg-muted motion-safe:animate-pulse`} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

const NO_FILES: File[] = [];

/** Object URLs for local files, released when the files change or the page closes. */
function useObjectUrls(files: File[]): string[] {
  const urls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => urls.forEach((u) => URL.revokeObjectURL(u)), [urls]);
  return urls;
}

/** Clears the photos, place and date, and the answer, for the next find. */
function StartOverButton({ onClick, disabled, testId, children = "Start over" }: {
  onClick: () => void;
  disabled?: boolean;
  testId: string;
  children?: ReactNode;
}) {
  return (
    <Button type="button" variant="outline" size="sm" onClick={onClick} disabled={disabled}
            data-testid={testId}>
      <RotateCcw /> {children}
    </Button>
  );
}

function PhotoPicker({ files, setFiles, zone, onStartOver, canStartOver }: {
  files: File[];
  setFiles: (f: File[]) => void;
  zone: RefObject<HTMLDivElement | null>;
  onStartOver: () => void;
  canStartOver: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const urls = useObjectUrls(files);
  const add = (list: FileList | null) => {
    if (!list) return;
    const images = Array.from(list).filter((f) => f.type.startsWith("image/"));
    setFiles([...files, ...images].slice(0, MAX_PHOTOS));
  };
  return (
    <div>
      <div
        ref={zone}
        role="button"
        tabIndex={0}
        aria-label="Add photos"
        className={`rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-myco-green ${
          over ? "border-myco-green bg-myco-green/5" : "border-[#A87146]/30 hover:border-myco-green"
        }`}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            input.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          add(e.dataTransfer.files);
        }}
      >
        <ImagePlus className="mx-auto h-8 w-8 text-myco-brown" />
        <p className="mt-2 font-medium text-[#4a3728]">Add photos</p>
        <p className="text-xs text-muted-foreground">
          <span className="sm:hidden">Take or choose</span>
          <span className="hidden sm:inline">Drop or choose</span> up to {MAX_PHOTOS}, all of the same find
        </p>
        <input
          ref={input}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            add(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {files.length > 0 && (
        <div className="mt-3 grid grid-cols-3 sm:grid-cols-4 gap-2">
          {urls.map((u, i) => (
            <div key={u} className="relative aspect-square rounded-md overflow-hidden border">
              <img src={u} alt={`Photo ${i + 1}`} className="h-full w-full object-cover" />
              <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 text-[10px] text-white">
                {i + 1}
              </span>
              <button
                type="button"
                className="absolute top-0 right-0 flex h-9 w-9 items-start justify-end p-1 text-white"
                onClick={() => setFiles(files.filter((_, j) => j !== i))}
                aria-label={`Remove photo ${i + 1}`}
              >
                <span className="rounded-full bg-black/60 p-1 hover:bg-black/80"><X className="h-3.5 w-3.5" /></span>
              </button>
            </div>
          ))}
        </div>
      )}
      {files.length > 0 && (
        <div className="mt-2 flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">
            {files.length} of {MAX_PHOTOS} photos
          </span>
          <StartOverButton onClick={onStartOver} disabled={!canStartOver} testId="button-start-over-picker" />
        </div>
      )}
    </div>
  );
}

function WhereWhen({ where, setWhere, noPlace }: {
  where: Where;
  setWhere: Dispatch<SetStateAction<Where>>;
  /** Set on a phone whose photos carry no place: offer the current location. */
  noPlace: { tip: string } | null;
}) {
  const [locating, setLocating] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  // The map sheet; focus goes back to the button that opened it, or to the one by the
  // fields when that's gone (the offer box closes once a place is set).
  const [picking, setPicking] = useState(false);
  const opener = useRef<HTMLElement | null>(null);
  const pickButton = useRef<HTMLButtonElement>(null);
  const openMap = (e: { currentTarget: HTMLElement }) => {
    opener.current = e.currentTarget;
    setPicking(true);
  };
  const closeMap = () => {
    setPicking(false);
    setTimeout(() => (opener.current?.isConnected ? opener.current : pickButton.current)?.focus(), 0);
  };
  const locate = () => {
    if (!navigator.geolocation) return setFailed(locateError(2));
    setLocating(true);
    setFailed(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        // Applied to the form as it is when the position arrives, not as it was when asked.
        setWhere((w) => ({ ...w, lat: pos.coords.latitude.toFixed(4), lng: pos.coords.longitude.toFixed(4),
                           placeFromPhoto: undefined, placeFromDevice: true, placeFromMap: undefined }));
        setLocating(false);
      },
      (err) => {
        setLocating(false);
        setFailed(locateError(err.code));
      },
      { timeout: 15_000, maximumAge: 60_000 },
    );
  };
  const note = placeNote(where);
  const input = "h-9 w-full rounded-md border border-input bg-background px-2 text-sm";
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Where and when</CardTitle>
        <p className="text-sm text-muted-foreground">
          Optional. Filled in from your photos when they carry a place or date; change them if
          they're wrong.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {noPlace && (
          <div className="rounded-md border border-myco-green/40 bg-myco-green/5 p-3 space-y-2"
               data-testid="box-no-photo-place">
            <p className="text-sm font-medium text-[#4a3728]">Your photos don't say where they were taken.</p>
            <Button type="button" className="w-full" onClick={locate} disabled={locating}
                    data-testid="button-use-current-location">
              {locating ? <Loader2 className="animate-spin" /> : <LocateFixed />} Use my current location
            </Button>
            <Button type="button" variant="outline" className="w-full bg-white" onClick={openMap}
                    data-testid="button-pick-on-map-offer">
              <MapIcon /> Pick on map
            </Button>
            <p className="text-xs text-muted-foreground">
              Current location only if you're still where you found it. Otherwise pick it on the
              map, type its coordinates below, or leave them empty.
            </p>
            <p className="text-xs text-muted-foreground" data-testid="text-no-place-tip">{noPlace.tip}</p>
          </div>
        )}
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-myco-brown shrink-0" />
          <input className={`${input} min-w-0`} placeholder="Latitude" inputMode="decimal" value={where.lat ?? ""}
                 onChange={(e) => setWhere({ ...where, lat: e.target.value, placeFromPhoto: undefined, placeFromDevice: undefined, placeFromMap: undefined })} aria-label="Latitude" />
          <input className={`${input} min-w-0`} placeholder="Longitude" inputMode="decimal" value={where.lng ?? ""}
                 onChange={(e) => setWhere({ ...where, lng: e.target.value, placeFromPhoto: undefined, placeFromDevice: undefined, placeFromMap: undefined })} aria-label="Longitude" />
          <Button type="button" variant="outline" size="icon" className="h-10 w-10 shrink-0" onClick={locate}
                  disabled={locating} title="Use my location" aria-label="Use my location">
            {locating ? <Loader2 className="animate-spin" /> : <LocateFixed />}
          </Button>
        </div>
        {!noPlace && (
          <Button ref={pickButton} type="button" variant="outline" size="sm" onClick={openMap}
                  data-testid="button-pick-on-map">
            <MapIcon /> Pick on map
          </Button>
        )}
        <div className="flex items-center gap-2">
          <CalendarDays className="h-4 w-4 text-myco-brown shrink-0" />
          <input className={input} type="date" value={where.observedOn ?? ""}
                 onChange={(e) => setWhere({ ...where, observedOn: e.target.value, dateFromPhoto: undefined })} aria-label="Date found" />
        </div>
        {failed && <p className="text-xs text-destructive" data-testid="text-locate-failed">{failed}</p>}
        {note && <p className="text-xs text-muted-foreground" data-testid="text-place-source">{note}</p>}
      </CardContent>
      {picking && (
        <Suspense fallback={
          <div className="fixed inset-0 z-[60] grid place-items-center bg-black/40" role="status" aria-label="Loading the map">
            <Loader2 className="h-8 w-8 animate-spin text-white" />
          </div>
        }>
          <LocationPicker where={where} onClose={closeMap}
                          onUse={(pin) => { setWhere((w) => applyPick(w, pin)); closeMap(); }} />
        </Suspense>
      )}
    </Card>
  );
}

function ContextLine({ used, sent, placeModel }: { used: ContextUsed; sent?: Where; placeModel: boolean }) {
  if (!used.place_from && !used.date_from) {
    return (
      <p className="min-w-0 flex-1 text-sm text-muted-foreground">
        {placeModel
          ? "No place or date given or found in the photos, so models with a range-and-season score used the photos alone."
          : "From the photos alone (no place or date given)."}
      </p>
    );
  }
  // Places and dates filled in from a photo reach the server as entered; say where they came from.
  const from = (f: ContextUsed["place_from"], photo?: number, device?: boolean, map?: boolean) =>
    sourceWords(f, photo, device, map);
  return (
    <p className="min-w-0 flex-1 text-sm text-muted-foreground flex flex-wrap gap-x-4 gap-y-1">
      {used.place_from && (
        <span className="inline-flex items-center gap-1">
          <MapPin className="h-3.5 w-3.5" /> near {used.latitude}, {used.longitude} ({from(used.place_from, sent?.placeFromPhoto, sent?.placeFromDevice, sent?.placeFromMap)})
        </span>
      )}
      {used.date_from && (
        <span className="inline-flex items-center gap-1">
          <CalendarDays className="h-3.5 w-3.5" /> {used.observed_on} ({from(used.date_from, sent?.dateFromPhoto)})
        </span>
      )}
    </p>
  );
}

const selectClass =
  "w-full rounded-md border bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[hsl(var(--myco-green))]";

/**
 * One scoring method (default: the trained classifier, where offered) with two
 * add-on switches, and optionally a second model to compare side by side (the page
 * never runs more than MAX_COMPARED, nor the server's max_models). Reports the
 * backbone/method keys to run.
 */
function ModelPicker({ models, onChange }: {
  models: UseQueryResult<ModelsInfo>;
  onChange: (keys: string[]) => void;
}) {
  const offered = models.data?.methods ?? [];
  const ready = models.data?.ready ?? [];
  const bases = offeredBases(offered);
  const [backbone, setBackbone] = useState<string | null>(null);
  const [choice, setChoice] = useState<Choice | null>(null);
  const [comparing, setComparing] = useState(false);
  const [second, setSecond] = useState<{ backbone: string; method: string } | null>(null);

  // Defaults once the server says what it offers.
  useEffect(() => {
    if (!backbone && ready.length) setBackbone(ready[0]);
    if (!choice && offered.length) setChoice(defaultChoice(offered, models.data?.default_method));
  }, [ready, offered, backbone, choice]);

  const method = choice ? resolveMethod(choice, offered) : null;
  const first = backbone && method ? modelKey(backbone, method) : null;
  // The second model starts as something different from the first: another method
  // on the same backbone, else the same method on another backbone.
  useEffect(() => {
    if (!comparing || second || !backbone || !method) return;
    const otherMethod = offered.find((m) => m !== method);
    const otherBackbone = ready.find((b) => b !== backbone);
    if (otherMethod) setSecond({ backbone, method: otherMethod });
    else if (otherBackbone) setSecond({ backbone: otherBackbone, method });
  }, [comparing, second, backbone, method, offered, ready]);
  const secondKey = comparing && second ? modelKey(second.backbone, second.method) : null;
  const keys = [first, secondKey].filter((k, i, all): k is string => !!k && all.indexOf(k) === i);
  useEffect(() => onChange(keys), [keys.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps

  const switches = choice ? switchesFor(choice.base, offered) : { weighNearest: false, usePlace: false };
  const cap = Math.min(MAX_COMPARED, models.data?.max_models ?? MAX_COMPARED);
  const canCompare = cap >= 2 && ready.length * offered.length > 1;
  const info = models.data?.backbones.find((b) => b.backbone === backbone);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {models.isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {models.isError && (
          <p className="text-sm text-destructive">
            The identifier isn't answering. Please try again in a minute.
          </p>
        )}
        {models.data && (!ready.length || !bases.length) && (
          <p className="text-sm text-muted-foreground">
            No model is ready yet. Please try again in a minute.
          </p>
        )}
        {choice && ready.length > 0 && (
          <fieldset className="space-y-3">
            {ready.length > 1 && (
              <select className={selectClass} value={backbone ?? ""} onChange={(e) => setBackbone(e.target.value)}
                aria-label="Backbone" data-testid="select-backbone">
                {ready.map((b) => <option key={b} value={b}>{publicModelName(b)}</option>)}
              </select>
            )}
            <select className={selectClass} value={choice.base} aria-label="Model"
              onChange={(e) => setChoice({ ...choice, base: e.target.value as Base })} data-testid="select-method">
              {bases.map((b) => <option key={b} value={b}>{BASE_LABEL[b]}</option>)}
            </select>
            {switches.weighNearest && (
              <label className="flex items-start gap-2 text-sm cursor-pointer">
                <input type="checkbox" className="mt-1 accent-[hsl(var(--myco-green))]" checked={choice.weighNearest}
                  onChange={(e) => setChoice({ ...choice, weighNearest: e.target.checked })} data-testid="switch-nearest" />
                <span>Also weigh the nearest specimens</span>
              </label>
            )}
            {switches.usePlace && (
              <label className="flex items-start gap-2 text-sm cursor-pointer">
                <input type="checkbox" className="mt-1 accent-[hsl(var(--myco-green))]" checked={choice.usePlace}
                  onChange={(e) => setChoice({ ...choice, usePlace: e.target.checked })} data-testid="switch-place" />
                <span>Use place and date (range and season)</span>
              </label>
            )}
            {info && (
              <p className="text-xs text-muted-foreground">
                {publicModelName(info.backbone)} · {num(info.embedded_photos)} reference photos
              </p>
            )}
          </fieldset>
        )}
        {canCompare && (
          <details className="pt-1" open={comparing} onToggle={(e) => setComparing((e.target as HTMLDetailsElement).open)}>
            <summary className="cursor-pointer text-sm text-muted-foreground" data-testid="toggle-compare">
              Compare with a second model
            </summary>
            {second && (
              <div className="mt-2 space-y-2">
                {ready.length > 1 && (
                  <select className={selectClass} value={second.backbone} aria-label="Second model backbone"
                    onChange={(e) => setSecond({ ...second, backbone: e.target.value })} data-testid="select-second-backbone">
                    {ready.map((b) => <option key={b} value={b}>{publicModelName(b)}</option>)}
                  </select>
                )}
                <select className={selectClass} value={second.method} aria-label="Second model"
                  onChange={(e) => setSecond({ ...second, method: e.target.value })} data-testid="select-second-method">
                  {offered.map((m) => <option key={m} value={m}>{capitalize(publicMethodLabel(m))}</option>)}
                </select>
                {secondKey === first && (
                  <p className="text-xs text-muted-foreground">Same as the first model; pick another to compare.</p>
                )}
              </div>
            )}
          </details>
        )}
      </CardContent>
    </Card>
  );
}

/** Shown before anything else when identifying needs a mycomap.org account. */
function SignInFirst() {
  return (
    <Card className="border-myco-green/40 bg-myco-green/5" data-testid="card-signin-first">
      <CardContent className="pt-5 space-y-3 text-sm text-[#5c4a3a]">
        <p className="font-semibold text-[#4a3728] text-base">Sign in to identify</p>
        <p>
          Identifying uses your free mycomap.org account. Sign in first: you come straight back
          here to add your photos.
        </p>
        <Button asChild className="w-full">
          <a href={signInUrl()}>Sign in with mycomap.org</a>
        </Button>
      </CardContent>
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-[#A87146]/20 bg-[#faf9f7] p-8 text-[#5c4a3a]">
      <h2 className="font-display font-semibold text-2xl text-[#4a3728]">What you'll get</h2>
      <ul className="mt-4 space-y-2 text-sm list-disc pl-5">
        <li>
          An answer at every rank (family, genus, species), each with its own confidence. Often
          the genus is clear when the species isn't.
        </li>
        <li>
          Species known so far only from DNA, under provisional names, that identifiers trained
          on community names can't suggest.
        </li>
        <li>The closest DNA-verified specimens, with their photos, so you can compare by eye.</li>
        <li>How each of your photos scores on its own, to see which ones agree.</li>
        <li>Advice on what would firm up the identification.</li>
        <li>Optionally a second model side by side, to see where they agree.</li>
      </ul>
      <p className="mt-4 text-xs">
        A preview. Every answer comes from DNA-verified MycoMap records, and its confidence is
        calibrated on the newest ones; see <a className="underline" href="/about">how it works and
        how good it is</a>.
      </p>
      <p className="mt-2 text-xs font-semibold text-[#4a3728]">{SAFETY_LINE}</p>
    </div>
  );
}

/** Open until photos are chosen, then folded to one line (it can still be opened). */
function PhotoChecklist({ hasPhotos }: { hasPhotos: boolean }) {
  const [open, setOpen] = useState(!hasPhotos);
  useEffect(() => setOpen(!hasPhotos), [hasPhotos]);
  return (
    <details className="group rounded-lg border border-[#A87146]/20 bg-[#faf9f7] p-4" data-testid="list-photo-checklist"
             open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="flex cursor-pointer list-none items-center justify-between text-sm text-[#4a3728] [&::-webkit-details-marker]:hidden">
        <span>
          <span className="font-semibold">What to photograph</span>
          <span className="text-muted-foreground group-open:hidden"> · {PHOTO_CHECKLIST.length} views</span>
        </span>
        <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-90" aria-hidden />
      </summary>
      <ul className="mt-2 space-y-1.5 text-sm text-[#5c4a3a]">
        {PHOTO_CHECKLIST.map((c) => (
          <li key={c.what} className="flex gap-2">
            <Camera className="h-4 w-4 mt-0.5 shrink-0 text-myco-brown" aria-hidden />
            <span>
              <span className="font-medium">{c.what}</span>
              <span className="text-muted-foreground">: {c.why}</span>
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-muted-foreground">
        One find per identification. Sharp, close and in daylight works best.
      </p>
    </details>
  );
}

/** After an answer: how to go from a suggestion to a name you can rely on (gate.whatNext
 *  words it for photos that may not show a fungus). */
function WhatNext({ results }: { results: IdentifyResult[] }) {
  const next = whatNext(results);
  return (
    <div className="mt-6 rounded-lg border border-[#A87146]/20 bg-[#faf9f7] p-4 text-sm text-[#5c4a3a]"
         data-testid="card-what-next">
      <p className="font-semibold text-[#4a3728]">{next.title}</p>
      <ol className="mt-2 list-decimal pl-5 space-y-1">
        {next.steps.map((step) => {
          const [before, after] = step.split("{sequencing}");
          return (
            <li key={step}>
              {before}
              {after !== undefined && (
                <>
                  <a className="underline" href={SEQUENCING_URL} target="_blank" rel="noreferrer">
                    free DNA sequencing
                  </a>
                  {after}
                </>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function ResultCard({ r, urls, photos }: { r: IdentifyResult; urls: string[]; photos: number }) {
  // null: all photos together (the main answer). A number: that photo on its own.
  const [view, setView] = useState<number | null>(null);
  const [details, setDetails] = useState(false);
  const [everyRank, setEveryRank] = useState(false);
  const photo = view == null ? null : (r.per_photo.find((p) => p.photo === view) ?? null);
  const ranks = photo ? photo.ranks : r.ranks;
  const specimens = photo ? photo.specimens : r.specimens;
  // The not-a-fungus gate (lib/gate.ts): its notice, and whether the chance is withheld.
  const banner = gateBanner(r);
  const gate = { notFungus: banner?.verdict === "not_fungus", withheld: chanceWithheld(r) };
  const answer = answerFor(r, gate);
  const hints = pageHints(r);
  const noRanks = gate.notFungus || (photo != null && leftOut(photo));
  const found = gate.withheld ? "look-alikes" : "specimens";
  const coverage = coverageSummary(r.likely);
  // Every rank's list is detail once there is an answer; without one (or for one photo) it is the answer.
  // The open/closed state is kept while switching photos.
  const showRanks = breakdownOpen(everyRank, answer != null, noRanks);
  return (
    <Card className="overflow-hidden scroll-mt-20" data-result-card>
      <CardContent className="p-0">
        {/* The gate's notice goes first, above everything else in the card. */}
        {banner && <div className="px-4 pt-4 sm:px-6"><GateNotice banner={banner} /></div>}
        {answer && <AnswerHero a={answer} r={{ ...r, hints }} photos={photos} />}
        <div className="space-y-6 px-4 py-5 sm:px-6">
          {answer && <SafetyNote />}
          {r.per_photo.length > 1 && (
            <PhotoSwitcher r={r} urls={urls} view={view} setView={setView} />
          )}
          {photo && <PhotoVerdict p={photo} />}
          <div className="space-y-4">
          {!noRanks && answer && (
            <button type="button" onClick={() => setEveryRank(!everyRank)} aria-expanded={everyRank}
                    className="flex w-full items-center justify-between rounded-md border border-[#A87146]/20 px-3 py-2 text-sm text-[#5c4a3a] hover:bg-[#faf9f7]"
                    data-testid="button-every-rank">
              <span>{breakdownLabel(r.per_photo.length, photo?.photo ?? null)}</span>
              <ChevronRight className={`h-4 w-4 transition-transform ${everyRank ? "rotate-90" : ""}`} aria-hidden />
            </button>
          )}
          {showRanks && (
            <div className="space-y-5">
              {RANKS.map((rank) => (
                photo == null && r.likely?.[rank]?.names.length
                  ? <LikelyBlock key={rank} rank={rank} set={r.likely[rank]!} candidates={ranks[rank]} />
                  : <RankBlock key={rank} rank={rank} candidates={ranks[rank]} withheld={gate.withheld} />
              ))}
              <p className="text-xs text-muted-foreground" data-testid="text-coverage">
                {photo
                  ? "One photo on its own is a rougher guide than all of them together."
                  : gate.withheld
                    ? "No chance is stated: it is calibrated on photos of fungi only."
                    : coverage
                      ? `${coverage} ${LIKELY_COPY.noReference}`
                      : publicNote(r.confidence_note)}
              </p>
            </div>
          )}
          </div>
          {!photo && !answer && hints.length > 0 && <Hints hints={hints} />}
          {specimens.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-[#4a3728] mb-2">
                {photo
                  ? `Closest DNA-verified ${found} to photo ${photo.photo + 1}`
                  : `Closest DNA-verified ${found}`}
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {specimens.map((s) => (
                  <SpecimenCard key={s.observation_id} s={s} urls={urls} showMatched={!photo} />
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-t border-[#A87146]/10 bg-[#faf9f7] px-4 py-2.5 text-xs text-muted-foreground sm:px-6">
          <span>{publicModelName(r.model.backbone)} · {publicMethodLabel(r.model.method)}</span>
          <button type="button" className="underline decoration-dotted underline-offset-2 hover:text-myco-green"
                  aria-expanded={details} onClick={() => setDetails(!details)} data-testid="button-model-details">
            {details ? "Hide details" : "About this answer"}
          </button>
          {details && (
            <div className="w-full space-y-1 pt-1" data-testid="text-model-details">
              <p>
                {num(r.reference.records)} DNA-verified records · {num(r.reference.species)} names ·{" "}
                {num(r.reference.photos)} photos
              </p>
              <p>{publicNote(r.confidence_note)}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/** The not-a-fungus gate's verdict on the whole find. */
function GateNotice({ banner }: { banner: GateBanner }) {
  const strong = banner.verdict === "not_fungus";
  return (
    <div
      role="status"
      data-testid={`gate-${banner.verdict}`}
      className={`flex gap-3 rounded-md border p-3 text-sm ${
        strong ? "border-amber-300 bg-amber-50 text-amber-900" : "border-[#A87146]/30 bg-[#fbf7f0] text-[#5c4a3a]"
      }`}
    >
      <CircleHelp className={`h-5 w-5 shrink-0 ${strong ? "text-amber-600" : "text-myco-tan"}`} aria-hidden />
      <div className="space-y-1">
        <p className="font-semibold">{banner.title}</p>
        <p>{banner.body}</p>
      </div>
    </div>
  );
}

function Hints({ hints }: { hints: string[] }) {
  return (
    <div className="rounded-md border border-myco-tan-light/50 bg-[#fbf7f0] p-3 space-y-1.5" data-testid="list-hints">
      {hints.map((h) => (
        <p key={h} className="flex gap-2 text-sm text-[#5c4a3a]">
          <Lightbulb className="h-4 w-4 mt-0.5 shrink-0 text-myco-tan" />
          {h}
        </p>
      ))}
    </div>
  );
}

/** What we know about one name, one line each, strongest first: what DNA says it is often
 *  confused with, then the place cue. `size` "xs": a name still in play (smaller, warnings only). */
function Facts({ n, extra, size = "sm" }: { n: Named; extra?: string | null; size?: "sm" | "xs" }) {
  const confused = n.rank === "species" ? confusedNote(n) : null;
  const far = placeWarns(n.place);
  const place = placeLines(n.place, size === "xs");
  if (!confused && !place.length && !extra) return null;
  const text = size === "sm" ? "text-sm" : "text-xs";
  return (
    <ul className={`mt-2 space-y-1 ${text}`} data-testid={`facts-${n.rank}`}>
      {confused && (
        <li className="flex items-start gap-2 text-[#8a4b0f]" data-testid="text-confused-with">
          <Dna className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>
            {confused}{" "}
            <a href="https://mycomap.org/misid" target="_blank" rel="noreferrer"
               className="whitespace-nowrap underline decoration-dotted underline-offset-2">how we know</a>
          </span>
        </li>
      )}
      {place.map((l) => (
        <li key={l} className={`flex items-start gap-2 ${far ? "text-[#8a4b0f]" : "text-[#5c4a3a]"}`}>
          <MapPin className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />{l}
        </li>
      ))}
      {extra && (
        <li className="flex items-start gap-2 text-[#5c4a3a]">
          <Camera className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />{extra}
        </li>
      )}
    </ul>
  );
}

/** The top of each result, and the first thing the eye should land on: the finest settled
 *  rank's name, its chance, the ranks above it, what we know about it, and what is still in play. */
function AnswerHero({ a, r, photos }: { a: Answer; r: IdentifyResult; photos: number }) {
  const italic = (rank: Rank) => (rank !== "family" ? "sci" : "");
  const h = a.headline;
  return (
    <section className="border-b border-[#4f7a43]/15 bg-[#f1f6ec] px-4 py-5 sm:px-6 scroll-mt-20 text-[#4a3728]"
             data-answer-box data-testid="box-answer">
      <p className="text-xs font-semibold uppercase tracking-wider text-[#4f7a43]">
        {a.lookAlike
          ? "Closest DNA-verified look-alike"
          : `Best answer from your ${photos === 1 ? "photo" : `${photos} photos`}`}
      </p>
      {h ? (
        <>
          <div className="mt-1 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <NameLink rank={h.rank} name={h.name}
                      className={`text-3xl font-semibold leading-tight text-[#2f2419] ${italic(h.rank)}`} />
            {h.confidence != null && (
              <span className="rounded-full bg-[#4f7a43] px-3 py-1 text-lg font-semibold text-white tabular-nums"
                    data-testid="text-answer-chance">
                {pct(h.confidence)}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-[#5c4a3a]">
            {RANK_LABEL[h.rank]}
            {a.above.map((n) => (
              <span key={n.rank}>
                {" · "}{RANK_LABEL[n.rank]} <span className={italic(n.rank)}>{n.name}</span>
                {n.confidence != null && ` ${pct(n.confidence)}`}
              </span>
            ))}
            {a.lookAlike && " · no chance is stated for these photos"}
          </p>
          <Facts n={h} extra={photoOnlyNote(r, h)} />
        </>
      ) : (
        <p className="mt-1 text-2xl font-semibold text-[#2f2419]">No firm answer at any rank yet</p>
      )}
      {a.inPlay && (
        <div className={h ? "mt-4 border-t border-[#4f7a43]/15 pt-4" : "mt-3"} data-testid="text-in-play">
          <p className="text-sm font-semibold text-[#4a3728]">
            {h ? `${RANK_LABEL[a.inPlay.rank]} not settled. Most likely:` : `${RANK_LABEL[a.inPlay.rank]} most likely:`}
          </p>
          <ul className="mt-2 space-y-3">
            {a.inPlay.names.map((n) => (
              <li key={n.name}>
                <div className="flex items-baseline justify-between gap-3">
                  <NameLink rank={n.rank} name={n.name}
                            className={`text-lg font-medium text-[#2f2419] ${italic(n.rank)}`} />
                  {n.confidence != null && (
                    <span className="text-base font-semibold text-[#4f7a43] tabular-nums">{pct(n.confidence)}</span>
                  )}
                </div>
                <Facts n={n} size="xs" />
              </li>
            ))}
          </ul>
          {a.inPlay.more > 0 && (
            <p className="mt-2 text-xs text-muted-foreground">and {a.inPlay.more} more in the full breakdown</p>
          )}
        </div>
      )}
      {r.hints.length > 0 && (
        <div className="mt-4 space-y-1.5 rounded-md bg-white/70 p-3 text-sm text-[#5c4a3a]" data-testid="list-hints">
          {boxHints(r.hints, a.inPlay?.rank === "species").map((t) => (
            <p key={t} className="flex gap-2">
              <Lightbulb className="h-4 w-4 mt-0.5 shrink-0 text-myco-tan" aria-hidden />
              {t}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}

/** "Often confused with …" under a species name in the lists. */
function ConfusedLine({ c }: { c: { confused_with?: Candidate["confused_with"] } }) {
  const note = confusedNote(c);
  if (!note) return null;
  return (
    <p className="mt-0.5 flex items-start gap-1 text-xs text-[#8a4b0f]" data-testid="text-confused-with">
      <Dna className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />{note}
    </p>
  );
}

const ordinal = (n: number) => {
  const v = n % 100;
  return `${n}${v >= 11 && v <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th"}`;
};

/** All photos together, or one photo on its own. Each tile shows how that photo alone
 *  rates the overall species answer, so a photo pulling the other way stands out. */
function PhotoSwitcher({ r, urls, view, setView }: {
  r: IdentifyResult;
  urls: string[];
  view: number | null;
  setView: (v: number | null) => void;
}) {
  const top = r.ranks.species[0]?.name;
  const tile = (on: boolean) =>
    `flex shrink-0 flex-col items-center gap-1 rounded-md p-1 transition-colors ${
      on ? "bg-myco-green/10 ring-2 ring-myco-green" : "hover:bg-muted"
    }`;
  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">
        {top ? (
          <>
            Under each photo: how it rates <span className="sci">{top}</span> on its own. Pick a
            photo to see its own results.
          </>
        ) : (
          "Pick a photo to see its own results."
        )}
      </p>
      <div className="flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Results for">
        <button type="button" role="tab" aria-selected={view == null} className={tile(view == null)}
                onClick={() => setView(null)} aria-label="All photos combined">
          <span className="flex h-14 w-14 flex-col items-center justify-center rounded border border-[#A87146]/30 bg-[#f8f5f0] text-[#4a3728] leading-tight">
            <span className="text-sm font-semibold">All {r.per_photo.length}</span>
            <span className="text-[10px]">photos</span>
          </span>
          <span className="text-[11px] text-muted-foreground">combined</span>
        </button>
        {r.per_photo.map((p) => {
          const place = p.overall_top.species;
          const agrees = place?.position === 1;
          const label = photoGateLabel(p);
          return (
            <button
              key={p.photo}
              type="button"
              role="tab"
              aria-selected={view === p.photo}
              className={tile(view === p.photo)}
              onClick={() => setView(p.photo)}
              title={leftOut(p)
                ? `Photo ${p.photo + 1} doesn't look like a fungus and was left out`
                : place ? `Photo ${p.photo + 1} on its own puts ${place.name} ${ordinal(place.position)}` : undefined}
            >
              <span className="relative h-14 w-14 overflow-hidden rounded border bg-muted">
                {urls[p.photo] && (
                  <img src={urls[p.photo]} alt={`Your photo ${p.photo + 1}`} className="h-full w-full object-cover" />
                )}
                <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 text-[10px] text-white">
                  {p.photo + 1}
                </span>
              </span>
              {label ? (
                <span className="text-[11px] font-medium text-amber-700" data-testid={`photo-gate-${p.photo}`}>
                  {label}
                </span>
              ) : (
                <span className={`text-[11px] tabular-nums ${agrees ? "font-semibold text-myco-green" : "text-muted-foreground"}`}>
                  {place ? (place.confidence == null ? `#${place.position}` : pct(place.confidence)) : "–"}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PhotoVerdict({ p }: { p: PhotoResult }) {
  const species = p.overall_top.species;
  const genus = p.overall_top.genus;
  const n = p.photo + 1;
  if (leftOut(p)) {
    return (
      <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
        <span className="font-semibold">Photo {n} doesn't look like a fungus,</span> so it was left
        out of the answer.
      </div>
    );
  }
  const chance = (c: number | null) => (c == null ? "" : ` (${pct(c)})`);
  return (
    <div className="rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-3 text-sm text-[#5c4a3a]">
      <span className="font-semibold text-[#4a3728]">Photo {n} on its own.</span>{" "}
      {species &&
        (species.position === 1 ? (
          <>
            It agrees with the answer from all photos, <span className="sci">{species.name}</span>
            {chance(species.confidence)}.
          </>
        ) : (
          <>
            It puts <span className="sci">{species.name}</span>, the answer from all photos,{" "}
            {ordinal(species.position)}{chance(species.confidence)}.
          </>
        ))}
      {genus && genus.position > 1 && (
        <>
          {" "}At genus level it puts <span className="sci">{genus.name}</span>{" "}
          {ordinal(genus.position)}.
        </>
      )}
    </div>
  );
}


/** A rank's likely list first: every name worth considering with its calibrated chance,
 *  and how often such a list held the right name in testing. */
/** One rank in the full breakdown: the names worth considering, each with its bar (the top one
 *  strongest), then the rest in one line. `fitted`: the list is the model's likely set (its
 *  coverage was measured); otherwise the names at 5% or more. Names under 1% are left out. */
function LikelyBlock({ rank, set, candidates, fitted = true }: {
  rank: Rank; set: LikelySet; candidates: Candidate[]; fitted?: boolean;
}) {
  const italic = rank !== "family";
  const others = alsoPossible(candidates, set.names);
  const plural = set.names.length > 1;
  return (
    <div data-testid={`likely-${rank}`}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs uppercase tracking-wider text-muted-foreground">
          {RANK_LABEL[rank]}{plural ? (fitted ? ": likely names" : ": most likely") : ""}
        </span>
        <span className="text-xs text-muted-foreground">calibrated chance</span>
      </div>
      <div className="mt-1 space-y-1.5">
        {set.names.map((n, i) => (
          <div key={n.name}>
            <div className="flex items-baseline justify-between gap-3">
              <NameLink rank={rank} name={n.name}
                className={`${i === 0 ? "text-lg font-semibold" : "text-base font-medium"} text-[#4a3728] ${italic ? "sci" : ""}`} />
              <span className={`${i === 0 ? "text-lg" : "text-sm"} font-semibold text-myco-green tabular-nums`}>
                {pct(n.confidence)}
              </span>
            </div>
            <Bar value={n.confidence} strong={i === 0} />
            {rank === "species" && <ConfusedLine c={n} />}
          </div>
        ))}
      </div>
      {/* How often lists like this held the right name is said once, under all the ranks. */}
      {set.capped && <p className="mt-2 text-xs text-muted-foreground">{LIKELY_COPY.capped}</p>}
      {others.length > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          {LIKELY_COPY.alsoPossible}{" "}
          {others.map((c, i) => (
            <span key={c.name}>
              {i > 0 && ", "}
              <span className={italic ? "sci" : ""}>{c.name}</span> {pct(c.confidence ?? 0)}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

/** A rank without a fitted likely set (one photo on its own, or a model not yet calibrated):
 *  the same look as LikelyBlock, listing the top name and any others at 5% or more. With the
 *  chance withheld (the gate isn't sure it's a fungus), names only. */
function RankBlock({ rank, candidates, withheld }: {
  rank: Rank;
  candidates: Candidate[];
  /** Names only, no chance: the gate isn't sure the photos show a fungus. */
  withheld?: boolean;
}) {
  const [top, ...rest] = candidates;
  if (!top) return null;
  const italic = rank !== "family";
  if (!withheld && top.confidence != null) {
    return <LikelyBlock rank={rank} candidates={candidates} fitted={false}
                        set={{ coverage: 0, checked_coverage: null, capped: false, names: listedNames(candidates) }} />;
  }
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs uppercase tracking-wider text-muted-foreground">{RANK_LABEL[rank]}</span>
        <span className="text-xs text-muted-foreground">
          {num(top.reference_records)} DNA-verified record{top.reference_records === 1 ? "" : "s"}
        </span>
      </div>
      <div className="flex items-baseline justify-between gap-3">
        <NameLink rank={rank} name={top.name} className={`text-lg font-semibold text-[#4a3728] ${italic ? "sci" : ""}`} />
        <span className="text-xs text-muted-foreground">closest look-alike</span>
      </div>
      {rank === "species" && <ConfusedLine c={top} />}
      <div className="mt-2 space-y-1">
        {rest.slice(0, 3).map((c) => (
          <div key={c.name} className="text-sm">
            <NameLink rank={rank} name={c.name} className={`truncate text-[#5c4a3a] ${italic ? "sci" : ""}`} />
            {rank === "species" && <ConfusedLine c={c} />}
          </div>
        ))}
      </div>
    </div>
  );
}

function NameLink({ rank, name, className }: { rank: Rank; name: string; className?: string }) {
  if (rank !== "species") return <span className={className}>{name}</span>;
  return (
    <a
      href={`https://mycomap.org/species/${encodeURIComponent(name)}`}
      className={`${className} hover:text-myco-green hover:underline underline-offset-2`}
      title="Open on mycomap.org"
    >
      {name}
    </a>
  );
}

function Bar({ value, strong }: { value: number; strong?: boolean }) {
  return (
    <div className={`w-full overflow-hidden rounded-full bg-muted ${strong ? "h-2 mt-1" : "h-1.5"}`}>
      <div
        className={`h-full rounded-full ${strong ? "bg-myco-green" : "bg-myco-green/60"}`}
        style={{ width: `${Math.max(2, Math.round(value * 100))}%` }}
      />
    </div>
  );
}

function SpecimenCard({ s, urls, showMatched }: { s: Specimen; urls: string[]; showMatched: boolean }) {
  const matched = showMatched && urls[s.matched_query_photo];
  const mo = s.source === "mo";
  const href = s.record_url ?? s.inat_url ?? undefined;
  const site = mo ? "Mushroom Observer" : "iNaturalist";
  const shownId = s.source_id || s.observation_id;
  return (
    <div className="rounded-md border overflow-hidden bg-white">
      <a href={href} target="_blank" rel="noreferrer" className="block aspect-square bg-muted">
        {s.photo_url ? (
          <img src={s.photo_url} alt={s.name ?? s.species} loading="lazy" className="h-full w-full object-cover" />
        ) : s.photo_withheld ? (
          <span className="flex h-full w-full flex-col items-center justify-center gap-1 p-3 text-center text-[11px] leading-snug text-muted-foreground"
                title="All rights reserved, and the photographer hasn't given permission to show it here.">
            <Lock className="h-4 w-4" aria-hidden />
            <span>Photo on {site} only</span>
            <span className="opacity-80">no permission to show it here</span>
          </span>
        ) : null}
      </a>
      <div className="p-2 space-y-1 min-w-0">
        {s.species_url ? (
          <a href={s.species_url} className="sci block text-sm font-semibold text-[#4a3728] leading-tight hover:text-myco-green">
            {s.species}
          </a>
        ) : (
          // Named only to genus or family on mycomap.org: no species, so no species page.
          <span className="sci block text-sm font-semibold text-[#4a3728] leading-tight">
            {s.name ?? (s.genus || s.family)}
          </span>
        )}
        <Badge variant="outline" className="px-1.5 py-0 font-normal whitespace-nowrap">
          {pct(s.similarity)} alike
        </Badge>
        {matched && (
          <p className="text-xs text-muted-foreground">matches your photo {s.matched_query_photo + 1}</p>
        )}
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-0.5 text-xs text-muted-foreground hover:text-myco-green"
        >
          {mo ? "MO" : "iNat"} {shownId} <ArrowUpRight className="h-3 w-3" />
        </a>
        {s.photo_owner && <p className="text-[10px] text-muted-foreground truncate">© {s.photo_owner}</p>}
      </div>
    </div>
  );
}
