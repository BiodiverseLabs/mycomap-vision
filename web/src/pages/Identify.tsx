import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation, type UseQueryResult } from "@tanstack/react-query";
import { ArrowUpRight, CalendarDays, ImagePlus, Lightbulb, Loader2, LocateFixed, MapPin,
         Microscope, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/Layout";
import {
  api,
  modelKey,
  modelLabel,
  num,
  pct,
  RANKS,
  type Candidate,
  type ContextUsed,
  type IdentifyResult,
  type ModelsInfo,
  type PhotoResult,
  type Where,
  type Rank,
  type Specimen,
} from "@/lib/api";
import { fillFromPhotos, readPhotoPlaceDate, type PhotoPlaceDate } from "@/lib/photoPlaceDate";
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

  // Read each new photo's place and date, then fill the fields once every photo is read.
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
  const run = useMutation({
    mutationFn: (sent: { where: Where; files: File[] }) => api.identify(sent.files, chosen, sent.where),
  });
  const sentUrls = useObjectUrls(run.variables?.files ?? NO_FILES);

  return (
    <>
      <PageHeader title="Identify a fungus">
        Add every photo you have of one find: the cap from above, the underside (gills, pores or
        teeth), the stem and the habitat. Each photo is compared with DNA-verified MycoMap records.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 grid gap-8 lg:grid-cols-[360px_1fr]">
        <div className="space-y-6">
          <PhotoPicker files={files} setFiles={setFiles} />
          <WhereWhen where={where} setWhere={setWhere} />
          <ModelPicker models={models} onChange={setChosen} />
          <Button
            size="lg"
            className="w-full"
            disabled={!files.length || !chosen.length || run.isPending}
            onClick={() => run.mutate({ where, files })}
          >
            {run.isPending ? <Loader2 className="animate-spin" /> : <Microscope />}
            {run.isPending ? "Identifying…" : `Identify from ${files.length || "your"} photo${files.length === 1 ? "" : "s"}`}
          </Button>
          {run.isError && <p className="text-sm text-destructive">{(run.error as Error).message}</p>}
        </div>

        <div className="min-w-0">
          {!run.data && !run.isPending && <EmptyState />}
          {run.data && <ContextLine used={run.data.context_used} sent={run.variables?.where} />}
          {run.data && (
            <div
              className={`grid gap-6 ${run.data.results.length > 1 ? "xl:grid-cols-2" : ""}`}
            >
              {run.data.results.map((r) => (
                <ResultCard key={`${modelKey(r.model.backbone, r.model.method)}@${run.submittedAt}`}
                            r={r} urls={sentUrls} />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

const NO_FILES: File[] = [];

/** Object URLs for local files, released when the files change or the page closes. */
function useObjectUrls(files: File[]): string[] {
  const urls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => urls.forEach((u) => URL.revokeObjectURL(u)), [urls]);
  return urls;
}

function PhotoPicker({ files, setFiles }: { files: File[]; setFiles: (f: File[]) => void }) {
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
        className={`rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition-colors ${
          over ? "border-myco-green bg-myco-green/5" : "border-[#A87146]/30 hover:border-myco-green"
        }`}
        onClick={() => input.current?.click()}
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
          Drop or choose up to {MAX_PHOTOS}, all of the same find
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
        <div className="mt-3 grid grid-cols-4 gap-2">
          {urls.map((u, i) => (
            <div key={u} className="relative aspect-square rounded-md overflow-hidden border">
              <img src={u} alt={`Photo ${i + 1}`} className="h-full w-full object-cover" />
              <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 text-[10px] text-white">
                {i + 1}
              </span>
              <button
                className="absolute top-0.5 right-0.5 rounded-full bg-black/60 p-0.5 text-white hover:bg-black/80"
                onClick={() => setFiles(files.filter((_, j) => j !== i))}
                aria-label={`Remove photo ${i + 1}`}
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WhereWhen({ where, setWhere }: { where: Where; setWhere: (w: Where) => void }) {
  const [locating, setLocating] = useState(false);
  const locate = () => {
    if (!navigator.geolocation) return;
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setWhere({ ...where, lat: pos.coords.latitude.toFixed(4), lng: pos.coords.longitude.toFixed(4),
                   placeFromPhoto: undefined });
        setLocating(false);
      },
      () => setLocating(false),
      { timeout: 10_000 },
    );
  };
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
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-myco-brown shrink-0" />
          <input className={input} placeholder="Latitude" inputMode="decimal" value={where.lat ?? ""}
                 onChange={(e) => setWhere({ ...where, lat: e.target.value, placeFromPhoto: undefined })} aria-label="Latitude" />
          <input className={input} placeholder="Longitude" inputMode="decimal" value={where.lng ?? ""}
                 onChange={(e) => setWhere({ ...where, lng: e.target.value, placeFromPhoto: undefined })} aria-label="Longitude" />
          <Button type="button" variant="outline" size="icon" onClick={locate} disabled={locating}
                  title="Use my location" aria-label="Use my location">
            {locating ? <Loader2 className="animate-spin" /> : <LocateFixed />}
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <CalendarDays className="h-4 w-4 text-myco-brown shrink-0" />
          <input className={input} type="date" value={where.observedOn ?? ""}
                 onChange={(e) => setWhere({ ...where, observedOn: e.target.value, dateFromPhoto: undefined })} aria-label="Date found" />
        </div>
        {(where.placeFromPhoto || where.dateFromPhoto) && (
          <p className="text-xs text-muted-foreground">
            {where.placeFromPhoto === where.dateFromPhoto
              ? `Place and date from photo ${where.placeFromPhoto}.`
              : [where.placeFromPhoto && `Place from photo ${where.placeFromPhoto}.`,
                 where.dateFromPhoto && `Date from photo ${where.dateFromPhoto}.`].filter(Boolean).join(" ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function ContextLine({ used, sent }: { used: ContextUsed; sent?: Where }) {
  if (!used.place_from && !used.date_from) {
    return (
      <p className="mb-4 text-sm text-muted-foreground">
        No place or date given or found in the photos, so models with a range-and-season
        score used the photos alone.
      </p>
    );
  }
  // Places and dates filled in from a photo reach the server as entered; say where they came from.
  const from = (f: string | null, photo?: number) =>
    f === "photo" ? "from your photo" : photo ? `from photo ${photo}` : "as entered";
  return (
    <p className="mb-4 text-sm text-muted-foreground flex flex-wrap gap-x-4 gap-y-1">
      {used.place_from && (
        <span className="inline-flex items-center gap-1">
          <MapPin className="h-3.5 w-3.5" /> near {used.latitude}, {used.longitude} ({from(used.place_from, sent?.placeFromPhoto)})
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
    if (!choice && offered.length) setChoice(defaultChoice(offered));
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
            The API isn't answering. Start it with <code>mv serve</code>.
          </p>
        )}
        {models.data && (!ready.length || !bases.length) && (
          <p className="text-sm text-muted-foreground">
            No model has embedded photos yet. Run <code>mv embed</code> first.
          </p>
        )}
        {choice && ready.length > 0 && (
          <fieldset className="space-y-3">
            {ready.length > 1 && (
              <select className={selectClass} value={backbone ?? ""} onChange={(e) => setBackbone(e.target.value)}
                aria-label="Backbone" data-testid="select-backbone">
                {ready.map((b) => <option key={b} value={b}>{b}</option>)}
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
                {num(info.embedded_photos)} reference photos · {backbone && method ? modelLabel(backbone, method).how : ""}
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
                    {ready.map((b) => <option key={b} value={b}>{b}</option>)}
                  </select>
                )}
                <select className={selectClass} value={second.method} aria-label="Second model"
                  onChange={(e) => setSecond({ ...second, method: e.target.value })} data-testid="select-second-method">
                  {offered.map((m) => <option key={m} value={m}>{modelLabel(second.backbone, m).how}</option>)}
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

function EmptyState() {
  return (
    <div className="rounded-lg border border-[#A87146]/20 bg-[#faf9f7] p-8 text-[#5c4a3a]">
      <h2 className="font-display font-semibold text-2xl text-[#4a3728]">What you'll get</h2>
      <ul className="mt-4 space-y-2 text-sm list-disc pl-5">
        <li>An answer at every rank (family, genus, species), each with its own confidence.</li>
        <li>The closest DNA-verified specimens, with their photos, so you can compare by eye.</li>
        <li>How each of your photos scores on its own, to see which ones agree.</li>
        <li>Advice on what would firm up the identification.</li>
        <li>Optionally a second model side by side, to see where they agree.</li>
      </ul>
      <p className="mt-4 text-xs">
        Early development: the reference set grows as photos are downloaded, and confidence is
        not yet calibrated.
      </p>
    </div>
  );
}

function ResultCard({ r, urls }: { r: IdentifyResult; urls: string[] }) {
  // null: all photos together (the main answer). A number: that photo on its own.
  const [view, setView] = useState<number | null>(null);
  const photo = view == null ? null : (r.per_photo.find((p) => p.photo === view) ?? null);
  const ranks = photo ? photo.ranks : r.ranks;
  const specimens = photo ? photo.specimens : r.specimens;
  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-[#f8f5f0] border-b border-[#A87146]/10 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base text-[#4a3728]">
            {r.model.backbone}{" "}
            <span className="font-normal text-muted-foreground">
              · {modelLabel(r.model.backbone, r.model.method).how}
            </span>
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            {num(r.reference.records)} records · {num(r.reference.species)} names ·{" "}
            {num(r.reference.photos)} photos
          </span>
        </div>
      </CardHeader>
      <CardContent className="pt-5 space-y-6">
        {r.per_photo.length > 1 && (
          <PhotoSwitcher r={r} urls={urls} view={view} setView={setView} />
        )}
        {photo && <PhotoVerdict p={photo} />}
        <div className="space-y-5">
          {RANKS.map((rank) => (
            <RankBlock key={rank} rank={rank} candidates={ranks[rank]} />
          ))}
          <p className="text-xs text-muted-foreground">
            {r.confidence_note}
            {photo && " One photo on its own is a rougher guide than all of them together."}
          </p>
        </div>
        {!photo && r.hints.length > 0 && (
          <div className="rounded-md border border-myco-tan-light/50 bg-[#fbf7f0] p-3 space-y-1.5">
            {r.hints.map((h) => (
              <p key={h} className="flex gap-2 text-sm text-[#5c4a3a]">
                <Lightbulb className="h-4 w-4 mt-0.5 shrink-0 text-myco-tan" />
                {h}
              </p>
            ))}
          </div>
        )}
        <div>
          <h3 className="text-sm font-semibold text-[#4a3728] mb-2">
            {photo
              ? `Closest DNA-verified specimens to photo ${photo.photo + 1}`
              : "Closest DNA-verified specimens"}
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {specimens.map((s) => (
              <SpecimenCard key={s.observation_id} s={s} urls={urls} showMatched={!photo} />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
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
            Under each photo: how it rates <span className="sci">{top}</span> on its own. Click a
            photo to see its own results.
          </>
        ) : (
          "Click a photo to see its own results."
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
          return (
            <button
              key={p.photo}
              type="button"
              role="tab"
              aria-selected={view === p.photo}
              className={tile(view === p.photo)}
              onClick={() => setView(p.photo)}
              title={place ? `Photo ${p.photo + 1} on its own puts ${place.name} ${ordinal(place.position)} (${pct(place.confidence)})` : undefined}
            >
              <span className="relative h-14 w-14 overflow-hidden rounded border bg-muted">
                {urls[p.photo] && (
                  <img src={urls[p.photo]} alt={`Your photo ${p.photo + 1}`} className="h-full w-full object-cover" />
                )}
                <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 text-[10px] text-white">
                  {p.photo + 1}
                </span>
              </span>
              <span className={`text-[11px] tabular-nums ${agrees ? "font-semibold text-myco-green" : "text-muted-foreground"}`}>
                {place ? pct(place.confidence) : "–"}
              </span>
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
  return (
    <div className="rounded-md border border-[#A87146]/20 bg-[#faf9f7] p-3 text-sm text-[#5c4a3a]">
      <span className="font-semibold text-[#4a3728]">Photo {n} on its own.</span>{" "}
      {species &&
        (species.position === 1 ? (
          <>
            It agrees with the answer from all photos, <span className="sci">{species.name}</span>{" "}
            ({pct(species.confidence)}).
          </>
        ) : (
          <>
            It puts <span className="sci">{species.name}</span>, the answer from all photos,{" "}
            {ordinal(species.position)} ({pct(species.confidence)}).
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

function RankBlock({ rank, candidates }: { rank: Rank; candidates: Candidate[] }) {
  const [top, ...rest] = candidates;
  if (!top) return null;
  const italic = rank !== "family";
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
        <span className="text-lg font-semibold text-myco-green tabular-nums">{pct(top.confidence)}</span>
      </div>
      <Bar value={top.confidence} strong />
      <div className="mt-2 space-y-1">
        {rest.slice(0, 3).map((c) => (
          <div key={c.name} className="grid grid-cols-[1fr_72px_40px] items-center gap-2 text-sm">
            <NameLink rank={rank} name={c.name} className={`truncate text-[#5c4a3a] ${italic ? "sci" : ""}`} />
            <Bar value={c.confidence} />
            <span className="text-right text-xs text-muted-foreground tabular-nums">{pct(c.confidence)}</span>
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
  return (
    <div className="rounded-md border overflow-hidden bg-white">
      <a href={s.inat_url} target="_blank" rel="noreferrer" className="block aspect-square bg-muted">
        {s.photo_url ? (
          <img src={s.photo_url} alt={s.species} loading="lazy" className="h-full w-full object-cover" />
        ) : s.photo_withheld ? (
          <span className="flex h-full w-full items-center justify-center p-3 text-center text-xs text-muted-foreground">
            Photo not shown here: all rights reserved, and the photographer hasn't given permission. View it on iNaturalist.
          </span>
        ) : null}
      </a>
      <div className="p-2 space-y-1">
        <a href={s.species_url} className="sci block text-sm font-semibold text-[#4a3728] leading-tight hover:text-myco-green">
          {s.species}
        </a>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <Badge variant="outline" className="px-1.5 py-0 font-normal">
            {pct(s.similarity)} alike
          </Badge>
          {matched && <span>your photo {s.matched_query_photo + 1}</span>}
        </div>
        <a
          href={s.inat_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-0.5 text-xs text-muted-foreground hover:text-myco-green"
        >
          iNat {s.observation_id} <ArrowUpRight className="h-3 w-3" />
        </a>
        {s.photo_owner && <p className="text-[10px] text-muted-foreground truncate">© {s.photo_owner}</p>}
      </div>
    </div>
  );
}
