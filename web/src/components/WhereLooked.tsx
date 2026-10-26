// "Where Vision looked" on an answer (looked.py; Steve, 2026-10-26, from the approved mockup):
// a button that asks for each photo's map of where its evidence for the name was, then the
// photo with that map over the square the model sees, and the specimen photo it matched.
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ScanEye } from "lucide-react";
import { api, ApiError, lookedApi, type IdentifyResult, type LookedPhoto, type LookedView } from "@/lib/api";
import {
  cellColours, cropBox, firstPhoto, markers, ownLine, POLL_MS, refusalLine, squareOf, unseenStrips,
  workingLine,
} from "@/lib/whereLooked";

type Style = "spot" | "glow";
type Ready = Extract<LookedView, { status: "ready" }>;

/** The button, and the panel once the maps are there. Shown only where the server has the
 *  feature on and the answer has its own link (`runId`). */
export function WhereLooked({ runId, r, urls }: { runId: string; r: IdentifyResult; urls: string[] }) {
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  const [view, setView] = useState<LookedView | null>(null);
  const [open, setOpen] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);

  // While they are made, ask again every POLL_MS until they are ready or have failed.
  useEffect(() => {
    if (view?.status !== "working") return;
    const t = setTimeout(() => {
      lookedApi.read(runId).then(setView).catch((e) => {
        setProblem(refusalLine(e instanceof ApiError ? e.status : 0, String(e.message ?? "")));
        setView(null);
      });
    }, POLL_MS);
    return () => clearTimeout(t);
  }, [view, runId]);
  useEffect(() => {
    if (view?.status === "failed") setProblem(refusalLine(503, ""));
  }, [view]);

  if (!me.data?.features?.where_looked) return null;

  const ask = async () => {
    setOpen(true);
    setProblem(null);
    if (view?.status === "ready") return;
    setAsking(true);
    try {
      setView(await lookedApi.ask(runId));
    } catch (e) {
      const status = e instanceof ApiError ? e.status : 0;
      // Signed out: the maps may have been made already for someone else with the link.
      const kept = status === 401 ? await lookedApi.read(runId).catch(() => null) : null;
      if (kept?.status === "ready") setView(kept);
      else setProblem(refusalLine(status, e instanceof Error ? e.message : ""));
    } finally {
      setAsking(false);
    }
  };

  const shown = open && view?.status === "ready";
  return (
    <div className="space-y-2" data-testid="where-vision-looked">
      {!shown && (
        <button type="button" onClick={ask} disabled={asking || view?.status === "working"}
                aria-expanded={false}
                className="flex w-full items-center gap-2 rounded-md border border-myco-green bg-[#f4f9ee] px-3 py-2 text-sm font-semibold text-[#3f6b1d] hover:bg-[#eaf4df] disabled:opacity-80"
                data-testid="button-where-looked">
          <ScanEye className="h-4 w-4 shrink-0" aria-hidden />
          <span>Show where Vision looked</span>
          <span className="ml-auto text-xs font-normal text-[#6b7a5c]">
            {view?.status === "working" ? workingLine(view) : asking ? "asking…" : "takes a few seconds"}
          </span>
        </button>
      )}
      {problem && <p className="text-xs text-amber-800" role="status" data-testid="text-where-looked-problem">{problem}</p>}
      {shown && <LookedPanel view={view as Ready} r={r} urls={urls} onHide={() => setOpen(false)} />}
    </div>
  );
}

/** The panel itself, from the kept maps (rendered alone in tests). */
export function LookedPanel({ view, r, urls, onHide }: {
  view: Ready; r: IdentifyResult; urls: string[]; onHide?: () => void;
}) {
  const [cur, setCur] = useState<number | null>(() => firstPhoto(view.photos, r.per_photo));
  const [style, setStyle] = useState<Style>("spot");
  const [strength, setStrength] = useState(75);
  const entry = view.photos.find((p) => p.photo === cur) ?? view.photos[0];
  if (!entry) return null;
  const own = ownLine(entry.photo, r.per_photo);
  const seg = (on: boolean) =>
    `rounded-full border px-3 py-1 text-[13px] ${on ? "border-[#4a7a22] bg-[#4a7a22] text-white" : "border-[#A87146]/25 bg-white text-[#4a3728]"}`;
  return (
    <section className="rounded-lg border border-[#A87146]/20 bg-white p-3 sm:p-4" aria-label="Where Vision looked"
             data-testid="panel-where-looked">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-display text-base font-semibold text-[#4a3728]">Where Vision looked</h3>
        {onHide && (
          <button type="button" onClick={onHide} className="rounded-md border border-[#A87146]/25 px-2 py-0.5 text-xs text-[#5c4a3a]">
            Hide
          </button>
        )}
      </div>
      <p className="mt-1 text-[13px] text-[#5c4a3a]">
        The bright parts of each photo are where the evidence for <span className="sci">{view.name}</span> was:
        hiding them lowers Vision's score the most. It shows <b>where</b>, not <b>why</b>.
      </p>
      {view.photos.length > 1 && (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Photo">
          {view.photos.map((p) => (
            <button key={p.photo} type="button" role="tab" aria-selected={p.photo === entry.photo}
                    onClick={() => setCur(p.photo)} aria-label={`Photo ${p.photo + 1}`}
                    className={`relative h-14 w-14 shrink-0 overflow-hidden rounded-md border-2 bg-muted ${p.photo === entry.photo ? "border-myco-green" : "border-transparent"}`}>
              {urls[p.photo] && <img src={urls[p.photo]} alt="" className="h-full w-full object-cover" />}
              <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 text-[10px] text-white">{p.photo + 1}</span>
            </button>
          ))}
        </div>
      )}
      <div className="mt-3 grid items-start gap-4 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <figure className="relative m-0 overflow-hidden rounded-md bg-[#111]" data-testid="figure-where-looked">
          {urls[entry.photo] && <img src={urls[entry.photo]} alt={`Your photo ${entry.photo + 1}`} className="block w-full" />}
          <div className="absolute" style={cropBox(entry.crop)}>
            <MapLayer entry={entry} style={style} opacity={strength / 100} />
          </div>
          {unseenStrips(entry.crop).map((s) => (
            <div key={s.side} className="absolute flex items-center justify-center bg-[repeating-linear-gradient(135deg,rgba(255,255,255,.55)_0_6px,rgba(255,255,255,.35)_6px_12px)]"
                 style={s.style}>
              {(s.side === "top" || s.side === "left") && (
                <span className="rounded-full bg-white/90 px-2 text-[11px] text-[#4a3728]">Vision doesn't see this strip</span>
              )}
            </div>
          ))}
        </figure>
        <div>
          <div className="mb-3">
            <div className="mb-1 text-[11px] uppercase tracking-wide text-[#7a6a5a]">Show as</div>
            <div className="flex flex-wrap gap-1.5">
              <button type="button" className={seg(style === "spot")} onClick={() => setStyle("spot")}>Spotlight</button>
              <button type="button" className={seg(style === "glow")} onClick={() => setStyle("glow")}>Glow</button>
            </div>
          </div>
          <label className="mb-3 block">
            <span className="mb-1 block text-[11px] uppercase tracking-wide text-[#7a6a5a]">Strength</span>
            <input type="range" min={0} max={100} value={strength} onChange={(e) => setStrength(+e.target.value)}
                   className="w-full accent-[#5e9a2c]" aria-label="Strength of the map" />
          </label>
          {own && (
            <p className="text-[13px] text-[#5c4a3a]" data-testid="text-where-looked-own">
              {own.lead}<span className="sci">{own.name}</span>{own.tail}
            </p>
          )}
          <p className="mt-1 text-xs text-[#7a6a5a]">
            {style === "spot"
              ? "Clear: this photo's strongest evidence for the name. Dimmed: what mattered less."
              : "Bright: what pushed this photo toward the name. Nothing: what barely mattered."}
          </p>
        </div>
      </div>
      {entry.match && <Match entry={entry} urls={urls} />}
      <details className="mt-3 rounded-md border border-[#A87146]/20 px-3 py-2 text-[13px] text-[#4a3728]">
        <summary className="cursor-pointer font-semibold">What this map does and doesn't mean</summary>
        <ul className="mt-2 list-disc space-y-1 pl-4">
          <li><b>It does show</b> which parts of the photo Vision's score depended on. On 314 photos we
            checked, hiding the brightest tenth of a photo cost Vision about six times as much score as
            hiding a random tenth.</li>
          <li><b>It doesn't show why.</b> Vision doesn't know "pores" or "colour"; it compares patterns
            with photos of DNA-verified specimens. A bright patch means "this part looked like the
            specimens", nothing more.</li>
          <li><b>Bright on the background, a hand or a label?</b> Then that photo may be matching for the
            wrong reason. A clearer photo of the fungus itself would help.</li>
          <li>Each photo is scored on its own centre square; the answer above combines all your photos.</li>
        </ul>
      </details>
      <p className="mt-2 text-[11px] text-[#8a7b6c]">Made when someone first asked, and kept with this result's link.</p>
    </section>
  );
}

/** The map over the square the model sees: a canvas of grid x grid cells, scaled up smoothly. */
function MapLayer({ entry, style, opacity }: { entry: LookedPhoto; style: Style; opacity: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const g = entry.grid;
    const img = ctx.createImageData(g, g);
    cellColours(entry.map, style).forEach(([r, gr, b, a], i) => {
      img.data.set([r, gr, b, Math.round(a * 255)], i * 4);
    });
    ctx.putImageData(img, 0, 0);
  }, [entry, style]);
  return <canvas ref={ref} width={entry.grid} height={entry.grid} className="block h-full w-full"
                 style={{ opacity, imageRendering: "auto" }} aria-hidden />;
}

/** The photo's seen square beside the specimen photo's, with the numbered matching spots. */
function Match({ entry, urls }: { entry: LookedPhoto; urls: string[] }) {
  const m = entry.match!;
  const spots = markers(m.pairs);
  const s = m.specimen;
  const square = (src: string | null | undefined, crop: LookedPhoto["crop"] | null, side: "photo" | "specimen",
                  alt: string) => (
    <div className="relative aspect-square overflow-hidden rounded-md bg-[#111]">
      {src && (crop
        ? <img src={src} alt={alt} className="absolute max-w-none" style={squareOf(crop)} />
        : <img src={src} alt={alt} className="h-full w-full object-cover" />)}
      {spots.map((k) => (
        <span key={k.n} className="absolute" style={{ left: `${k[side].x}%`, top: `${k[side].y}%` }}>
          <span className="absolute h-7 w-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,.35)]" />
          <span className="absolute left-2 -top-5 h-5 w-5 rounded-full border-2 border-[#3f6b1d] bg-white text-center text-[11px] font-semibold leading-4 text-[#3f6b1d]">
            {k.n}
          </span>
        </span>
      ))}
    </div>
  );
  return (
    <div className="mt-4 border-t border-[#A87146]/15 pt-3" data-testid="section-where-looked-match">
      <h4 className="text-sm font-semibold text-[#4a3728]">What it matched</h4>
      <p className="mt-0.5 text-[13px] text-[#5c4a3a]">
        Photo {entry.photo + 1} beside the DNA-verified specimen it was closest to:{" "}
        <span className="sci">{s.name}</span>{s.similarity != null && `, ${Math.round(s.similarity * 100)}% alike`}.
      </p>
      <div className="mt-2 grid max-w-xl grid-cols-2 gap-3">
        <figure className="m-0">
          {square(urls[entry.photo], entry.crop, "photo", `Your photo ${entry.photo + 1}, the part Vision saw`)}
          <figcaption className="mt-1 text-xs text-[#6f6257]">Your photo {entry.photo + 1}</figcaption>
        </figure>
        <figure className="m-0">
          {square(s.photo_url, m.crop, "specimen", "The DNA-verified specimen")}
          <figcaption className="mt-1 text-xs text-[#6f6257]">
            {s.record_url
              ? <a href={s.record_url} target="_blank" rel="noreferrer" className="underline">{s.source === "mo" ? "MO" : "iNat"} {s.observation_id}</a>
              : `${s.source === "mo" ? "MO" : "iNat"} ${s.observation_id}`}
            {s.photo_owner && <> · © {s.photo_owner}</>}
          </figcaption>
        </figure>
      </div>
      <p className="mt-2 text-xs text-[#7a6a5a]">
        Numbered spots are pairs of patches that look most alike to the model. Similar patterns, not
        proof: two photos can match on colour or light as well as on the fungus.
      </p>
    </div>
  );
}
