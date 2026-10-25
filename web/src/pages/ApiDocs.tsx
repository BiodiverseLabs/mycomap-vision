import { useEffect, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import type { Me } from "@/lib/api";
import { docParts, endpointCounts, endpointIndex, inlineParts, isFieldTable, isStatusTable, statusClass,
         type Block, type DocPart, type EndpointBlock, type Outline } from "@/lib/apiOutline";
import { VISION_KEYS_URL } from "@/lib/getInvolved";
import { activeSection } from "@/lib/researchView";
import { CopyButton, ResearchHero, ResearchShell, StatusPill } from "@/components/ResearchKit";
import { PRIMARY_BUTTON } from "@/components/Explainer";
import { useMe, useOutline } from "@/components/researchQueries";

// Research > API (Steve, 2026-10-25: the approved redesign): a docs layout over the API's
// outline (GET /api/docs/v1.json, apidocs.py on the server; mycomap.org/api-docs renders the
// same outline, so the text lives in one place). Sticky contents, a quickstart, an endpoint
// index, each endpoint's fields beside its code panel, the answer field by field, errors.

function Text({ text }: { text: string }) {
  return (
    <>
      {inlineParts(text).map((p, i) => {
        if (p.kind === "code") return <code key={i} className="ic2">{p.text}</code>;
        if (p.kind === "bold") return <strong key={i}>{p.text}</strong>;
        if (p.kind === "link") return <a key={i} href={p.href} className="text-[#4f7a16] underline">{p.text}</a>;
        return <span key={i}>{p.text}</span>;
      })}
    </>
  );
}

const MTH: Record<string, string> = { GET: "get", POST: "post", PATCH: "patch", DELETE: "del" };
const Method = ({ m }: { m: string }) => <span className={`mth ${MTH[m] ?? "get"}`}>{m}</span>;

/** An endpoint's code panel: the request, the response, one at a time, with a copy button. */
function CodePanel({ request, response }: { request?: string; response?: string }) {
  const tabs = [request && ["req", "curl", request], response && ["resp", "Response", response]]
    .filter(Boolean) as [string, string, string][];
  const [on, setOn] = useState(tabs[0]?.[0]);
  const cur = tabs.find((t) => t[0] === on) ?? tabs[0];
  if (!cur) return null;
  return (
    <div className="codep" data-testid="code-panel">
      <div className="bar">
        {tabs.map(([k, l]) => <button key={k} type="button" className={k === cur[0] ? "on" : ""} aria-pressed={k === cur[0]} onClick={() => setOn(k)}>{l}</button>)}
        <CopyButton text={cur[2]} />
      </div>
      <pre>{cur[2]}</pre>
    </div>
  );
}

function EndpointView({ b }: { b: EndpointBlock }) {
  return (
    <div className="mt-5" data-testid={`endpoint-${b.method}-${b.path}`}>
      <div className="flex flex-wrap items-center gap-2.5">
        <Method m={b.method} /><code className="ic2" style={{ fontSize: 15 }}>{b.path}</code>
        <StatusPill status={b.status === "available" ? "running" : "planned"} label={b.status === "available" ? "Available" : "Planned"} />
      </div>
      <p className="small mt-1.5"><Text text={b.summary} />{b.auth && <> Needs: {b.auth}.</>}</p>
      <div className="ep mt-3">
        {b.fields && b.fields.length > 0 && (
          <div className="tcard"><div className="th"><b>Fields</b></div><div className="scrollx"><table className="dt">
            <thead><tr><th className="l">Field</th><th className="l">Type</th><th className="l">Required</th><th className="l">Notes</th></tr></thead>
            <tbody>{b.fields.map(([name, type, req, notes]) => (
              <tr key={name}><td className="l"><code className="ic2">{name}</code></td><td className="l">{type}</td><td className="l">{req}</td><td className="l"><Text text={notes} /></td></tr>
            ))}</tbody></table></div></div>
        )}
        <CodePanel request={b.request} response={b.response} />
      </div>
      {b.notes?.map((n, i) => <p key={i} className="small mt-2"><Text text={n} /></p>)}
    </div>
  );
}

function BlockView({ block }: { block: Block }) {
  switch (block.type) {
    case "p": return <p className="mb-2.5"><Text text={block.text} /></p>;
    case "list": return <ul className="mb-2.5 list-disc space-y-1 pl-6">{block.items.map((t, i) => <li key={i}><Text text={t} /></li>)}</ul>;
    case "code": return <pre className="code">{block.text}</pre>;
    case "endpoint": return <EndpointView b={block} />;
    case "table":
      if (isFieldTable(block)) {
        return (
          <div className="afs mt-3" data-testid="field-list">
            {block.rows.map((r, i) => <div key={i} className="af"><span><Text text={r[0]} /></span><span><Text text={r[1]} /></span></div>)}
          </div>
        );
      }
      return (
        <div className="tcard mt-3"><div className="scrollx"><table className="dt">
          <thead><tr>{block.columns.map((c) => <th key={c} className="l">{c}</th>)}</tr></thead>
          <tbody>{block.rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className="l">{j === 0 && isStatusTable(block) ? <span className={`st ${statusClass(c)}`}>{c}</span> : <Text text={c} />}</td>)}</tr>
          ))}</tbody></table></div></div>
      );
  }
}

const KICKER = "text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a] mb-1.5";
const H2 = "font-display font-semibold text-[25px] sm:text-[30px] leading-tight text-[#4a3728] mb-2";
const CARD = "min-w-0 rounded-[10px] border border-[#A87146]/20 bg-white p-4";
const H3 = "font-display font-semibold text-lg text-[#4a3728]";

function Quickstart({ o }: { o: Outline }) {
  const warn = o.sections.find((s) => s.id === "overview")?.blocks.find((b) => b.type === "list");
  return (
    <div className="asec" id="quickstart" data-testid="api-quickstart">
      <p className={KICKER}>Quickstart</p><h2 className={H2}>Identify a find in three steps</h2>
      <div className="qs3 mt-3.5">
        <div className={CARD}><span className="n">1</span><h3 className={H3}>Get a token</h3><p className="small">Request one on mycomap.org with a short note on what it is for; it works once an admin approves it.</p></div>
        <div className={CARD}><span className="n">2</span><h3 className={H3}>Send photos</h3><p className="small"><code className="ic2">POST /identify</code> with 1-10 photos of one find, and its place and date if you have them.</p></div>
        <div className={CARD}><span className="n">3</span><h3 className={H3}>Read the answer</h3><p className="small"><code className="ic2">answer</code> is the most precise rank Vision is confident at; <code className="ic2">ranks</code> holds the rest.</p></div>
      </div>
      {warn?.type === "list" && <div className="callout warm mt-3.5" data-testid="api-warning"><div>{warn.items.map((t, i) => <span key={i}>{i > 0 && " "}<Text text={t} /></span>)}</div></div>}
    </div>
  );
}

function EndpointIndex({ o }: { o: Outline }) {
  const all = endpointIndex(o);
  if (!all.length) return null;
  return (
    <div className="asec" id="endpoints" data-testid="api-endpoint-index">
      <p className={KICKER}>Endpoints</p><h2 className={H2}>All endpoints</h2>
      <div className="tcard mt-3"><div className="scrollx"><table className="dt">
        <thead><tr><th className="l">Method</th><th className="l">Path</th><th className="l">What it does</th><th className="l">Status</th></tr></thead>
        <tbody>{all.map((b) => (
          <tr key={`${b.method} ${b.path}`}><td className="l"><Method m={b.method} /></td><td className="l"><code className="ic2">{b.path}</code></td>
            <td className="l"><Text text={b.summary} /></td>
            <td className="l"><StatusPill status={b.status === "available" ? "running" : "planned"} label={b.status === "available" ? "Available" : "Planned"} /></td></tr>
        ))}</tbody></table></div></div>
    </div>
  );
}

function Part({ p }: { p: DocPart }) {
  if (p.kind === "cards") {
    return (
      <div className="asec" id={p.id} data-testid="api-cards">
        <p className={KICKER}>{p.title}</p><h2 className={H2}>Using it well</h2>
        <div className="three mt-3">
          {p.cards.map((c) => (
            <div key={c.title} className={CARD}>
              <h3 className={H3}>{c.title}</h3>
              <ul className="mt-1.5 list-disc pl-[18px] text-[13.5px] leading-normal">{c.items.map((t, i) => <li key={i}><Text text={t} /></li>)}</ul>
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (p.id === "overview") {
    // The overview's warning list is the quickstart's callout; its paragraphs stay here.
    return (
      <div className="asec" id={p.id}>
        <p className={KICKER}>{p.title}</p>
        {p.blocks.filter((b) => b.type !== "list").map((b, i) => <BlockView key={i} block={b} />)}
      </div>
    );
  }
  return (
    <div className="asec" id={p.id}>
      <p className={KICKER}>{p.title}</p><h2 className={H2}>{p.title}</h2>
      {p.blocks.map((b, i) => <BlockView key={i} block={b} />)}
    </div>
  );
}

/** The contents in the side bar: the quickstart, the endpoint index (before the first part
 *  with an endpoint), then the outline's own parts. */
export function apiContents(o: Outline): { id: string; label: string }[] {
  const items = [{ id: "quickstart", label: "Quickstart" }];
  for (const p of docParts(o)) {
    if (p.kind === "section" && p.blocks.some((b) => b.type === "endpoint") && !items.some((x) => x.id === "endpoints")) {
      items.push({ id: "endpoints", label: "Endpoints" });
    }
    items.push({ id: p.id, label: p.title });
  }
  return items;
}

export function OutlineView({ outline }: { outline: Outline }) {
  const contents = apiContents(outline);
  const [cur, setCur] = useState(0);
  useEffect(() => {
    const ids = apiContents(outline).map((c) => c.id);
    const on = () => setCur(activeSection(ids.map((id) => {
      const el = document.getElementById(id);
      return el ? el.getBoundingClientRect().top + window.scrollY : null;
    }), window.scrollY, 150));
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, [outline]);
  let indexShown = false;
  return (
    <div className="apil" data-testid="api-docs">
      <nav className="side" aria-label="On this page">
        <b>On this page</b>
        {contents.map((c, i) => <a key={c.id} href={`#${c.id}`} className={i === cur ? "on" : ""}>{c.label}</a>)}
      </nav>
      <div className="min-w-0">
        <Quickstart o={outline} />
        {docParts(outline).map((p) => {
          const first = !indexShown && p.kind === "section" && p.blocks.some((b) => b.type === "endpoint");
          if (first) indexShown = true;
          return <div key={p.id}>{first && <EndpointIndex o={outline} />}<Part p={p} /></div>;
        })}
      </div>
    </div>
  );
}

export function ApiDocsPage() {
  const me = useMe();
  const outline = useOutline();
  return <ApiDocs me={me.data} outline={outline.data} loading={outline.isPending} error={outline.error as Error | null} />;
}

export function ApiDocs({ me, outline, loading, error }: { me?: Me | null; outline?: Outline; loading?: boolean; error?: Error | null }) {
  const counts = outline ? endpointCounts(outline) : null;
  const base = outline?.base_url ?? "https://vision.mycomap.org/api/v1";
  return (
    <ResearchShell me={me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "API" }]} title={outline?.title ?? "MycoMap Vision API"}
                    top={counts ? <span className="chip" data-testid="api-endpoint-count"><span className="dot" />{outline!.version} · {counts.available} of {counts.total} endpoints available</span> : undefined}
                    lede={outline?.summary ?? "Identify fungi from photos in your own app or site."}>
        <div className="mt-5 flex flex-wrap items-center gap-2.5">
          <span className="baseurl" data-testid="api-base-url"><span>{base}</span><CopyButton text={base} /></span>
          <a className={PRIMARY_BUTTON} href={VISION_KEYS_URL}>Request a Vision token <ArrowUpRight className="h-3.5 w-3.5" /></a>
        </div>
      </ResearchHero>
      <section className="mx-auto max-w-[1220px] px-4 py-[30px] sm:px-8 sm:py-10 text-[#5c4a3a] leading-relaxed">
        {loading && !error && <p>Loading the API outline…</p>}
        {error && <p data-testid="api-error">{error.message}</p>}
        {outline && <OutlineView outline={outline} />}
      </section>
    </ResearchShell>
  );
}
