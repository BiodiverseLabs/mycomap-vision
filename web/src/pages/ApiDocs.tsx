import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/Layout";
import { type Block, fetchOutline, inlineParts, type Outline } from "@/lib/apiOutline";

// The API's outline (GET /api/docs/v1.json, apidocs.py on the server). mycomap.org/api-docs
// renders the same outline on its "MycoMap Vision API" tab, so the text lives in one place.

function Text({ text }: { text: string }) {
  return (
    <>
      {inlineParts(text).map((p, i) => {
        if (p.kind === "code")
          return <code key={i} className="px-1 py-0.5 rounded bg-[#f3eee6] text-[#4a3728] text-[0.9em]">{p.text}</code>;
        if (p.kind === "bold") return <strong key={i}>{p.text}</strong>;
        if (p.kind === "link")
          return <a key={i} href={p.href} className="text-myco-green underline">{p.text}</a>;
        return <span key={i}>{p.text}</span>;
      })}
    </>
  );
}

const METHOD_COLOURS: Record<string, string> = {
  GET: "bg-sky-100 text-sky-800",
  POST: "bg-emerald-100 text-emerald-800",
  PATCH: "bg-amber-100 text-amber-800",
  DELETE: "bg-rose-100 text-rose-800",
};

function Code({ text }: { text: string }) {
  return (
    <pre className="mt-2 p-3 rounded bg-[#2f241b] text-[#f8f5f0] text-xs overflow-x-auto whitespace-pre">
      {text}
    </pre>
  );
}

function BlockView({ block, base }: { block: Block; base: string }) {
  switch (block.type) {
    case "p":
      return <p><Text text={block.text} /></p>;
    case "list":
      return (
        <ul className="list-disc pl-6 space-y-1">
          {block.items.map((t, i) => <li key={i}><Text text={t} /></li>)}
        </ul>
      );
    case "code":
      return <Code text={block.text} />;
    case "table":
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr>{block.columns.map((c) => <th key={c} className="text-left font-semibold text-[#4a3728] border-b border-[#A87146]/20 py-2 pr-4">{c}</th>)}</tr>
            </thead>
            <tbody>
              {block.rows.map((r, i) => (
                <tr key={i} className="align-top border-b border-[#A87146]/10">
                  {r.map((c, j) => <td key={j} className="py-2 pr-4"><Text text={c} /></td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "endpoint":
      return (
        <div className="rounded-lg border border-[#A87146]/20 bg-white p-4" data-testid={`endpoint-${block.method}-${block.path}`}>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`px-2 py-0.5 rounded text-xs font-bold ${METHOD_COLOURS[block.method] ?? "bg-gray-100"}`}>{block.method}</span>
            <code className="text-sm text-[#4a3728] break-all">{block.path}</code>
            <span className={`ml-auto px-2 py-0.5 rounded-full text-xs ${block.status === "available" ? "bg-myco-green/15 text-myco-green" : "bg-gray-100 text-gray-600"}`}>
              {block.status === "available" ? "Available" : "Planned"}
            </span>
          </div>
          <p className="mt-2"><Text text={block.summary} /></p>
          {block.auth && <p className="mt-1 text-sm text-[#5c4a3a]/80">Needs: {block.auth}</p>}
          {block.fields && block.fields.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[#4a3728]">
                    <th className="py-1 pr-3">Field</th><th className="py-1 pr-3">Type</th>
                    <th className="py-1 pr-3">Required</th><th className="py-1">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {block.fields.map(([name, type, req, notes]) => (
                    <tr key={name} className="align-top border-t border-[#A87146]/10">
                      <td className="py-1 pr-3"><code>{name}</code></td><td className="py-1 pr-3">{type}</td>
                      <td className="py-1 pr-3">{req}</td><td className="py-1"><Text text={notes} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {block.request && (<><p className="mt-3 text-sm font-semibold text-[#4a3728]">Example</p><Code text={block.request} /></>)}
          {block.response && (
            <details className="mt-3">
              <summary className="cursor-pointer text-sm font-semibold text-[#4a3728]">Response</summary>
              <Code text={block.response} />
            </details>
          )}
          {block.notes?.map((n, i) => <p key={i} className="mt-2 text-sm"><Text text={n} /></p>)}
          <p className="sr-only">{base}{block.path}</p>
        </div>
      );
  }
}

export function OutlineView({ outline }: { outline: Outline }) {
  return (
    <div className="space-y-10">
      <nav className="flex flex-wrap gap-x-4 gap-y-1 text-sm" aria-label="On this page">
        {outline.sections.map((s) => (
          <a key={s.id} href={`#${s.id}`} className="text-myco-green hover:underline">{s.title}</a>
        ))}
      </nav>
      {outline.sections.map((s) => (
        <section key={s.id} id={s.id} className="space-y-4 scroll-mt-24">
          <h2 className="font-display font-semibold text-2xl text-[#4a3728]">{s.title}</h2>
          {s.blocks.map((b, i) => <BlockView key={i} block={b} base={outline.base_url} />)}
        </section>
      ))}
    </div>
  );
}

export function ApiDocsPage() {
  const outline = useQuery({ queryKey: ["api-outline"], queryFn: () => fetchOutline(), retry: 1 });
  return (
    <>
      <PageHeader title="MycoMap Vision API">
        {outline.data?.summary ?? "Identify fungi from photos in your own app or site."}
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-4xl text-[#5c4a3a] leading-relaxed" data-testid="api-docs">
        {outline.isPending && <p>Loading the API outline...</p>}
        {outline.error && <p>{String((outline.error as Error).message)}</p>}
        {outline.data && <OutlineView outline={outline.data} />}
      </div>
    </>
  );
}
