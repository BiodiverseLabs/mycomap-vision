import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// Markdown as the site writes it (the paper draft, experiment write-ups): headings, tables,
// lists and links in the site's type and colours.

const md: Components = {
  h1: ({ children }) => (
    <h1 className="font-display font-semibold text-2xl md:text-3xl text-[#4a3728] mt-10 mb-4">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="font-display font-semibold text-xl md:text-2xl text-[#4a3728] mt-10 mb-3 pt-4 border-t border-[#A87146]/15">
      {children}
    </h2>
  ),
  h3: ({ children }) => <h3 className="font-semibold text-lg text-[#4a3728] mt-6 mb-2">{children}</h3>,
  h4: ({ children }) => <h4 className="font-semibold text-[#4a3728] mt-4 mb-1">{children}</h4>,
  p: ({ children }) => <p className="my-3">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-6 my-3 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-6 my-3 space-y-1">{children}</ol>,
  a: ({ href, children }) => (
    <a href={href} className="text-myco-green underline underline-offset-2 break-words">
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-4 border-l-4 border-myco-green/40 bg-myco-green/5 px-4 py-2 rounded-r">
      {children}
    </blockquote>
  ),
  code: ({ children }) => (
    <code className="rounded bg-[#f3eee6] px-1 py-0.5 text-[0.9em] break-words">{children}</code>
  ),
  table: ({ children }) => (
    <div className="my-4 overflow-x-auto">
      <table className="min-w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border-b-2 border-[#A87146]/30 px-2 py-1.5 text-left font-semibold align-bottom">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="border-b border-[#A87146]/10 px-2 py-1.5 align-top">{children}</td>,
  hr: () => <hr className="my-8 border-[#A87146]/20" />,
};

export function Markdown({ children }: { children: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{children}</ReactMarkdown>;
}
