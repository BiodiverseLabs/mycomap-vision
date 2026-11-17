// A tiny Markdown reader for the pinned instructions (instructions.md): ## and ### headings,
// paragraphs, nested "- " lists, **bold** and *italic*, and nothing else. It returns plain
// data that the page renders as React elements, so no HTML in the text is ever passed
// through: a "<" stays a character.

export type Inline = { kind: "text" | "bold" | "italic"; text?: string; children?: Inline[] };
export type ListItem = { content: Inline[]; children: ListItem[] };
export type Block =
  | { kind: "heading"; level: 2 | 3; content: Inline[] }
  | { kind: "paragraph"; content: Inline[] }
  | { kind: "list"; items: ListItem[] };

/** **bold** (which may hold *italic*) and *italic*; an unmatched marker stays text. */
export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let plain = "";
  let i = 0;
  const flush = () => {
    if (plain) out.push({ kind: "text", text: plain });
    plain = "";
  };
  while (i < text.length) {
    if (text.startsWith("**", i)) {
      const end = text.indexOf("**", i + 2);
      if (end > i + 2) {
        flush();
        out.push({ kind: "bold", children: parseInline(text.slice(i + 2, end)) });
        i = end + 2;
        continue;
      }
    } else if (text[i] === "*") {
      let end = text.indexOf("*", i + 1);
      while (end > 0 && text[end + 1] === "*") end = text.indexOf("*", end + 2);
      if (end > i + 1) {
        flush();
        out.push({ kind: "italic", children: [{ kind: "text", text: text.slice(i + 1, end) }] });
        i = end + 1;
        continue;
      }
    }
    plain += text[i];
    i += 1;
  }
  flush();
  return out;
}

const LIST_LINE = /^( *)- (.*)$/;
const HEADING = /^(#{2,3}) (.*)$/;

export function parseMarkdown(source: string): Block[] {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];
  let stack: { indent: number; items: ListItem[] }[] = [];

  const endPara = () => {
    if (para.length) blocks.push({ kind: "paragraph", content: parseInline(para.join(" ")) });
    para = [];
  };
  const endList = () => {
    if (stack.length) blocks.push({ kind: "list", items: stack[0].items });
    stack = [];
  };

  for (const line of lines) {
    const list = LIST_LINE.exec(line);
    if (list) {
      endPara();
      const indent = list[1].length;
      const item: ListItem = { content: parseInline(list[2].trim()), children: [] };
      if (!stack.length) stack.push({ indent, items: [] });
      while (stack.length > 1 && indent < stack[stack.length - 1].indent) stack.pop();
      const top = stack[stack.length - 1];
      if (indent > top.indent && top.items.length) {
        const parent = top.items[top.items.length - 1];
        stack.push({ indent, items: parent.children });
      }
      stack[stack.length - 1].items.push(item);
      continue;
    }
    if (!line.trim()) {
      endPara();
      endList();
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      endPara();
      endList();
      blocks.push({ kind: "heading", level: heading[1].length as 2 | 3, content: parseInline(heading[2].trim()) });
      continue;
    }
    endList();
    para.push(line.trim());
  }
  endPara();
  endList();
  return blocks;
}

/** The text of some inline content, markers dropped (for tests and labels). */
export function plainText(content: Inline[]): string {
  return content.map((c) => (c.kind === "text" ? c.text ?? "" : plainText(c.children ?? []))).join("");
}
