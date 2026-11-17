import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { parseInline, parseMarkdown, plainText, type Block, type ListItem } from "./markdown";

// The instructions the reviewers read, pinned (Vision Coordinator, 16 Nov 2026). The page shows
// this file and nothing else as its instructions; any change to it must change this pin.
const PINNED_SHA256 = "14d4906be0d1a96562c571454b8c18c9554c37dd671304c4ac1d3dc9e8b4f1a3";
const source = readFileSync(new URL("./instructions.md", import.meta.url));

test("the committed instructions are the pinned text, byte for byte", () => {
  assert.equal(createHash("sha256").update(source).digest("hex"), PINNED_SHA256);
});

test("the instructions read as their headings, paragraphs and nested lists", () => {
  const blocks = parseMarkdown(source.toString("utf8"));
  const headings = blocks.filter((b) => b.kind === "heading").map((b) => [b.level, plainText(b.content)]);
  assert.deepEqual(headings, [
    [2, "Instructions"], [3, "What each item shows"], [3, "What we ask for each item"], [3, "Please"], [3, "Saving"],
  ]);
  const ask = blocks.filter((b): b is Extract<Block, { kind: "list" }> => b.kind === "list")[1];
  assert.equal(plainText(ask.items[0].content), "Verdict.");
  assert.deepEqual(ask.items[0].children.map((c) => plainText(c.content).split(":")[0]), ["Supported", "Wrong", "Cannot tell"]);
  assert.equal(ask.items.length, 6);
  // No Markdown marker is left in what is shown.
  const flat = (items: ListItem[]): string[] => items.flatMap((i) => [plainText(i.content), ...flat(i.children)]);
  const shown = blocks.flatMap((b) => (b.kind === "list" ? flat(b.items) : [plainText(b.content)])).join("\n");
  assert.ok(!shown.includes("*") && !shown.includes("#"), "a marker was left in the text");
  assert.ok(shown.includes("Genus sp. 'CA01'"));
});

test("bold and italic are read, bold may hold italic, and an unmatched star stays a star", () => {
  assert.deepEqual(parseInline("a **b** c"), [
    { kind: "text", text: "a " }, { kind: "bold", children: [{ kind: "text", text: "b" }] }, { kind: "text", text: " c" },
  ]);
  assert.deepEqual(parseInline("*Genus* sp. 'CA01'"), [
    { kind: "italic", children: [{ kind: "text", text: "Genus" }] }, { kind: "text", text: " sp. 'CA01'" },
  ]);
  assert.deepEqual(parseInline("**x *y* z**")[0].children?.map((c) => c.kind), ["text", "italic", "text"]);
  assert.deepEqual(parseInline("2 * 3"), [{ kind: "text", text: "2 * 3" }]);
});

test("HTML in the text is only text: nothing is passed through as markup", () => {
  const blocks = parseMarkdown('Hello <script src="x"></script> <img src=x onerror=alert(1)>');
  assert.deepEqual(blocks, [{ kind: "paragraph", content: [{ kind: "text", text: 'Hello <script src="x"></script> <img src=x onerror=alert(1)>' }] }]);
});

test("a nested list hangs under the item above it and the outer list carries on after it", () => {
  const blocks = parseMarkdown("- one\n  - one.a\n  - one.b\n- two\n\nAfter.");
  assert.equal(blocks.length, 2);
  const list = blocks[0] as Extract<Block, { kind: "list" }>;
  assert.deepEqual(list.items.map((i) => plainText(i.content)), ["one", "two"]);
  assert.deepEqual(list.items[0].children.map((i) => plainText(i.content)), ["one.a", "one.b"]);
  assert.equal(blocks[1].kind, "paragraph");
});
