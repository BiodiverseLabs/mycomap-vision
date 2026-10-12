import { test } from "node:test";
import assert from "node:assert/strict";
import { inlineParts } from "./apiOutline";

test("the outline's inline marks become code, bold and links", () => {
  assert.deepEqual(inlineParts("Send `photos` to **available** [API keys](https://mycomap.org/api-docs) now"), [
    { kind: "text", text: "Send " },
    { kind: "code", text: "photos" },
    { kind: "text", text: " to " },
    { kind: "bold", text: "available" },
    { kind: "text", text: " " },
    { kind: "link", text: "API keys", href: "https://mycomap.org/api-docs" },
    { kind: "text", text: " now" },
  ]);
});

test("only https links are links; anything else stays text", () => {
  for (const text of ["[x](javascript:alert(1))", "[x](http://example.com)", "plain"]) {
    assert.deepEqual(inlineParts(text), [{ kind: "text", text }]);
  }
});

test("an empty string has no parts", () => {
  assert.deepEqual(inlineParts(""), []);
});
