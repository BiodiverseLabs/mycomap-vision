import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { PUBLIC_METHOD_LABEL, publicMethodLabel, publicModelName } from "./publicNames";
import { releaseNote } from "./publicView";
import { BASE_LABEL } from "./modelChoice";

// Steve, 2026-10-11: outside the Research section the site says "MycoMap Vision", never
// the backbone it was built on.
const BACKBONE_WORDS = /bioclip|fine-?tun/i;

const BACKBONES = ["bioclip-2-ft-20261007-165400", "bioclip-2-ft-20270115-090000", "bioclip-2",
                   "bioclip-2-ft-sample", "dinov2-b14", "dinov3-l16"];

test("every model is MycoMap Vision to visitors, dated by the records it learned from", () => {
  assert.equal(publicModelName("bioclip-2-ft-20261007-165400"), "MycoMap Vision · October 2026 model");
  assert.equal(publicModelName("bioclip-2-ft-20270115-090000"), "MycoMap Vision · January 2027 model");
  assert.equal(publicModelName("bioclip-2-ft-sample"), "MycoMap Vision · test model");
  assert.equal(publicModelName("bioclip-2"), "MycoMap Vision");
  for (const b of BACKBONES) {
    assert.doesNotMatch(publicModelName(b), BACKBONE_WORDS, b);
    assert.doesNotMatch(releaseNote({ backbone: b, embedded_photos: 1, trained_through: "2026-09-07" }),
                        BACKBONE_WORDS, b);
  }
  assert.equal(publicModelName("external:inat-cv"), "iNaturalist's computer vision");
});

test("methods are named in plain words, the place model included", () => {
  assert.equal(publicMethodLabel("nearest+mean"), "closest specimens + species average");
  assert.equal(publicMethodLabel("nearest+mean+place"), "closest specimens + species average + place and date");
  assert.equal(publicMethodLabel("nearest"), "closest specimen");
  assert.equal(publicMethodLabel("some+new"), "some + new", "an unknown method still reads as words");
  for (const label of [...Object.values(PUBLIC_METHOD_LABEL), ...Object.values(BASE_LABEL)]) {
    assert.doesNotMatch(label, BACKBONE_WORDS);
    assert.doesNotMatch(label, /nearest/i, "\"closest\", as in \"Closest DNA-verified specimens\"");
  }
});

// Every page outside Research, found from the routes so a new page is covered too, plus
// the layout around them. The public answer is rendered by the Identify page.
const SRC = new URL("../", import.meta.url);
const read = (rel: string) => readFileSync(new URL(rel, SRC), "utf-8");

function publicSources(): string[] {
  const app = read("App.tsx");
  const fileOf = new Map<string, string>();
  for (const m of app.matchAll(/import \{([^}]+)\} from "@\/(pages\/\w+)"/g)) {
    for (const name of m[1].split(",").map((s) => s.trim()).filter(Boolean)) fileOf.set(name, `${m[2]}.tsx`);
  }
  const pages = [...app.matchAll(/<Route path="([^"]+)" component=\{(\w+)\}/g)]
    .filter(([, path]) => !path.startsWith("/research"))
    .map(([, , component]) => fileOf.get(component)!);
  assert.ok(pages.includes("pages/Identify.tsx") && pages.includes("pages/About.tsx")
            && pages.includes("pages/GetInvolved.tsx"), `routes found: ${pages}`);
  // Components those pages use, one level down, and the layout.
  const components = new Set(["components/Layout.tsx"]);
  for (const page of pages) {
    for (const m of read(page).matchAll(/from "@\/(components\/\w+)"/g)) components.add(`${m[1]}.tsx`);
  }
  return [...new Set([...pages, ...components])];
}

test("no page outside Research names the backbone", () => {
  const files = publicSources();
  for (const f of files) assert.doesNotMatch(read(f), BACKBONE_WORDS, f);
  assert.doesNotMatch(readFileSync(new URL("../index.html", SRC), "utf-8"), BACKBONE_WORDS, "index.html");
});

test("pages outside Research use the public names and never show a raw backbone id", () => {
  for (const f of publicSources()) {
    const src = read(f);
    assert.doesNotMatch(src, /\bmodel(Name|Label)\(/, `${f}: use publicModelName / publicMethodLabel`);
    assert.doesNotMatch(src, /title=\{[^}]*backbone\}/, `${f}: a backbone id in a tooltip`);
    assert.doesNotMatch(src, />\s*\{[^}]*\.backbone\}\s*</, `${f}: a backbone id as text`);
  }
});
