import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { OPEN_V1, PUBLIC_METHOD_LABEL, publicMethodLabel, publicModelName, SERVED_V1 } from "./publicNames";
import { MODEL_CREDIT_LINE, modelName, releaseNote } from "./publicView";
import { modelCredit } from "./howItWorks";
import { modelLabel } from "./api";
import { BASE_LABEL } from "./modelChoice";
import { twoModelLabel, TWO_MODELS } from "./twoModels";
import { TwoModelsSection } from "../pages/Results";

// Steve, 2026-10-11: outside the Research section the site says "MycoMap Vision", never
// the backbone it was built on.
const BACKBONE_WORDS = /bioclip|fine-?tun/i;

const BACKBONES = [SERVED_V1, OPEN_V1, "bioclip-2-ft-20261007-165400", "bioclip-2-ft-20261014-003539",
                   "bioclip-2-448-ft-20261030-120000", "bioclip-2", "bioclip-2-ft-sample", "dinov2-b14",
                   "dinov3-l16"];

test("every model is MycoMap Vision to visitors, by its version", () => {
  assert.equal(publicModelName(SERVED_V1), "MycoMap Vision v1");
  assert.equal(publicModelName(OPEN_V1), "MycoMap Vision v1-open");
  assert.equal(publicModelName("bioclip-2-ft-20261007-165400"), "MycoMap Vision v0");
  assert.equal(publicModelName("bioclip-2-ft-20261014-003539"), "MycoMap Vision v0-open");
  assert.equal(publicModelName("bioclip-2-448-ft-20261030-120000"), "MycoMap Vision (development, 448 px)");
  assert.equal(publicModelName("bioclip-2-ft-sample"), "MycoMap Vision (development, 224 px)");
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

// Steve's decision D (2026-10-11): no internal ids such as comparison runs in public text.
test("pages outside Research show no comparison run id", () => {
  for (const f of publicSources()) {
    const src = read(f);
    assert.doesNotMatch(src, /\b\d{8}-\d{6}-[0-9a-f]{6}\b/, `${f}: a comparison id written out`);
    assert.doesNotMatch(src, /\{[^}]*comparisonId[^}]*\}/, `${f}: a comparison id rendered`);
  }
});

// The one exception (Steve, 2026-10-22, the approved How it works redesign): its small model
// credit and "Model in use" card name the base model, through modelCredit (howItWorks.ts).
// No other page outside Research uses it, and the page's own text still never names it.
test("only How it works' model credit names the base model, and only through modelCredit", () => {
  const users = publicSources().filter((f) => /\bmodelCredit\(/.test(read(f)));
  assert.deepEqual(users, ["pages/About.tsx"]);
});

// Steve, 2026-10-23: the models' names. The site's 336 px fine-tune is MycoMap Vision v1, the
// paper's open-data one v1-open, and the unchanged base model is BioCLIP 2, said as such.
test("the served model is MycoMap Vision v1, the paper's open model v1-open, the base model BioCLIP 2 unchanged", () => {
  assert.equal(SERVED_V1, "bioclip-2-336-ft-20261018-171420");
  assert.equal(OPEN_V1, "bioclip-2-336-ft-20261020-173106");
  assert.equal(publicModelName(SERVED_V1), "MycoMap Vision v1");
  assert.equal(modelName(SERVED_V1), "MycoMap Vision v1 (public + non-public training data)");
  assert.equal(publicModelName(OPEN_V1), "MycoMap Vision v1-open");
  assert.equal(modelName(OPEN_V1), "MycoMap Vision v1-open (open data only)");
  assert.deepEqual(TWO_MODELS.models.map((m) => m.backbone), [SERVED_V1, OPEN_V1]);
  for (const base of ["bioclip-2", "bioclip-2-336", "bioclip-2-448"]) {
    assert.ok(modelName(base).startsWith("BioCLIP 2 (base model, not fine-tuned)"), modelName(base));
  }
});

const WEB_SRC = new URL("../", import.meta.url);
function sources(dir = WEB_SRC): { file: string; text: string }[] {
  const out: { file: string; text: string }[] = [];
  for (const name of readdirSync(dir)) {
    const url = new URL(name, dir);
    if (statSync(url).isDirectory()) out.push(...sources(new URL(`${name}/`, dir)));
    else if (/\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name)) {
      out.push({ file: url.pathname.split("/src/").pop()!, text: readFileSync(url, "utf-8") });
    }
  }
  return out;
}
/** Source text without its comments: what can reach a page. */
const code = (src: string) => src.replace(/\/\*[^]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

test("no fine-tuned model is named BioCLIP 2 anywhere: it is the base model, the credit line or a citation", () => {
  const all = sources();
  assert.ok(all.some((s) => s.file === "pages/Results.tsx"), all.map((s) => s.file).join(","));
  // Every fine-tune id the site's code knows, plus each kind of name.
  const ids = new Set(["bioclip-2-448-ft-20261030-120000", "bioclip-2-ft-sample", "dinov3-l16-ft-20261101-000000"]);
  for (const { text } of all) for (const m of text.matchAll(/\b[a-z0-9-]+-ft-(?:\d{8}-\d{6}|sample)\b/g)) ids.add(m[0]);
  assert.ok(ids.has(SERVED_V1) && ids.has(OPEN_V1));
  for (const id of ids) {
    const credit = modelCredit(id);
    for (const label of [modelName(id), publicModelName(id), modelLabel(id, "nearest").name, credit.name, credit.title]) {
      assert.doesNotMatch(label, /bioclip|fine-?tun|frozen/i, `${id}: ${label}`);
      assert.match(label, /^MycoMap Vision/, `${id}: ${label}`);
    }
  }
  for (const m of TWO_MODELS.models) assert.doesNotMatch(twoModelLabel(m), /bioclip/i);
  // Written out in the code, "BioCLIP 2" is only the base model as itself, the credit line, its
  // citation (its text or BibTeX title), its zero-shot gate, or the base model's own name in a table of bases.
  const allowed = /BioCLIP 2(?= \(base model, not fine-tuned| \(Imageomics\) fine-tuned on DNA-verified records\.|: [eE]mergent [pP]roperties|'s zero-shot| \(Gu et al\. 2025\)|")/g;
  for (const { file, text } of all) {
    const left = code(text).replace(allowed, "");
    assert.doesNotMatch(left, /BioCLIP 2/, `${file}: BioCLIP 2 named outside the base model and its credit`);
    assert.doesNotMatch(code(text), /BioCLIP 2,[^\n]*fine-?tuned|frozen BioCLIP|BioCLIP 2 frozen/i, file);
  }
});

test("the credit line is said where the models are explained: Research's two models, Protocols / For researchers, How it works", () => {
  assert.equal(MODEL_CREDIT_LINE, "MycoMap Vision is BioCLIP 2 (Imageomics) fine-tuned on DNA-verified records.");
  const html = renderToStaticMarkup(createElement(TwoModelsSection));
  const text = html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'");
  assert.ok(text.includes(MODEL_CREDIT_LINE));
  assert.equal((text.match(/BioCLIP/g) ?? []).length, 1, "the two models section names BioCLIP 2 only in the credit line");
  assert.match(text, /MycoMap Vision v1\b/);
  assert.match(text, /public \+ non-public training data/);
  assert.match(text, /MycoMap Vision v1-open/);
  assert.match(text, /open data only/);
  // Protocols and For researchers both say the line where they explain the models.
  for (const page of ["pages/Protocols.tsx", "pages/ForResearchers.tsx", "pages/ResearchLanding.tsx"]) {
    assert.match(read(page), /\{MODEL_CREDIT_LINE\}/, page);
  }
  assert.equal(modelCredit(SERVED_V1).madeBy, MODEL_CREDIT_LINE, "How it works' model credit");
});
