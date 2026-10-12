// Mint a Vision app token with mycomap.org's own signer (its services/visionAccess.ts), for
// vision_e2e.py --private-pem. Run from mycomap.org's artifacts/api-server folder
// (VISION_E2E_MINT_CWD), which has tsx:
//   node --import tsx mint.mts <private.pem> <aud> <sub> <plan> <limit|null> [now-offset-seconds]
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const src = path.resolve(process.cwd(), "src/services/visionAccess.ts");
const { signVisionAppToken } = await import(pathToFileURL(src).href);
const [pem, aud, sub, plan, limit, offset] = process.argv.slice(2);
const key = crypto.createPrivateKey(fs.readFileSync(pem));
const now = Math.floor(Date.now() / 1000) + Number(offset ?? 0);
const { token } = signVisionAppToken(
  { sub, aud, plan, limit: limit === "null" ? null : Number(limit) }, key, now);
process.stdout.write(token);
