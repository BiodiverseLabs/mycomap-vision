# Deploying vision.mycomap.org

Vision's public site runs on its own Lightsail instance, separate from
mycomap.org's boxes. The box only **answers**: it turns an uploaded photo into
a vector on its CPU, compares it with the reference vectors, and returns the
result. That takes a few seconds per photo. All the heavy work stays on
on-demand GPU instances that shut themselves down: downloading photos,
embedding them, fine-tuning, and comparing models (see
[`deploy/aws/README.md`](../deploy/aws/README.md)). Their results reach the box
as a **release** in S3.

| Piece | Where |
| --- | --- |
| Instance | Lightsail, Ohio (us-east-2, the bucket's region), Ubuntu 24.04, **16 GB RAM**, 4 vCPUs (as of 2026-10-25; built on the 4 GB, 2 vCPU plan), + 4 GB swap |
| App | `/var/www/mycomap-vision`, a clone of the repo; `.venv` with CPU-only torch |
| Research pages | `/var/www/mycomap-vision-research`, a clone of the private research repo (experiments, paper draft) |
| Server | systemd unit `mycomap-vision`: `mv serve` on 127.0.0.1:8010 (API **and** the built site) |
| Settings | `/etc/mycomap-vision/vision.env` (600), linked as the repo's `.env` |
| Secrets | `/etc/mycomap-vision/session-secret` (made on the box), `signin-public.pem` (public key, not secret), `org-vision-key` (mycomap.org's `VISION_API_KEY`, typed by a person), `~/.aws/credentials` (read-only key, typed by a person) |
| Data | `/srv/mycomap-vision/releases/<id>/`, the live one named in `current.txt`; model weights in `/srv/mycomap-vision/hf` |
| Front | nginx + Let's Encrypt; Cloudflare DNS for `vision.mycomap.org` |

**Sign-in.** Only people with a mycomap.org account can use the site before launch.
mycomap.org vouches for them through its sign-in bridge, which its dev site also
uses (`MV_SIGNIN`, see `src/mycomap_vision/signin.py`):

- `all`: every page and API call (pre-launch).
- `identify`: only identifications need a sign-in.
- `off`: open to everyone.

The box holds only the bridge's **public** key. It can check a sign-in but never
create one.

## First-time setup

Steps marked **(Steve)** involve a console, a key or production and are done by
the project owner; the rest are run over SSH by an operator.

### 1. Create the instance (Steve, Lightsail console)

1. Create instance: **Ohio (us-east-2)**, Linux/Unix, **OS only → Ubuntu 24.04 LTS**,
   the **16 GB RAM** plan (the box began on 4 GB; this guide's memory notes were
   written for that). Name it `mycomap-vision`.
2. Networking → **attach a static IP**.
3. Networking → firewall (IPv4 and IPv6): SSH 22, HTTP 80, HTTPS 443.
4. Snapshots → turn on **automatic snapshots**.
5. Add each operator's SSH public key to `~/.ssh/authorized_keys` on the box.

### 2. Clone the repo onto the box (Steve: the deploy key)

As `ubuntu` on the box:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "vision box deploy key"
cat ~/.ssh/id_ed25519.pub
```

Add that key at GitHub → `BiodiverseLabs/mycomap-vision` → Settings → Deploy
keys, **read access only**. Then:

```bash
ssh -o StrictHostKeyChecking=accept-new -T git@github.com   # expect "successfully authenticated"
sudo mkdir -p /var/www && sudo chown ubuntu:ubuntu /var/www
git clone git@github.com:BiodiverseLabs/mycomap-vision.git /var/www/mycomap-vision
cd /var/www/mycomap-vision
bash deploy/lightsail/provision.sh
```

`provision.sh` installs Python with CPU-only torch, Node (for the web build),
nginx, certbot and the AWS CLI. It adds swap, makes the folders and the session
secret, installs the systemd unit, and downloads the BioCLIP 2 weights. It is
safe to re-run.

### 2b. The private research pages (Steve: a second deploy key)

Research > Experiments and Paper show unpublished results, so they are not in this
public repo. They come from the private repo `BiodiverseLabs/mycomap-vision-research`,
cloned next to the app as `/var/www/mycomap-vision-research` (`config.RESEARCH_DIR`;
`MV_RESEARCH_DIR` overrides it). GitHub refuses one deploy key on two repos, so the box
gets a second key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_research -N "" -C "vision box research key"
cat ~/.ssh/id_research.pub
```

Add that key at GitHub → `BiodiverseLabs/mycomap-vision-research` → Settings → Deploy
keys, **read access only**. Then:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github-research
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_research
  IdentitiesOnly yes
EOF
git clone github-research:BiodiverseLabs/mycomap-vision-research.git /var/www/mycomap-vision-research
```

`deploy.sh` pulls it on every deploy. A copied folder works too (it is then left as it
is), and without either the two pages are empty and the deploy says so.

### 3. Settings

In `/etc/mycomap-vision/vision.env`, replace `<BUCKET>` and `<REGION>` with the
values from the laptop's `.env` (the folder is root's, so edit with sudo:
`sudo sed -i 's/<BUCKET>/.../; s/<REGION>/.../' /etc/mycomap-vision/vision.env`,
or `sudo nano`). Then put the sign-in bridge's public key in
place. It is the same public key mycomap.org's dev site uses to check its
sign-ins. It is public, so pasting it is fine:

```bash
sudo tee /etc/mycomap-vision/signin-public.pem >/dev/null   # paste the PEM, then Ctrl-D
sudo chown ubuntu:ubuntu /etc/mycomap-vision/signin-public.pem
```

### 4. Read-only AWS key for the box (Steve)

Lightsail instances cannot use an EC2 instance role, so the box gets its own IAM
user. The user can read `releases/` in the bucket and nothing else.

1. On the laptop: `.venv/Scripts/mv aws-policies` prints `box-policy.json`.
2. IAM → Users → Create user `mycomap-vision-box` (no console access) → Attach
   policies directly → Create policy → JSON: paste `box-policy.json` → attach.
3. The user → Security credentials → Create access key → Command Line Interface.
4. On the box, typed by you:

```bash
aws configure --profile mycomap-vision-box      # region: the bucket's; output: json
```

### 5. Let mycomap.org sign people in here (Steve, production)

On the mycomap.org server, add this to the API process's environment. It takes
effect at mycomap.org's next deploy:

```js
SIGNIN_BRIDGE_SITES: "https://vision.mycomap.org",
```

### 5b. Photographers' answers from mycomap.org (Steve, production + box)

The box reads who granted or withdrew permission from mycomap.org every 5
minutes (`permissions.py`). Until it can, all-rights-reserved photos are never
shown. One key, known to both sides:

1. On the laptop (Git Bash): `openssl rand -hex 32`
2. On the mycomap.org box, in the same `env` block as above:
   `VISION_API_KEY: "<key>",` (takes effect at the next routine deploy).
3. On this box, typed by you:

```bash
sudo tee /etc/mycomap-vision/org-vision-key >/dev/null   # paste the key, then Ctrl-D
sudo chown ubuntu:ubuntu /etc/mycomap-vision/org-vision-key
sudo chmod 600 /etc/mycomap-vision/org-vision-key
```

`vision.env` already has `MV_ORG_BASE_URL`, `MV_ORG_VISION_KEY_FILE` and
`MV_LICENSE_REFRESH_HOURS=24`. After a restart, `.venv/bin/mv permissions` shows
the last read.

Only a well-formed answer replaces the box's copy. mycomap.org never deletes an
answer, so an answer that leaves out anyone the box already has (an empty one
included) is refused as suspicious: the last good list stays, withdrawals
included, and the refusal shows under `last_attempt` in `mv permissions` and as an
error in `journalctl -u mycomap-vision`. If people really were removed on
mycomap.org, take the shorter list once with
`.venv/bin/mv permissions --sync --accept-shrink`. The last good list is also kept
in `/srv/mycomap-vision/state/photo-permissions.json`, outside the releases, and
put back at startup when it is newer than the new release's manifest.

### 6. First release (laptop, then box)

A release is what the site serves: a snapshot of the manifest, the served
backbones' vectors, and any fine-tuned weights. On the laptop:

```
.venv/Scripts/mv release --backbones bioclip-2 --label bioclip2 --make-current
```

On the box:

```bash
cd /var/www/mycomap-vision
.venv/bin/mv pull-release
```

`pull-release` downloads into `releases/<id>/` and checks every file's size and
sha256. Only then does it switch `current.txt`, so a broken download never goes
live. It keeps the previous release for rollback.

### 7. First deploy (app only; nginx comes next)

```bash
BASE=http://127.0.0.1:8010 bash deploy/lightsail/deploy.sh
```

### 8. DNS and certificate (Steve: Cloudflare)

In Cloudflare, zone `mycomap.org`: add an **A record `vision` → the static IP**,
proxy status **DNS only** (grey cloud) for now. When it resolves
(`dig +short vision.mycomap.org`), on the box:

```bash
sudo certbot certonly --webroot -w /var/www/html -d vision.mycomap.org \
  --email info@mycomap.org --agree-tos --no-eff-email
```

Expiry warnings go to info@mycomap.org (Steve, 2026-09-29; the Library's account
uses the same address).

### 9. nginx site

```bash
sudo cp deploy/lightsail/nginx-vision-headers.conf /etc/nginx/snippets/mycomap-vision-headers.conf
sudo cp deploy/lightsail/nginx-vision.conf.template /etc/nginx/sites-available/mycomap-vision
sudo sed -i 's/<PORT>/8010/' /etc/nginx/sites-available/mycomap-vision
sudo ln -sf /etc/nginx/sites-available/mycomap-vision /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

`deploy.sh` never touches nginx. When either file changes in the repo, copy **both** again
with the first three lines above, then `sudo nginx -t && sudo systemctl reload nginx`.
They change together: the snippet holds no Referrer-Policy, and the site file sets it,
`no-referrer` with no access log for the E17 review page's `/review/` and `/api/e17/`.

### 10. Verify

```bash
bash deploy/lightsail/verify.sh
RESOLVE_IP=127.0.0.1 bash deploy/lightsail/verify.sh
```

Then sign in from a browser. Optionally, switch the Cloudflare record to
**Proxied**: `cloudflare-realip.sh` has already told nginx to take the visitor's
address from Cloudflare. If every check then fails with 403, Cloudflare is
blocking curl from the box itself, as it does on mycomap.org. That is not an
nginx fault; use `RESOLVE_IP=127.0.0.1`.

## Routine deploy (code)

```bash
git push origin main      # from the laptop
# then on the box:
cd /var/www/mycomap-vision && bash deploy/lightsail/deploy.sh
```

It pulls `main`, installs, builds the web app into a new folder before swapping
it in, restarts the server, waits for it and runs `verify.sh`. If the pull changed
`deploy.sh` itself, it starts the new copy first (bash would otherwise finish the
old one). The server loads the model and index at startup (`MV_PRELOAD=default`,
about 30 s); the deploy waits for that and prints how long it took, and
`/api/health` shows it under `preload`. The deploy is
done when verify passes.

## New release (model data)

```
.venv/Scripts/mv release --backbones bioclip-2 --label <what changed> --make-current   # laptop
```
```bash
.venv/bin/mv pull-release && sudo systemctl restart mycomap-vision                     # box
```

To roll back, run `.venv/bin/mv pull-release --release <previous id>` and
restart.

### After sealing a benchmark

The box learns which records are held out (`benchmark_holdouts`, holdouts.py) only
from a release's manifest: its nightly update keeps adding a sealed benchmark's
records until it serves a release made after the seal. So, with this code deployed to
the box, cut and pull a release right after `mv heldout freeze --holdout` or
`mv holdout add` on the laptop (the two commands above). The release carries the
held-out list but none of the benchmark's own tables (`heldout_*`), which stay on the
laptop.

### A calibration borrowed from another model

A model served before it has a calibration of its own (the v2 beta: Steve, 2026-11-07,
the step-1 BioCLIP 2.5 fine-tune's) gets that model's calibration as rows of its own with
`mv calibration copy` (calcopy.py), written exactly where serving reads them
(`evaluate.latest_calibration`), each with where it came from, when, who and why, and no
scores (the scoreboard never lists them). On the box the manifest is the nightly layer's,
the live one, so stop the service, back it up, dry-run, then write:

```bash
sudo systemctl stop mycomap-vision
sqlite3 $L/manifest.sqlite ".backup $R/state/layer-manifest.bak-$STAMP-pre-calcopy.sqlite"
.venv/bin/mv calibration copy --from <source> --to <target> --from-file <source>.calibration-v1-val.json \
    --note "<why>" --by <who> --dry-run
.venv/bin/mv calibration copy ... (the same, without --dry-run)
sudo systemctl start mycomap-vision
```

`--from-file` takes the step-1 calibration file (its `photos_only` and `place` blocks: the
served by-specimen method and its place variant); without it the source's own rows are
copied (`--from-manifest` reads them from another manifest, read-only). It refuses if the
target already has rows for those methods (`--replace` adds the copies as newer rows and
deletes nothing), and prints what serving would read afterwards, with a warning when the
served method (`MV_DEFAULT_METHOD`) would still read none. Undo: restore the backup with the
service stopped, or delete the rows it printed.

## Switching the default model

Which model answers an identification that names none is set per caller, so the site's
visitors and the app can move to a new model while mycomap.org's auto-validation stays on
the old one until its own switch (Steve, 2026-11-24). v1 = `bioclip-2-336-ft-20261018-171420`,
v2 = `bioclip-2.5-336-ft-20261122-171257`.

### The settings

| Setting | What it decides | Unset |
|---|---|---|
| `MV_ALLOWED_BACKBONES` | which models can be served at all | (required) |
| `MV_DEFAULT_BACKBONE` | mycomap.org's model and everything that isn't the site or the app: API v1 with the server key or a developer key (identify, by-observation, kept answers, `/api/v1/models`), taxon counts, the nightly advance predictions, Human Feedback re-asks, excluded-photo maps, How it works (`/api/how-good`), the public health's "MycoMap Vision" | the first ready model **by name** (fragile: `bioclip-2-336…` sorts before `bioclip-2.5…`) |
| `MV_SITE_DEFAULT_BACKBONE` | the site's visitors and the app: the Identify page's first choice, the site's identifications that name no model (uploads and "from a link"), and API v1 with an app token (the app's sessions and identify). An app token may also name it through API v1 | `MV_DEFAULT_BACKBONE`: exactly as before |
| `MV_API_V1_BACKBONES` | which models API v1 answers mycomap.org's server key and developer keys with (an app token also gets `MV_SITE_DEFAULT_BACKBONE`) | the default model only |
| `MV_PRELOAD` | which models load at startup (`default` is `MV_DEFAULT_BACKBONE`'s) | `default` |
| `MV_PHOTO_TAGGER` | tagger files, comma list; the **first** drives the nightly photo review and health's `photo_tagger` | no tagging |
| `MV_CROSS_BASE_GATE` | a model that borrows another base's gate (v2 → `bioclip-2-336`, a second pass through v1's tower) | none: such a model is refused |

Who is which caller is told by the credential, not the path: the app and developer keys
both call `POST /api/v1/identify` with no model, so an app token (`mvt.`, kind `app`)
gets the site's default and a `vision_` key (kind `key`) or mycomap.org's server key
(kind `server`) gets `MV_DEFAULT_BACKBONE`.

A named model's label follows the setting: v2 is "MycoMap Vision v2 (beta)" while it is
opt-in, and "MycoMap Vision v2" once it is `MV_SITE_DEFAULT_BACKBONE` (Steve,
2026-11-24), on the site (the page shell carries `<meta name="mv-site-default">`), in the
app's answers (`model.name`) and in the public `/api/health`. An answer carries a model
card only for a model in the release's `release.json` backbones, so v2's answers keep
`version: null` and `data: null` until a release includes it, default or not.

### Refused, never another model

- A named default that isn't served here (not allowed, no embeddings, refused for want of its
  gate) is refused on its callers' paths: the site 503, API v1 503 `not_served`, predictions
  `skipped`. Nothing answers in its place, and the other callers are unaffected. A
  `[default]` line at startup and `default_backbone` (with `site` when the site default is
  set) in the server-key `/api/health` say why.
- Settings under which some caller's default is a model it may not use are said at startup
  as `[default] REFUSED: ...` and listed in the server-key `/api/health` under
  `default_backbone.conflicts` (never in the public health): `MV_DEFAULT_BACKBONE` not in
  `MV_ALLOWED_BACKBONES` or not in `MV_API_V1_BACKBONES` (trap 1 below),
  `MV_SITE_DEFAULT_BACKBONE` not in `MV_ALLOWED_BACKBONES`, or `MV_SITE_DEFAULT_BACKBONE` set
  without `MV_DEFAULT_BACKBONE`. After every restart that changes these, check:

```bash
sudo journalctl -u mycomap-vision --since -5min | grep '\[default\]'
# The key goes in a header made by printf (a shell builtin), never on a command line.
curl -s -H @<(printf 'Authorization: Bearer %s\n' "$(cat /etc/mycomap-vision/org-vision-key)") \
    http://127.0.0.1:8010/api/health | python3 -c 'import json,sys; print(json.load(sys.stdin)["default_backbone"])'
```

Expect no `REFUSED` line and no `conflicts` key.

### Who uses the default, and what a switch does to them

| Caller | Path | Setting | Switch risk |
|---|---|---|---|
| Site visitors | Identify page preselection, uploads and "from a link" naming no model | site | low; the other model stays offered |
| The mobile app | API v1 sessions and identify, app token | site | the app shows `model.name`; with no card it loses the "trained on data to" line |
| **mycomap.org auto-validation** | API v1 by-observation, server key, no `model` | default | **high**: its agreement thresholds were set for v1 |
| mycomap.org record pages, kept answers | `/api/v1/answers`, `/api/v1/models` (answers version) | default | kept answers carry an answers version per model |
| Developer keys (the validation tool) | API v1 with a `vision_` key | default | the tool's agreement rules were set for v1 |
| The paper's prospective test | nightly advance predictions | default | changing model splits the series |
| Human Feedback | re-asks, excluded-photo maps | default | reviewers would see the new model's answers |
| How it works / Get involved | `/api/how-good` | default | a model with no benchmark rows shows an empty section |
| Taxon counts | `/api/v1/taxon-counts` | default | minor |

A site "from a link" answer and mycomap.org's by-observation answer are kept per model, so
while the two defaults differ they are two answers: the site's no longer reuses the one
mycomap.org's Validate Records kept, and is answered afresh once.

### Traps already hit

1. **Default v2 + API v1 list v1 = mycomap.org broken.** With `MV_DEFAULT_BACKBONE=v2` and
   `MV_API_V1_BACKBONES=v1`, every API v1 call naming no model is refused 503 `not_served`,
   and mycomap.org names no model. Now said as `[default] REFUSED`; never set it.
2. **Name-order default.** Unset `MV_DEFAULT_BACKBONE` picks by name. Always set it on a
   multi-model box (and always with `MV_SITE_DEFAULT_BACKBONE`).
3. **Model card.** Only the release's own backbones carry a card (above).
4. **Calibration is per backbone + method + place setting.** A model's borrowed rows
   ("A calibration borrowed from another model") cover only the methods copied; changing
   `MV_DEFAULT_METHOD` or `MV_PLACE_SETTING` needs new rows first, or confidences fall back
   to the method's own temperature.
5. **Tagger order.** v1's tagger file stays first in `MV_PHOTO_TAGGER` while v1 is the
   review model; the nightly photo review uses the first.
6. **Predictions stop if the default isn't ready**: recorded `skipped`, never another model.
7. **The gate depends on v1's tower.** v2's gate is v1's 336 gate read from a second pass
   through v1's tower; without v1 in `MV_ALLOWED_BACKBONES` the frozen `bioclip-2-336`
   tower is loaded instead (more memory).
8. **Memory.** Two models need the raised cap (about 9.7 GB steady, 11-11.5 GB peak,
   estimated); re-measure after each switch.
9. **Paper rules.** The paper's PC agrees before the prospective series' model changes, and
   nothing compares the two models in public before the paper allows it.
10. **Nightly timing.** Never restart during the nightly (08:00 UTC, 0.3-2.5 h).
11. **Preload.** `default` in `MV_PRELOAD` is `MV_DEFAULT_BACKBONE`'s model; list the site's
    default too, or its first visitor waits for it to load.

### The order of switches

**Stage 0: v2 opt-in.** `MV_DEFAULT_BACKBONE=v1`, API v1 v1 only, `MV_SITE_DEFAULT_BACKBONE`
unset. v2 is "MycoMap Vision v2 (beta)" in the model list.

**Stage 1: site visitors and the app on v2.** Add one line, everything else unchanged:

```
MV_SITE_DEFAULT_BACKBONE=bioclip-2.5-336-ft-20261122-171257
```

with `MV_DEFAULT_BACKBONE=bioclip-2-336-ft-20261018-171420` and `MV_API_V1_BACKBONES` unset or
`=bioclip-2-336-ft-20261018-171420`, and v2 in `MV_PRELOAD`. Back up vision.env, restart
outside the nightly, run the checks above, then: `/api/models` has `site_default_backbone` v2
and `site_default_ready` true; an upload on the site names "MycoMap Vision v2"; a
server-key `GET /api/v1/models` still answers "MycoMap Vision" with the release's card.
Rollback: delete the line, restart.

**Stage 2: mycomap.org auto-validation on v2.** Only after about two weeks of paired data
on the same records, mycomap.org's thresholds reviewed (a .org change of its own), its
stored answers and kept-answer versions checked, the PC agreeing, and a dry run of
server-key calls naming v2 on recent records. Then set `MV_API_V1_BACKBONES=v2,v1` **and**
`MV_DEFAULT_BACKBONE=v2` together, in one restart (v2 first in the list is not enough: the
default decides), so no request can meet a default API v1 doesn't offer; v1 stays offered
by name for a fallback. `MV_SITE_DEFAULT_BACKBONE` may then be deleted (it equals the
default). Rollback: restore the vision.env backup, restart.

**Stage 3: v2 everywhere.** Advance predictions, Human Feedback and How it works follow
`MV_DEFAULT_BACKBONE` already; then the two-model serving release, and whether v1 stays
served at all (it carries the gate's second pass: trap 7).

## Nightly update (between releases)

With `MV_NIGHTLY=1` the server keeps the reference set in step with mycomap.org
every night at 3 a.m. Eastern (`nightly.py`): records that stopped being green
leave it and renamed records carry their new name that same night. Records new to
the green list get their iNat details and photos, are embedded on this box's CPU
with the served model (about 0.7 s a photo, at most `MV_NIGHTLY_MAX_PHOTOS` a
night), and join it. Their photos are deleted once embedded; only the vectors stay.

The same night embeds the specimen crops Human Feedback's **Recovered labels** queue keeps
(`recovered_labels.py`): each crop of a photo a person left out as a field slip, unless a
reviewer rejected it, synced into the manifest's `feedback_photo_crops` before the night. The
photo is fetched within the usual download budgets, its box is applied in memory (no cropped
file is ever written), the crop becomes a vector of its own (a negative id), and the photo is
deleted once its crops are embedded. At most `MV_NIGHTLY_MAX_CROPS` (3000) a night; 0 turns it
off. The proposals come from a GPU batch job on the laptop (`mv feedback slip-crops`, OWLv2 +
SAM: about 1.3 s a photo there, 30–55 s on this box's CPU), seeded here with
`mv feedback seed --slip-crops <tsv>`.

The queue's **Priority: last photo** tranche (Steve, 2026-11-01) is a list seeded on its own:
the slip photos of v1 train/val records that every saved photo exclusion leaves with no photo,
ranked slip-only records first, then records with other exclusions too, and within each the
species that would leave training entirely first (`recovered_labels.rank_priority`). Seed it
after any crops it needs, from a store backup taken just before: `mv feedback seed --slip-crops
<tsv> --priority <list.tsv>`, then `mv feedback seal-check`. The list REPLACES the tranche (an
empty list clears it) and only marks items (table `recovered_priority`): no item or decision
changes, and the paper's exports are byte-identical with or without it. The laptop job can be
limited to the list's photos with `mv feedback slip-crops --photos <list.tsv>`. The page opens
on the tranche while it has open photos.

The release is never written. The server works on a layer beside it,
`/srv/mycomap-vision/state/nightly/<release id>/` (a copy of the release's
manifest, the nightly shards and iNat's answers). A new release starts a fresh
layer, and the old one is deleted when the server starts on the new release. When
a backbone's nightly photos pass 20% of the release's, `mv nightly` says a new
release is due.

**Switching releases.** A fresh layer holds what the laptop exported for the release,
often days before the switch. So that the site doesn't go a day without what
mycomap.org validated, renamed or un-validated since (2026-10-18 and -19: 368 new and
887 renamed records each time), the server gives a layer that has never had a good
night, and whose records were exported more than `MV_NIGHTLY_CATCHUP_AFTER_HOURS` (24)
ago, one **catch-up run**: an ordinary night, `MV_NIGHTLY_CATCHUP_DELAY_MINUTES` (10)
after startup and only once the preload is done, so its index rebuild never runs beside
the startup one. It runs on the nightly thread (never beside a scheduled night), with
the usual `MV_NIGHTLY_MAX_PHOTOS` cap, removal guard and lock; a refused or failed one
is tried again at the next restart or the next night. The journal says
`[nightly] one catch-up run from …` at startup. `MV_NIGHTLY_CATCHUP_AFTER_HOURS=0`
turns it off. Its memory is a night's: the green list read from mycomap.org and,
while the rebuilt index is swapped in, the old and new search index side by side
(~0.5 GB each for v1 int8).

The "often confused with" table is mycomap.org's, not the release's: a new layer
takes the previous layer's copy, and a layer with none (the first one) reads
mycomap.org's at startup. `mv confused-with --fetch` after a switch is no longer
needed.

Before turning it on:

1. mycomap.org serves `GET /api/vision/green-records` (the .org branch
   `feat/vision-green-records`, deployed there), with the same `VISION_API_KEY` as
   the photographers' answers (5b).
2. mycomap.org's firewall lets this box's server-to-server calls through to it (Steve).
3. On this box, `.venv/bin/mv nightly --plan` reads mycomap.org and prints what the
   first night would change, changing nothing.

Then add `MV_NIGHTLY=1` to `/etc/mycomap-vision/vision.env` and restart. The first
start copies the release's manifest (a few seconds).

```bash
.venv/bin/mv nightly                          # last runs, layer size, next run
.venv/bin/mv nightly --plan                   # what tonight would change
.venv/bin/mv nightly --now                    # ask the server to run within a minute
.venv/bin/mv nightly --now --accept-removals  # take a large removal it refused
```

An answer that is missing, malformed, empty or short of the total mycomap.org
announced changes nothing. One that removes more than 500 records (or 2% of the
list, when that is more) is refused as suspicious until you take it with
`--accept-removals`. Every run, refused ones included, is listed by `mv nightly`
and logged under `[nightly]` in `journalctl -u mycomap-vision`.

`/api/health` carries a `nightly` block: `state` (`never`, `running`, `ok`,
`failed`), when the last run started and the last one succeeded, what it changed,
the next run, and each backbone's layer size with `new_release_due`. It holds no
error text (the page is public); `mv nightly` on the box has the reason.
`verify.sh` prints it as `ok` / `WARN` lines (`mycomap_vision/nightly_notes.py`):
a failed or refused last run, no successful run for 26 hours, a run going on
3 hours, and a layer past 20% of the release are warnings, never failures.

**Upload meter** (`uploadmeter.py`; Steve, 2026-10-19). Read with mycomap.org's server key
only, `/api/health` also carries `upload_matches`: how often an uploaded photo is one of
Vision's reference photos (within 3 dHash bits of a non-template photo in the release's
`twins.npz`). It's counted per UTC day and path (`site`, `api`, `session`) as uploads,
`metered` (checked against a table), `matched`, photos, `photos_metered` and
`photos_matched`, for the last 7 days with a total. Only those counters are kept, in
`state/vision-usage.sqlite`; no photo, hash, person or record id. Answers are unchanged.
Without a twin table (`"table": false`) uploads are counted but nothing is hashed.

**Identification log** (`identifylog.py`; Steve, 2026-10-20). Every answer the box gives or
serves (site uploads and links, API v1, by-observation and its kept-answer read, app
sessions, Human Feedback re-asks) adds one row to `state/identify-log.sqlite`: when, the
path, the observation where there is one, fresh or kept, answer key, answers version,
model and release, who asked (a site user's mycomap.org id and name; a key id or the
member's pseudonymous subject) and the run link. It is for the paper's sealed test: did
Vision answer a test observation before its first green validation (docs/api.md §11)?
Insert-only (the table refuses UPDATE and DELETE), written on its own thread so it never
fails or slows an answer; `/api/health` with the server key shows `identify_log` counters,
and `failed` or `dropped` above 0 is worth a look in `journalctl -u mycomap-vision`
(`[identify-log]`). It holds people's ids and names: **back it up with the other `state/`
files, never copy it off the box except for the paper's read, and keep it until the paper
closes** (no expiry for now). Read it on the box:

```bash
.venv/bin/mv identify-log                         # rows per path
.venv/bin/mv identify-log first --out first.tsv   # per observation, its first answer
```

or `GET /api/identify-log/first` with the server key. **Once, after the first deploy
with the log**, add what older stores hold, labelled as recovered ("answered at or
before"), never as live rows:

```bash
.venv/bin/mv identify-log import --answers --runs
```

After the reference set, the night makes **advance predictions**: up to
`MV_NIGHTLY_PREDICT` (300) records still awaiting validation on mycomap.org
(`GET /api/vision/pending-records`), newest first, are identified from their iNat
photos with the model people get by default (`MV_DEFAULT_BACKBONE`, as preloaded; unset,
the first in `MV_PRELOAD`; skipped, never another model, while a named default isn't
served), photos
held in memory only. Each prediction is scored once its record turns green, and
only if it was made before: `.venv/bin/mv prospective` on the box. That is the
honest test of the model, since nothing could have seen the answer. A failure here
is recorded under `predictions` in `mv nightly` and never undoes the night's update.

When a new release is due, build one from an up-to-date manifest and publish and
pull it as under "New release". It already holds what the nights added, so its
fresh layer starts (nearly) empty.

## Memory on the 4 GB box

> **Status as of 2026-10-25:** this section was written for the original 4 GB box. The box
> is now Lightsail 16 GB / 4 vCPU, with a systemd drop-in that raises the server's `MemoryMax` to 12G (an override file under
> `/etc/systemd/system/mycomap-vision.service.d/`). The measurements below still describe
> what the server uses.

Since 2026-10-12 (Steve: v1 stays on the 4 GB box) the reference vectors can ship as int8
(`mv release --int8-vectors`): half the vector memory, the same answers within the
pre-declared check (docs/retrain-v1.md section 5); `/api/health` → `vectors` says which a
served index holds. The measurements below are of float16 vectors.

Measured in a Linux container with the box's packages, on the full photo set
(593,224 reference photos; `anon` is memory only the process holds, `RSS` also
counts file pages mapped from disk, which the kernel can drop and re-read):

| | anon | RSS |
|---|---|---|
| BioCLIP 2 + index, steady | 0.6 GB | 2.7 GB |
| peak while rebuilding the index (a withdrawal) | 1.1 GB | 4.0 GB* |
| peak at startup (loading the model and building the index) | 2.1 GB | 3.0 GB |
| an identification with 10 phone photos | +0.1 GB | |

\* RSS counts the vector files twice while the old and new index both map them;
they are one copy in memory. Before `fix/serving-memory-and-withdrawals` the same
steps took 1.8 GB anon steady and 4.8 GB anon (6.1 GB RSS) while rebuilding.

What is where:
- BioCLIP 2's image tower, ~1.2 GB, read from its safetensors file.
- The reference vectors, float16, ~1.5 KB per photo (0.9 GB for all photos),
  memory-mapped from the release's embedding files (`serving.py`), never copied
  into memory. A rebuilt index maps the same files, and is built on its own
  thread and swapped in whole; identifications wait for it (up to
  `MV_INDEX_WAIT_SECONDS`, default 20) or get "try again shortly".
- The index's own tables, ~70 MB (`identify.Specimens`). Building one needs
  ~0.5 GB for a few seconds while the records are read.
- Uploaded photos, kept at twice the model's input size (672 px for the 336 px v1),
  ~1 MB each (`uploads.py`).

`MemoryMax=3500M` in the unit (12G on the 16 GB box, through its drop-in) makes the
server restart rather than starve sshd.
`journalctl -u mycomap-vision` shows when that happens.

## Human Feedback queues and the paper's seal

No review queue may hold a v1-test record: one first green after dataset release v1's label
snapshot, 2026-10-12 13:17:51 UTC (`MV_PROSPECTIVE_SEAL`; set it to `off` only after the
paper's freeze). Turning it off opens only the photo queues (Photos that show no specimen, Photo
views: `feedback.POST_SEAL_QUEUES`); every other queue keeps the fixed seal until submission,
because public Human Feedback shows Vision's answers beside their names (the paper coordinator,
15 Nov 2026). A private hold list (`MV_FEEDBACK_HOLD_KEYS`, pinned by `MV_FEEDBACK_HOLD_SHA256`)
keeps its records out of every queue whatever the seal. Every way an item is added goes through
the seal:

- seeding (`mv feedback seed`);
- a reviewer's photo item;
- anything else that adds items.

It refuses a record first green after the seal. It also refuses, as unknown, a record green
on the seal's own day, with no date, or missing from the manifest, unless the record is on
the pinned list of v1 record keys. Refusals are counted in the seed's output with their
records.

Once: write the pinned list where the dataset release is (a laptop; the box has no dataset
release), copy it to the box, and point the setting at it.

```
mv dataset record-keys v1 --out data/research-release-extras/v1/v1-record-keys.txt   # prints its sha256
# copy it to <release root>/state/v1-record-keys.txt on the box, check its sha256, then
# vision.env: MV_FEEDBACK_V1_KEYS=<release root>/state/v1-record-keys.txt
```

Without the list, the seal still holds, but it refuses v1 records green on the seal's day
(1,070 of v1's) or with no date as unknown.

Every re-seed, on the box:

1. Back up the store:
   `sqlite3 <release root>/state/human-feedback.sqlite ".backup <release root>/state/human-feedback.sqlite.bak-<UTC time>"`.
2. Seed, then read the output's `refused_sealed`, `refused_unknown` and `refused_records`.
3. Check: `mv feedback seal-check`. It must print `"ok": true` (exit 0). It lists any failing
   item with its record and the decisions made on it.
4. For the paper's record, also run the research check, `s21_seal_check.py`, in the private
   research repo (experiments/v1-miss-review/). It judges against the dataset release v1
   itself and .org's first-green dates.
   - Run it on a copy of the store with research-releases/v1/release.sqlite
     (sha256 f2590352…) and a manifest.
   - Never run it against a serving manifest.


## Opening the site

- Anyone with a mycomap.org account (today): `MV_SIGNIN=all`.
- Open pages, but identifying needs a sign-in: set `MV_SIGNIN=identify`, restart,
  and verify with `SIGNIN=identify bash deploy/lightsail/verify.sh`.
- Search engines: delete the `X-Robots-Tag` line in `nginx-vision-headers.conf`
  (and its test), copy the file to the box and reload nginx.
- Results show a photo only when it is Creative Commons licensed or its
  photographer granted permission on mycomap.org (`identify.SHOWN_LICENSES`,
  `permissions.py`); any other record is linked to its source without a photo.
