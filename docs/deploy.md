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
| Instance | Lightsail, Ohio (us-east-2, the bucket's region), Ubuntu 24.04, **4 GB RAM**, 2 vCPUs, + 4 GB swap |
| App | `/var/www/mycomap-vision`, a clone of the repo; `.venv` with CPU-only torch |
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

Steps marked **(Steve)** involve a console, a key or production; the rest can be
run by Claude over SSH.

### 1. Create the instance (Steve, Lightsail console)

1. Create instance: **Ohio (us-east-2)**, Linux/Unix, **OS only → Ubuntu 24.04 LTS**,
   the **4 GB RAM** plan. Name it `mycomap-vision`.
2. Networking → **attach a static IP**.
3. Networking → firewall (IPv4 and IPv6): SSH 22, HTTP 80, HTTPS 443.
4. Snapshots → turn on **automatic snapshots**.
5. For Claude's access (full, as on the Library box), add Claude's public key to
   `~/.ssh/authorized_keys` on the box; Claude adds an `mycomap-vision` alias to
   its ssh config once it has the static IP.

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

### 3. Settings

In `/etc/mycomap-vision/vision.env`, replace `<BUCKET>` and `<REGION>` with the
values from the laptop's `.env` (the folder is root's, so edit with sudo:
`sudo sed -i 's/<BUCKET>/.../; s/<REGION>/.../' /etc/mycomap-vision/vision.env`,
or `sudo nano`). Then put the sign-in bridge's public key in
place. It is the same key the dev site uses (`DEV_BRIDGE_PUBLIC_KEY` in the dev
box's ecosystem file). It is public, so pasting it is fine:

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

On the mycomap.org box, add this to the `env` block of
`/var/www/mycomap-next/ecosystem.staging.config.cjs`. It takes effect at the next
routine deploy, once the `feat/signin-bridge-sites` change is on main:

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
it in, restarts the server, waits for it and runs `verify.sh`. The deploy is
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

## Memory on the 4 GB box

The server holds three things in RAM:
- BioCLIP 2, about 1.7 GB with its unused text tower;
- the reference vectors, float16, about 1.5 KB per photo, so about 0.9 GB for all ~593k photos;
- the nearest-specimen index, which today is a float32 copy of those vectors, about 1.8 GB for all photos.

A sample-sized release fits easily. The full photo set does not fit as the code
stands. Before publishing a full release, either:
- trim the server's memory: drop the text tower, keep the nearest-specimen copy
  in float16, and free the vectors once the index is built; or
- move to the 8 GB plan: take a snapshot, create a bigger instance from it,
  move the static IP.

`MemoryMax=3500M` in the unit makes the server restart rather than starve sshd.
`journalctl -u mycomap-vision` shows when that happens.

## Opening the site

- Anyone with a mycomap.org account (today): `MV_SIGNIN=all`.
- Open pages, but identifying needs a sign-in: set `MV_SIGNIN=identify`, restart,
  and verify with `SIGNIN=identify bash deploy/lightsail/verify.sh`.
- Search engines: delete the `X-Robots-Tag` line in `nginx-vision-headers.conf`
  (and its test), copy the file to the box and reload nginx.
- Results only ever show Creative Commons photos (`identify.SHOWN_LICENSES`); a
  record with only all-rights-reserved photos is linked to iNat without a photo.
