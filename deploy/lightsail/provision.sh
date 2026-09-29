#!/usr/bin/env bash
# =============================================================================
# One-time provisioning for the MycoMap Vision Lightsail instance
# (vision.mycomap.org). Run ON a fresh Ubuntu 24.04 instance as `ubuntu`:
#
#     bash deploy/lightsail/provision.sh      # from the clone in /var/www/mycomap-vision
#
# Idempotent: safe to re-run. Installs Python (CPU-only torch), Node for the web
# build, nginx and certbot, adds swap, makes the folders and the session secret,
# installs the systemd unit and pre-fetches the served model's weights.
#
# It does NOT write the settings file's deployment values, the sign-in public
# key, the AWS key, or the nginx site (needs a certificate first): docs/deploy.md
# walks through those.
# =============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/var/www/mycomap-vision}"
DATA_ROOT="${DATA_ROOT:-/srv/mycomap-vision}"
ETC_DIR="${ETC_DIR:-/etc/mycomap-vision}"
NODE_VERSION="${NODE_VERSION:-24}"
SWAP_GB="${SWAP_GB:-4}"
# Same versions as the laptop and the GPU trainer; the CPU wheels have no CUDA.
TORCH="${TORCH:-torch==2.14.0 torchvision==0.29.0}"
MODEL_LIBS="${MODEL_LIBS:-timm==1.0.30 open_clip_torch==3.3.0 numpy==2.5.2 pillow==12.3.0}"
PREFETCH="${PREFETCH:-imageomics/bioclip-2}"

if [ "$(id -un)" = "root" ]; then
  echo "!!  Run as the 'ubuntu' user, not root."
  exit 2
fi
if [ ! -f "$APP_DIR/pyproject.toml" ]; then
  echo "!!  Clone the repo into $APP_DIR first (docs/deploy.md, step 2)."
  exit 1
fi

echo "==> System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl build-essential python3 python3-venv python3-dev \
  sqlite3 nginx certbot
command -v aws >/dev/null || sudo snap install aws-cli --classic

# The model (~1.7 GB), the reference vectors and the manifest share 4 GB of RAM;
# swap keeps a spike from killing the server.
if ! swapon --show | grep -q /swapfile; then
  echo "==> Adding ${SWAP_GB} GB swap"
  sudo fallocate -l "${SWAP_GB}G" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> Node $NODE_VERSION via nvm, pnpm via corepack (only to build the web app)"
if [ ! -d "$HOME/.nvm" ]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
nvm install "$NODE_VERSION" >/dev/null
nvm alias default "$NODE_VERSION" >/dev/null
corepack enable

echo "==> Folders: releases in $DATA_ROOT, settings in $ETC_DIR"
sudo mkdir -p "$DATA_ROOT/releases" "$DATA_ROOT/hf" "$ETC_DIR"
sudo chown -R ubuntu:ubuntu "$DATA_ROOT"
sudo chown root:ubuntu "$ETC_DIR"
sudo chmod 750 "$ETC_DIR"
if [ ! -f "$ETC_DIR/session-secret" ]; then
  # Signs this site's cookies. Made here, never printed, never leaves the box.
  (umask 077; openssl rand -base64 48 | sudo tee "$ETC_DIR/session-secret" >/dev/null)
  sudo chown ubuntu:ubuntu "$ETC_DIR/session-secret"
  sudo chmod 600 "$ETC_DIR/session-secret"
fi
if [ ! -f "$ETC_DIR/vision.env" ]; then
  sudo cp "$APP_DIR/deploy/lightsail/vision.env.template" "$ETC_DIR/vision.env"
  sudo chown ubuntu:ubuntu "$ETC_DIR/vision.env"
  sudo chmod 600 "$ETC_DIR/vision.env"
fi
# The CLI reads <repo>/.env; the service reads the same file through systemd.
ln -sfn "$ETC_DIR/vision.env" "$APP_DIR/.env"

echo "==> Python venv (CPU-only torch)"
cd "$APP_DIR"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --require-hashes -r requirements/box.txt
# shellcheck disable=SC2086
.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu $TORCH
# shellcheck disable=SC2086
.venv/bin/pip install -q $MODEL_LIBS
.venv/bin/pip install -q -e . --no-deps

echo "==> Model weights into $DATA_ROOT/hf (downloaded once, not loaded)"
for repo in $PREFETCH; do
  HF_HOME="$DATA_ROOT/hf" .venv/bin/python -c \
    "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$repo"
done

echo "==> Cloudflare's addresses, so nginx sees the visitor's IP behind the proxy"
bash deploy/lightsail/cloudflare-realip.sh

echo "==> systemd unit"
sudo cp deploy/lightsail/mycomap-vision.service /etc/systemd/system/mycomap-vision.service
sudo systemctl daemon-reload
sudo systemctl enable mycomap-vision >/dev/null

cat <<NEXT

=============================================================================
Provisioning done. The remaining steps are in docs/deploy.md:
  1. Fill in $ETC_DIR/vision.env (bucket, region) and the sign-in public key
  2. AWS key for the box:      aws configure --profile mycomap-vision-box   (typed by a person)
  3. First release:            .venv/bin/mv pull-release
  4. First deploy:             bash deploy/lightsail/deploy.sh
  5. DNS + certificate + nginx site, then verify.sh
=============================================================================
NEXT
