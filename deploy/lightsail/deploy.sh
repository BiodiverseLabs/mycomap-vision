#!/usr/bin/env bash
# =============================================================================
# Routine deploy of Vision, ON the Lightsail box, as `ubuntu`:
#     cd /var/www/mycomap-vision && bash deploy/lightsail/deploy.sh
#
# Pulls the branch, installs, builds the web app into a new folder before swapping
# it in, restarts the server, waits for it, then runs verify.sh. Stops at the first
# failure; a failed build leaves the previous site in place.
#
# Releases (the model data) are separate: `mv pull-release` then
# `sudo systemctl restart mycomap-vision`. docs/deploy.md has both.
# =============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/var/www/mycomap-vision}"
BRANCH="${BRANCH:-main}"
PORT="${PORT:-8010}"
ENV_FILE="${ENV_FILE:-/etc/mycomap-vision/vision.env}"
SELF="$APP_DIR/deploy/lightsail/deploy.sh"

export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd "$APP_DIR"

if grep -q '<BUCKET>\|<REGION>' "$ENV_FILE"; then
  echo "!!  $ENV_FILE still has <...> placeholders."
  exit 1
fi

echo "==> git pull ($BRANCH)"
before="$(sha256sum "$SELF" 2>/dev/null || true)"
git fetch --quiet origin "$BRANCH"
git checkout --quiet "$BRANCH"
git merge --ff-only --quiet "origin/$BRANCH"
echo "    now at $(git log --oneline -1)"

# bash keeps reading the script it started with, so a pull that changes this file
# would otherwise run the rest of the OLD deploy (it once built the web app the old
# way). Start the new one instead; it pulls again (nothing new) and carries on.
if [ -z "${DEPLOY_REEXEC:-}" ] && [ "$(sha256sum "$SELF" 2>/dev/null || true)" != "$before" ]; then
  echo "    deploy.sh changed in this pull: running the new one"
  DEPLOY_REEXEC=1 exec bash "$SELF" "$@"
fi
if [ -n "${DEPLOY_STOP_AFTER_PULL:-}" ]; then exit 0; fi   # tests: the pull step only

echo "==> Python packages"
.venv/bin/pip install -q --require-hashes -r requirements/box.txt
.venv/bin/pip install -q -e . --no-deps

echo "==> building the web app"
rm -rf web/dist-next
# From inside web/: corepack reads the pnpm version pinned there ("packageManager");
# run from the repo root it would fetch the newest pnpm, which refuses the pin.
(cd web && pnpm install --frozen-lockfile && pnpm exec vite build --outDir dist-next --emptyOutDir)
rm -rf web/dist-prev
[ -d web/dist ] && mv web/dist web/dist-prev
mv web/dist-next web/dist

echo "==> restarting the server"
sudo systemctl restart mycomap-vision

# Health answers within seconds; the model loads behind it (MV_PRELOAD). Checking
# before health is up would report 502s that are false alarms.
echo "==> waiting for the server"
up=0
for i in $(seq 1 60); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -q '"ok":true'; then
    echo "    up after ${i} s"; up=1; break
  fi
  sleep 1
done
if [ "$up" -ne 1 ]; then
  echo "!!  The server did not answer in 60 s:"
  sudo journalctl -u mycomap-vision -n 40 --no-pager
  exit 1
fi

# Not a failure if it takes longer: identifications wait for it. Just report it.
echo "==> waiting for the model to load (MV_PRELOAD)"
for i in $(seq 1 180); do
  # The preload block's own state: health has other "state" fields (the nightly update's).
  state="$(curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health" 2>/dev/null \
           | grep -o '"preload":{"state":"[a-z]*"' | cut -d'"' -f6)"
  case "$state" in
    ready)  echo "    ready after ${i} s"; break ;;
    off)    echo "    no preload configured; the first identification loads it"; break ;;
    failed) echo "!!  preload failed (identifications still load on demand):"
            curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health"; echo; break ;;
  esac
  sleep 1
done
bash deploy/lightsail/verify.sh
