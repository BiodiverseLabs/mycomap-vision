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

export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
cd "$APP_DIR"

if grep -q '<BUCKET>\|<REGION>' /etc/mycomap-vision/vision.env; then
  echo "!!  /etc/mycomap-vision/vision.env still has <...> placeholders."
  exit 1
fi

echo "==> git pull ($BRANCH)"
git fetch --quiet origin "$BRANCH"
git checkout --quiet "$BRANCH"
git merge --ff-only --quiet "origin/$BRANCH"
echo "    now at $(git log --oneline -1)"

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

# The model loads on the first identification, not at start, so health comes
# up in a few seconds; verify.sh reporting 502s before that would be false alarms.
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
bash deploy/lightsail/verify.sh
