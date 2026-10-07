#!/usr/bin/env bash
# =============================================================================
# Post-deploy checks for vision.mycomap.org. Run after every deploy (deploy.sh
# does). Exits non-zero on any failure.
#
#   bash deploy/lightsail/verify.sh
#   RESOLVE_IP=127.0.0.1 bash deploy/lightsail/verify.sh   # bypass Cloudflare, hit local nginx
#   BASE=http://127.0.0.1:8010 bash deploy/lightsail/verify.sh  # the app only, before nginx exists
#   SIGNIN=identify bash deploy/lightsail/verify.sh        # after the site opens up
# Run from the repo root (deploy.sh does). The nightly update's state is printed as
# ok/WARN lines and never fails the check.
# =============================================================================
set -uo pipefail

HOST="${HOST:-vision.mycomap.org}"
BASE="${BASE:-https://$HOST}"
SIGNIN="${SIGNIN:-all}"
ISSUER="${ISSUER:-https://mycomap.org}"
RESOLVE=()
if [ -n "${RESOLVE_IP:-}" ]; then RESOLVE=(--resolve "$HOST:443:$RESOLVE_IP" --resolve "$HOST:80:$RESOLVE_IP"); fi
fail=0

check() { # name, command...
  local name="$1"; shift
  if out="$("$@" 2>&1)"; then echo "  ok    $name"; else echo "  FAIL  $name: $out"; fail=1; fi
}
get() { curl -sS "${RESOLVE[@]}" --max-time 30 "$@"; }
status_is() { local want="$1"; shift; local got; got="$(get -o /dev/null -w '%{http_code}' "$@")"; [ "$got" = "$want" ] || { echo "HTTP $got, expected $want"; return 1; }; }
body_has() { local want="$1"; shift; get "$@" | grep -q -- "$want" || { echo "response lacks '$want'"; return 1; }; }
header_has() { local want="$1"; shift; get -D - -o /dev/null "$@" | grep -qi -- "$want" || { echo "headers lack '$want'"; return 1; }; }
aud_is_this_site() { local loc; loc="$(get -o /dev/null -w '%{redirect_url}' "$BASE/auth/signin")"; [[ "$loc" == *"&aud=https%3A%2F%2F$HOST" ]] || { echo "audience in '$loc' is not https://$HOST"; return 1; }; }
redirects_to() { local want="$1"; shift; local loc; loc="$(get -o /dev/null -w '%{redirect_url}' "$@")"; [[ "$loc" == "$want"* ]] || { echo "redirects to '$loc', expected '$want...'"; return 1; }; }

echo "Verifying $BASE (sign-in: $SIGNIN)"
check "health"                            body_has '"ok":true' "$BASE/api/health"
check "sign-in mode is $SIGNIN"           body_has "\"signin\":\"$SIGNIN\"" "$BASE/api/me"

if [ "$SIGNIN" = "all" ]; then
  check "API refuses before sign-in"      status_is 401 "$BASE/api/stats"
  check "identify refuses before sign-in" status_is 401 -X POST "$BASE/api/identify"
  check "pages send people to sign in"    redirects_to "$BASE/auth/signin?returnTo=" "$BASE/models"
fi
if [ "$SIGNIN" = "identify" ]; then
  check "site open"                       body_has '"records":' "$BASE/api/stats"
  check "identify refuses before sign-in" status_is 401 -X POST "$BASE/api/identify"
fi
if [ "$SIGNIN" != "off" ]; then
  check "sign-in goes to mycomap.org for this site" \
    redirects_to "$ISSUER/auth/dev-bridge/authorize?nonce=" "$BASE/auth/signin"
  check "sign-in names this site as audience" aud_is_this_site
  check "a made-up token signs nobody in"  status_is 401 "$BASE/auth/dev-bridge/callback?token=a.b"
fi

if [[ "$BASE" == https://* ]]; then
  check "http redirects to https"         status_is 301 "http://$HOST/"
  check "security headers"                header_has 'content-security-policy' "$BASE/api/health"
  check "kept out of search engines"      header_has 'x-robots-tag: noindex' "$BASE/api/health"
  check "certificate good for 14+ days" bash -c \
    "echo | openssl s_client -servername $HOST -connect ${RESOLVE_IP:-$HOST}:443 2>/dev/null | openssl x509 -noout -checkend 1209600"
fi

# The nightly update (nightly.py): reported, never a failure. A night can fail
# because mycomap.org was briefly down; the next one makes it good.
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
get "$BASE/api/health" | PYTHONPATH=src "$PY" -m mycomap_vision.nightly_notes | sed 's/^/  /'

# On the box itself: the service and a release.
if systemctl list-unit-files mycomap-vision.service >/dev/null 2>&1; then
  check "service active"                  systemctl is-active --quiet mycomap-vision
  check "a release is current"            test -s /srv/mycomap-vision/current.txt
fi

if [ "$fail" -ne 0 ]; then echo "VERIFY FAILED"; exit 1; fi
echo "All checks passed."
