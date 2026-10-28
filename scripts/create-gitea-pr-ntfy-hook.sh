#!/usr/bin/env bash
# One-off: adds a Gitea *system* webhook (fires for every repository) that
# sends pull_request events to ntfy topic `gitea`, formatted by the gitea-pr
# template (apps/ntfy/homelabapp.yaml). The template rejects everything but
# "opened", so those deliveries show as failed (HTTP 400) in Gitea's hook
# history -- expected.
# Auth: Gitea API as the local admin patrickjmcd (password from the gitea-1pw
# Secret, prompted if that fails); ntfy with the publisher token NTFY_TOKEN
# from the ntfy-1pw Secret, stored in the hook's Authorization header.
# Re-running replaces the hook (e.g. after rotating the ntfy token).
set -euo pipefail

GITEA=https://git.x.pmcd.io/api/v1
NTFY_URL='http://ntfy.ntfy.svc.cluster.local/gitea?template=gitea-pr'

NTFY_TOKEN=$(kubectl -n ntfy get secret ntfy-1pw -o jsonpath='{.data.NTFY_TOKEN}' | base64 -d)
[ -n "$NTFY_TOKEN" ] || { echo "ntfy-1pw has no NTFY_TOKEN" >&2; exit 1; }

PASS=$(kubectl -n gitea get secret gitea-1pw -o jsonpath='{.data.ADMIN_PASSWORD}' | base64 -d)
if [ "$(curl -s -o /dev/null -w '%{http_code}' -u "patrickjmcd:$PASS" "$GITEA/user")" != 200 ]; then
  read -r -s -p "Gitea password for patrickjmcd: " PASS; echo
fi
api() { curl -sf -u "patrickjmcd:$PASS" -H 'Content-Type: application/json' "$@"; }

for id in $(api "$GITEA/admin/hooks" | jq -r --arg u "$NTFY_URL" '.[] | select(.config.url == $u) | .id'); do
  api -X DELETE "$GITEA/admin/hooks/$id" >/dev/null && echo "removed old hook $id"
done

jq -n --arg url "$NTFY_URL" --arg auth "Bearer $NTFY_TOKEN" '{
  type: "gitea", name: "ntfy: new pull requests", active: true,
  events: ["pull_request"], authorization_header: $auth,
  config: {url: $url, content_type: "json"}
}' | api -X POST "$GITEA/admin/hooks" -d @- | jq -r '"created system hook \(.id) -> \(.config.url)"'
