#!/usr/bin/env bash
# One-off (re-runnable): secrets + Gitea webhook for konflate
# (clusters/home/apps/konflate.yaml).
#   1. creates a Gitea access token "konflate" for patrickjmcd (scopes read:user,
#      which konflate needs to resolve its own identity;
#      write:repository for commit statuses, write:issue for PR comments;
#      replaces an existing one) and a random webhook secret;
#   2. stores both in 1Password item vaults/Kubernetes/items/konflate
#      (KONFLATE_TOKEN = KONFLATE_WRITE_TOKEN = the token,
#      KONFLATE_WEBHOOK_SECRET), never printing them, and forces the sync;
#   3. adds (or replaces) the k8s-argo repo webhook -> konflate's /hooks for
#      pull_request (open/edit/close) and pull_request_sync (new commits).
set -euo pipefail

GITEA=https://git.x.pmcd.io/api/v1
REPO=patrickjmcd/k8s-argo
HOOK_URL=http://konflate.konflate.svc.cluster.local:8080/hooks

PASS=$(kubectl -n gitea get secret gitea-1pw -o jsonpath='{.data.ADMIN_PASSWORD}' | base64 -d)
if [ "$(curl -s -o /dev/null -w '%{http_code}' -u "patrickjmcd:$PASS" "$GITEA/user")" != 200 ]; then
  read -r -s -p "Gitea password for patrickjmcd: " PASS; echo
fi
api() { curl -sf -u "patrickjmcd:$PASS" -H 'Content-Type: application/json' "$@"; }

api -X DELETE "$GITEA/users/patrickjmcd/tokens/konflate" >/dev/null 2>&1 && echo "removed old Gitea token konflate" || true
TOKEN=$(jq -n '{name: "konflate", scopes: ["write:repository", "write:issue", "read:user"]}' |
  api -X POST "$GITEA/users/patrickjmcd/tokens" -d @- | jq -r .sha1)
[ -n "$TOKEN" ] && [ "$TOKEN" != null ] || { echo "token creation failed" >&2; exit 1; }
echo "created Gitea token konflate"
SECRET=$(openssl rand -hex 32)

eval "$(op signin)"
fields=("KONFLATE_TOKEN[password]=$TOKEN" "KONFLATE_WRITE_TOKEN[password]=$TOKEN" "KONFLATE_WEBHOOK_SECRET[password]=$SECRET")
if op item get konflate --vault Kubernetes >/dev/null 2>&1; then
  op item edit konflate --vault Kubernetes "${fields[@]}" >/dev/null && echo "updated 1Password item konflate"
else
  op item create --vault Kubernetes --category password --title konflate "${fields[@]}" >/dev/null && echo "created 1Password item konflate"
fi
kubectl -n konflate annotate onepassworditem konflate-1pw force-sync="$(date +%s)" --overwrite >/dev/null
echo "waiting for the konflate-1pw Secret..."
for _ in $(seq 60); do
  [ "$(kubectl -n konflate get secret konflate-1pw -o jsonpath='{.data.KONFLATE_WEBHOOK_SECRET}' 2>/dev/null | base64 -d)" = "$SECRET" ] && break
  sleep 3
done

for id in $(api "$GITEA/repos/$REPO/hooks" | jq -r --arg u "$HOOK_URL" '.[] | select(.config.url == $u) | .id'); do
  api -X DELETE "$GITEA/repos/$REPO/hooks/$id" >/dev/null && echo "removed old hook $id"
done
jq -n --arg url "$HOOK_URL" --arg secret "$SECRET" '{
  type: "gitea", active: true, events: ["pull_request_only", "pull_request_sync"],
  config: {url: $url, content_type: "json", secret: $secret}
}' | api -X POST "$GITEA/repos/$REPO/hooks" -d @- | jq -r '"created hook \(.id) -> \(.config.url) \(.events)"'
# The first install times out while the Secret doesn't exist yet; --force
# retries it. On later runs Reloader has already restarted the pod.
flux reconcile helmrelease konflate -n flux-system --force
kubectl -n konflate rollout status deploy/konflate --timeout=120s
