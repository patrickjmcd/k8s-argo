#!/usr/bin/env bash
# One-off: wires Gitea's k8s-argo push webhook to the Flux Receiver
# (clusters/home/receiver.yaml).
#   1. creates vaults/Kubernetes/items/flux-receiver with a random `token`
#      (generated here and piped into 1Password, never printed);
#   2. waits for the operator to sync it and the Receiver to become Ready;
#   3. adds (or replaces) the repo webhook in Gitea, pointing at the in-cluster
#      webhook-receiver Service, with the same token as its secret.
# Gitea API auth: the local admin patrickjmcd, password from the gitea-1pw
# Secret (ADMIN_PASSWORD); you're prompted if that no longer works.
set -euo pipefail

GITEA=https://git.x.pmcd.io/api/v1
REPO=patrickjmcd/k8s-argo
HOOK_BASE=http://webhook-receiver.flux-system.svc.cluster.local

eval "$(op signin)"
if op item get flux-receiver --vault Kubernetes >/dev/null 2>&1; then
  echo "1Password item flux-receiver exists; reusing it"
else
  op item create --vault Kubernetes --category password --title flux-receiver \
    "token[password]=$(openssl rand -hex 32)" >/dev/null
  echo "created: flux-receiver"
fi
TOKEN=$(op read "op://Kubernetes/flux-receiver/token")

kubectl -n flux-system annotate onepassworditem flux-receiver-1pw \
  force-sync="$(date +%s)" --overwrite >/dev/null
echo "waiting for the Receiver..."
kubectl -n flux-system wait receiver/gitea-k8s-argo --for=condition=Ready --timeout=180s >/dev/null
HOOK_PATH=$(kubectl -n flux-system get receiver gitea-k8s-argo -o jsonpath='{.status.webhookPath}')
[ -n "$HOOK_PATH" ] || { echo "Receiver has no webhookPath" >&2; exit 1; }

PASS=$(kubectl -n gitea get secret gitea-1pw -o jsonpath='{.data.ADMIN_PASSWORD}' | base64 -d)
if [ "$(curl -s -o /dev/null -w '%{http_code}' -u "patrickjmcd:$PASS" "$GITEA/user")" != 200 ]; then
  read -r -s -p "Gitea password for patrickjmcd: " PASS; echo
fi
api() { curl -sf -u "patrickjmcd:$PASS" -H 'Content-Type: application/json' "$@"; }

# Replace any earlier hook to the receiver (e.g. after a token rotation).
for id in $(api "$GITEA/repos/$REPO/hooks" | jq -r --arg b "$HOOK_BASE" '.[] | select(.config.url | startswith($b)) | .id'); do
  api -X DELETE "$GITEA/repos/$REPO/hooks/$id" >/dev/null && echo "removed old hook $id"
done

jq -n --arg url "$HOOK_BASE$HOOK_PATH" --arg secret "$TOKEN" '{
  type: "gitea", active: true, events: ["push"], branch_filter: "main",
  config: {url: $url, content_type: "json", secret: $secret}
}' | api -X POST "$GITEA/repos/$REPO/hooks" -d @- | jq -r '"created hook \(.id) -> \(.config.url)"'

# Gitea's "test delivery" sends a push event; Flux should log a reconcile request.
id=$(api "$GITEA/repos/$REPO/hooks" | jq -r --arg b "$HOOK_BASE" '.[] | select(.config.url | startswith($b)) | .id' | head -1)
api -X POST "$GITEA/repos/$REPO/hooks/$id/tests" >/dev/null && echo "test delivery sent"
