#!/usr/bin/env bash
# Gitea pull requests -> n8n -> ntfy (topic `gitea`, with Review/Merge/Close
# buttons). Replaces create-gitea-pr-ntfy-hook.sh, which posted to ntfy's
# gitea-pr template directly. Two steps, each re-runnable:
#
#   secrets  creates Gitea access token "n8n-pr-actions" for patrickjmcd
#            (write:repository: merge/close PRs, read repo merge settings;
#            replaces an existing one), stores it in 1Password item
#            vaults/Kubernetes/items/gitea-pr-n8n (GITEA_TOKEN) and in the n8n
#            credential "Gitea API" (header auth, Authorization: token <...>).
#   hook     mints a new webhook secret (so re-running rotates it), stores it in
#            the same 1Password item (HOOK_SECRET) and in the n8n credential
#            "Gitea PR hook" (Authorization: Bearer <...>), then points Gitea
#            system hook 3 (all repos, pull_request_only) at the
#            "Gitea PR -> ntfy" workflow's webhook with that header. Run after
#            the workflow is published; deliveries 403 for the second between
#            the n8n and Gitea updates.
#
# n8n credentials go through n8n's public API (key from
# op://Kubernetes/n8n/N8N_API_KEY, or prompted), updated in place by name.
# Secret values are never printed.
#
# The Merge/Close buttons don't use a shared secret: the workflow mints a
# random single-use token per PR (n8n data table gitea_pr_action_tokens) and
# the "Gitea PR action" webhook only acts on a matching repo/number/token.
# Gitea API quirks: without config.is_system_webhook="true" the admin API
# makes a *default* hook (copied into new repos only), and "pull_request"
# expands to every PR sub-event (sync, review, comment...); "pull_request_only"
# is the open/close/reopen/edit event alone.
set -euo pipefail

GITEA=https://git.x.pmcd.io/api/v1
N8N=https://n8n.x.pmcd.io/api/v1
HOOK_ID=3
HOOK_URL=http://n8n.default.svc.cluster.local/webhook/gitea-pr
ITEM=gitea-pr-n8n

gitea_login() {
  PASS=$(kubectl -n gitea get secret gitea-1pw -o jsonpath='{.data.ADMIN_PASSWORD}' | base64 -d)
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -u "patrickjmcd:$PASS" "$GITEA/user")" != 200 ]; then
    read -r -s -p "Gitea password for patrickjmcd: " PASS; echo
  fi
}
api() { curl -sf -u "patrickjmcd:$PASS" -H 'Content-Type: application/json' "$@"; }

op_store() {  # op_store FIELD VALUE
  if op item get "$ITEM" --vault Kubernetes >/dev/null 2>&1; then
    op item edit "$ITEM" --vault Kubernetes "$1[password]=$2" >/dev/null && echo "stored $1 in 1Password item $ITEM"
  else
    op item create --vault Kubernetes --category password --title "$ITEM" "$1[password]=$2" >/dev/null && echo "created 1Password item $ITEM with $1"
  fi
}

n8n_cred() {  # n8n_cred NAME HEADER_VALUE: header-auth credential, updated in place
  N8N_KEY=${N8N_KEY:-$(op read op://Kubernetes/n8n/N8N_API_KEY 2>/dev/null || true)}
  [ -n "$N8N_KEY" ] || { read -r -s -p "n8n API key (Settings > n8n API): " N8N_KEY; echo; }
  n8n() { curl -sf -H "X-N8N-API-KEY: $N8N_KEY" -H 'Content-Type: application/json' "$@"; }
  local id body
  id=$(n8n "$N8N/credentials?limit=250" | jq -r --arg n "$1" '[.data[] | select(.name == $n)][0].id // empty')
  body=$(jq -n --arg n "$1" --arg v "$2" '{name: $n, type: "httpHeaderAuth", data: {name: "Authorization", value: $v}}')
  if [ -n "$id" ]; then
    n8n -X PATCH "$N8N/credentials/$id" -d "$body" >/dev/null && echo "updated n8n credential $1 ($id)"
  else
    n8n -X POST "$N8N/credentials" -d "$body" | jq -r '"created n8n credential \(.name) (\(.id))"'
  fi
}

secrets() {
  gitea_login
  api -X DELETE "$GITEA/users/patrickjmcd/tokens/n8n-pr-actions" >/dev/null 2>&1 && echo "removed old Gitea token n8n-pr-actions" || true
  TOKEN=$(jq -n '{name: "n8n-pr-actions", scopes: ["write:repository"]}' |
    api -X POST "$GITEA/users/patrickjmcd/tokens" -d @- | jq -r .sha1)
  [ -n "$TOKEN" ] && [ "$TOKEN" != null ] || { echo "token creation failed" >&2; exit 1; }
  echo "created Gitea token n8n-pr-actions"
  eval "$(op signin)"
  op_store GITEA_TOKEN "$TOKEN"
  n8n_cred "Gitea API" "token $TOKEN"
}

hook() {
  gitea_login
  eval "$(op signin)"
  SECRET=$(openssl rand -hex 32)
  op_store HOOK_SECRET "$SECRET"
  n8n_cred "Gitea PR hook" "Bearer $SECRET"
  jq -n --arg url "$HOOK_URL" --arg auth "Bearer $SECRET" '{
    active: true, events: ["pull_request_only"], authorization_header: $auth,
    config: {url: $url, content_type: "json", is_system_webhook: "true"}
  }' | api -X PATCH "$GITEA/admin/hooks/$HOOK_ID" -d @- | jq -r '"system hook \(.id) -> \(.config.url) (\(.events | join(",")))"'
}

case "${1:-}" in
  secrets) secrets ;;
  hook) hook ;;
  *) echo "usage: $0 secrets|hook" >&2; exit 2 ;;
esac
