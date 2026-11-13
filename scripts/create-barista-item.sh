#!/usr/bin/env bash
# One-off: creates vaults/Kubernetes/items/barista for apps/barista.
# Create the PocketID OIDC client "Barista" first (callback
# https://coffee.x.pmcd.io/auth/callback; restrict it to your group so only
# you can open /admin). The ntfy token is copied from the ntfy item and never
# printed. ADMIN_EMAILS is optional: leave it empty to rely on PocketID.
set -euo pipefail

eval "$(op signin)"

ask() { local v; read -r -p "$1: " v; printf '%s' "$v"; }
asks() { local v; read -r -s -p "$1: " v; echo >&2; printf '%s' "$v"; }

ID=$(ask "Barista PocketID client ID")
SECRET=$(asks "Barista PocketID client secret")
ADMINS=$(ask "ADMIN_EMAILS (comma-separated, or empty)")

# Other publishers read NTFY_TOKEN; older items only have the
# NTFY_AUTH_TOKENS line (user:token:label).
NTFY_TOKEN=$(op read "op://Kubernetes/ntfy/NTFY_TOKEN" 2>/dev/null ||
  op read "op://Kubernetes/ntfy/NTFY_AUTH_TOKENS" | cut -d: -f2)

op item create --vault Kubernetes --category login --title barista \
  "OIDC_CLIENT_ID[text]=$ID" \
  "OIDC_CLIENT_SECRET[password]=$SECRET" \
  "NTFY_TOKEN[password]=$NTFY_TOKEN" \
  "ADMIN_EMAILS[text]=$ADMINS" >/dev/null
echo "created: barista"
