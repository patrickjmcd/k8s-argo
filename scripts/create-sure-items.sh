#!/usr/bin/env bash
# One-off: creates vaults/Kubernetes/items/sure and garage-sure for apps/sure.
# Create the PocketID OIDC client "Sure" first (callback
# https://sure.x.pmcd.io/auth/openid_connect/callback). Every other secret is
# generated here and never printed.
set -euo pipefail

eval "$(op signin)"

ask() { local v; read -r -p "$1: " v; printf '%s' "$v"; }
asks() { local v; read -r -s -p "$1: " v; echo >&2; printf '%s' "$v"; }

ID=$(ask "Sure PocketID client ID")
SECRET=$(asks "Sure PocketID client secret")

# SECRET_KEY_BASE also derives the Active Record encryption keys: losing or
# changing it makes encrypted columns unreadable.
op item create --vault Kubernetes --category login --title sure \
  "SECRET_KEY_BASE[password]=$(openssl rand -hex 64)" \
  "OIDC_CLIENT_ID[text]=$ID" \
  "OIDC_CLIENT_SECRET[password]=$SECRET" >/dev/null
echo "created: sure"

# Garage key format: GK + 24 hex id, 64 hex secret.
op item create --vault Kubernetes --category login --title garage-sure \
  "GENERIC_S3_ACCESS_KEY_ID[text]=GK$(openssl rand -hex 12)" \
  "GENERIC_S3_SECRET_ACCESS_KEY[password]=$(openssl rand -hex 32)" >/dev/null
echo "created: garage-sure"
