#!/usr/bin/env bash
# One-off: creates vaults/Kubernetes/items/flux-web for the Flux web UI's
# PocketID SSO (clusters/home/flux-web.yaml). Create the OIDC client in PocketID
# first (callback URL https://flux.x.pmcd.io/oauth2/callback), then paste its
# client ID and secret at the prompts.
set -euo pipefail

eval "$(op signin)"
read -r -p "PocketID client ID: " CLIENT_ID
read -r -s -p "PocketID client secret: " CLIENT_SECRET; echo

op item create --vault Kubernetes --category login --title flux-web \
  "clientID[text]=$CLIENT_ID" \
  "clientSecret[password]=$CLIENT_SECRET" >/dev/null
echo "created: flux-web"
