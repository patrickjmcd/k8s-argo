#!/usr/bin/env bash
# One-off: creates the 1Password items for apps/media/koito and
# apps/media/multi-scrobbler. Koito's admin password is generated (read it
# from 1Password to log in). The Spotify client ID/secret come from a Spotify
# developer app (developer.spotify.com/dashboard, owned by a Premium account)
# with redirect URI https://multi-scrobbler.x.pmcd.io/callback (press Enter
# at the prompts to skip them for now).
# KOITO_TOKEN starts as a placeholder: fill it in with a Koito API key once
# Koito is up (Reloader restarts multi-scrobbler when the field syncs).
set -euo pipefail

eval "$(op signin)"
read -r -p "Spotify client ID: " SID
read -r -s -p "Spotify client secret: " SSECRET; echo
# Blank is fine: fill both in later, before authorizing Spotify.
SID=${SID:-replace-me}; SSECRET=${SSECRET:-replace-me}

PW=$(openssl rand -base64 24)
op item create --vault Kubernetes --category login --title koito \
  "KOITO_DEFAULT_PASSWORD[password]=$PW" >/dev/null
echo "created: koito"

op item create --vault Kubernetes --category login --title multi-scrobbler \
  "SPOTIFY_CLIENT_ID[text]=$SID" \
  "SPOTIFY_CLIENT_SECRET[password]=$SSECRET" \
  "KOITO_TOKEN[password]=replace-me" >/dev/null
echo "created: multi-scrobbler"
