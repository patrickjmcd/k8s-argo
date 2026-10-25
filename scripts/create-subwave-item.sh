#!/usr/bin/env bash
# One-off: creates vaults/Kubernetes/items/subwave for apps/media/subwave.
# ADMIN_PASS is generated here and piped straight into 1Password (never
# printed); read it back with `op item get subwave --fields ADMIN_PASS --reveal`.
set -euo pipefail

eval "$(op signin)"
op item create --vault Kubernetes --category login --title subwave \
  "username=admin" \
  "ADMIN_PASS[password]=$(openssl rand -hex 16)" >/dev/null
echo "created: subwave"
