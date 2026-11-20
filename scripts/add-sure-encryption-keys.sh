#!/usr/bin/env bash
# One-off: pins Sure's Active Record encryption keys in 1Password
# (vaults/Kubernetes/items/sure). Without them Sure derives the keys from
# SECRET_KEY_BASE, and security:backfill_encryption refuses to run, so
# existing rows stay plaintext. The values written are the ones Sure derives
# today, so anything already encrypted stays readable. Never printed.
set -euo pipefail

eval "$(op signin)"

KEYS=$(kubectl -n sure exec deploy/sure -- bin/rails runner \
  'c = Rails.application.config.active_record.encryption
   puts [c.primary_key, c.deterministic_key, c.key_derivation_salt].join(" ")' \
  2>/dev/null | tail -1)
read -r PK DK SALT <<<"$KEYS"
for v in "$PK" "$DK" "$SALT"; do
  [[ ${#v} -eq 64 ]] || { echo "unexpected key output from the sure pod" >&2; exit 1; }
done

op item edit sure --vault Kubernetes \
  "ACTIVE_RECORD_ENCRYPTION_PRIMARY_KEY[password]=$PK" \
  "ACTIVE_RECORD_ENCRYPTION_DETERMINISTIC_KEY[password]=$DK" \
  "ACTIVE_RECORD_ENCRYPTION_KEY_DERIVATION_SALT[password]=$SALT" >/dev/null

kubectl -n sure annotate onepassworditem sure-1pw force-sync="$(date +%s)" --overwrite
echo "Saved. Reloader restarts sure and sure-worker once sure-1pw updates."
