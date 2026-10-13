#!/bin/sh
# Runs before Gitea starts (init container). Idempotent: renders app.ini from
# the GITEA__* env, migrates the database, then makes sure the local admin
# and the PocketID login source exist (updating the OIDC client if it does).
set -eu
environment-to-ini --config /etc/gitea/app.ini
gitea migrate
if gitea admin user list --admin | awk 'NR>1 {print $2}' | grep -qx patrickjmcd; then
  echo "admin patrickjmcd exists"
else
  gitea admin user create --admin --username patrickjmcd \
    --email patrickjmcd@gmail.com --password "$ADMIN_PASSWORD" \
    --must-change-password=false
fi
ARGS="--provider openidConnect --key $OIDC_CLIENT_ID --secret $OIDC_CLIENT_SECRET
  --auto-discover-url https://auth.pmcd.dev/.well-known/openid-configuration
  --scopes openid --scopes email --scopes profile --skip-local-2fa"
ID=$(gitea admin auth list | awk '$2=="pocketid" {print $1}')
if [ -n "$ID" ]; then
  gitea admin auth update-oauth --id "$ID" --name pocketid $ARGS
  echo "login source pocketid updated"
else
  gitea admin auth add-oauth --name pocketid $ARGS
  echo "login source pocketid added"
fi
