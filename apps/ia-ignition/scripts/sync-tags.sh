#!/usr/bin/env sh
set -eu

# Keeps a local checkout of the ignition-ha repo (source of truth for tag
# configuration) up to date on a shared volume the ia-ignition container also
# mounts read-only. That's this script's entire job -- it never talks to the
# gateway. The actual import is done by a Gateway Timer Script (configured in
# Designer; see ignition-ha/README.md) calling system.tag.importTags()
# against the checked-out tags/tags.json, on its own polling schedule.
#
# One-directional (git -> gateway) by design: exporting stays a manual
# Designer -> PR -> merge step.

REPO_DIR="/data/repo"
SSH_KEY="/data/id_ed25519"

# The 1Password field stores the deploy key base64-encoded on a single line --
# concealed fields aren't guaranteed to round-trip an embedded-newline PEM
# block faithfully (observed: ssh-keygen/ssh both failed with "error in
# libcrypto" on a key stored raw), so this decodes it once into a writable
# volume rather than relying on the multiline Secret mount directly.
if [ ! -f "${SSH_KEY}" ]; then
  base64 -d /run/secrets/ignition-ha-deploy-key/id_ed25519.b64 > "${SSH_KEY}"
  chmod 600 "${SSH_KEY}"
fi

export GIT_SSH_COMMAND="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"

clone_or_pull() {
  if [ -d "${REPO_DIR}/.git" ]; then
    git -C "${REPO_DIR}" fetch origin main
    git -C "${REPO_DIR}" reset --hard origin/main
  else
    git clone --depth 1 "${IGNITION_HA_REPO}" "${REPO_DIR}"
  fi
}

echo "ignition-tag-sync: watching ${IGNITION_HA_REPO} every ${SYNC_INTERVAL_SECONDS:-300}s"

while true; do
  if clone_or_pull; then
    echo "Synced $(git -C "${REPO_DIR}" rev-parse --short HEAD)"
  else
    echo "git clone/pull failed, will retry next cycle" >&2
  fi
  sleep "${SYNC_INTERVAL_SECONDS:-300}"
done
