#!/usr/bin/env sh
set -eu

# Polls the ignition-ha repo (source of truth for tag configuration) and, on
# a new commit, calls the Tag CICD module's import endpoint so the running
# gateway picks up the change. One-directional (git -> gateway) by design:
# exporting stays a manual Designer -> PR -> merge step, see
# ignition-ha/README.md.
#
# The clone lives on a volume shared read-only with the ia-ignition
# container (mounted there at /usr/local/bin/ignition/${GATEWAY_LOCAL_REPO_PATH}),
# because the import endpoint's filePath is resolved by the gateway process
# against its own filesystem, not this container's.
#
# NOTE: the import endpoint's exact request contract wasn't fully spelled out
# in the Tag CICD module's docs at the time this was written -- confirm
# provider/baseTagPath/filePath/collisionPolicy query-param names and
# behavior against the real gateway once Tag-CICD.modl is installed, and
# adjust the curl call below if it differs.

REPO_DIR="/data/repo"
STATE_FILE="/data/.last-synced-sha"
SSH_KEY="/data/id_ed25519"

# The 1Password field stores the deploy key base64-encoded on a single line --
# concealed fields aren't guaranteed to round-trip an embedded-newline PEM
# block faithfully (observed: ssh-keygen/ssh both fail with "error in
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

import_tags() {
  config_file="${REPO_DIR}/tag-cicd/export-config.json"
  if [ ! -f "${config_file}" ]; then
    echo "No tag-cicd/export-config.json in ignition-ha yet, skipping import"
    return 0
  fi

  status=0
  entries_file="/tmp/tag-cicd-entries.$$"
  jq -c '.[]' "${config_file}" > "${entries_file}"
  # Read from a file, not a pipe -- piping into `while` would run it in a
  # subshell in this shell (ash), and the `status` assignment below would be
  # lost on exit instead of surviving to the `return` after the loop.
  while IFS= read -r entry; do
    provider=$(echo "${entry}" | jq -r '.provider')
    baseTagPath=$(echo "${entry}" | jq -r '.baseTagPath')
    collisionPolicy=$(echo "${entry}" | jq -r '.collisionPolicy')

    # sourcePath in export-config.json is git-repo-relative (e.g. "../../tags"
    # from tag-cicd/export-config.json); translate to the gateway-local mount.
    filePath="${GATEWAY_LOCAL_REPO_PATH}/tags"

    echo "Importing provider='${provider}' baseTagPath='${baseTagPath}' from ${filePath}"
    curl -sf -u "${TAG_CICD_API_TOKEN}" -G -X POST \
      "${GATEWAY_URL}/data/tag-cicd/tags/import" \
      --data-urlencode "provider=${provider}" \
      --data-urlencode "baseTagPath=${baseTagPath}" \
      --data-urlencode "filePath=${filePath}" \
      --data-urlencode "collisionPolicy=${collisionPolicy}" \
      || status=1
  done < "${entries_file}"
  rm -f "${entries_file}"
  return "${status}"
}

echo "ignition-tag-sync: watching ${IGNITION_HA_REPO} every ${SYNC_INTERVAL_SECONDS:-300}s"

while true; do
  if clone_or_pull; then
    current_sha="$(git -C "${REPO_DIR}" rev-parse HEAD)"
    last_sha="$(cat "${STATE_FILE}" 2>/dev/null || echo '')"
    if [ "${current_sha}" != "${last_sha}" ]; then
      echo "New commit ${current_sha} (was ${last_sha:-none}) -- importing tags"
      if import_tags; then
        echo "${current_sha}" > "${STATE_FILE}"
      else
        echo "Import failed, will retry next cycle" >&2
      fi
    fi
  else
    echo "git clone/pull failed, will retry next cycle" >&2
  fi
  sleep "${SYNC_INTERVAL_SECONDS:-300}"
done
