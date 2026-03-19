#!/usr/bin/env bash
# setup.sh - Copy pgBackRest config files to their correct locations on the PostgreSQL server.
# Assumes this repo (k8s-argo) has been cloned to the server and this script is run from its location.
# Must be run with sudo (or as root).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/configs"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)." >&2
    exit 1
fi

# ============================================================================
# CONFIGURATION
# Collect values needed to substitute into installed config files.
# Pass as environment variables to skip prompts:
#   HA_IP=192.168.1.21 NAS_IP=192.168.1.253 sudo bash setup.sh
# ============================================================================

prompt_if_unset() {
    local var_name="$1"
    local prompt_text="$2"
    local current_val="${!var_name:-}"

    if [[ -z "$current_val" ]]; then
        read -r -p "$prompt_text: " current_val
    fi

    if [[ -z "$current_val" ]]; then
        echo "WARNING: $var_name not provided — placeholder YOUR_${var_name} will remain in installed files." >&2
        current_val="YOUR_${var_name}"
    fi

    printf '%s' "$current_val"
}

echo "==> Configuration"
echo "    (press Enter to leave a placeholder and fill in manually later)"
echo ""

HA_IP="$(prompt_if_unset HA_IP  "Home Assistant IP (e.g. 192.168.1.21)")"
NAS_IP="$(prompt_if_unset NAS_IP "NAS IP           (e.g. 192.168.1.253)")"

echo ""

# Helper: copy a file and substitute placeholders in the destination
install_file() {
    local src="$1"
    local dest="$2"
    cp "$src" "$dest"
    sed -i "s/YOUR_HA_IP/${HA_IP}/g"  "$dest"
    sed -i "s/YOUR_NAS_IP/${NAS_IP}/g" "$dest"
}

# ============================================================================
# INSTALL
# ============================================================================

echo "==> Creating required directories..."
mkdir -p /var/spool/pgbackrest
chown postgres:postgres /var/spool/pgbackrest
chmod 770 /var/spool/pgbackrest

mkdir -p /etc/postgresql/18/main/conf.d
mkdir -p /usr/local/bin/pgbackrest-scripts
mkdir -p /mnt/backups
mkdir -p /var/log/pgbackrest
chown postgres:postgres /var/log/pgbackrest

echo "==> Installing pgBackRest main config..."
install_file "$CONFIGS_DIR/pgbackrest.conf" /etc/pgbackrest.conf
chmod 640 /etc/pgbackrest.conf
chown postgres:postgres /etc/pgbackrest.conf

echo "==> Installing PostgreSQL WAL archiving config..."
install_file "$CONFIGS_DIR/postgresql-pgbackrest.conf" /etc/postgresql/18/main/conf.d/pgbackrest.conf
chown postgres:postgres /etc/postgresql/18/main/conf.d/pgbackrest.conf

echo "==> Installing backup scripts..."
install_file "$CONFIGS_DIR/full-backup.sh" /usr/local/bin/pgbackrest-scripts/full-backup.sh
install_file "$CONFIGS_DIR/diff-backup.sh" /usr/local/bin/pgbackrest-scripts/diff-backup.sh
chmod +x /usr/local/bin/pgbackrest-scripts/full-backup.sh
chmod +x /usr/local/bin/pgbackrest-scripts/diff-backup.sh

echo "==> Installing restore test script..."
install_file "$CONFIGS_DIR/pgbackrest_restore_test.sh" /usr/local/bin/pgbackrest_restore_test.sh
chmod +x /usr/local/bin/pgbackrest_restore_test.sh

echo "==> Installing cron jobs..."
install_file "$CONFIGS_DIR/pgbackrest.cron" /etc/cron.d/pgbackrest
install_file "$CONFIGS_DIR/pgbackrest-restore-test.cron" /etc/cron.d/pgbackrest-restore-test
chmod 644 /etc/cron.d/pgbackrest
chmod 644 /etc/cron.d/pgbackrest-restore-test

echo ""
echo "==> Files installed. Remaining manual steps:"
echo ""
echo "  1. Add NFS mount to /etc/fstab:"
echo "       echo '${NAS_IP}:/var/nfs/shared/PGBackup /mnt/backups nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab"
echo "       sudo mount -a"
echo ""
echo "  2. Create pgbackrest backup directory on the NFS mount:"
echo "       sudo mkdir -p /mnt/backups/pgbackrest"
echo "       sudo chown -R postgres:postgres /mnt/backups/pgbackrest"
echo "       sudo chmod 750 /mnt/backups/pgbackrest"
echo ""
echo "  3. Restart PostgreSQL to apply WAL archiving config:"
echo "       sudo pg_ctlcluster 18 main restart"
echo ""
echo "  4. Initialize the pgBackRest stanza:"
echo "       sudo -u postgres pgbackrest --stanza=main-db stanza-create"
echo "       sudo -u postgres pgbackrest --stanza=main-db check"
echo ""
echo "  5. Run the initial full backup:"
echo "       sudo -u postgres pgbackrest --stanza=main-db --type=full backup"
echo ""
echo "Done."
