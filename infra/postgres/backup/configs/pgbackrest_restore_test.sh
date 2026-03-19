#!/bin/bash
#
# pgBackRest Restore Test Script
# Tests that backups can actually be restored
# Run weekly via cron to verify backup integrity
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

STANZA_NAME="main-db"
PG_VERSION="18"
PG_PORT="5432"

# Test restore location (separate from production)
TEST_RESTORE_DIR="/var/tmp/pgbackrest-restore-test"
TEST_PG_PORT="5433"  # Different port from production

# Notification webhook (Home Assistant)
WEBHOOK_URL="${WEBHOOK_URL:-http://YOUR_HA_IP:8123/api/webhook/pgbackrest_restore_test}"

# Log file
LOG_FILE="/var/log/pgbackrest/restore-test.log"

# ============================================================================
# FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

notify() {
    local message="$1"
    local status="${2:-info}"  # info, success, error, warning
    
    log "$message"
    
    if [ -n "${WEBHOOK_URL}" ]; then
        curl -s -X POST "${WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"$message\", \"status\": \"$status\", \"timestamp\": \"$(date -Iseconds)\"}" \
            2>/dev/null || true
    fi
}

cleanup() {
    log "Cleaning up test environment..."
    
    # Stop test postgres if running
    if [ -f "${TEST_RESTORE_DIR}/postmaster.pid" ]; then
        sudo -u postgres /usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl \
            -D "${TEST_RESTORE_DIR}" stop -m fast 2>/dev/null || true
    fi
    
    # Remove test directory
    rm -rf "${TEST_RESTORE_DIR}"
    
    log "Cleanup complete"
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

log "=========================================="
log "pgBackRest Restore Test Starting"
log "=========================================="

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Check if production postgres is running
if ! systemctl is-active --quiet postgresql; then
    notify "Production PostgreSQL is not running - aborting test" "error"
    exit 1
fi

# Get latest backup info
log "Checking for available backups..."
BACKUP_INFO=$(sudo -u postgres pgbackrest --stanza="${STANZA_NAME}" info --output=json 2>/dev/null || echo "")

if [ -z "$BACKUP_INFO" ]; then
    notify "No backups found for stanza ${STANZA_NAME}" "error"
    exit 1
fi

# Get latest backup label
LATEST_BACKUP=$(echo "$BACKUP_INFO" | jq -r '.[0].backup[-1].label' 2>/dev/null || echo "")

if [ -z "$LATEST_BACKUP" ] || [ "$LATEST_BACKUP" = "null" ]; then
    notify "Could not determine latest backup" "error"
    exit 1
fi

log "Latest backup: ${LATEST_BACKUP}"

# Get backup details
BACKUP_TYPE=$(echo "$BACKUP_INFO" | jq -r '.[0].backup[-1].type' 2>/dev/null || echo "unknown")
BACKUP_SIZE=$(echo "$BACKUP_INFO" | jq -r '.[0].backup[-1].info.size' 2>/dev/null || echo "0")
BACKUP_SIZE_MB=$((BACKUP_SIZE / 1024 / 1024))

log "Backup type: ${BACKUP_TYPE}, Size: ${BACKUP_SIZE_MB}MB"

# Clean test directory if it exists
if [ -d "${TEST_RESTORE_DIR}" ]; then
    log "Removing existing test directory..."
    rm -rf "${TEST_RESTORE_DIR}"
fi

mkdir -p "${TEST_RESTORE_DIR}"
chown postgres:postgres "${TEST_RESTORE_DIR}"

# Perform restore
log "Restoring backup ${LATEST_BACKUP} to test directory..."
RESTORE_START=$(date +%s)

if sudo -u postgres pgbackrest --stanza="${STANZA_NAME}" \
    --pg1-path="${TEST_RESTORE_DIR}" \
    --type=immediate \
    --target-action=promote \
    restore; then
    
    RESTORE_END=$(date +%s)
    RESTORE_DURATION=$((RESTORE_END - RESTORE_START))
    log "Restore completed in ${RESTORE_DURATION} seconds"
else
    notify "Restore FAILED - check logs at /var/log/pgbackrest/${STANZA_NAME}-restore.log" "error"
    exit 1
fi

# Create minimal postgresql.conf for test instance
log "Configuring test PostgreSQL instance..."

cat > "${TEST_RESTORE_DIR}/postgresql.conf" << 'EOF'
# Minimal test instance configuration
listen_addresses = ''
port = 5433
unix_socket_directories = '/tmp'
max_connections = 100
shared_buffers = 256MB
dynamic_shared_memory_type = posix
max_wal_size = 1GB
min_wal_size = 80MB
log_timezone = 'UTC'
datestyle = 'iso, mdy'
timezone = 'UTC'
lc_messages = 'C'
lc_monetary = 'C'
lc_numeric = 'C'
lc_time = 'C'
default_text_search_config = 'pg_catalog.english'
archive_mode = off
EOF

chown postgres:postgres "${TEST_RESTORE_DIR}/postgresql.conf"
chmod 600 "${TEST_RESTORE_DIR}/postgresql.conf"

# Create pg_hba.conf for test instance
cat > "${TEST_RESTORE_DIR}/pg_hba.conf" << 'EOF'
# Test instance access control
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF

chown postgres:postgres "${TEST_RESTORE_DIR}/pg_hba.conf"
chmod 600 "${TEST_RESTORE_DIR}/pg_hba.conf"

# Create pg_ident.conf (also required)
cat > "${TEST_RESTORE_DIR}/pg_ident.conf" << 'EOF'
# Test instance ident mapping
EOF

chown postgres:postgres "${TEST_RESTORE_DIR}/pg_ident.conf"
chmod 600 "${TEST_RESTORE_DIR}/pg_ident.conf"

# Update postgresql.auto.conf with test overrides
cat >> "${TEST_RESTORE_DIR}/postgresql.auto.conf" << EOF

# Test instance overrides
port = ${TEST_PG_PORT}
unix_socket_directories = '/tmp'
listen_addresses = ''
archive_mode = off
EOF

# Start test postgres instance
log "Starting test PostgreSQL instance on port ${TEST_PG_PORT}..."
if sudo -u postgres /usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl \
    -D "${TEST_RESTORE_DIR}" \
    -l "${TEST_RESTORE_DIR}/logfile" \
    start; then
    
    log "Test instance started successfully"
    sleep 3  # Wait for startup
else
    notify "Failed to start test PostgreSQL instance" "error"
    exit 1
fi

# Run verification queries
log "Running verification queries..."

# Test 1: Can connect?
if sudo -u postgres psql -p "${TEST_PG_PORT}" -d postgres -c "SELECT version();" > /dev/null 2>&1; then
    log "✓ Connection test passed"
else
    notify "Connection test FAILED" "error"
    exit 1
fi

# Test 2: Check database count
DB_COUNT=$(sudo -u postgres psql -p "${TEST_PG_PORT}" -d postgres -t -c \
    "SELECT count(*) FROM pg_database WHERE datistemplate = false;" | xargs)
log "✓ Found ${DB_COUNT} databases in restored instance"

# Test 3: Sample data verification (if you have specific tables to check)
# Example: Check if a known table exists and has data
# Customize this for your databases
SAMPLE_CHECK=$(sudo -u postgres psql -p "${TEST_PG_PORT}" -d postgres -t -c \
    "SELECT count(*) FROM pg_tables WHERE schemaname = 'public';" | xargs)
log "✓ Found ${SAMPLE_CHECK} tables in public schema"

# Test 4: Check for critical system catalogs
CATALOG_CHECK=$(sudo -u postgres psql -p "${TEST_PG_PORT}" -d postgres -t -c \
    "SELECT count(*) FROM pg_class WHERE relname IN ('pg_database', 'pg_tables', 'pg_roles');" | xargs)

if [ "$CATALOG_CHECK" -eq 3 ]; then
    log "✓ System catalogs intact"
else
    notify "System catalog check FAILED - expected 3, got ${CATALOG_CHECK}" "error"
    exit 1
fi

# Test 5: WAL replay check (ensure recovery completed)
RECOVERY_STATUS=$(sudo -u postgres psql -p "${TEST_PG_PORT}" -d postgres -t -c \
    "SELECT pg_is_in_recovery();" | xargs)

if [ "$RECOVERY_STATUS" = "f" ]; then
    log "✓ Database is not in recovery mode (as expected)"
else
    log "⚠ Database is still in recovery mode (may be expected for PITR)"
fi

# Stop test instance
log "Stopping test PostgreSQL instance..."
sudo -u postgres /usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl \
    -D "${TEST_RESTORE_DIR}" stop -m fast

# Calculate total test time
TEST_END=$(date +%s)
TEST_START=$(date +%s)
TOTAL_DURATION=$((TEST_END - RESTORE_START))

# Success!
log "=========================================="
log "Restore Test PASSED"
log "=========================================="
log "Backup: ${LATEST_BACKUP} (${BACKUP_TYPE}, ${BACKUP_SIZE_MB}MB)"
log "Restore Time: ${RESTORE_DURATION}s"
log "Total Test Time: ${TOTAL_DURATION}s"
log "Databases: ${DB_COUNT}"

notify "Restore test PASSED - Backup ${LATEST_BACKUP} verified (${RESTORE_DURATION}s restore, ${DB_COUNT} databases)" "success"

exit 0
