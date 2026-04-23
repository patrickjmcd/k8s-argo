# Restoring a Single PostgreSQL Database

This procedure recovers one database (e.g., `lidarr-main`) from a pgBackRest backup without touching other databases on the production cluster.

Because pgBackRest backs up the entire PostgreSQL cluster, the process is:

1. Restore the full cluster to a **temporary location**
2. Start a temporary PostgreSQL instance on a different port
3. `pg_dump` the target database from the temp instance
4. `pg_restore` into the production instance
5. Clean up

---

## Prerequisites

- SSH access to the PostgreSQL server (Ubuntu VM)
- At least one available backup: `sudo -u postgres pgbackrest --stanza=main-db info`
- Free disk space for the restored cluster (check with `df -h /var/tmp`)
- The production PostgreSQL service can stay **running** throughout

---

## Step 1 — Confirm the target database exists in the backup

```bash
# Show available backups and their timestamps
sudo -u postgres pgbackrest --stanza=main-db info
```

Note the backup label you want (e.g., `20260423-020000F` for the latest full, or a differential label).

---

## Step 2 — Restore the cluster to a temporary directory

```bash
TEMP_DIR="/var/tmp/pgbackrest-single-db-restore"
PG_VERSION="18"

# Remove any previous temp restore
sudo rm -rf "${TEMP_DIR}"
sudo mkdir -p "${TEMP_DIR}"
sudo chown postgres:postgres "${TEMP_DIR}"

# Restore latest backup (remove --set flag to use latest)
sudo -u postgres pgbackrest \
    --stanza=main-db \
    --pg1-path="${TEMP_DIR}" \
    --type=immediate \
    --target-action=promote \
    restore
```

To restore from a **specific backup label** instead of latest:

```bash
sudo -u postgres pgbackrest \
    --stanza=main-db \
    --pg1-path="${TEMP_DIR}" \
    --set=20260423-020000F \
    --type=immediate \
    --target-action=promote \
    restore
```

To restore to a **specific point in time** (e.g., just before a bad migration ran):

```bash
sudo -u postgres pgbackrest \
    --stanza=main-db \
    --pg1-path="${TEMP_DIR}" \
    --type=time \
    --target="2026-04-23 14:30:00" \
    --target-action=promote \
    restore
```

---

## Step 3 — Configure and start the temporary PostgreSQL instance

```bash
TEMP_DIR="/var/tmp/pgbackrest-single-db-restore"
PG_VERSION="18"
TEMP_PORT="5433"

# postgresql.conf is NOT in the data dir on Ubuntu (lives in /etc/postgresql/...)
# so we must create a minimal one for the temp instance
sudo -u postgres tee "${TEMP_DIR}/postgresql.conf" > /dev/null << EOF
listen_addresses = ''
port = ${TEMP_PORT}
unix_socket_directories = '/tmp'
max_connections = 100
shared_buffers = 256MB
dynamic_shared_memory_type = posix
max_wal_size = 1GB
min_wal_size = 80MB
log_timezone = 'UTC'
datestyle = 'iso, mdy'
timezone = 'UTC'
default_text_search_config = 'pg_catalog.english'
archive_mode = off
EOF

sudo -u postgres tee "${TEMP_DIR}/pg_hba.conf" > /dev/null << 'EOF'
local   all   all   trust
host    all   all   127.0.0.1/32   trust
host    all   all   ::1/128        trust
EOF

sudo -u postgres touch "${TEMP_DIR}/pg_ident.conf"

# Start the temp instance
sudo -u postgres /usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl \
    -D "${TEMP_DIR}" \
    -l "${TEMP_DIR}/logfile" \
    start

# Wait for startup and confirm it's listening
sleep 3
sudo -u postgres psql -h /tmp -p ${TEMP_PORT} -c "SELECT current_timestamp, pg_is_in_recovery();"
```

If the last command returns `f` for `pg_is_in_recovery`, the instance is ready.

---

## Step 4 — Verify the target database is present

```bash
# List all databases in the restored instance
sudo -u postgres psql -h /tmp -p 5433 -l

# Confirm row counts look right for the target DB (example: lidarr-main)
sudo -u postgres psql -h /tmp -p 5433 -d lidarr-main -c "\dt"
sudo -u postgres psql -h /tmp -p 5433 -d lidarr-main -c "SELECT count(*) FROM \"Artists\";"
```

If the data looks wrong (missing rows, wrong timestamp), stop here, shut down the temp instance, and repeat Step 2 with a different `--set` label or `--target` time.

---

## Step 5 — Dump the target database from the temp instance

```bash
TARGET_DB="lidarr-main"
DUMP_FILE="/var/tmp/${TARGET_DB}-$(date +%Y%m%d-%H%M%S).dump"

sudo -u postgres pg_dump \
    -h /tmp -p 5433 \
    --format=custom \
    --no-privileges \
    --no-owner \
    "${TARGET_DB}" \
    -f "${DUMP_FILE}"

echo "Dump written to: ${DUMP_FILE}"
ls -lh "${DUMP_FILE}"
```

---

## Step 6 — Stop the temporary instance

```bash
sudo -u postgres /usr/lib/postgresql/18/bin/pg_ctl \
    -D /var/tmp/pgbackrest-single-db-restore \
    stop -m fast
```

---

## Step 7 — Restore the dump into production

> **If the database still exists in production** and you want to replace it, drop and recreate it first. Otherwise, skip the DROP/CREATE lines.

```bash
TARGET_DB="lidarr-main"
DUMP_FILE="/var/tmp/${TARGET_DB}-<timestamp>.dump"   # use actual filename from Step 5

# Drop and recreate (adjust owner/template as needed)
sudo -u postgres psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${TARGET_DB}';"
sudo -u postgres psql -c "DROP DATABASE IF EXISTS \"${TARGET_DB}\";"
sudo -u postgres psql -c "CREATE DATABASE \"${TARGET_DB}\";"

# Restore
sudo -u postgres pg_restore \
    --no-privileges \
    --no-owner \
    -d "${TARGET_DB}" \
    "${DUMP_FILE}"
```

If the app uses a specific role as the DB owner, set it after restore:

```bash
sudo -u postgres psql -c "ALTER DATABASE \"${TARGET_DB}\" OWNER TO <app_role>;"
```

---

## Step 8 — Verify the restored data in production

```bash
TARGET_DB="lidarr-main"

sudo -u postgres psql -d "${TARGET_DB}" -c "\dt"
sudo -u postgres psql -d "${TARGET_DB}" -c "SELECT count(*) FROM \"Artists\";"
```

Restart the app pod that owns this database so it reconnects cleanly:

```bash
# Example for lidarr running in Kubernetes
kubectl rollout restart deployment lidarr -n media
```

---

## Step 9 — Clean up

```bash
sudo rm -rf /var/tmp/pgbackrest-single-db-restore
sudo rm -f /var/tmp/lidarr-main-*.dump   # or whatever DUMP_FILE was named
```

---

## Quick reference

| Task | Command |
|---|---|
| List backups | `sudo -u postgres pgbackrest --stanza=main-db info` |
| Restore to temp (latest) | `sudo -u postgres pgbackrest --stanza=main-db --pg1-path=<dir> --type=immediate --target-action=promote restore` |
| Start temp instance | `sudo -u postgres pg_ctl -D <dir> -l <dir>/logfile start` |
| List DBs in temp | `sudo -u postgres psql -h /tmp -p 5433 -l` |
| Dump from temp | `sudo -u postgres pg_dump -h /tmp -p 5433 --format=custom <db> -f <file>` |
| Stop temp instance | `sudo -u postgres pg_ctl -D <dir> stop -m fast` |
| Restore dump to prod | `sudo -u postgres pg_restore --no-privileges --no-owner -d <db> <file>` |

---

## Notes

- The production PostgreSQL service stays running throughout — only the target database is touched.
- Port `5433` is the same port used by the automated restore test script (`pgbackrest_restore_test.sh`). Do not run them simultaneously.
- If the temp restore fails with a WAL error, add `--type=immediate` to skip WAL replay and promote immediately from the latest consistent backup state.
- Dump files use `--format=custom` (binary). To inspect the contents without restoring: `sudo -u postgres pg_restore -l <file>`.
