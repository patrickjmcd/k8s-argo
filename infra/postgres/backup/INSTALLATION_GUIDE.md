# PostgreSQL Backup with pgBackRest - Complete Setup Guide

**Last Updated:** 2026-03-19 09:57:56
**Environment:** PostgreSQL 18 on Ubuntu 22.04, UniFi NAS, Home Assistant
**Configuration Files:** [k8s-argo/infra/postgres/backup](https://github.com/patrickjmcd/k8s-argo/tree/main/infra/postgres/backup)

This guide documents the complete setup process for enterprise-grade PostgreSQL backups using pgBackRest with automated restore testing and Home Assistant notifications.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Setup (Recommended)](#quick-setup-recommended)
4. [Detailed Manual Installation](#detailed-manual-installation)
   - [Step 1: Configure NAS Storage](#step-1-configure-nas-storage)
   - [Step 2: Mount NFS Share on PostgreSQL Server](#step-2-mount-nfs-share-on-postgresql-server)
   - [Step 3: Create PostgreSQL Backup User](#step-3-create-postgresql-backup-user)
   - [Step 4: Install and Configure pgBackRest](#step-4-install-and-configure-pgbackrest)
   - [Step 5: Configure PostgreSQL for WAL Archiving](#step-5-configure-postgresql-for-wal-archiving)
   - [Step 6: Initialize pgBackRest Stanza](#step-6-initialize-pgbackrest-stanza)
   - [Step 7: Set Up Backup Scripts](#step-7-set-up-backup-scripts)
   - [Step 8: Configure Automated Backups (Cron)](#step-8-configure-automated-backups-cron)
   - [Step 9: Set Up Restore Testing](#step-9-set-up-restore-testing)
   - [Step 10: Configure Home Assistant Notifications](#step-10-configure-home-assistant-notifications)
5. [Verification and Testing](#verification-and-testing)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## Overview

### What We're Building

- **Backup Strategy:** Weekly full backups, daily differential backups
- **Retention:** 4 full + 4 differential (≈6 weeks)
- **Point-in-Time Recovery:** Continuous WAL archiving
- **Storage:** NFS mount to UniFi NAS with dual redundancy
- **Testing:** Automated weekly restore tests
- **Monitoring:** Home Assistant notifications for success/failure

### Architecture
```
PostgreSQL Server (Ubuntu VM)
    ↓ pgBackRest writes
NFS Mount (/mnt/backups)
    ↓ mounted from
UniFi NAS (YOUR_NAS_IP:/var/nfs/shared/PGBackup)
    ↓ replicates to
Secondary Storage (handled by UniFi)

Notifications
    ↓ webhooks to
Home Assistant (YOUR_HA_IP:8123)
```

### Key Benefits Over pg_dump

- **3-5x faster backups** (parallel processing)
- **10-100x faster restores** (delta restore)
- **80% less storage** (differential backups)
- **Point-in-time recovery** (WAL archiving)
- **Automated integrity testing**

---

## Prerequisites

### Required

- PostgreSQL server (Ubuntu 22.04 or similar)
- NAS with NFS support (UniFi NAS, Synology, TrueNAS, etc.)
- Network connectivity between PostgreSQL server and NAS
- Root/sudo access to PostgreSQL server
- Home Assistant instance (optional, for notifications)

### Software Versions Used

- PostgreSQL: 18.3
- pgBackRest: 2.58.0
- Ubuntu: 22.04 LTS
- UniFi OS: Latest

---

## Quick Setup (Recommended)

This approach uses `setup.sh` to automate all config file placement and directory creation. A few steps still require manual action (NAS configuration, NFS mounting, PostgreSQL user creation, and stanza initialization).

### What the script handles automatically
- Creates required directories with correct permissions (`/var/spool/pgbackrest`, `/var/log/pgbackrest`, `/usr/local/bin/pgbackrest-scripts`, etc.)
- Installs `/etc/pgbackrest.conf` with correct ownership and permissions
- Installs the PostgreSQL WAL archiving config into `conf.d`
- Installs backup scripts (`full-backup.sh`, `diff-backup.sh`) and the restore test script
- Installs cron jobs for automated backups and restore testing

### What you still do manually
- Configure NAS storage and NFS export (one-time, on the NAS)
- Mount the NFS share and add it to `/etc/fstab`
- Create the PostgreSQL backup user
- Install the `pgbackrest` package
- Initialize the pgBackRest stanza and run the first backup
- Configure Home Assistant notifications (optional)

---

### Quick Setup Steps

**Step 1 — Configure NAS storage** (one-time, on the UniFi console)

See [Step 1: Configure NAS Storage](#step-1-configure-nas-storage) for full instructions.

---

**Step 2 — Mount the NFS share**

```bash
sudo apt install nfs-common -y

# Verify the NFS export is visible
showmount -e YOUR_NAS_IP

# Create the mount point and test
sudo mkdir -p /mnt/backups
sudo mount -t nfs YOUR_NAS_IP:/var/nfs/shared/PGBackup /mnt/backups
df -T | grep backups  # Must show: nfs

# Test symlink support (required by pgBackRest)
cd /mnt/backups && ln -s test testlink && ls -la testlink && rm testlink && cd ~

# Test write access as postgres user
sudo -u postgres touch /mnt/backups/test && sudo -u postgres rm /mnt/backups/test

# Unmount (will be remounted via fstab in a later step)
sudo umount /mnt/backups
```

See [Step 2](#step-2-mount-nfs-share-on-postgresql-server) for persistent mount and fstab instructions.

---

**Step 3 — Create the PostgreSQL backup user**

See [Step 3: Create PostgreSQL Backup User](#step-3-create-postgresql-backup-user) for full SQL instructions.

---

**Step 4 — Install pgBackRest**

```bash
sudo apt update && sudo apt install pgbackrest -y
pgbackrest version
```

---

**Step 5 — Clone the repo and run setup.sh**

```bash
# Clone the repo to the postgres server (if not already present)
git clone https://github.com/patrickjmcd/k8s-argo.git ~/k8s-argo
```

Run the setup script. It will prompt for your Home Assistant IP and NAS IP, substitute them into all installed files, then print the remaining steps.

```bash
sudo bash ~/k8s-argo/infra/postgres/backup/setup.sh
```

Or pass the values as environment variables to skip the prompts:

```bash
sudo HA_IP=192.168.1.21 NAS_IP=192.168.1.253 bash ~/k8s-argo/infra/postgres/backup/setup.sh
```

If you skip a value (just press Enter), the placeholder (`YOUR_HA_IP` / `YOUR_NAS_IP`) will remain in the installed files and you can fill them in manually later.

---

**Step 6 — Add NFS mount to /etc/fstab**

```bash
# Append the fstab entry
sudo sh -c 'cat ~/k8s-argo/infra/postgres/backup/configs/fstab.backups >> /etc/fstab'

# Mount and verify
sudo mount -a
mountpoint /mnt/backups
df -T | grep backups  # Must say "nfs"
```

---

**Step 7 — Create the backup directory on the NFS mount**

```bash
sudo mkdir -p /mnt/backups/pgbackrest
sudo chown -R postgres:postgres /mnt/backups/pgbackrest
sudo chmod 750 /mnt/backups/pgbackrest
```

---

**Step 8 — Restart PostgreSQL to apply WAL archiving config**

```bash
sudo pg_ctlcluster 18 main restart
sudo tail -50 /var/log/postgresql/postgresql-18-main.log
```

---

**Step 9 — Initialize the stanza and run the first backup**

```bash
sudo -u postgres pgbackrest --stanza=main-db stanza-create
sudo -u postgres pgbackrest --stanza=main-db check
sudo -u postgres pgbackrest --stanza=main-db --type=full backup
sudo -u postgres pgbackrest --stanza=main-db info
```

---

**Step 10 — Configure Home Assistant notifications** (optional)

See [Step 10: Configure Home Assistant Notifications](#step-10-configure-home-assistant-notifications).

---

## Detailed Manual Installation

The sections below document each step in full detail. Use these if you prefer not to use the setup script, or want to understand exactly what the script does.

---

### Step 1: Configure NAS Storage

#### On UniFi Console

1. Navigate to **UniFi OS Console → Storage → Shared Folders**

2. Create a new shared folder:
   - **Name:** `PGBackup`
   - **Protocols:** Enable **NFS**

3. Configure NFS permissions:
   - **Allowed clients:** Add your PostgreSQL server IP (e.g., `YOUR_PG_SERVER_IP`)
   - **Permissions:** Read/Write
   - **Squash:** No root squash

4. Note the NFS export path:
```bash
   # /var/nfs/shared/PGBackup    YOUR_PG_SERVER_IP
```
---

### Step 2: Mount NFS Share on PostgreSQL Server

#### Install NFS Client
```bash
sudo apt update
sudo apt install nfs-common -y
```

#### Test NFS Connection
```bash
# Verify NFS exports are visible
showmount -e YOUR_NAS_IP

# Expected output (your path may vary):
# Export list for YOUR_NAS_IP:
# /var/nfs/shared/PGBackup YOUR_PG_SERVER_IP
```

#### Create Mount Point
```bash
sudo mkdir -p /mnt/backups
```

#### Test Manual Mount
```bash
# Replace with your actual NFS path from showmount output
sudo mount -t nfs YOUR_NAS_IP:/var/nfs/shared/PGBackup /mnt/backups

# Verify mount
df -h | grep backups
mount | grep backups

# IMPORTANT: Verify NFS (not CIFS)
df -T | grep backups
# Must show: nfs

# Test symlink support (critical for pgBackRest)
cd /mnt/backups
ln -s test testlink
ls -la testlink
rm testlink
cd ~

# Test write access
sudo touch /mnt/backups/test
sudo rm /mnt/backups/test

# If successful, unmount (we'll add to fstab next)
sudo umount /mnt/backups
```

#### Add to /etc/fstab for Persistent Mount

See: [`fstab.backups`](./configs/fstab.backups)
```bash
sudo nvim /etc/fstab
```

Add this line (replace with your NFS path):
```
YOUR_NAS_IP:/var/nfs/shared/PGBackup /mnt/backups nfs defaults,_netdev 0 0
```

**Important:** The `_netdev` option ensures the mount waits for network availability.

#### Mount and Verify
```bash
# Mount using fstab
sudo mount -a

# Verify
df -h | grep backups
df -T | grep backups  # Should say "nfs"
mountpoint /mnt/backups  # Should say "is a mountpoint"

# Test as postgres user
sudo -u postgres touch /mnt/backups/test
sudo -u postgres rm /mnt/backups/test
```

---

### Step 3: Create PostgreSQL Backup User

#### Why a Dedicated Backup User?

- Security: Limited permissions (read-only)
- Auditing: Track backup operations separately
- Best practice: Never use postgres superuser for backups

#### Create the User (PostgreSQL 15+)
```bash
# Connect as postgres superuser
sudo -u postgres psql
```

```sql
-- Create backup user
CREATE USER backup_user WITH PASSWORD 'your_secure_password_here';

-- Grant read-all privileges (PostgreSQL 15+)
GRANT pg_read_all_data TO backup_user;

-- Grant connect to all databases you want to backup
GRANT CONNECT ON DATABASE database1 TO backup_user;
GRANT CONNECT ON DATABASE database2 TO backup_user;
-- Repeat for each database

-- For each database, grant read access
\c database1
GRANT pg_read_all_data TO backup_user;

\c database2
GRANT pg_read_all_data TO backup_user;

-- Security hardening
ALTER ROLE backup_user CONNECTION LIMIT 5;
ALTER ROLE backup_user NOCREATEDB NOCREATEROLE;

-- Verify
\du backup_user
\q
```

#### Test the Backup User
```bash
# Test connection and read access
psql -h localhost -U backup_user -d database1 -c "SELECT version();"
psql -h localhost -U backup_user -d database1 -c "SELECT count(*) FROM pg_tables WHERE schemaname = 'public';"

# Test that writes fail (should error)
psql -h localhost -U backup_user -d database1 -c "CREATE TABLE test (id int);"
# Expected: ERROR: permission denied for schema public
```

---

### Step 4: Install and Configure pgBackRest

#### Add PostgreSQL APT Repository (if not already added)
```bash
# Install prerequisites
sudo apt update
sudo apt install curl ca-certificates -y

# Add PostgreSQL GPG key
sudo install -d /usr/share/postgresql-common/pgdg
curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
  --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc

# Add repository
sudo sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
  https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list'

# Update package list
sudo apt update
```

#### Install pgBackRest
```bash
sudo apt install pgbackrest -y

# Verify installation
pgbackrest version
# Should show: pgBackRest 2.58.0 or newer
```

#### Create Required Directories

**CRITICAL:** Create the spool directory for lock files (must persist across reboots):
```bash
# Create persistent spool directory (NOT /tmp which clears on reboot)
sudo mkdir -p /var/spool/pgbackrest
sudo chown postgres:postgres /var/spool/pgbackrest
sudo chmod 770 /var/spool/pgbackrest

# Verify
ls -la /var/spool/ | grep pgbackrest
```

#### Create pgBackRest Configuration

See: [`pgbackrest.conf`](./configs/pgbackrest.conf)
```bash
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/pgbackrest.conf /etc/pgbackrest.conf
```

Or create manually:
```bash
sudo nvim /etc/pgbackrest.conf
```

**Critical Notes:**
- `lock-path=/var/spool/pgbackrest` - This is REQUIRED and must persist across reboots
- Without this, WAL archiving will fail with exit code 41
- Replace `18` with your PostgreSQL version
- Adjust `pg1-path` if your data directory is different

#### Set Permissions
```bash
sudo chmod 640 /etc/pgbackrest.conf
sudo chown postgres:postgres /etc/pgbackrest.conf
```

#### Create Backup Directories
```bash
sudo mkdir -p /mnt/backups/pgbackrest
sudo chown -R postgres:postgres /mnt/backups/pgbackrest
sudo chmod 750 /mnt/backups/pgbackrest
```

---

### Step 5: Configure PostgreSQL for WAL Archiving

See: [`postgresql-pgbackrest.conf`](./configs/postgresql-pgbackrest.conf)

WAL (Write-Ahead Log) archiving enables point-in-time recovery.

#### Install the Configuration File
```bash
sudo mkdir -p /etc/postgresql/18/main/conf.d

# Copy from repo
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/postgresql-pgbackrest.conf \
    /etc/postgresql/18/main/conf.d/pgbackrest.conf
sudo chown postgres:postgres /etc/postgresql/18/main/conf.d/pgbackrest.conf
```

Or create manually:
```bash
sudo nvim /etc/postgresql/18/main/conf.d/pgbackrest.conf
```

#### Restart PostgreSQL
```bash
# Restart to apply WAL archiving settings
sudo pg_ctlcluster 18 main restart

# Check for errors
sudo tail -50 /var/log/postgresql/postgresql-18-main.log
```

---

### Step 6: Initialize pgBackRest Stanza

A "stanza" is pgBackRest's configuration for a specific PostgreSQL database cluster.

#### Create the Stanza
```bash
sudo -u postgres pgbackrest --stanza=main-db stanza-create
```

Expected output:
```
P00   INFO: stanza-create command begin
P00   INFO: stanza-create command end: completed successfully
```

#### Verify Configuration
```bash
sudo -u postgres pgbackrest --stanza=main-db check
```

Expected output:
```
P00   INFO: check command begin
P00   INFO: check repo1 configuration (primary)
P00   INFO: check repo1 archive for WAL (primary)
P00   INFO: WAL segment ... successfully archived
P00   INFO: check command end: completed successfully
```

If this fails with "Permission denied" on `/tmp/pgbackrest/main-db.stop`, verify:
1. You created `/var/spool/pgbackrest` with correct permissions
2. You added `lock-path=/var/spool/pgbackrest` to `/etc/pgbackrest.conf`

#### View Stanza Information
```bash
sudo -u postgres pgbackrest --stanza=main-db info
```

Should show:
```
stanza: main-db
    status: ok
    cipher: none

    db (current)
        wal archive min/max (18): ...
```

#### Perform Initial Full Backup
```bash
sudo -u postgres pgbackrest --stanza=main-db --type=full backup
```

This will take several minutes depending on database size. Expected output:
```
P00   INFO: backup command begin
P00   INFO: execute backup start
P00   INFO: backup start archive = ...
P00   INFO: execute backup stop
P00   INFO: new backup label = 20260318-020000F
P00   INFO: backup command end: completed successfully
```

#### Verify Backup
```bash
sudo -u postgres pgbackrest --stanza=main-db info
```

Should now show your backup:
```
stanza: main-db
    status: ok

    db (current)
        wal archive min/max (18): ...

        full backup: 20260318-020000F
            timestamp start/stop: ...
            database size: X.XGB
            repo1: backup size: X.XGB
```

---

### Step 7: Set Up Backup Scripts

See configuration files:
- [`full-backup.sh`](./configs/full-backup.sh)
- [`diff-backup.sh`](./configs/diff-backup.sh)

#### Create Scripts Directory
```bash
sudo mkdir -p /usr/local/bin/pgbackrest-scripts
```

#### Install Scripts
```bash
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/full-backup.sh \
    /usr/local/bin/pgbackrest-scripts/full-backup.sh
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/diff-backup.sh \
    /usr/local/bin/pgbackrest-scripts/diff-backup.sh

sudo chmod +x /usr/local/bin/pgbackrest-scripts/full-backup.sh
sudo chmod +x /usr/local/bin/pgbackrest-scripts/diff-backup.sh
```

#### Test Scripts
```bash
# Test differential backup (full already exists)
sudo -u postgres /usr/local/bin/pgbackrest-scripts/diff-backup.sh

# Verify in Home Assistant (if configured) or check logs
sudo -u postgres pgbackrest --stanza=main-db info
```

---

### Step 8: Configure Automated Backups (Cron)

See: [`pgbackrest.cron`](./configs/pgbackrest.cron)

#### Install Cron Job
```bash
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/pgbackrest.cron \
    /etc/cron.d/pgbackrest
sudo chmod 644 /etc/cron.d/pgbackrest
```

Or create manually:
```bash
sudo nvim /etc/cron.d/pgbackrest
```

#### Verify Cron Jobs Are Scheduled
```bash
# Check cron file
cat /etc/cron.d/pgbackrest

# Verify cron service
sudo systemctl status cron
```

---

### Step 9: Set Up Restore Testing

See: [`pgbackrest_restore_test.sh`](./configs/pgbackrest_restore_test.sh)

Automated restore testing ensures backups are actually recoverable.

#### Install Dependencies
```bash
sudo apt install jq -y
```

#### Install Restore Test Script
```bash
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/pgbackrest_restore_test.sh \
    /usr/local/bin/pgbackrest_restore_test.sh
sudo chmod +x /usr/local/bin/pgbackrest_restore_test.sh
```

**Key features of the restore test script:**
- Restores latest backup to `/var/tmp/pgbackrest-restore-test`
- Starts test PostgreSQL instance on port 5433
- Runs verification queries (connection, database count, system catalogs)
- Sends results to Home Assistant webhook
- Cleans up automatically

#### Test Manually
```bash
sudo /usr/local/bin/pgbackrest_restore_test.sh
```

Expected output:
```
[TIMESTAMP] Restore Test PASSED
[TIMESTAMP] Backup: 20260318-193259F (full, 967MB)
[TIMESTAMP] Restore Time: 9s
[TIMESTAMP] Databases: 19
```

#### Install Restore Test Cron Job

See: [`pgbackrest-restore-test.cron`](./configs/pgbackrest-restore-test.cron)
```bash
sudo cp /path/to/k8s-argo/infra/postgres/backup/configs/pgbackrest-restore-test.cron \
    /etc/cron.d/pgbackrest-restore-test
sudo chmod 644 /etc/cron.d/pgbackrest-restore-test
```

Or create manually:
```bash
sudo nvim /etc/cron.d/pgbackrest-restore-test
```

---

### Step 10: Configure Home Assistant Notifications

### Create Input Text Helpers

In Home Assistant:

1. **Settings → Devices & Services → Helpers → Create Helper → Text**

Create two helpers:

**Helper 1:**
- Name: `Last Postgres Backup`
- Entity ID: `input_text.last_postgres_backup`
- Max length: 255

**Helper 2:**
- Name: `Last pgBackRest Restore Test`
- Entity ID: `input_text.last_pgbackrest_restore_test`
- Max length: 255

### Add Automations

**Configuration → Automations → Add Automation → Skip → YAML Mode**

Copy the two automations from [`home-assistant-automations.yaml`](./configs/home-assistant-automations.yaml) in the repo:

1. **pgBackRest Backup Status** - Monitors backup success/failure
2. **pgBackRest Restore Test Notification** - Monitors weekly restore test results

### Test Webhooks
```bash
# Test backup webhook
curl -X POST "http://YOUR_HA_IP:8123/api/webhook/pgbackrest_backup_status" \
    -H "Content-Type: application/json" \
    -d '{"status":"success","message":"Test backup notification","type":"full"}'

# Test restore webhook
curl -X POST "http://YOUR_HA_IP:8123/api/webhook/pgbackrest_restore_test" \
    -H "Content-Type: application/json" \
    -d '{"status":"success","message":"Test restore notification","timestamp":"2026-03-19T02:00:00Z"}'

# Check Home Assistant for notifications
```

---

## Verification and Testing

### Complete Verification Checklist

Run these commands to verify everything is working:
```bash
# 1. Verify NFS mount
mountpoint /mnt/backups
df -h | grep backups
df -T | grep backups  # Must say "nfs" not "cifs"

# 2. Test symlink support on NFS
cd /mnt/backups && ln -s test testlink && ls -la testlink && rm testlink

# 3. Test write access
sudo -u postgres touch /mnt/backups/test
sudo -u postgres rm /mnt/backups/test

# 4. Verify spool directory exists
ls -la /var/spool/ | grep pgbackrest

# 5. Check pgBackRest configuration
sudo -u postgres pgbackrest --stanza=main-db check

# 6. View backups
sudo -u postgres pgbackrest --stanza=main-db info

# 7. Check cron jobs
cat /etc/cron.d/pgbackrest
cat /etc/cron.d/pgbackrest-restore-test

# 8. Verify backup logs
tail -50 /var/log/pgbackrest/main-db-backup.log

# 9. Check restore test logs
tail -50 /var/log/pgbackrest/restore-test.log

# 10. Test manual differential backup
sudo -u postgres pgbackrest --stanza=main-db --type=diff backup

# 11. Run restore test
sudo /usr/local/bin/pgbackrest_restore_test.sh

# 12. Check PostgreSQL is archiving WALs
sudo tail -50 /var/log/postgresql/postgresql-18-main.log | grep archive
```

### Expected Results

All of these should return success:

- ✅ NFS mount is active (type: nfs)
- ✅ Symlinks work on NFS mount
- ✅ postgres user can write to backup location
- ✅ /var/spool/pgbackrest exists with correct permissions
- ✅ pgBackRest stanza check passes
- ✅ At least one full backup exists
- ✅ Cron jobs are configured
- ✅ Backup scripts are executable
- ✅ Restore test completes successfully
- ✅ Home Assistant receives webhook notifications
- ✅ WAL archiving is working

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: NFS Mount Shows as CIFS

**Problem:** `df -T | grep backups` shows `cifs` instead of `nfs`

**Solution:**
```bash
# Unmount
sudo umount /mnt/backups

# Edit /etc/fstab - ensure it says "nfs" not "cifs"
sudo nvim /etc/fstab

# Should be:
# YOUR_NAS_IP:/var/nfs/shared/PGBackup /mnt/backups nfs defaults,_netdev 0 0

# Remount
sudo mount -a

# Verify
df -T | grep backups
```

#### Issue: WAL Archiving Fails with Exit Code 41

**Problem:** PostgreSQL logs show `archive command failed with exit code 41`

**Root Cause:** pgBackRest needs a persistent lock directory. Exit code 41 = "Permission denied" or "Operation not supported" - typically because `/tmp/pgbackrest` doesn't exist or has wrong permissions, and `/tmp` is cleared on reboot.

**Solutions:**

1. **Check if lock directory exists:**
```bash
ls -la /var/spool/pgbackrest
# If missing:
sudo mkdir -p /var/spool/pgbackrest
sudo chown postgres:postgres /var/spool/pgbackrest
sudo chmod 770 /var/spool/pgbackrest
```

2. **Verify lock-path in pgbackrest.conf:**
```bash
grep lock-path /etc/pgbackrest.conf
# Should show: lock-path=/var/spool/pgbackrest
# If missing, add it to the [global] section
```

3. **Restart PostgreSQL after fixing:**
```bash
sudo systemctl restart postgresql
```

4. **Test manually:**
```bash
sudo -u postgres pgbackrest --stanza=main-db --log-level-console=debug \
    archive-push /var/lib/postgresql/18/main/pg_wal/000000010000000400000001
```

#### Issue: Stanza Creation Fails

**Problem:** `ERROR: unable to find primary cluster`

**Solution:**
```bash
# Verify PostgreSQL is running
systemctl status postgresql

# Check pg1-path in /etc/pgbackrest.conf matches actual data directory
sudo -u postgres psql -c "SHOW data_directory;"

# Verify version in config matches PostgreSQL version
psql --version  # Should match pg1-path
```

#### Issue: Backup Fails with "Permission denied"

**Problem:** Backup fails with permission errors

**Solution:**
```bash
# Check NFS mount permissions
sudo -u postgres touch /mnt/backups/test
sudo -u postgres rm /mnt/backups/test

# Check backup directory ownership
ls -la /mnt/backups/
sudo chown -R postgres:postgres /mnt/backups/pgbackrest

# Check spool directory
ls -la /var/spool/pgbackrest
sudo chown postgres:postgres /var/spool/pgbackrest
```

#### Issue: Restore Test Fails to Start

**Problem:** `could not start server`

**Solution:**
```bash
# Check logfile for actual error
cat /var/tmp/pgbackrest-restore-test/logfile

# Common issues:
# - max_connections too low (script sets it to 100)
# - Missing postgresql.conf, pg_hba.conf, pg_ident.conf (script creates these)
# - Port 5433 already in use

# Check port availability
sudo lsof -i :5433
```

#### Issue: Home Assistant Not Receiving Webhooks

**Problem:** No notifications appearing

**Solution:**
```bash
# Test webhook manually
curl -v -X POST "http://YOUR_HA_IP:8123/api/webhook/pgbackrest_backup_status" \
    -H "Content-Type: application/json" \
    -d '{"status":"test","message":"Manual test"}'

# Check Home Assistant logs
# Settings → System → Logs
# Search for "webhook" or "pgbackrest"

# Verify automation is enabled
# Settings → Automations → Check pgBackRest automations are ON

# Check webhook URL in scripts
grep WEBHOOK_URL /usr/local/bin/pgbackrest-scripts/full-backup.sh
grep WEBHOOK_URL /usr/local/bin/pgbackrest_restore_test.sh
```

---

## Maintenance

### Daily Operations
```bash
# Check backup status
sudo -u postgres pgbackrest --stanza=main-db info

# View recent backup logs
tail -50 /var/log/pgbackrest/main-db-backup.log

# Check storage usage
du -sh /mnt/backups/pgbackrest/
```

### Weekly Tasks
```bash
# Review restore test results (automatic Sunday 4 AM)
tail -100 /var/log/pgbackrest/restore-test.log

# Check for failed cron jobs
sudo grep pgbackrest /var/log/syslog
```

### Monthly Tasks
```bash
# Verify all databases are being backed up
sudo -u postgres psql -l
sudo -u postgres pgbackrest --stanza=main-db info

# Review retention and adjust if needed
sudo nvim /etc/pgbackrest.conf
# Update repo1-retention-full and repo1-retention-diff
```

### Restore a Backup

**Full restore (latest backup):**
```bash
# 1. Stop PostgreSQL
sudo systemctl stop postgresql

# 2. Restore
sudo -u postgres pgbackrest --stanza=main-db --delta restore

# 3. Start PostgreSQL
sudo systemctl start postgresql

# 4. Verify
sudo -u postgres psql -l
```

**Point-in-time recovery:**
```bash
sudo systemctl stop postgresql

sudo -u postgres pgbackrest --stanza=main-db --delta \
    --type=time --target="2026-03-19 14:30:00" \
    --target-action=promote \
    restore

sudo systemctl start postgresql
```

---

## Summary

You now have a production-grade PostgreSQL backup system with:

- ✅ Enterprise-grade backups (pgBackRest)
- ✅ Automated daily backups (cron)
- ✅ Point-in-time recovery (WAL archiving)
- ✅ Weekly restore testing (automated)
- ✅ Home Assistant notifications (success/failure alerts)
- ✅ Dual redundancy (UniFi NAS replication)
- ✅ 6 weeks retention (4 full + 4 differential)
- ✅ NFS storage with full symlink support
- ✅ Persistent lock directory (/var/spool/pgbackrest)

### Configuration Files Repository

All configuration files are stored in: **[k8s-argo/infra/postgres/backup](https://github.com/patrickjmcd/k8s-argo/tree/main/infra/postgres/backup)**

- `setup.sh` - Automated config deployment script
- `configs/pgbackrest.conf` - pgBackRest main configuration
- `configs/postgresql-pgbackrest.conf` - PostgreSQL WAL archiving config
- `configs/full-backup.sh` - Full backup script with notifications
- `configs/diff-backup.sh` - Differential backup script with notifications
- `configs/pgbackrest_restore_test.sh` - Automated restore testing script
- `configs/pgbackrest.cron` - Backup cron schedule
- `configs/pgbackrest-restore-test.cron` - Restore test cron schedule
- `configs/fstab.backups` - NFS mount configuration

### Key Files and Locations on Server

- **pgBackRest config:** `/etc/pgbackrest.conf`
- **PostgreSQL WAL config:** `/etc/postgresql/18/main/conf.d/pgbackrest.conf`
- **Lock/spool directory:** `/var/spool/pgbackrest` (CRITICAL - must persist)
- **Backup scripts:** `/usr/local/bin/pgbackrest-scripts/`
- **Restore test:** `/usr/local/bin/pgbackrest_restore_test.sh`
- **Cron jobs:** `/etc/cron.d/pgbackrest`, `/etc/cron.d/pgbackrest-restore-test`
- **Backup storage:** `/mnt/backups/pgbackrest/`
- **Logs:** `/var/log/pgbackrest/`

### Quick Reference Commands
```bash
# Check backup status
sudo -u postgres pgbackrest --stanza=main-db info

# Manual backup
sudo -u postgres pgbackrest --stanza=main-db --type=full backup

# Run restore test
sudo /usr/local/bin/pgbackrest_restore_test.sh

# View logs
tail -f /var/log/pgbackrest/main-db-backup.log

# Test WAL archiving
sudo tail -50 /var/log/postgresql/postgresql-18-main.log | grep archive
```

---

**End of Setup Guide**

*Last Updated: 2026-03-19 09:57:56*
*Configuration Files: https://github.com/patrickjmcd/k8s-argo/tree/main/infra/postgres/backup*
