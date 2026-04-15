# Ansible

Manages bare-metal server configuration outside the Kubernetes cluster.

## Inventory

| Host  | IP             | Role            |
|-------|----------------|-----------------|
| db01  | 192.168.1.11   | PostgreSQL 18   |

## Roles

### `postgres`
Manages PostgreSQL configuration files:
- `postgresql.conf` — main server config
- `conf.d/pgbackrest.conf` — WAL archiving settings (triggers restart on change)

### `pgbackrest`
Manages pgBackRest backup tooling:
- `/etc/pgbackrest.conf` — stanza and repository config
- `/usr/local/bin/pgbackrest-scripts/full-backup.sh` — weekly full backup
- `/usr/local/bin/pgbackrest-scripts/diff-backup.sh` — daily differential backup
- `/usr/local/bin/pgbackrest_restore_test.sh` — weekly restore verification
- `/etc/cron.d/pgbackrest` — backup and restore test schedule

## Usage

```bash
cd ansible

# Dry run
ansible-playbook playbooks/postgres.yml --check --diff

# Apply
ansible-playbook playbooks/postgres.yml
```

## Key Variables

All tunable values are in `inventory/group_vars/db_servers.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `pg_version` | `18` | PostgreSQL major version |
| `pg_port` | `5432` | PostgreSQL port |
| `pgbackrest_stanza` | `main-db` | pgBackRest stanza name |
| `pgbackrest_repo_path` | `/mnt/backups/pgbackrest` | Backup repository (NFS mount) |
| `pgbackrest_retention_full` | `4` | Full backups to retain |
| `pgbackrest_retention_diff` | `4` | Differential backups to retain |
| `pgbackrest_full_backup_schedule` | `0 2 * * 0` | Full backup cron (Sun 2 AM) |
| `pgbackrest_diff_backup_schedule` | `0 2 * * 1-6` | Diff backup cron (Mon–Sat 2 AM) |
| `pgbackrest_restore_test_schedule` | `0 8 * * 6` | Restore test cron (Sat 8 AM) |
| `ha_backup_webhook` | — | Home Assistant webhook for backup notifications |
| `ha_restore_test_webhook` | — | Home Assistant webhook for restore test notifications |

## Backup Strategy

- **Full backup** weekly (Sunday 2 AM), retained for 4 weeks
- **Differential backup** daily (Mon–Sat 2 AM), retained for 4 cycles
- **WAL archiving** continuous to `/mnt/backups/pgbackrest` (NFS mount to UniFi NAS)
- **Restore test** weekly (Saturday 8 AM) — restores latest backup to a temporary instance on port 5433, runs verification queries, reports to Home Assistant
