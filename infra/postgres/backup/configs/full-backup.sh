#!/bin/bash
WEBHOOK_URL="http://YOUR_HA_IP:8123/api/webhook/pgbackrest_backup_status"

if /usr/bin/pgbackrest --stanza=main-db --type=full backup; then
    curl -s -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{"status":"success","message":"Full backup completed successfully","type":"full"}'
else
    curl -s -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{"status":"error","message":"Full backup FAILED","type":"full"}'
    exit 1
fi
