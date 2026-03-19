#!/bin/bash
WEBHOOK_URL="http://YOUR_HA_IP:8123/api/webhook/pgbackrest_backup_status"

if /usr/bin/pgbackrest --stanza=main-db --type=diff backup; then
    curl -s -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{"status":"success","message":"Differential backup completed successfully","type":"diff"}'
else
    curl -s -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{"status":"error","message":"Differential backup FAILED","type":"diff"}'
    exit 1
fi
