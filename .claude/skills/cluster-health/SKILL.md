---
name: cluster-health
description: Run a read-only health check of the homelab k3s cluster (nodes, memory/swap, pods, ArgoCD, Longhorn volumes and backups, alerts and flapping, control-plane memory, certs). Use when the user asks to check on the cluster, asks why alerts keep firing, or runs /cluster-health.
argument-hint: "[focus area, e.g. alerts | longhorn | control-plane]"
allowed-tools: Bash(kubectl get:*), Bash(kubectl top:*), Bash(kubectl describe:*), Bash(kubectl logs:*), Bash(.claude/skills/cluster-health/promq.sh:*), Bash(jq:*)
---

# Cluster health check

Read-only sweep of the cluster. **Don't change anything during the check.** Report findings, rank them, and ask before fixing anything. If `$ARGUMENTS` names a focus area, do that section in depth and the rest briefly.

PromQL helper (goes through the apiserver proxy, no port-forward needed):

```bash
P=.claude/skills/cluster-health/promq.sh
$P '<promql>' | jq -r '.data.result[] | "\(.metric) \(.value[1])"'
```

Zsh globs `?` and `[`, so always pass PromQL in single quotes through `$P` rather than inline in a `kubectl get --raw` URL.

Run independent checks in parallel. Skip a section only if its tooling is absent, and say so.

## 1. Nodes

- `kubectl get nodes` — everything should be Ready. Note any `SchedulingDisabled`.
- `kubectl top nodes`.
- Memory headroom and swapping. Swap is enabled on the x86 control-plane nodes, so they degrade silently before they OOM:
  - `100*node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes` (flag <15%)
  - `node_memory_SwapTotal_bytes-node_memory_SwapFree_bytes` (flag >1Gi)
  - `rate(node_vmstat_pgmajfault[30m])` (flag >500/s; that's thrashing)

## 2. Control plane

kube-n3160 and kube-macmini have only ~7.6G of RAM and also run etcd.

- `kubectl get --raw='/readyz?verbose' | grep -E '\[-\]|etcd'` — expect `etcd ok`.
- `max by (instance) (process_resident_memory_bytes{job="apiserver"})` and `go_memstats_heap_inuse_bytes{job="apiserver"}`. Compare against the same query with ` offset 3d`. Steady growth means a leak.
- `sum by (instance, resource, subresource, verb) (apiserver_longrunning_requests)`, top 10. A `pods/log` CONNECT count above ~1000 means a log shipper or viewer is fanning out cluster-wide. Oct 2026 incident: Alloy tailed every pod from every node; fixed in `core/alloy.yaml` with a node-local selector. dozzle (`DOZZLE_MODE=k8s`) can do the same.
- `max by (instance) (process_start_time_seconds{job="apiserver"})` gives the apiserver uptime per node.

## 3. Workloads

- Pods not Running/Completed: `kubectl get pods -A --no-headers | awk '$4!="Running" && $4!="Completed"'`.
- Recent restarts: `topk(15, increase(kube_pod_container_status_restarts_total[24h]) > 0)`. Lifetime restart counts are misleading here, so ignore them.
- `kubectl get pvc -A` not Bound. `kubectl get certificates -A` not Ready.
- ArgoCD: `kubectl get applications -n argocd --no-headers | awk '$2!="Synced" || $3!="Healthy"'`. For each one, list its non-Synced resources with `.status.resources[] | select(.status!="Synced")`.

## 4. Longhorn

- Volume health: `kubectl get volumes.longhorn.io -n longhorn-system -o json | jq -r '.items[]|"\(.status.state)/\(.status.robustness)"' | sort | uniq -c`. Detached/unknown is normal. Name any attached volume that isn't healthy.
- `kubectl get nodes.longhorn.io -n longhorn-system`.
- Setting drift: `kubectl get settings.longhorn.io -n longhorn-system -o json | jq -r '.items[]|select(.status.applied==false)|.metadata.name'`. A changed `taint-toleration` never applies while volumes are attached.
- **Backups.** Every PVC labelled `recurring-job-group.longhorn.io/b2-backup: enabled` must also carry `recurring-job.longhorn.io/source: enabled`, or the label never reaches the volume. For each such PVC:
  - Check the bound volume has the b2-backup label.
  - Check its `.status.lastBackupAt` is within the last 36h.
  - Flag missing labels and stale or empty backups.
- Any `backups.longhorn.io` not in `Completed` state.

## 5. Alerts

Alerts reach the user via `core/prometheus-assets/alertmanager-discord-config.yaml` → n8n → Discord. Everything except `Watchdog|InfoInhibitor|KubeSchedulerDown|KubeProxyDown|KubeControllerManagerDown` and `severity=info` pages them, with `groupInterval: 5m`. A flapping alert therefore pages roughly every 5 minutes.

- Currently paging: `ALERTS{alertstate="firing",severity!~"none|info",alertname!~"Watchdog|InfoInhibitor|KubeSchedulerDown|KubeProxyDown|KubeControllerManagerDown"}`. The three `*Down` alerts are always firing on k3s, which runs those components in-process, so ignore them.
- Flapping over the last 24h: `sort_desc(sum by (alertname, severity) (changes(ALERTS_FOR_STATE[24h]) + 1))`. Then break down the top 3 by `instance`/`namespace`/`pod`. Find the shared root cause; several alerts usually trace back to one starved node.

## 6. Report

Lead with a one-line verdict: healthy, degraded, or needs action now.

Then:

- **Needs action**: ranked by impact. For each one: the evidence (numbers), the likely cause, and the fix you'd propose.
- **Watch**: trending items that aren't urgent.
- **Healthy**: one short line per section.

Keep it scannable. Don't dump raw command output.

Proposing fixes:
- Prefer GitOps changes, then a push to `main` (allowed in this repo).
- For host-level actions, such as restarting k3s, SSH with `SSH_AUTH_SOCK=/run/user/1000/ssh-agent.sock` (gnome-keyring refuses the passphrase-protected `swarm` key).
- Restart control-plane nodes one at a time. Confirm etcd `ok` between them.
