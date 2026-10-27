# k8s-argo GitOps Repository

Homelab Kubernetes cluster managed with k3s and **Flux** (Argo CD until Oct 27
2026; the repo name is historical). All changes are GitOps — push to `main` and
Flux reconciles within about a minute. Migration history and per-component
notes: `docs/flux-migration.md`.

## Where the repo lives (since Oct 2026)

- **Primary: Gitea in the cluster** (`apps/gitea`), `https://git.x.pmcd.io/patrickjmcd/k8s-argo`
  (git remote `origin`; push with a Gitea token over HTTPS, or over SSH as
  `git@git-ssh.x.pmcd.io:patrickjmcd/<repo>.git` -- MetalLB `.210`, LAN/tailnet only).
- **Flux reads** `http://gitea.gitea.svc.cluster.local:3000/patrickjmcd/k8s-argo.git`
  (in-cluster, anonymous -- the repo is public) -- not via Traefik, so a broken Traefik can't
  stop Flux from fixing it. The URL is set once, in `clusters/home/flux-instance.yaml`
  (`spec.sync`); the Flux Operator creates the root GitRepository/Kustomization `flux-system`
  from it, and every other Kustomization uses that source.
- **GitHub is a push mirror** (`github` remote): Gitea force-pushes every commit there. Never
  commit or merge on GitHub -- it gets overwritten.
- **If Gitea (or the cluster) is down:** point Flux at GitHub, in git *and* live (otherwise the
  first reconcile from GitHub re-applies the Gitea URL):
  ```bash
  sed -i 's#http://gitea.gitea.svc.cluster.local:3000/patrickjmcd/k8s-argo.git#https://github.com/patrickjmcd/k8s-argo.git#' clusters/home/flux-instance.yaml
  git commit -am "flux: sync from GitHub while Gitea is down" && git push github main
  kubectl -n flux-system patch fluxinstance flux --type merge \
    -p '{"spec":{"sync":{"url":"https://github.com/patrickjmcd/k8s-argo.git"}}}'
  ```
  Gitea reconciles itself from this repo (`clusters/home/apps/gitea.yaml`). Its data restores
  from B2 (Longhorn volume + shared-postgres `gitea` DB).
- Flux web UI: https://flux.x.pmcd.io (PocketID SSO). Flux errors reach ntfy via
  Alertmanager (`clusters/home/notifications.yaml`).

## Directory Structure

```
clusters/home/  # What Flux applies: FluxInstance, Kustomizations, HelmReleases
  apps/         #   one file per app (Kustomization and/or HelmRelease)
  core/         #   one file per core component; helm-repositories.yaml
  cronjobs.yaml #   Kustomization for cronjobs/
apps/           # App manifests (HomelabApp instances, plain kustomize)
core/           # Core component manifests and chart extras
cronjobs/       # Scheduled jobs (namespace cron)
charts/         # homelab-app chart (retired, unused)
infra/          # Terraform for Proxmox VMs
scripts/        # Utility scripts
docs/           # Runbooks (flux-migration.md, cilium-migration.md, ...)
```

## How Flux is wired

- `clusters/home/kustomization.yaml` lists everything Flux manages; a file in
  `clusters/home/apps/` or `clusters/home/core/` does nothing until it is
  listed in that directory's `kustomization.yaml`.
- Each directory's `kustomization.yaml` patches shared settings onto every
  Flux Kustomization in it (interval 10m, retryInterval 2m, timeout 5m,
  `sourceRef: flux-system`, `prune: true`, `wait: true`; apps also get a
  HomelabApp health check). So an app's file only needs `name` + `path`
  (+ `dependsOn`).
- Ordering is `dependsOn`, not sync waves. Every Kustomization containing a
  HomelabApp needs `dependsOn: [{name: kro-definitions}]`, set per file
  (patching it in the shared settings would replace other `dependsOn` lists).
- The media apps are one ResourceSet (`clusters/home/apps/media.yaml`): add
  the directory name under `apps/media/` to its `inputs`.
- Upstream charts are HelmReleases in `clusters/home/{apps,core}/<name>.yaml`,
  with their HelmRepository in `clusters/home/core/helm-repositories.yaml`
  (or next to the release). Extra manifests for a chart go in a second
  Kustomization `<name>-extras`.
- Former Argo hook Jobs are plain Jobs in `core/<x>/jobs/` applied by a
  `<x>-jobs` Kustomization (`dependsOn` + `force: true`); no TTL (Flux would
  recreate a TTL-deleted Job), and a hashed script ConfigMap so edits re-run it.
- Force a reconcile: `flux reconcile kustomization <name> --with-source`
  (or `flux reconcile helmrelease <name>`). Pause one: `flux suspend
  kustomization <name>` (`flux resume` to undo).
- Renovate can bump a chart version before Flux's cached HelmRepository index
  (12h interval) has it ("no chart with version matching"):
  `flux reconcile source helm <repo>`.

## Defining Applications

### HomelabApp (kro) — preferred for single-container apps

`core/kro-definitions/homelabapp.yaml` is a kro ResourceGraphDefinition that
creates a `HomelabApp` CRD. Each instance gets a Deployment, Service, HTTPRoutes
(internal + optional public), OnePasswordItem, ConfigMap and ServiceMonitor.
The schema at the top of the RGD is the reference for every field.

```yaml
# clusters/home/apps/myapp.yaml (and list it in clusters/home/apps/kustomization.yaml)
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: myapp
  namespace: flux-system
spec:
  path: ./apps/myapp
  dependsOn:
    - name: kro-definitions
```

```yaml
# apps/myapp/homelabapp.yaml (listed in apps/myapp/kustomization.yaml)
apiVersion: homelab.pmcd.io/v1alpha1
kind: HomelabApp
metadata:
  name: myapp
  namespace: default
spec:
  image: ghcr.io/example/app:v1.2.3
  port: 80
  containerPort: 8080
  env: {TZ: America/Chicago}
  onePassword: {enabled: true}          # <name>-1pw from vaults/Kubernetes/items/<name>
  probes:
    type: httpGet                       # one handler for startup/readiness/liveness
    path: /health
    liveness: {enabled: true}
  route:
    hostnames: [myapp.x.pmcd.io]
  persistence:
    claimName: longhorn-myapp-data      # existing PVC, defined next to it
    mountPath: /data
  glance: {monitor: true}               # Glance widget + blackbox probe (scripts/generate_glance_monitors.py)
```

Flux never creates namespaces: an app in its own namespace needs an
`apps/<name>/namespace.yaml` with `kustomize.toolkit.fluxcd.io/prune: disabled`
(deleting a Namespace deletes everything in it).

Rules:
- **Persistence mounts a PVC you define yourself** in `apps/<name>/pvc.yaml`
  (and `pv.yaml` for SMB). Give it backup labels and
  `kustomize.toolkit.fluxcd.io/prune: disabled`, so removing the manifest
  (or the app's Kustomization) never deletes the data. kro never owns the PVC:
  it deletes `includeWhen` resources when the condition turns false, which
  would lose data if persistence were toggled off.
- HomelabApp covers extraVolumes/extraVolumeMounts (PVC, ConfigMap, Secret,
  hostPath, emptyDir), per-probe type/path overrides, command/args,
  nodeSelector/tolerations/hostNetwork, podAnnotations, dnsNdots, a basic
  securityContext, route middlewares and ServiceMonitor basic auth. Extra
  manifests go in the app's kustomization as plain files.
  `loadBalancer.enabled` adds a `<name>-lan` LoadBalancer Service
  (optionally pinned `loadBalancer.ip`) next to the ClusterIP one, and
  `preferredNodes` sets a soft node preference. Not covered: sidecars and
  initContainers (use plain Kustomize).
- kro owns the Deployment's replicas, so `kubectl scale` is undone within
  a minute. To stop an app (e.g. for a restore), set `replicas: 0` on the
  HomelabApp (in git, or patched live after `flux suspend kustomization <name>`).
- Every HomelabApp Deployment carries `reloader.stakater.com/auto: "true"`;
  Reloader (`clusters/home/core/reloader.yaml`, Oct 2026) restarts it when its
  `<name>-1pw` Secret or ConfigMap changes, so editing a 1Password field
  reaches the pod once the operator syncs (force it with the `force-sync`
  annotation below).
- `${VAR}` inside instance values (configs, commands) passes through kro
  literally; only `${` written in the RGD itself is evaluated.
- When converting a chart app, merge its values over the chart defaults first
  (Helm does), and use the Helm **release name** as the HomelabApp name.
- Children carry ownerReferences to their HomelabApp; the apps Kustomizations
  report HomelabApp health through kro's Ready condition (`healthCheckExprs`
  in `clusters/home/apps/kustomization.yaml`).
- Editing the RGD: build the container as one CEL expression and use the
  optional map-entry syntax (`?"key": cond ? optional.of(v) : optional.none()`),
  sort map keys before turning them into lists, and validate as a throwaway
  RGD (different name/kind/group) before touching the real one. Comments at
  the top of the RGD explain why. Never edit it with `yq -i` (it reflows the
  CEL template); use an editor or sed.

### Upstream Helm chart (HelmRelease)

For apps an upstream chart covers (immich, nats, every core chart). Copy the
shape of an existing one, e.g. `clusters/home/apps/immich.yaml`: HelmRelease in
`flux-system` with `targetNamespace`/`storageNamespace`, `serverSideApply`,
`driftDetection.mode: enabled`, values inline. Keep `remediation.retries: 0`
with `remediateLastFailure: false` on releases adopted from live objects:
remediating a failed install means uninstalling what was running. Check the
chart for CRDs shipped as templates and keep them
(`helm.sh/resource-policy: keep` via values or a `postRenderers` patch).
Helm 4 (helm-controller) keeps `null` values that Helm 3 dropped, and a local
`helm template` is not proof of what helm-controller renders.

### Kustomize-only app (initContainers, multi-container pods, CronJobs, StatefulSets)

A Flux Kustomization with `path: ./apps/myapp` over plain manifests. Copy the
live `kubectl.kubernetes.io/restartedAt` of an existing workload into git
before adopting it, or Flux strips it and the pods roll.

## homelab-app Chart (retired)

`charts/homelab-app/` is no longer used by any app (all were converted to
HomelabApp or upstream HelmReleases in Oct 2026); kept for reference only.

## Secrets — 1Password Operator

All secrets managed via `OnePasswordItem` CRD. The operator creates a Kubernetes `Secret` where each field in the 1Password item becomes a secret key.

```yaml
apiVersion: onepassword.com/v1
kind: OnePasswordItem
metadata:
  name: myapp-1pw
  namespace: myapp
spec:
  itemPath: "vaults/Kubernetes/items/myapp"
```

**Important**: 1Password field names become Kubernetes secret keys exactly. The Connect server has a cache — force re-sync with:
```
kubectl annotate onepassworditem <name> -n <ns> force-sync=$(date +%s) --overwrite
```

If a field consistently syncs as empty, restart the Connect server:
```
kubectl rollout restart deployment onepassword-connect -n default
```

## Networking — Gateway API

Use HTTPRoute, not Ingress. Gateway is `traefik-gateway` in `default` namespace.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: myapp
  namespace: myapp
spec:
  parentRefs:
    - name: traefik-gateway
      namespace: default
  hostnames:
    - myapp.example.com
  rules:
    - backendRefs:
        - name: myapp
          port: 8080
```

Cert-manager issues certs via Let's Encrypt DNS-01 (Cloudflare). Certs live in `default` namespace and are referenced by the Gateway.

## Storage

### Longhorn (block storage)

Default storage class. Used for databases and stateful apps. PVC name convention: `longhorn-<release>-data`.

### SMB (network storage)

For media and shared files. CSI driver in `csi-smb-provisioner` namespace. PV name: `smb-<release>`, PVC: `smb-<release>-claim`. Credentials in `csi-smb-provisioner` namespace from 1Password.

### PostgreSQL (CNPG)

3-node CloudNativePG cluster in `postgres` namespace. Connect via `shared-postgres-rw.postgres.svc.cluster.local`. Databases and roles managed as CNPG `Database`/`Role` CRDs in `core/postgres/`.

## Kyverno Policies

Kyverno mutates pods automatically:
- **node-failure-tolerations**: Adds 30s tolerations for `node.kubernetes.io/not-ready` and `node.kubernetes.io/unreachable` to all pods
- **pi5-preference**: Scheduling preference for Pi 5 nodes
- **goldilocks-namespace-label**: Auto-labels namespaces for VPA recommendations

Audit-only (never blocks admission):
- **cpu-limit-guard**: Flags containers with a CPU limit under 100m. See "Resource requests and limits" below. Findings: `kubectl get polr -A | grep cpu-limit-guard`

Kyverno's webhooks skip `kube-system`, `kyverno` and `flux-system` (`config.webhooks` in
`clusters/home/core/kyverno.yaml`), so a broken Kyverno can't stop Flux from recreating its
own pods; the mutations above don't apply there.

If Kyverno webhook gets stuck (failurePolicy=Fail + context deadline exceeded), temporarily patch to `Ignore`, fix root cause, then Kyverno restores `Fail`.

## Resource requests and limits

**Goldilocks CPU numbers go in `requests`, never `limits`.** Its figures are VPA
recommendations, which describe requests; the dashboard's "Guaranteed" column just
mirrors the target into limits too. Applying them as limits forms a ratchet — the
limit caps observed usage, the recommender then recommends the cap, and it gets
reapplied as the limit — so the recommendation collapses to a point estimate and
can never climb back.

A CPU limit is a quota per 100ms scheduling period: 100m = 10ms, 15m = 1.5ms. A
bursty process needs a few contiguous ms per event, exhausts a small quota
immediately, and stalls until the next period — while its *average* usage still
reads well under the limit. **Low average utilisation is not evidence that a CPU
limit is safe.**

This caused a real outage-adjacent incident in July 2026: flannel at 49m throttled
53–94% on every node while averaging 5–29m, and all three cert-manager components
at 15m throttled 18–35% while averaging 1–3m — both on critical paths (pod
networking, API admission). By then cert-manager's recommendation had collapsed to
`lowerBound == target == upperBound == 15m`, while argocd-repo-server under a roomy
500m limit still showed a healthy 15m/35m/49m spread. Once collapsed, the
dashboard's "Burstable" column is no help either — it reads lowerBound/upperBound,
by then the same number.

Rules of thumb:
- Latency-sensitive or critical-path components (CNI, webhooks, CSI): **no CPU
  limit**. Eviction protection comes from `priorityClassName`, not Guaranteed QoS.
- Everything else: a CPU limit only as a deliberate runaway guard, sized well above
  observed peak — not derived from average usage.
- Memory is incompressible and not period-scheduled, so memory limits stay correct.
  Give them real headroom; don't pin them a few Mi above requests, which is one
  spike away from an OOMKill.

## Synthetic monitoring (blackbox exporter)

`core/blackbox-exporter` (replaced uptime-kuma, Oct 2026) probes every app with
`glance.monitor` set -- `scripts/generate_glance_monitors.py` writes
`probes-generated.yaml` alongside the Glance widget list (the pre-commit hook
in `.githooks` reruns it; enable per clone with `git config core.hooksPath .githooks`) -- plus hand-written targets in `probes-static.yaml`
(Home Assistant, PocketID, LAN devices over TCP). Module `http_up` counts
redirects and 401/403 as up. Alerts: `ProbeFailed` (5m) and
`ProbeTlsCertExpiringSoon` (<14d) in `rules.yaml`. It runs in-cluster, so it
can't report a whole-cluster outage.

## MetalLB

IP pool: `192.168.8.200–192.168.8.210`. Traefik LoadBalancer gets `.200`. L2 advertisement mode.

## Hardware

- **Pi 5 nodes**: `pi5-kube0`, `pi5-kube1`, `pi5-kube2` — general workloads
- **x86 VMs**: `kube-leader`, `kube-worker-{1-4}` — provisioned via Terraform on Proxmox

The cluster has no GPU nodes. The Jetson Nano (`jetson-nano-kube0`) was retired — it could not run an eBPF CNI datapath or attach Longhorn volumes on its L4T/Tegra kernel, and its GPU workloads (piper TTS, openwakeword) were dropped along with the NVIDIA device plugin and runtime config.

## Node Bootstrap Requirements (all nodes)

**CNI** — the cluster runs **Cilium** (Flux HelmRelease in `clusters/home/core/cilium.yaml`) as its only CNI; flannel was removed in Sept 2026 (`docs/cilium-migration.md`). k3s still runs with `--flannel-backend=none` — required, or k3s starts its built-in flannel alongside Cilium. Cilium's `install-cni-binaries` init container only drops in `cilium-cni` and writes `/etc/cni/net.d/05-cilium.conflist`, so k3s does NOT provide the standard CNI plugins either: every new node needs the base plugins (`loopback`, `portmap`, …) in `/opt/cni/bin` before pods can start. Symptom when missing: `FailedCreatePodSandBox … failed to find plugin "loopback" in path [/opt/cni/bin]`.

```bash
sudo apt-get install -y containernetworking-plugins
sudo mkdir -p /opt/cni/bin
sudo cp /usr/lib/cni/* /opt/cni/bin/
```

Also install the storage clients so CSI mounts work: `nfs-common` and `cifs-utils`.

**No flannel leftovers** — a node that ever ran flannel keeps its host state until cleaned or rebooted: a `cni0` bridge holding the `.1` of the node's pod CIDR, `flannel.1`, `/etc/cni/net.d/10-flannel.conflist`, `/run/flannel` and `FLANNEL-*` iptables chains. The stale `cni0` is not cosmetic: Cilium (`ipam.mode: kubernetes`) hands out the same `.1`, and a pod that gets it can't receive replies because the host owns that address — seen Sept 2026 as a Traefik pod on `pi5-kube2` crashlooping on "timed out waiting for controller caches to sync". Clean-up steps are in `docs/cilium-migration.md` §6.5.

**inotify instance limit** — `fs.inotify.max_user_instances` is a per-UID limit shared by every root-owned container on the node (not namespaced per-pod), and most system/sidecar containers run as root. A node hosting many containers with fsnotify-based watchers (config reloaders, log tailers, etc.) can silently approach the default cap of 128 as container density grows, at which point the next container to request an inotify instance crash-loops with `OSError: [Errno 24] inotify instance limit reached` — a red herring that looks like an app bug. Confirmed on `pi5-kube0/1/2` in Sept 2026 (91, 72, and 127 of 128 respectively) when the `youtubedl` postprocessor's `watchdog` Observer tipped `pi5-kube2` over the edge. It recurred in Sept 2026 via Grafana Alloy (`clusters/home/core/alloy.yaml`, tails every pod's log file over inotify) exhausting the limit and breaking Argo CD's CronJob log stream (Argo was still in use then).

As of Sept 2026 this is enforced **declaratively**, not just at bootstrap: the `node-tuning` DaemonSet (`core/node-tuning/daemonset.yaml`, applied by the Flux Kustomization in `clusters/home/core/node-tuning.yaml`) runs on every node — including control-plane/tainted ones — and sets both `fs.inotify.max_user_instances=1024` and `fs.inotify.max_user_watches=1048576` via a privileged initContainer, writing the same values into `/etc/sysctl.d/99-inotify.conf` on the host so they survive a reboot before the pod is rescheduled. It self-heals: Flux re-applies it every 10 minutes and the DaemonSet covers every node, so drift (or a newly joined node) gets corrected automatically. `fs.inotify.*` are node-level (non-namespaced) sysctls, so they can't be set via a pod's `securityContext.sysctls` — hence the privileged container, scoped to only that DaemonSet.

The manual commands below remain the fallback for a node that hasn't joined the cluster yet (pre-bootstrap), or for ad-hoc verification:

```bash
echo 'fs.inotify.max_user_instances=1024' | sudo tee /etc/sysctl.d/99-inotify.conf
sudo sysctl -p /etc/sysctl.d/99-inotify.conf
```

Verify with: `sudo sysctl fs.inotify.max_user_instances` → should be `1024`. Check current usage per node with:
```bash
sudo sh -c 'for f in /proc/[0-9]*/fd; do pid=${f%/fd}; pid=${pid#/proc/}; n=$(ls -la "$f" 2>/dev/null | grep -c inotify); [ "$n" -gt 0 ] && stat -c %u /proc/$pid; done' | sort | uniq -c
```

## Control-plane node memory (k3s GOMEMLIMIT)

kube-n3160 and kube-macmini have only ~7.6 GB RAM, and `k3s-server` (apiserver
watch cache of every object + embedded etcd) settles around 4.5 GiB within a
day -- enough to push them into swap (Oct 2026). Both run k3s with a Go soft
memory limit, set **on the host, not in git**:

```bash
# /etc/default/k3s  (k3s.service reads it; the installer doesn't touch it)
GOMEMLIMIT=4GiB
```

Apply with `sudo systemctl restart k3s`, one control-plane node at a time,
checking `kubectl get --raw='/readyz?verbose' | grep etcd` between them. Verify
with `sudo cat /proc/$(pidof k3s-server)/environ | tr '\0' '\n' | grep GO`.
A rebuilt control-plane node needs this re-added. The limit is soft: if live
heap ever exceeds it Go spends more CPU on GC rather than OOMing, so watch
apiserver CPU (`process_cpu_seconds_total{job="apiserver"}`) after changes.
Keep stored data small too -- Trivy SBOM reports (214 MiB) were disabled for
the same reason.

## Control-plane timezone

All three control-plane nodes run in `America/Chicago` (`timedatectl
set-timezone`, then restart k3s). A CronJob without `spec.timeZone` follows the
host timezone of whichever node holds the controller-manager lease, so mixed
timezones shift schedules on every leader change. On Oct 13 2026 kube-leader-2
was still UTC: the new leader backfilled "missed" runs, and the Postgres backup
and restore test fired at the same second. Set `timeZone: America/Chicago` on
new CronJobs anyway; Longhorn's recurring-job CronJobs can't be pinned, so a
rebuilt control-plane node must get the timezone too.

## Node Bootstrap Requirements (RPi nodes)

When adding a new Raspberry Pi node (Pi 4 or Pi 5, Debian trixie, kernel 6.12.x), apply this before joining the cluster:

**Conntrack checksum fix** — RPi 6.12.x kernels have TX checksum offload enabled on veth/cni interfaces. With `nf_conntrack_checksum=1` (default), the kernel re-validates checksums on forwarded packets and marks them invalid, causing conntrack `clash_resolve` storms that poison DNS reply tracking and silently drop UDP responses.

```bash
echo 'net.netfilter.nf_conntrack_checksum=0' | sudo tee /etc/sysctl.d/99-conntrack.conf
sudo sysctl -w net.netfilter.nf_conntrack_checksum=0
```

Verify with: `sudo sysctl net.netfilter.nf_conntrack_checksum` → should be `0`.

## Conventions

- Every Flux Kustomization/HelmRelease applies server-side; HelmReleases use
  `serverSideApply` on install and upgrade
- Guard anything whose deletion loses data or takes others with it
  (Namespaces, PVCs, PVs, CRDs, CNPG clusters, Longhorn settings) with
  `kustomize.toolkit.fluxcd.io/prune: disabled`
- Prefer `prune: true` (the default via the shared patches) and drift detection
- Never use `latest` image tags
- For a new chart, add its HelmRepository to `clusters/home/core/helm-repositories.yaml`
  (Renovate's `flux` manager updates chart versions under `clusters/`)
