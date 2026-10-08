# k8s-argo GitOps Repository

Homelab Kubernetes cluster managed with k3s and ArgoCD. All changes are GitOps — push to `main` and ArgoCD reconciles automatically.

## Directory Structure

```
argo/           # ArgoCD app-of-apps and AppProjects
apps/           # User applications
charts/         # Local Helm charts (homelab-app)
core/           # Core infrastructure components
cronjobs/       # Scheduled jobs
infra/          # Terraform for Proxmox VMs
scripts/        # Utility scripts
```

## App-of-Apps Pattern

Three root Applications in `argo/applications/` each point to a directory:
- `core.yaml` → `core/` — infrastructure (cert-manager, traefik, metallb, longhorn, etc.)
- `apps.yaml` → `apps/` — user applications
- `cronjobs.yaml` → `cronjobs/` — scheduled jobs (sync disabled, manual only)

Any `.yaml` file dropped into `core/` or `apps/` is automatically picked up as an ArgoCD Application.

## ArgoCD Projects

- **core-components** — infrastructure apps (`core/`)
- **apps** — user applications (`apps/`)
- **cronjobs** — scheduled jobs

## Sync Waves

Order matters; set via `argocd.argoproj.io/sync-wave` annotation:
- `-2`: ArgoCD self
- `-1`: Longhorn, cert-manager, metallb, kyverno, CSI drivers
- `0`: Traefik, monitoring
- (default): Applications

## Defining Applications

### HomelabApp (kro) — preferred for single-container apps

`core/kro-definitions/homelabapp.yaml` is a kro ResourceGraphDefinition that
creates a `HomelabApp` CRD. Each instance gets a Deployment, Service, HTTPRoutes
(internal + optional public), OnePasswordItem, ConfigMap and ServiceMonitor. It
is replacing the homelab-app chart (migration started Oct 2026); the schema
at the top of the RGD is the reference for every field.

```yaml
# apps/myapp.yaml — single-source kustomize Application
spec:
  syncPolicy:
    automated: {prune: true, selfHeal: true}
    syncOptions:
      - ServerSideApply=true
      - SkipDryRunOnMissingResource=true   # CRD comes from kro (fresh cluster)
  source:
    path: apps/myapp
    repoURL: https://github.com/patrickjmcd/k8s-argo.git
    targetRevision: main
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
  glance: {monitor: true}               # scripts/generate_glance_monitors.py
```

Rules:
- **Persistence mounts a PVC you define yourself** in `apps/<name>/pvc.yaml`
  (and `pv.yaml` for SMB). Give it backup labels and
  `argocd.argoproj.io/sync-options: Delete=false` -- which only keeps the PVC
  when the *Application* is deleted; removing the manifest from git still
  prunes it (seen Oct 2026), so for that use `Prune=false` or move the data
  first. kro never owns the PVC:
  it deletes `includeWhen` resources when the condition turns false, which
  would lose data if persistence were toggled off.
- HomelabApp covers extraVolumes/extraVolumeMounts (PVC, ConfigMap, Secret,
  hostPath, emptyDir), per-probe type/path overrides, command/args,
  nodeSelector/tolerations/hostNetwork, podAnnotations, dnsNdots, a basic
  securityContext, route middlewares and ServiceMonitor basic auth. Extra
  manifests (the chart's extraObjects) go in the app's kustomization as plain
  files. `loadBalancer.enabled` adds a `<name>-lan` LoadBalancer Service
  (optionally pinned `loadBalancer.ip`) next to the ClusterIP one, and
  `preferredNodes` sets a soft node preference. Not covered: sidecars and
  initContainers (use plain Kustomize).
- kro owns the Deployment's replicas, so `kubectl scale` is undone within
  a minute. To stop an app (e.g. for a restore), set `replicas: 0` on the
  HomelabApp (in git, or patched with the Application's auto-sync off).
- HomelabApp Applications set
  `argocd.argoproj.io/compare-options: ServerSideDiff=true`: CRD defaults
  filled into list items (extraVolumes, tolerations) otherwise read as drift.
- `${VAR}` inside instance values (configs, commands) passes through kro
  literally; only `${` written in the RGD itself is evaluated.
- When converting a chart app, merge its values over the chart defaults first
  (Helm does), and use the Helm **release name** as the HomelabApp name (it
  can differ from the Application name, e.g. homelable-backend -> backend).
- Children carry ownerReferences to their HomelabApp, so Argo CD shows them in
  the app's tree; a Lua health check in `core/argocd.yaml` maps kro's Ready
  condition to Argo health.
- Editing the RGD: build the container as one CEL expression and use the
  optional map-entry syntax (`?"key": cond ? optional.of(v) : optional.none()`),
  sort map keys before turning them into lists, and validate as a throwaway
  RGD (different name/kind/group) before touching the real one. Comments at
  the top of the RGD explain why.
- After an RGD **schema** change (new defaulted fields), some HomelabApp apps
  go permanently OutOfSync in Argo with no real diff: Argo caches server-side
  diff results in Redis keyed by the live resourceVersion, so objects that
  haven't changed keep a prediction made under the old schema (restarting
  the controller doesn't help). Bump every instance's resourceVersion:
  `kubectl annotate homelabapp -A --all homelab.pmcd.io/rgd-touch=$(date +%s) --overwrite`

### Helm app using homelab-app chart (only for what HomelabApp can't express)

```yaml
# apps/myapp.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
  annotations:
    # Multi-source: list the chart and the values dir explicitly. "." only
    # covers the chart source, so a values-only commit would be served from
    # Argo's manifest cache and never deployed.
    argocd.argoproj.io/manifest-generate-paths: /charts/homelab-app;/apps/myapp
spec:
  destination:
    namespace: myapp
    name: in-cluster
  project: apps
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true
  sources:
    - repoURL: https://github.com/patrickjmcd/k8s-argo.git
      targetRevision: main
      ref: values
    - chart: homelab-app
      repoURL: https://patrickjmcd.github.io/homelab-app
      targetRevision: <version>
      helm:
        releaseName: myapp
        valueFiles:
          - $values/apps/myapp/values.yaml
    - path: apps/myapp
      repoURL: https://github.com/patrickjmcd/k8s-argo.git
      targetRevision: main
```

The third source applies the `kustomization.yaml` in `apps/myapp/` for any extra resources (OnePasswordItem, PVCs, HTTPRoutes, etc.).

### Kustomize-only app (for apps needing initContainers, CronJobs, StatefulSets, etc.)

```yaml
sources:
  - path: apps/myapp
    repoURL: https://github.com/patrickjmcd/k8s-argo.git
    targetRevision: main
```

Apps that **stay as Kustomize**: anything needing initContainers, multi-container pods, CronJob kind, or complex StatefulSets.

## homelab-app Chart

Generic chart at `charts/homelab-app/`. Key values:

```yaml
image:
  repository: ghcr.io/example/app
  tag: latest

service:
  port: 8080
  containerPort: 0  # set if container port ≠ service port
  type: ClusterIP   # LoadBalancer for non-HTTP LAN services (e.g. apps/voice-assistant)
  loadBalancerIP: "" # pin a MetalLB pool address
  annotations: {}   # e.g. kube-vip.io/ignore: "true" on LoadBalancer services

httpRoute:
  enabled: true
  hostnames:
    - myapp.example.com
  parentRefs:
    - name: traefik-gateway
      namespace: default

probes:
  startup:
    type: httpGet   # httpGet | tcpSocket | exec
    path: /health
    port: 8080
  readiness: ...
  liveness: ...

persistence:
  longhorn:
    enabled: true
    size: 5Gi
    mountPath: /data
  smb:
    enabled: true
    mountPath: /mnt/media
    claimName: smb-media-claim

onePassword:
  enabled: true
  # secret name defaults to <release>-1pw
  # itemPath defaults to vaults/Kubernetes/items/<release>

env:
  MY_VAR: value

configMap:
  enabled: true
  mountPath: /config
  subPath: config.yml   # for single-file mounts
  data:
    config.yml: |
      ...
```

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

## MetalLB

IP pool: `192.168.8.200–192.168.8.210`. Traefik LoadBalancer gets `.200`. L2 advertisement mode.

## Hardware

- **Pi 5 nodes**: `pi5-kube0`, `pi5-kube1`, `pi5-kube2` — general workloads
- **x86 VMs**: `kube-leader`, `kube-worker-{1-4}` — provisioned via Terraform on Proxmox

The cluster has no GPU nodes. The Jetson Nano (`jetson-nano-kube0`) was retired — it could not run an eBPF CNI datapath or attach Longhorn volumes on its L4T/Tegra kernel, and its GPU workloads (piper TTS, openwakeword) were dropped along with the NVIDIA device plugin and runtime config.

## Node Bootstrap Requirements (all nodes)

**CNI** — the cluster runs **Cilium** (`core/cilium.yaml`) as its only CNI; flannel was removed in Sept 2026 (`docs/cilium-migration.md`). k3s still runs with `--flannel-backend=none` — required, or k3s starts its built-in flannel alongside Cilium. Cilium's `install-cni-binaries` init container only drops in `cilium-cni` and writes `/etc/cni/net.d/05-cilium.conflist`, so k3s does NOT provide the standard CNI plugins either: every new node needs the base plugins (`loopback`, `portmap`, …) in `/opt/cni/bin` before pods can start. Symptom when missing: `FailedCreatePodSandBox … failed to find plugin "loopback" in path [/opt/cni/bin]`.

```bash
sudo apt-get install -y containernetworking-plugins
sudo mkdir -p /opt/cni/bin
sudo cp /usr/lib/cni/* /opt/cni/bin/
```

Also install the storage clients so CSI mounts work: `nfs-common` and `cifs-utils`.

**No flannel leftovers** — a node that ever ran flannel keeps its host state until cleaned or rebooted: a `cni0` bridge holding the `.1` of the node's pod CIDR, `flannel.1`, `/etc/cni/net.d/10-flannel.conflist`, `/run/flannel` and `FLANNEL-*` iptables chains. The stale `cni0` is not cosmetic: Cilium (`ipam.mode: kubernetes`) hands out the same `.1`, and a pod that gets it can't receive replies because the host owns that address — seen Sept 2026 as a Traefik pod on `pi5-kube2` crashlooping on "timed out waiting for controller caches to sync". Clean-up steps are in `docs/cilium-migration.md` §6.5.

**inotify instance limit** — `fs.inotify.max_user_instances` is a per-UID limit shared by every root-owned container on the node (not namespaced per-pod), and most system/sidecar containers run as root. A node hosting many containers with fsnotify-based watchers (config reloaders, log tailers, etc.) can silently approach the default cap of 128 as container density grows, at which point the next container to request an inotify instance crash-loops with `OSError: [Errno 24] inotify instance limit reached` — a red herring that looks like an app bug. Confirmed on `pi5-kube0/1/2` in Sept 2026 (91, 72, and 127 of 128 respectively) when the `youtubedl` postprocessor's `watchdog` Observer tipped `pi5-kube2` over the edge. It recurred in Sept 2026 via Grafana Alloy (`core/alloy.yaml`, tails every pod's log file over inotify) exhausting the limit and breaking ArgoCD's CronJob log stream.

As of Sept 2026 this is enforced **declaratively**, not just at bootstrap: the `node-tuning` DaemonSet (`core/node-tuning/daemonset.yaml`, wired in via `core/node-tuning.yaml`) runs on every node — including control-plane/tainted ones — and sets both `fs.inotify.max_user_instances=1024` and `fs.inotify.max_user_watches=1048576` via a privileged initContainer, writing the same values into `/etc/sysctl.d/99-inotify.conf` on the host so they survive a reboot before the pod is rescheduled. It self-heals: sync-wave `-1` plus `selfHeal: true` mean any drift (or a newly joined node) gets corrected automatically. `fs.inotify.*` are node-level (non-namespaced) sysctls, so they can't be set via a pod's `securityContext.sysctls` — hence the privileged container, scoped to only that DaemonSet.

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

## Node Bootstrap Requirements (RPi nodes)

When adding a new Raspberry Pi node (Pi 4 or Pi 5, Debian trixie, kernel 6.12.x), apply this before joining the cluster:

**Conntrack checksum fix** — RPi 6.12.x kernels have TX checksum offload enabled on veth/cni interfaces. With `nf_conntrack_checksum=1` (default), the kernel re-validates checksums on forwarded packets and marks them invalid, causing conntrack `clash_resolve` storms that poison DNS reply tracking and silently drop UDP responses.

```bash
echo 'net.netfilter.nf_conntrack_checksum=0' | sudo tee /etc/sysctl.d/99-conntrack.conf
sudo sysctl -w net.netfilter.nf_conntrack_checksum=0
```

Verify with: `sudo sysctl net.netfilter.nf_conntrack_checksum` → should be `0`.

## Conventions

- Always use `ServerSideApply=true` in syncOptions
- Always set `CreateNamespace=true` when the namespace isn't pre-existing
- Use `argocd.argoproj.io/manifest-generate-paths: .` on single-source Applications; for multi-source (homelab-app chart + `$values`) use absolute paths `/charts/homelab-app;/apps/<name>`
- Prefer `selfHeal: true` and `prune: true` for automated apps
- Never use `latest` image tags
- HTTPRoute `backendRef` API defaults cause ArgoCD diff noise — suppress with `ignoreDifferences` if needed
- For apps with PVC resize conflicts, add `ServerSideApplyForceConflicts=true` to syncOptions
- If adding a new application via helm chart, make sure the chart is allowed in the `sourceRepos` list for the ArgoCD project
