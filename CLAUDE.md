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

### Helm app using homelab-app chart (preferred for simple apps)

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
    argocd.argoproj.io/manifest-generate-paths: .
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

Default storage class. Used for databases and stateful apps. Excludes Jetson Nano nodes. PVC name convention: `longhorn-<release>-data`.

**Jetson Nano can never attach Longhorn volumes, not just host replicas.** Its L4T/Tegra kernel (5.15.148-tegra, L4T R36.4.x) was built without `CONFIG_ISCSI_TCP` — `iscsi_tcp` doesn't exist as a loadable module, so Longhorn's iSCSI frontend fails outright when it tries to attach anything there (`allowScheduling: false` on that node only stops it from hosting *replicas*; it doesn't stop a pod with a Longhorn PVC from being scheduled there via nodeSelector and failing to attach). This is a structural kernel limitation, not a config bug — fixing it for real would mean cross-compiling a custom L4T kernel. Any workload placed on `jetson-nano-kube0` (e.g. for GPU access) must use a `hostPath` volume instead, same pattern as `apps/voice-assistant/piper` and `apps/voice-assistant/openwakeword`.

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

- **Pi 5 nodes**: `pi5-kube0`, `pi5-kube1`, `pi5-kube2` — general workloads, Frigate has Coral USB TPU on `pi5-kube1`
- **Jetson Nano**: `jetson-nano-kube0` — GPU workloads (NVIDIA device plugin). Least memory of any node (7.8Gi) and cannot attach Longhorn volumes (see Storage section) — keep it to stateless/GPU-only workloads with `hostPath` storage, not databases or other critical singletons.

**GPU runtime (`nvidia` RuntimeClass)**: managed by `apps/nvidia-runtime-config` (writes containerd config on the node via a privileged DaemonSet init container — see that manifest for the full story). Two non-obvious requirements, both confirmed by hand:
  - `SystemdCgroup = false` for the nvidia runtime specifically. `nvidia-container-runtime` fails sandbox creation with `SystemdCgroup = true` on this node, even though the plain `runc` handler uses `SystemdCgroup = true` fine. Don't "fix" this back to `true` without retesting.
  - Never set `default_runtime_name = "nvidia"`. That forces *every* pod on the node through `nvidia-container-runtime`, not just GPU ones — broke `kyverno`'s controllers (which run there with no HA elsewhere) and cascaded into a cluster-wide admission-webhook outage. `nvidia` must stay opt-in via `runtimeClassName` on individual pods only.
  - `nvidia-ctk runtime configure` (v1.16.2 on this node) always emits the containerd 1.x schema (`io.containerd.grpc.v1.cri`), which containerd 2.x (this node runs 2.2.2) silently ignores — looks like it worked, does nothing. Use the 2.x schema (`io.containerd.cri.v1.runtime`, `version = 3`) directly.
  - Test with a disposable pod (`runtimeClassName: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`) before pointing a real workload at it — check for `/dev/nvidia0`, `nvidiactl`, `nvhost-gpu`, `nvmap` inside the container.
- **x86 VMs**: `kube-leader`, `kube-worker-{1-4}` — provisioned via Terraform on Proxmox

## Node Bootstrap Requirements (all nodes)

**CNI plugins** — the cluster runs flannel as a DaemonSet (`kube-flannel`), not k3s's built-in flannel, so k3s does NOT provide the standard CNI plugins. Every new node needs the base plugins (`loopback`, `bridge`, `portmap`, …) present in `/opt/cni/bin` before pods can start; the flannel DaemonSet only drops in the `flannel` binary itself. Symptom when missing: `FailedCreatePodSandBox … failed to find plugin "loopback" in path [/opt/cni/bin]`.

```bash
sudo apt-get install -y containernetworking-plugins
sudo mkdir -p /opt/cni/bin
sudo cp /usr/lib/cni/* /opt/cni/bin/
```

Also install the storage clients so CSI mounts work: `nfs-common` and `cifs-utils`.

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
- Use `argocd.argoproj.io/manifest-generate-paths: .` annotation on all Applications
- Prefer `selfHeal: true` and `prune: true` for automated apps
- Never use `latest` image tags
- HTTPRoute `backendRef` API defaults cause ArgoCD diff noise — suppress with `ignoreDifferences` if needed
- For apps with PVC resize conflicts, add `ServerSideApplyForceConflicts=true` to syncOptions
- If adding a new application via helm chart, make sure the chart is allowed in the `sourceRepos` list for the ArgoCD project
