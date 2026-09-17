# flannel → Cilium migration

**Phase 1 (CNI + NetworkPolicy + Hubble): ✅ complete since 2026-09-11** (see
`core/cilium.yaml`'s header comment for the cutover history). Phase 2
(kube-proxy replacement) is planned below, not yet started.

## Phase 1 — CNI cutover

Replaces flannel with **Cilium v1.20.1** as the CNI and turns on NetworkPolicy +
Hubble. **kube-proxy, MetalLB, kube-vip and Traefik are all kept.** kube-proxy
replacement and LB-IPAM are deliberately out of scope (Phase 2/3).

The ArgoCD `Application` lives at `core/cilium.yaml` but describes the
**post-migration end state**. Do the out-of-band, node-by-node migration below
*first*, then merge that file so ArgoCD merely adopts the running install.

Prereq already done: the Jetson Nano is retired, so every remaining node
(`pi5-kube{0,1,2}`, `kube-leader`, `kube-worker-{1..4}`) is a modern kernel.

---

## 0. Preflight — confirm every node can run the eBPF datapath

On each node:

```bash
{ zcat /proc/config.gz 2>/dev/null || cat /boot/config-$(uname -r); } \
  | grep -E 'CONFIG_(BPF_SYSCALL|BPF_JIT|NET_CLS_BPF|NET_SCH_INGRESS|CGROUP_BPF|BPF_EVENTS)='
```

All should read `=y` (or `=m`). Or let the tooling check for you after install:

```bash
cilium install --dry-run-helm-values   # sanity-check rendered values
kubectl exec -n kube-system ds/cilium -- cilium-dbg status --verbose   # post-install
```

Do **not** proceed on any node that's missing `BPF_SYSCALL` / `BPF_JIT` — that was
the Jetson's failure mode.

---

## 1. Understand the connectivity window (important)

We reuse the existing cluster CIDR (`10.42.0.0/16`) via `ipam.mode: kubernetes`,
so each node keeps the same per-node `/24` k3s already assigned it — **no pod
renumber**. The trade-off: flannel's VXLAN and Cilium's VXLAN don't interoperate,
so while the cluster is *mixed*, **pods on already-migrated nodes cannot reach
pods on not-yet-migrated nodes** (and vice-versa). host-network pods, the API
server, and kube-proxy service VIPs keep working throughout.

⇒ Treat this as a **maintenance window** and migrate every node back-to-back in
one sitting. (If you need zero-downtime instead, use Cilium's officially
documented dual-CIDR migration with a temporary `cluster-pool` IPAM and
`CiliumNodeConfig` hybrid networking — more steps, and it converges to a
different interim IPAM than `core/cilium.yaml`. See
https://docs.cilium.io/en/stable/installation/k8s-install-migration/.)

---

## 2. k3s server flags

You already run `--flannel-backend=none` (flannel is a DaemonSet, not k3s's
built-in). For Phase 1 also ensure network-policy isn't double-owned, and **keep
kube-proxy** (do NOT add `--disable-kube-proxy` yet):

- `--flannel-backend=none`  ✅ already set
- `--disable-network-policy` — add if not present (Cilium owns policy now)
- kube-proxy: **left enabled**

Applied in the k3s systemd unit / config on the server node(s); restart k3s there
after changing.

---

## 3. Install Cilium (out-of-band, not yet via ArgoCD)

Install with the **same chart version and values** as `core/cilium.yaml` so the
later ArgoCD adoption is a no-op. Using Helm directly:

```bash
helm repo add cilium https://helm.cilium.io/
helm repo update

helm install cilium cilium/cilium --version 1.20.1 \
  --namespace kube-system \
  --set ipam.mode=kubernetes \
  --set routingMode=tunnel \
  --set tunnelProtocol=vxlan \
  --set kubeProxyReplacement=false \
  --set cni.exclusive=false \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true
```

`cni.exclusive=false` is what lets Cilium coexist with flannel's CNI config during
the cutover instead of clobbering it.

---

## 4. Migrate node-by-node

For **each** node (start with a non-critical worker, e.g. `kube-worker-4`; do a
server/`kube-leader` last):

```bash
NODE=kube-worker-4

kubectl cordon "$NODE"
kubectl drain "$NODE" --ignore-daemonsets --delete-emptydir-data

# On the node: restart so it picks up Cilium as its CNI and drops flannel's
# veths/routes cleanly.
ssh "$NODE" 'sudo systemctl restart k3s-agent'   # or `k3s` on the server node

# Wait for the cilium agent on this node to be Ready, then verify.
kubectl -n kube-system wait --for=condition=ready pod \
  -l k8s-app=cilium --field-selector spec.nodeName="$NODE" --timeout=120s

kubectl uncordon "$NODE"
```

After each node, sanity-check a pod on it gets a Cilium-managed IP and reaches
the API / same-node pods:

```bash
kubectl -n kube-system exec ds/cilium -- cilium-dbg status --brief
```

Roll through all nodes. Expect cross-node app traffic to be partially broken
until the *last* node is done — that's the window from §1.

---

## 5. Validate

```bash
cilium status --wait
cilium connectivity test          # full end-to-end suite; ~a few minutes
```

Then confirm the things Phase 1 must NOT have disturbed:

- MetalLB VIPs still answer: Traefik on `192.168.8.200`, nats on `.205`.
- HTTPRoutes resolve (hit any app hostname).
- DNS from a fresh pod (`kubectl run tmp --rm -it --image=busybox -- nslookup kubernetes`).

Land **one** proof-of-life policy before rolling segmentation out broadly — e.g. a
default-deny `CiliumNetworkPolicy` in a single low-risk namespace — and watch it in
Hubble.

---

## 6. Adopt into GitOps + remove flannel

1. Merge `core/cilium.yaml` (+ the `helm.cilium.io/` entry already added to
   `argo/projects/core.yaml`). ArgoCD adopts the running release; the sync should
   show no meaningful diff. Reconcile any trivial drift with a manual sync.
2. Delete `core/flannel.yaml` in a follow-up commit — ArgoCD prunes the flannel
   DaemonSet. Do this **only after every node is migrated and green**.
3. Flip `cni.exclusive` back to the chart default (`true`) once flannel is gone,
   so Cilium owns `/etc/cni/net.d` cleanly.
4. Update `CLAUDE.md`: the "CNI plugins / flannel DaemonSet" node-bootstrap note
   no longer applies; replace it with the Cilium bootstrap facts.

---

## 7. Rollback

Until `core/flannel.yaml` is deleted (step 6.2), rollback is per-node: cordon the
migrated node, `helm uninstall cilium` (or scale the agent off it), reboot back
onto flannel, uncordon. Keep flannel installed and healthy until a full node has
round-tripped and you trust the datapath.

---

## Phase 2 — kube-proxy replacement (planned, not started)

Replaces kube-proxy's iptables-based Service routing with Cilium's eBPF
implementation. MetalLB, kube-vip's **control-plane VIP**, and Traefik are all
kept — this phase touches Service dataplane only. Cilium LB-IPAM/BGP stays
out of scope (Phase 3).

### 0. Blocking prerequisite — fix cluster health first

**`kube-leader-2` is currently `NotReady`.** The cluster has 3 control-plane/etcd
members (`kube-leader-2` 192.168.8.32, `kube-macmini` 192.168.8.21, `kube-n3160`
192.168.8.23) and is running on **2/3 quorum right now**. This phase requires
restarting `k3s`/`k3s-agent` on every node, including all three control-plane
nodes, to pick up the new proxy flags. Restarting either healthy survivor while
`kube-leader-2` is down would drop etcd to 1/3 — non-quorate, API server down
cluster-wide.

Do not proceed past step 0 until `kubectl get nodes` shows all three
control-plane nodes `Ready` and an etcd `endpoint health --cluster` check (see
`docs/bare-metal-conversion-runbook.md` §1 for the etcdctl recipe) confirms
3/3. Take a fresh `k3s etcd-snapshot save` before starting §2 either
way, same as the bare-metal runbook requires.

### 1. Preflight

Cilium is already the sole CNI and healthy (`cilium-dbg status --brief` → OK),
so there's no CNI-swap risk this time — this phase only changes how Service
VIPs get routed. Still confirm before starting:

```bash
cilium status --wait
kubectl get nodes                                    # all Ready, correct count
kubectl -n kube-system get cm cilium-config -o jsonpath='{.data.kube-proxy-replacement}'   # currently "false"
```

Note today's actual topology for reference (this drifted from `CLAUDE.md`'s
hardware list — trust `kubectl get nodes`, not the doc, until it's updated):
control-plane = `kube-leader-2`, `kube-macmini`, `kube-n3160` (all amd64);
agents = `macpro-kube0`, `media-server`, `pi4-kube0`, `pi4-kube1`, `pi5-kube0`,
`pi5-kube1`, `pi5-kube2`.

### 2. Understand the rollout strategy (safer than Phase 1's cutover)

Unlike the CNI swap, Cilium's kube-proxy replacement does **not** require a
mixed-mode maintenance window. Cilium's own migration guidance is to enable
the eBPF replacement *while kube-proxy is still running* — the two don't
conflict, Cilium's BPF programs simply take over Service routing for traffic
they manage. That gives a validation window with a trivial rollback (flip the
Helm value back) before touching kube-proxy itself.

So this is two separate, independently-reversible stages, not one big-bang
cutover:

- **Stage A** — set `kubeProxyReplacement=true` in the Cilium Helm values
  (rolling restart of `cilium-agent`, no k3s/node changes). kube-proxy keeps
  running the whole time as a fallback. Validate thoroughly.
- **Stage B** — only after Stage A is proven, disable kube-proxy at the k3s
  level node-by-node and remove the DaemonSet. This is the step with real
  downtime risk per node (a k3s/k3s-agent restart), and the one where control
  plane nodes need the one-at-a-time care from step 0.

### 3. Stage A — enable Cilium's replacement, keep kube-proxy running

Bump the existing `core/cilium.yaml` Application values out-of-band first
(matching Phase 1's pattern of validating before merging), or just edit the
live Helm release directly since this is non-destructive and trivially
reversible:

```bash
helm upgrade cilium cilium/cilium --version 1.20.1 --namespace kube-system \
  --reuse-values \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=192.168.8.10 \
  --set k8sServicePort=6443
```

`k8sServiceHost`/`k8sServicePort` point at the **kube-vip control-plane VIP**
(`https://192.168.8.10:6443`, the same VIP `K3S_URL` already uses per
`docs/bare-metal-conversion-runbook.md`), not any single server — so this
survives losing one control-plane node. Setting these now (even before
disabling kube-proxy in Stage B) avoids the bootstrap chicken-and-egg problem
where cilium-agent can't resolve the `kubernetes` Service to reach the API
server once kube-proxy is gone later.

Validate:

```bash
cilium status --wait                                 # KubeProxyReplacement: True
kubectl -n kube-system exec ds/cilium -- cilium-dbg status --verbose | grep -A5 "KubeProxyReplacement"
cilium connectivity test                              # full suite, ~a few minutes
```

Then confirm real traffic, same checklist as Phase 1 §5:
- MetalLB VIPs still answer (Traefik `192.168.8.200`, nats `.205`).
- HTTPRoutes resolve.
- DNS from a fresh pod.
- A Service with `externalTrafficPolicy: Local` still preserves client source
  IP (check Traefik's access logs or a test Service) — this is the one
  behavior most likely to regress silently under eBPF Service routing.

Sit on Stage A for at least a few days before Stage B — it's the low-risk half
and the one worth letting soak.

### 4. Stage B — disable kube-proxy, node by node

k3s's kube-proxy is started per-node by the local `k3s`/`k3s-agent` process,
controlled by a `--disable-kube-proxy` flag — there's no central toggle. Per
`docs/bare-metal-conversion-runbook.md` §5, the authoritative flags for this
cluster live in **both** `/etc/rancher/k3s/config.yaml` *and* the systemd
unit's `ExecStart` (`systemctl cat k3s` / `k3s-agent`) — check both on every
node, don't assume they're empty.

Roll through **agents first**, control-plane nodes **last and one at a time**:

```bash
NODE=pi4-kube0   # start with a non-critical agent, same order as Phase 1 §4

kubectl cordon "$NODE"
kubectl drain "$NODE" --ignore-daemonsets --delete-emptydir-data

# on the node: add --disable-kube-proxy to the k3s-agent (or k3s, on servers)
# systemd unit / config.yaml, matching whichever mechanism the live flags use
ssh "$NODE" 'sudo systemctl restart k3s-agent'   # k3s on server nodes

kubectl -n kube-system wait --for=condition=ready pod \
  -l k8s-app=cilium --field-selector spec.nodeName="$NODE" --timeout=120s

kubectl uncordon "$NODE"
kubectl get pods -n kube-system -l app.kubernetes.io/name=kube-proxy -o wide \
  | grep -v "$NODE"   # confirm no kube-proxy pod remains on $NODE
```

For the **3 control-plane nodes** (`kube-leader-2`, `kube-macmini`,
`kube-n3160`): do them last, one at a time, and re-verify `etcd` is 3/3 healthy
*before starting each one* — not just before the first. A control-plane
`systemctl restart k3s` briefly drops that node's apiserver/etcd participation;
never start the next one until the previous is fully `Ready` again.

After the last node: delete the now-empty kube-proxy DaemonSet/ConfigMap.

### 5. Validate (repeat, more thoroughly than Stage A)

```bash
cilium connectivity test
```

Full checklist: MetalLB VIPs, HTTPRoutes, DNS, `externalTrafficPolicy: Local`
behavior, plus specifically re-test anything that talks to the API server via
the in-cluster `kubernetes` Service (e.g. ArgoCD, cert-manager, Kyverno) since
that's the path `k8sServiceHost`/`k8sServicePort` exists to protect.

### 6. Adopt into GitOps

Update `core/cilium.yaml`:

```yaml
kubeProxyReplacement: true
k8sServiceHost: "192.168.8.10"
k8sServicePort: "6443"
```

ArgoCD should show no diff (adopting the already-running config), same as
Phase 1 §6. Update `CLAUDE.md`'s "Resource requests and limits" / hardware
sections if kube-proxy or kube-vip references need correcting.

### 7. Optional follow-up — retire the `kube-vip-svc-ds` DaemonSet

Separate from the control-plane VIP (`kube-vip-ds`, keep this — it's what
`k8sServiceHost` points at), `kube-vip-svc-ds` runs on 8 of the 10 nodes doing
**Service** load-balancing that's redundant with MetalLB, which is the
documented/primary mechanism (`192.168.8.200-210`). Confirm nothing actually
depends on kube-vip's Service-LB mode (no Service annotated for it) before
deleting the DaemonSet — this is unrelated to kube-proxy replacement working
correctly, just cleanup of leftover "experiment" infra noted when Phase 1 was
written.

### 8. Rollback

- **Stage A**: `helm upgrade cilium cilium/cilium --reuse-values --set kubeProxyReplacement=false` — instant, kube-proxy was never touched.
- **Stage B**, per node: remove `--disable-kube-proxy`, restart `k3s`/`k3s-agent`
  on that node, confirm its kube-proxy pod comes back. Because Stage B is
  rolled out node-by-node, a bad node can be reverted without affecting
  others — unlike Phase 1's CNI swap, there's no cluster-wide mixed-mode
  connectivity break to worry about here.

## Phase 3 (not this migration)

Cilium LB-IPAM + L2/BGP announcements to replace MetalLB.
