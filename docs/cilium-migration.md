# flannel → Cilium migration (Phase 1)

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

## Later phases (not this migration)

- **Phase 2** — kube-proxy replacement: `kubeProxyReplacement=true` +
  `--disable-kube-proxy` on k3s + `k8sServiceHost`/`k8sServicePort`. Also lets the
  kube-vip experiment be retired.
- **Phase 3** — Cilium LB-IPAM + L2/BGP announcements to replace MetalLB.
