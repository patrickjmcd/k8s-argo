# k8s-argo

GitOps repository for the homelab k3s cluster, reconciled by
[Flux](https://fluxcd.io) (Argo CD until Oct 2026; the name is historical).
Push to `main` on Gitea and Flux applies it. Day-to-day conventions live in
[`CLAUDE.md`](CLAUDE.md); the migration record is
[`docs/flux-migration.md`](docs/flux-migration.md).

## Layout

```
clusters/home/            # Everything Flux applies
├── kustomization.yaml    # root (synced by the operator-created flux-system Kustomization)
├── flux-operator.yaml    # Flux Operator HelmRelease (+ web UI)
├── flux-instance.yaml    # FluxInstance: Flux version, components, git source
├── notifications.yaml    # Flux errors -> Alertmanager -> ntfy
├── cronjobs.yaml         # Kustomization for cronjobs/
├── apps/                 # one Kustomization / HelmRelease per app; media ResourceSet
└── core/                 # one per core component; helm-repositories.yaml
apps/                     # App manifests (HomelabApp instances, kustomize)
core/                     # Core component manifests and chart extras
cronjobs/                 # Scheduled jobs
```

Ordering between components is expressed with `dependsOn` (e.g. every app
with a HomelabApp depends on `kro-definitions`).

## Disaster recovery (fresh cluster)

Untested end to end; adapt as needed.

1. **k3s.** Install with Traefik/servicelb disabled and no built-in flannel
   (the cluster runs Cilium); see "Node Bootstrap Requirements" in
   `CLAUDE.md` for the CNI plugins and sysctls every node needs.
   ```bash
   curl -sfL https://get.k3s.io -o k3s-install.sh   # read it before running
   sh k3s-install.sh --disable traefik --disable servicelb \
     --flannel-backend=none --write-kubeconfig-mode 644
   ```
2. **Cilium, by hand.** Flux's controllers need pod networking, so install the
   CNI first with the values from `clusters/home/core/cilium.yaml`; Flux adopts
   the release later:
   ```bash
   yq 'select(.kind=="HelmRelease") | .spec.values' clusters/home/core/cilium.yaml > /tmp/cilium-values.yaml
   helm install cilium cilium --repo https://helm.cilium.io/ --version <chart version from that file> \
     -n kube-system -f /tmp/cilium-values.yaml
   ```
3. **1Password bootstrap Secrets.** Connect and its operator are a Flux
   HelmRelease (`clusters/home/core/onepassword-connect.yaml`), but the two
   Secrets they start from can't come from 1Password. Create them in `default`
   first:
   ```bash
   kubectl -n default create secret generic op-credentials \
     --from-file=1password-credentials.json=<path to the Connect server's credentials file>
   kubectl -n default create secret generic onepassword-token --from-literal=token=<Connect token>
   ```
   Every other secret comes from 1Password through `OnePasswordItem`s.
4. **Flux.** Gitea doesn't exist yet, so sync from the GitHub mirror. The URL
   must change in git too, or Flux's first reconcile re-applies the Gitea URL
   from `flux-instance.yaml`:
   ```bash
   sed -i 's#http://gitea.gitea.svc.cluster.local:3000/patrickjmcd/k8s-argo.git#https://github.com/patrickjmcd/k8s-argo.git#' clusters/home/flux-instance.yaml
   git commit -am "flux: sync from GitHub (DR)" && git push github main
   helm install flux-operator oci://ghcr.io/controlplaneio-fluxcd/charts/flux-operator \
     -n flux-system --create-namespace
   kubectl apply -f clusters/home/flux-instance.yaml
   ```
   Flux then installs everything else from git and adopts the Cilium release.
5. **Data.** Longhorn volumes restore from the B2 backup target; shared-postgres
   from its B2 barman backups; Gitea from its Longhorn volume + `gitea` DB.
6. **Back to Gitea.** Once Gitea is up with its data, push this repo to it
   (`origin`), revert the URL change there, and patch the live FluxInstance the
   same way back to
   `http://gitea.gitea.svc.cluster.local:3000/patrickjmcd/k8s-argo.git`.
   Gitea's push mirror then overwrites GitHub again.

## Key URLs

| Service | URL |
|---------|-----|
| Flux UI | https://flux.x.pmcd.io |
| Gitea | https://git.x.pmcd.io |
| Grafana | https://grafana.x.pmcd.io |
| Traefik Dashboard | https://traefik.x.pmcd.io/dashboard/ |

## Troubleshooting

### Something isn't deploying

```bash
flux get kustomizations -A --status-selector ready=false
flux get helmreleases -A --status-selector ready=false
flux reconcile kustomization <name> --with-source   # force a reconcile now
flux events --for Kustomization/<name> -n flux-system
```

A chart version Renovate just bumped can fail with "no chart with version
matching" until Flux's cached index refreshes: `flux reconcile source helm <repo>`.

### Secrets not syncing (1Password)

```bash
kubectl get onepassworditem -A
kubectl annotate onepassworditem <name> -n <ns> force-sync=$(date +%s) --overwrite
kubectl rollout restart deployment onepassword-connect -n default   # if a field stays empty
```

### Gateway API / HTTPRoute Issues

```bash
kubectl get gateways -A
kubectl get httproutes -A
kubectl describe httproute <name> -n <namespace>
```

### Node Prerequisites (Ubuntu/Debian)

Before joining a new node to the cluster, install required packages:

```bash
sudo apt-get update
sudo apt-get install -y containernetworking-plugins nfs-common cifs-utils
sudo mkdir -p /opt/cni/bin
sudo cp -a /usr/lib/cni/* /opt/cni/bin/
```

- `containernetworking-plugins` - Required for pod networking (CNI)
- `nfs-common` - Required for NFS CSI driver volumes
- `cifs-utils` - Required for CIFS/SMB volumes (if used)

#### Raspberry Pi: Enable cgroups

For Raspberry Pi nodes, enable cgroups by adding to `/boot/firmware/cmdline.txt`:

```bash
sudo sed -i '$ s/$/ cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1/' /boot/firmware/cmdline.txt
sudo reboot
```

After reboot, verify cgroups are enabled:

```bash
cat /proc/cgroups
```

### CNI Plugins Missing (Pods Stuck in ContainerCreating)

If pods are stuck in `ContainerCreating` on a node, verify CNI plugins exist on that node.

1. Check for CNI binaries and config:
```bash
ls -la /opt/cni/bin
ls -la /etc/cni/net.d
```

2. If missing, follow the Node Prerequisites steps above.

3. Restart k3s on the node (if needed):
```bash
sudo systemctl restart k3s
```

4. Verify:
```bash
kubectl get pods -A -o wide --field-selector spec.nodeName=<node-name>
```

## Development

### Security Commit Agent

This repository includes a pre-commit security agent that reviews staged changes before commit.

It blocks commits when it detects:
- Likely secrets (API keys, tokens, passwords, private keys)
- Sensitive local files staged by mistake (for example `1password-credentials.json`, `secrets.env.json`, `.env*`)
- High-risk security patterns (for example `privileged: true`, `allowPrivilegeEscalation: true`, `runAsNonRoot: false`, `:latest`, `curl | sh`)

Install once per clone:

```bash
./scripts/install-security-agent.sh
```

Bypass only for one-off exceptions:

```bash
SECURITY_AGENT_BYPASS=1 git commit -m "..."
```

### Renovate

Configuration is in `renovate.json`. The `flux` manager updates chart versions
under `clusters/`; image tags in `apps/` and `core/` manifests are updated by
the kubernetes, kustomize and custom regex managers.

### Adding New Applications

1. Put the manifests in `apps/<name>/` (a HomelabApp where possible, see `CLAUDE.md`).
2. Add `clusters/home/apps/<name>.yaml` (a Kustomization with `path: ./apps/<name>`,
   plus `dependsOn: kro-definitions` for a HomelabApp) and list it in
   `clusters/home/apps/kustomization.yaml`. Media apps: add the directory name
   to the `inputs` of `clusters/home/apps/media.yaml`.
3. Commit and push to Gitea; Flux applies it within about a minute.

---

## Helpers

### Resizing a PVE HD

#### 1st step: increase/resize disk from GUI console

From PVE GUI, select the VM -> Hardware -> Select the Disk -> Resize disk

#### 2nd step: Extend physical drive partition

1. Check free space:
```shell
sudo fdisk -l
```

2. Extend physical partition:
```shell
sudo growpart /dev/sda 3
```

3. Check physical drive:
```shell
sudo pvdisplay
```

4. Instruct LVM that disk size has changed:
```shell
sudo pvresize /dev/sda3
```

5. Check physical drive if has changed:
```shell
sudo pvdisplay
```

#### 3rd step: Extend Logical volume

1. Check Logical Volume:
```shell
sudo lvdisplay
```

2. Extend Logical Volume:
```shell
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
```

3. Check Logical Volume if has changed:
```shell
sudo lvdisplay
```

#### 4th step: Resize Filesystem

1. Resize filesystem:
```shell
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
```

2. Confirm results:
```shell
sudo fdisk -l
```
