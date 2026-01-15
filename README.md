# k8s-argo

GitOps repository for managing Kubernetes cluster with ArgoCD.

## Architecture

```
├── argo/                    # ArgoCD configuration
│   ├── apps.yaml           # App-of-apps for user applications
│   ├── core.yaml           # App-of-apps for core infrastructure
│   └── projects/           # ArgoCD AppProjects
├── apps/                    # User applications
│   ├── home/               # Home automation apps
│   ├── media/              # Media management apps
│   └── ...
└── core/                    # Core infrastructure components
    ├── argocd.yaml         # ArgoCD self-management
    ├── traefik.yaml        # Ingress controller
    ├── cert-manager.yaml   # TLS certificate management
    ├── external-secrets.yaml
    ├── metallb-system.yaml # Load balancer
    └── ...
```

## Sync Wave Order

Components deploy in this order based on sync waves:

| Wave | Components |
|------|------------|
| -2   | ArgoCD (self-management) |
| -1   | cert-manager, external-secrets, metallb |
| 0    | traefik, kyverno, csi drivers |
| 1    | monitoring, applications |

## Disaster Recovery Setup

### Prerequisites

- Fresh Ubuntu/Debian server(s)
- Access to external secrets provider (1Password, etc.)
- DNS configured for your domain

### Step 1: Install k3s

On the first/primary node:

```bash
curl -sfL https://get.k3s.io | sh -s - \
  --disable traefik \
  --disable servicelb \
  --write-kubeconfig-mode 644
```

For additional nodes (get token from `/var/lib/rancher/k3s/server/node-token`):

```bash
curl -sfL https://get.k3s.io | K3S_URL=https://<PRIMARY_IP>:6443 K3S_TOKEN=<TOKEN> sh -
```

### Step 2: Install ArgoCD

```bash
kubectl create namespace argocd

kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

Wait for ArgoCD to be ready:

```bash
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s
```

### Step 3: Create Required Secrets

ArgoCD needs certain secrets before it can manage itself and other apps:

#### ArgoCD OIDC Secret (if using OIDC authentication)

```bash
kubectl create secret generic argocd-secret -n argocd \
  --from-literal=oidc.pocketID.clientSecret='<YOUR_OIDC_SECRET>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

#### External Secrets Operator - 1Password Connect Credentials

```bash
kubectl create namespace external-secrets

kubectl create secret generic onepassword-connect-token -n external-secrets \
  --from-literal=token='<1PASSWORD_CONNECT_TOKEN>'
```

#### Cloudflare Tunnel Token (if using Cloudflare tunnels)

```bash
kubectl create namespace cloudflared

kubectl create secret generic cloudflared-token -n cloudflared \
  --from-literal=TUNNEL_TOKEN='<CLOUDFLARE_TUNNEL_TOKEN>'
```

### Step 4: Bootstrap with App-of-Apps

Apply the ArgoCD projects first:

```bash
kubectl apply -f argo/projects/
```

Then apply the root applications:

```bash
kubectl apply -f argo/core.yaml
kubectl apply -f argo/apps.yaml
```

ArgoCD will now:
1. Deploy itself via Helm (sync wave -2)
2. Deploy cert-manager, external-secrets, metallb (sync wave -1)
3. Deploy traefik and other core components (sync wave 0)
4. Deploy all user applications (sync wave 1)

### Step 5: Verify Deployment

Check ArgoCD UI (get initial admin password):

```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

Port-forward to access ArgoCD UI:

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Or access via your configured ingress URL once Traefik is running.

## Key URLs

| Service | URL |
|---------|-----|
| ArgoCD | https://argo.x.pmcd.io |
| Traefik Dashboard | https://traefik.x.pmcd.io/dashboard/ |

## Secrets Management

This cluster uses External Secrets Operator with 1Password Connect to manage secrets.

Secrets are defined as `ExternalSecret` resources that reference items in 1Password. The operator automatically syncs these to Kubernetes Secrets.

Example ExternalSecret:
```yaml
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: my-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: onepassword-connect
    kind: ClusterSecretStore
  target:
    name: my-secret
  data:
    - secretKey: password
      remoteRef:
        key: my-1password-item
        property: password
```

## Troubleshooting

### ArgoCD Application Stuck

If an application is stuck syncing:

```bash
# Check application status
kubectl get application -n argocd <app-name> -o yaml

# Force refresh
argocd app get <app-name> --refresh

# Hard refresh (invalidate cache)
argocd app get <app-name> --hard-refresh
```

### Sync Wave Issues

If components are deploying out of order, check sync wave annotations:

```bash
kubectl get applications -n argocd -o custom-columns='NAME:.metadata.name,WAVE:.metadata.annotations.argocd\.argoproj\.io/sync-wave'
```

### External Secrets Not Syncing

Check ExternalSecret status:

```bash
kubectl get externalsecret -A
kubectl describe externalsecret <name> -n <namespace>
```

Check 1Password Connect logs:

```bash
kubectl logs -n external-secrets deployment/onepassword-connect
```

### Gateway API / HTTPRoute Issues

Check Gateway status:

```bash
kubectl get gateways -A
kubectl get httproutes -A
kubectl describe httproute <name> -n <namespace>
```

## Backup and Restore

### What to Backup

1. **Secrets** - Most important! Back up to a secure location:
   - `argocd-secret` in `argocd` namespace
   - 1Password Connect token
   - Any manually created secrets

2. **Persistent Volumes** - Application data stored on PVCs

3. **This Git Repository** - All configuration is in Git

### Restore Process

1. Follow DR setup steps above
2. Restore secrets before bootstrapping
3. Apply app-of-apps to restore all applications
4. Restore PV data if needed

## Development

### Renovate

This repository uses Renovate for automated dependency updates. Configuration is in `renovate.json`.

Package rules:
- ArgoCD updates are grouped and require manual approval
- Infrastructure charts (traefik, cert-manager, external-secrets) are grouped

### Adding New Applications

1. Create application manifests in `apps/<category>/<app-name>/`
2. Either:
   - Add to an existing ApplicationSet, or
   - Create a new ArgoCD Application in `apps/<category>/`
3. Commit and push - ArgoCD will auto-sync

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
