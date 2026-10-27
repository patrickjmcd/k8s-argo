# Argo CD → Flux migration

**Status: Phase 1 complete (2026-10-27).** Flux v2.10.0 installed and
self-managing. On Flux: `iss-in-space`, `joshhawleygop`, `spoolman`,
`tronbyt-server`. Verified: in-place adoption (UIDs unchanged, no pod
restarts), drift correction, HomelabApp health via CEL, namespace-less objects,
a Flux-created Namespace, `prune: disabled` keeping an object that left git,
and error alerts reaching ntfy. Next: Phase 2.

Argo CD and Flux run side by side for the whole migration. Apps are handed over
one at a time, leaf apps first, core last, Argo itself last. Any app can go back
to Argo by restoring its `Application` manifest.

Why: Flux is lighter on the 7.6 GB control-plane nodes (no repo-server, Redis or
UI), runs real Helm releases (history, rollback, hooks), and has no manifest or
diff cache. The Argo-only problems in CLAUDE.md go away with it: values commits
served from cache, and the `rgd-touch` workaround for stale ServerSideDiff
results. The UI is what we give up; Headlamp (`core/headlamp.yaml`) plus its
Flux plugin replaces it.

HomelabApp (kro) stays. It is independent of the GitOps engine.

---

## 0. Scope: what exists today

| What | Count | Notes |
|---|---|---|
| Root Applications (`argo/applications/`) | 3 | `apps`, `core`, `cronjobs`; applied by hand |
| `apps/*.yaml` Applications | 42 | all plain path/Kustomize except `immich` and `nats` (Helm) |
| `apps/media` ApplicationSet elements | 24 | `media-apps-kustomize`, plus `media-base` |
| `core/*.yaml` Applications | 36 | 22 use an upstream Helm chart |
| HomelabApp instances | 46 | unchanged by the migration |
| PostSync hook Jobs | 7 | postgres grants/ownership fixes, timescale schema, garage bootstrap |
| `Delete=false` / `Prune=false` objects | ~35 | mostly PVCs |
| `Replace=true` objects | 10 | `core/longhorn-assets/longhorn-node-*.yaml` |
| Manual-sync Applications | 3 | `cronjobs`, `ia-ignition`, `monitoring` |

The `homelab-app` chart has no remaining users among Applications. Every former
chart app is now HomelabApp or plain Kustomize, so the only Helm adoption work is
for upstream charts.

---

## 1. Target layout

```
clusters/home/
  flux-system/            # gotk-components.yaml + gotk-sync.yaml (GitRepository "k8s-argo")
  infra.yaml              # Kustomizations for core/ (one per component, with dependsOn)
  apps.yaml               # Kustomizations for apps/ (one per app)
  media.yaml              # Kustomizations for apps/media/* (replaces the ApplicationSet)
  cronjobs.yaml           # suspended Kustomization for cronjobs/
  sources/                # HelmRepository / OCIRepository objects
```

The app directories (`apps/<name>/`, `core/<name>/`) don't change. Only the
`apps/<name>.yaml` / `core/<name>.yaml` Application files are replaced by
Kustomization or HelmRelease objects.

**One GitRepository for everything.** Every Kustomization points at the
`k8s-argo` GitRepository through `sourceRef`, so the "Gitea is down" fallback in
CLAUDE.md shrinks from a `sed` over ~100 files to one patch:

```bash
kubectl -n flux-system patch gitrepository k8s-argo --type merge \
  -p '{"spec":{"url":"https://github.com/patrickjmcd/k8s-argo.git"}}'
```

### Install

Don't use `flux bootstrap gitea`. It wants to push a deploy key and talk to
Gitea through its public hostname. Mirror what Argo does today instead: an
anonymous in-cluster URL, applied by hand once, then self-managed.

```bash
flux install --export > clusters/home/flux-system/gotk-components.yaml
# gotk-sync.yaml (hand-written):
#   GitRepository k8s-argo  url: http://gitea.gitea.svc.cluster.local:3000/patrickjmcd/k8s-argo.git  ref.branch: main  interval: 1m
#   Kustomization flux-system  path: ./clusters/home  prune: true
kubectl apply -f clusters/home/flux-system/
```

Add a Gitea push webhook to a notification-controller `Receiver` so commits
reconcile immediately rather than on the 1m poll.

Resources: `flux install --export` ships 50m requests and a 1000m CPU limit
per controller. That limit is a runaway guard far above observed use (~2m
idle), which CLAUDE.md allows, so the export is used unmodified.

**Stale RBAC from an old install (hit 2026-10-27).** The cluster ran Flux once
before (2023) and later Rancher; Rancher's `wrangler.cattle.io/auth-prov-v2-*`
finalizer left the old `*-flux-system` ClusterRoles/ClusterRoleBindings stuck
Terminating since Jan 2026, so the new flux-system Kustomization timed out
waiting on them. Fix was to strip that finalizer from the five `*-flux-system`
objects, then re-run `kubectl apply --server-side --force-conflicts -k
clusters/home/flux-system` (deleting `cluster-reconciler-flux-system` also
removes Flux's own permissions, so it can't recreate them itself). The other 51
Rancher-finalized RBAC objects (Rancher, fleet, CAPI, argo-rollouts,
open-feature-operator; one bound `cattle-system/rancher-webhook` to
cluster-admin, another granted `default/default` extra rights) and two knative
CRDs stuck since Jan 2024 were cleared the same day.

---

## 2. Concept mapping

| Argo CD | Flux | Notes |
|---|---|---|
| `Application` (path source) | `Kustomization` | `sourceRef: k8s-argo`, `path: ./apps/<name>` |
| `Application` (chart source) | `HelmRelease` + `HelmRepository`/`OCIRepository` | values inline in `spec.values`, or `valuesFrom` a ConfigMap |
| `ApplicationSet` (list generator) | one Kustomization per app | or Flux Operator `ResourceSet` if you want a generator; plain files are simpler |
| `spec.destination.namespace` | `kustomization.yaml` `namespace:`, or explicit `metadata.namespace` | **not** `targetNamespace` (see §4.3) |
| `CreateNamespace=true` | a `Namespace` manifest in the app dir, annotated `kustomize.toolkit.fluxcd.io/prune: disabled` | Flux never creates namespaces; never let it prune one (see `apps/spoolman/namespace.yaml`) |
| `sync-wave` on Applications | `dependsOn` between Kustomizations/HelmReleases | waits for Ready, not just applied |
| `selfHeal: true` | default for Kustomizations; HelmRelease needs `driftDetection.mode: enabled` | |
| `prune: true` | `spec.prune: true` on the Kustomization | |
| automated off (manual sync) | `spec.suspend: true`; deploy with `flux resume ks X`, then `flux suspend ks X` | |
| `sync-options: Delete=false` / `Prune=false` | `kustomize.toolkit.fluxcd.io/prune: disabled` | covers both removal from git and deletion of the Kustomization |
| `sync-options: Replace=true` | `kustomize.toolkit.fluxcd.io/force: enabled` | recreates on immutable-field conflicts |
| `hook: PostSync` + `BeforeHookCreation` | plain Job in a dependent Kustomization with `force: true` | §4.4 |
| `ignoreDifferences` (Kyverno tolerations, `restartedAt`, HTTPRoute defaults) | nothing | drift is detected by a server-side dry-run, which already includes webhook mutations and API defaults |
| `ignoreDifferences` on webhook `caBundle` (cert-manager) | `driftDetection.ignore` on the HelmRelease | |
| Lua health check for `HomelabApp` | `healthCheckExprs` on the Kustomization | §4.5 |
| `compare-options: ServerSideDiff=true`, `manifest-generate-paths` | drop | no Flux equivalent needed |
| `AppProject` | drop (or a ServiceAccount per Kustomization for impersonation) | single operator; not worth it |
| Argo notifications / UI | notification-controller → ntfy; Headlamp Flux plugin; `flux get`/`flux events` | |

---

## 3. Per-app handoff procedure

> ⚠️ **The one dangerous step.** Every Application carries
> `resources-finalizer.argocd.argoproj.io`. Deleting an Application that still
> has it **deletes everything it manages, PVCs included.** The root apps run
> `prune: true`, so removing `apps/<name>.yaml` from git deletes the Application.
> **A live patch alone is not enough** (learned on the iss-in-space pilot,
> 2026-10-27): turning off `automated` on the child makes it differ from git,
> so the root app's selfHeal re-applies the whole manifest from git within
> seconds, finalizer included. The finalizer has to leave git first, in its
> own commit, before the commit that deletes the file.

For each app:

1. **Pre-check the manifests** (§4.3):
   ```bash
   kubectl kustomize apps/<name> | yq 'select(.metadata.namespace == null) | .kind + "/" + .metadata.name'
   ```
   Every namespaced object needs a namespace from `metadata.namespace` or
   `kustomization.yaml`. Add a `Namespace` manifest if the app relied on
   `CreateNamespace=true`.
2. **Convert annotations** in the app dir: `Delete=false`/`Prune=false` →
   `kustomize.toolkit.fluxcd.io/prune: disabled`; `Replace=true` →
   `kustomize.toolkit.fluxcd.io/force: enabled`. Leave the Argo annotations in
   place for now; Argo is still managing the app.
3. **Disarm commit:** in `apps/<name>.yaml` delete the `finalizers:` block and
   the `automated:` block. Push, refresh the root app, and **gate** on the live
   object (a printed value isn't a gate; this exits non-zero):
   ```bash
   kubectl -n argocd annotate application apps argocd.argoproj.io/refresh=normal --overwrite
   kubectl -n argocd get application <name> -o json | jq -e '(.metadata.finalizers // []) | length == 0'
   ```
   (`core/argocd.yaml` ignores finalizer diffs on Applications, so the root app
   may show Synced without having removed it. If the gate fails, strip it live:
   `kubectl -n argocd patch application <name> --type json -p
   '[{"op":"remove","path":"/metadata/finalizers"}]'` and re-check. The root app
   no longer re-adds it once git has no finalizer.)
4. **Handoff commit:** delete `apps/<name>.yaml` and add
   `clusters/home/apps/<name>.yaml` (just `metadata.name` and `spec.path`; the
   shared settings come from the patch in `clusters/home/apps/kustomization.yaml`)
   plus its line in that file's `resources`. Re-run the step-3 gate in the same
   shell command as `git commit`, so a failing gate stops the commit.
   Several apps can share one disarm commit and one handoff commit.
5. **Verify:**
   ```bash
   flux get ks <name>                        # Ready=True, applied revision = HEAD
   kubectl -n argocd get application <name>  # gone
   kubectl get pvc -n <ns>                   # same PVCs, same age
   ```
   Compare UIDs of the main objects before and after (nothing recreated), then
   confirm the app works (blackbox probe green in Grafana / Glance).
6. **Drop Argo's field ownership** (§4.1) from every object in the app dir,
   i.e. everything `kubectl kustomize apps/<name>` renders (for a HomelabApp app
   that's the HomelabApp plus any extra files; kro's children were never Argo's):
   ```bash
   idx=$(kubectl -n <ns> get <kind> <obj> -o json --show-managed-fields \
     | jq '[.metadata.managedFields[].manager] | index("argocd-controller")')
   kubectl -n <ns> patch <kind> <obj> --type json -p "[
     {\"op\":\"test\",\"path\":\"/metadata/managedFields/$idx/manager\",\"value\":\"argocd-controller\"},
     {\"op\":\"remove\",\"path\":\"/metadata/managedFields/$idx\"}]"
   kubectl -n <ns> annotate <kind> <obj> argocd.argoproj.io/tracking-id-
   ```
7. **Rollback** if needed: `flux suspend ks <name>`, restore `apps/<name>.yaml`,
   push. Argo re-adopts through SSA.

### ApplicationSet-generated apps (`apps/media`)

The ApplicationSet controller re-adds the finalizer from the template, so the
live patch in step 3 doesn't stick. Do this once before touching any media app:

1. Remove `finalizers` from the template in
   `apps/media/media-apps.applicationset.yaml` and add
   `spec.syncPolicy.preserveResourcesOnDeletion: true`. Push and let `media` sync.
2. Confirm that no generated Application still has the finalizer:
   ```bash
   kubectl -n argocd get applications -o json | jq -r '.items[] | select(.metadata.finalizers) | .metadata.name'
   ```
3. Hand over each media app by removing its element from the list (step 4 above)
   instead of deleting a file.
4. Last of all, delete the ApplicationSet, `media-base` and `media`, checking
   each for finalizers first.

---

## 4. Known differences and gotchas

### 4.1 Field ownership after handoff

Argo applied with SSA as field manager `argocd-controller`. Flux applies as
`kustomize-controller` and takes over the fields it sets. Fields that Argo set
and that are no longer in git stay owned by `argocd-controller` and won't be
removed. **Confirmed on the pilot:** after adoption, `argocd-controller` still co-owned
every `spec` field of the HomelabApp alongside `kustomize-controller`. Flux
doesn't clean up that manager. Left alone, removing a field from git would not
remove it live. Step 6 of §3 drops the entry; the object is not modified
otherwise (generation unchanged).

### 4.2 Helm adoption (upstream charts)

Argo rendered every chart with `helm template`, so **no Helm release secrets
exist** and the live objects carry no Helm ownership metadata. A HelmRelease's
first `helm install` refuses to touch objects it doesn't own. Options:

- Use helm-controller's take-ownership support (Helm 3.17+ `--take-ownership`),
  if the installed Flux version exposes it. Check its release notes.
- Or pre-label every object the chart renders:
  ```bash
  helm template <release> <chart> --version <v> -n <ns> -f values.yaml \
    | kubectl annotate -f - --overwrite meta.helm.sh/release-name=<release> meta.helm.sh/release-namespace=<ns>
  # same again with: kubectl label -f - --overwrite app.kubernetes.io/managed-by=Helm
  ```

Also:
- Chart hooks will run for the first time (e.g. cert-manager startupapicheck,
  kube-prometheus-stack admission-webhook patch Jobs). Read each chart's hooks
  before cutover.
- Set `install.crds: CreateReplace` and `upgrade.crds: CreateReplace`; Helm
  otherwise never upgrades CRDs.
- Set `driftDetection.mode: enabled` (selfHeal equivalent).
- `install.remediation.retries` / `upgrade.remediation.retries` with
  `remediateLastFailure: true` gives automatic rollback. Leave it **off** for
  longhorn, cilium and cnpg: a rollback mid-migration of their CRDs is worse than
  a stuck release.
- Take a values diff first: `helm template` with Flux's merged values vs
  `kubectl get` of the live objects. Any unexplained diff blocks the cutover.

### 4.3 Namespaces: don't use `targetNamespace`

Argo's `destination.namespace` only fills in a namespace where the manifest
has none. Flux's `targetNamespace` **overrides every object's namespace**, which
would move things for apps like `pve-exporter` (Application ns `prometheus`)
and every app whose Application says `default` but whose objects set their own.
Put `namespace:` in the app's `kustomization.yaml`, or set it on each object.

Directories with namespace-less objects as of 2026-10-27 (check each with the
step-1 command; some are cluster-scoped kinds the quick filter missed):
`apps/dreeve`, `apps/immich`, `apps/joshhawleygop`, `apps/pihole`,
`apps/media/{base,kometa,lidarr,sabnzbd}`, `core/{cert-manager,cilium,
cloudflared-tunnel,cnpg,csi-driver-smb,dozzle,longhorn-assets}`. The quick
filter failed to parse `apps/ia-ignition`, `apps/media/youtarr`, `core/garage`,
`core/grafana-assets`, `core/postgres` and `core/timescale`; check those by hand.

### 4.4 PostSync hook Jobs

| Job | File |
|---|---|
| `shared-postgres-backup-grants` | `core/postgres/grants-job.yaml` |
| `{dispatcharr,n8n,outline,reedme}-ownership-fix` | `core/postgres/*-ownership-fix.yaml` |
| `homeassistant-schema` | `core/timescale/homeassistant-schema-job.yaml` |
| `garage-bootstrap` | `core/garage/bootstrap-job.yaml` |

Move each into its own sub-directory (e.g. `core/postgres/jobs/`) with a
Kustomization that `dependsOn` the parent and sets `force: true`, so a changed
Job (immutable spec) is deleted and recreated. That is the
`BeforeHookCreation` equivalent. Unlike Argo hooks, they no longer re-run on
every sync, only when the Job spec changes. Confirm each one is idempotent and
that this is acceptable (the ownership fixes are one-shot repairs and arguably
can be deleted).

### 4.5 HomelabApp health

Port the Lua check from `core/argocd.yaml` to each HomelabApp Kustomization
(or set it once with a kustomize patch over all of them in `clusters/home/`):

```yaml
spec:
  wait: true            # healthCheckExprs only apply to objects being waited on
  healthCheckExprs:
    - apiVersion: homelab.pmcd.io/v1alpha1
      kind: HomelabApp
      current: has(status.conditions) && status.conditions.exists(e, e.type == 'Ready' && e.status == 'True')
      failed: has(status.state) && status.state in ['ERROR', 'FAILED']
```

Every HomelabApp Kustomization gets `dependsOn: [kro-definitions]`. That
replaces `SkipDryRunOnMissingResource=true`, because Flux won't apply until the
CRD exists.

### 4.6 Ordering (`dependsOn` replacing sync waves)

```
cilium (-4) ─┬─ kube-vip, node-tuning (-2/-1)
             ├─ cert-manager ── traefik ── (all HTTPRoute users)
             ├─ metallb-system ── traefik
             ├─ longhorn, csi-driver-{nfs,smb} ── (all PVC users)
             ├─ kyverno
             ├─ cnpg ── shared-postgres ── postgres jobs; apps with a DB
             ├─ tailscale-operator, timescale
             └─ kro ── kro-definitions ── every HomelabApp
prometheus (CRDs) ── every ServiceMonitor/PrometheusRule user (blackbox-exporter, kro instances)
```

Keep `dependsOn` coarse (e.g. apps depend on `kro-definitions` and `longhorn`,
not on each other). Deep chains stall everything behind one failing component.

### 4.7 Monitoring and alerts

- Drop `core/prometheus-assets/argocd-servicemonitor.yaml` and
  `core/grafana-assets/dashboards/argocd.json` at the end.
- Add a PodMonitor for the Flux controllers and the upstream Flux Grafana
  dashboards. Add Flux resource state to kube-state-metrics (custom resource
  state config from the flux2-monitoring-example repo), then an alert
  `FluxReconcileFailed` (Ready=False for 15m) in `rules.yaml` → ntfy `alerts`.
- notification-controller (done, `clusters/home/notifications.yaml`): Flux
  v2.10 has no ntfy provider, so the `alertmanager` provider posts error events
  to Alertmanager, whose root route already sends non-`info` alerts through n8n
  to the ntfy `alerts` topic. Tested 2026-10-27 with a deliberately broken path
  (alert `FluxKustomizationArtifactfailed`, severity `error`).

### 4.8 Renovate

`renovate.json` configures the `argocd` manager, plus an `overridePackageName`
rule for OCI chart sources (see memory note on the OCI/Renovate mismatch).
Enable the `flux` manager (HelmRelease + HelmRepository/OCIRepository) and drop
the Argo rule once no chart Application remains. Check that Renovate picks up
`kro` (`oci://registry.k8s.io/kro/charts`) and `kubernetes-mcp-server`
(`oci://ghcr.io/containers/charts`) from their OCIRepository objects.

---

## 5. Phases

Each phase is done when everything in it reports `flux get all -A` Ready and
Argo has no Application for it.

### Phase 1: install and pilot

- Install Flux (§1). Kustomization `flux-system` only.
- Hand over **`iss-in-space`** (HomelabApp, stateless) using §3, including the
  HomelabApp health expression and the kro dependency. Check field ownership (§4.1).
- Hand over **`stubs`** or **`joshhawleygop`** (namespace-less objects) to test
  §4.3.
- Hand over one app with a PVC and `Delete=false`, e.g. **`spoolman`**, to test
  the prune annotation: remove it from git on a throwaway copy and confirm the
  PVC survives.
- Notification-controller → ntfy working.

**Go/no-go:** if anything here surprises us, stop and either fix the procedure
or abandon the migration. Rolling back three apps is cheap.

### Phase 2: `apps/` (low risk, in batches)

HomelabApp and Kustomize apps, about 5 at a time:

| Batch | Apps |
|---|---|
| 2a stateless | glance, go2rtc, homebox-companion, homelable, homelable-mcp, nut-exporter, pihole-exporter, pve-exporter, unifi-mcp, unpoller, reedme, kaneo, stubs |
| 2b stateful (PVC) | adsb, dreeve, hivemq, homebox, homebridge, homelable-backend, mealie, minecraft, n8n, ntfy, outline, romm, sparkyfitness, sparkyfitness-server, speech-to-phrase, tracearr, unifi-toolkit, voice-assistant |
| 2c media | prep the ApplicationSet (§3), then the 24 elements + media-base, ~6 at a time |
| 2d special | renovate; ia-ignition (manual → suspended); pihole (cluster DNS: do on a quiet evening; the secondary Pi-hole LXC covers clients) |
| 2e Helm | nats (HelmRelease, §4.2), immich (HelmRelease, §4.2; stateful) |

Hold until Phase 4: **gitea** (source of truth for both Argo and Flux).

### Phase 3: `core/` low risk

Kustomize-only: cloudflared-tunnel, dozzle, csi-driver-nfs-assets,
goldilocks-assets, pocketid, system-upgrade, garage (+ bootstrap job, §4.4),
timescale (+ schema job), monitoring (manual → suspended).

HelmRelease, low blast radius: reloader, goldilocks, vpa, headlamp, loki,
alloy, blackbox-exporter, external-dns, trivy-operator,
kubernetes-mcp-server, tailscale-operator.

### Phase 4: `core/` high risk, one per session with a fresh backup

Order: kro → kro-definitions → traefik → prometheus → kyverno → cert-manager →
shared-postgres (+ jobs) → cnpg → metallb-system → csi-driver-smb / csi-driver-nfs →
node-tuning → kube-vip → longhorn → cilium → gitea.

Per component:
- Before: take a Longhorn backup or CNPG backup if it's stateful; keep the
  `helm template` diff from §4.2 clean.
- **kro**: the RGD and CRD must survive. Mark the RGD and the `HomelabApp` CRD
  `prune: disabled`; deleting the CRD would delete all 46 instances and their
  children.
- **longhorn**: add `prune: disabled` to the chart's CRDs and the
  `longhorn-node-*` objects. The chart's pre-delete uninstall job is gated by
  `deletingConfirmationFlag`; confirm it is `false` before the HelmRelease
  first reconciles.
- **cnpg / shared-postgres**: `prune: disabled` on the `Cluster`, `Database`
  and `Role` objects and on the CRDs.
- **cilium**: do it last among CNI-adjacent components, in a maintenance window.
  A bad first Helm install here breaks pod networking cluster-wide, and with it
  Flux itself, so recovery has to work without the cluster: keep the
  `core/cilium.yaml` values and a workstation `helm upgrade` / `cilium install`
  command ready before starting.
- **gitea**: after handoff, Flux reconciles Gitea from Gitea. Same circularity
  as today; the §1 GitHub patch is the escape hatch.

### Phase 5: remove Argo

1. Confirm `kubectl -n argocd get applications` lists only the 3 roots and
   `argocd` itself, none with a finalizer.
2. Strip finalizers from those and delete them, then delete the Argo
   install itself: `core/argocd.yaml`, `core/argo-assets/` and `argo/`
   (removed by hand, since nothing manages them by then).
3. Remove Argo CRDs (`applications`, `applicationsets`, `appprojects`).
4. Remove Argo monitoring (§4.7), the Glance/blackbox entry for the Argo UI
   hostname, the PocketID OIDC client for Argo, and the Argo split-horizon
   DNS records.
5. Strip leftover annotations repo-wide:
   ```bash
   git grep -l 'argocd.argoproj.io/' -- apps core cronjobs
   ```
6. Rewrite CLAUDE.md: "Where the repo lives", "App-of-Apps Pattern", "ArgoCD
   Projects", "Sync Waves", "Defining Applications" and "Conventions", plus the
   `rgd-touch` paragraph (obsolete). Update the memory notes that mention Argo.

---

## 6. Inventory

Risk: **L** = stateless or trivially restorable, **M** = stateful or
cluster-visible, **H** = cluster-wide blast radius.

### apps/

| App | Namespace | Type | Risk | Notes |
|---|---|---|---|---|
| adsb | adsb | kustomize | M | PVC |
| dreeve | default | kustomize | M | PVC; namespace-less objects |
| gitea | gitea | kustomize | H | source of truth; Phase 4, last |
| glance | default | HomelabApp | L | |
| go2rtc | default | HomelabApp | L | |
| hivemq | hivemq | kustomize | M | PVC |
| homebox | default | HomelabApp | M | PVC |
| homebox-companion | default | HomelabApp | L | |
| homebridge | homebridge | HomelabApp | M | PVC |
| homelable | homelable | HomelabApp | L | |
| homelable-backend | homelable | HomelabApp | M | PVC |
| homelable-mcp | homelable | HomelabApp | L | |
| ia-ignition | default | kustomize | M | manual sync → suspended; PVC |
| immich | immich | Helm `immich@0.12.0` + kustomize | M | §4.2; PVC |
| iss-in-space | default | HomelabApp | L | ✅ Flux (Phase 1) |
| joshhawleygop | default | HomelabApp | L | ✅ Flux (Phase 1); outage fixed (see below) |
| kaneo | kaneo | HomelabApp | L | |
| mealie | default | HomelabApp | M | PVC |
| media | media | app-of-apps | M | parent of ApplicationSet + media-base; remove last in 2c |
| minecraft | minecraft | kustomize | M | PVC |
| n8n | default | HomelabApp | M | PVC; DB |
| nats | nats | Helm `nats@2.15.1` + kustomize | M | §4.2 |
| ntfy | ntfy | HomelabApp | M | PVC; alert path, so confirm alerts afterwards |
| nut-exporter | default | HomelabApp | L | |
| outline | default | HomelabApp | M | PVC; `markdown-export.yaml` has `Prune=false` |
| pihole | pihole | kustomize | M | cluster DNS; namespace-less objects |
| pihole-exporter | pihole | HomelabApp | L | |
| pve-exporter | prometheus | HomelabApp | L | |
| reedme | default | HomelabApp | L | DB |
| renovate | renovate | kustomize | L | update its config in the same phase (§4.8) |
| romm | romm | HomelabApp | M | PVC |
| sparkyfitness | sparkyfitness | HomelabApp | M | |
| sparkyfitness-server | sparkyfitness | HomelabApp | M | PVC |
| speech-to-phrase | voice-assistant | HomelabApp | M | PVC |
| spoolman | spoolman | HomelabApp | L | ✅ Flux (Phase 1); no PVC; Namespace manifest added |
| stubs | default | HomelabApp | L | |
| tracearr | tracearr | HomelabApp | M | PVC |
| tronbyt-server | default | HomelabApp | M | ✅ Flux (Phase 1); PVC prune-disabled |
| unifi-mcp | default | HomelabApp | L | |
| unifi-toolkit | default | HomelabApp | M | PVC |
| unpoller | default | HomelabApp | L | |
| voice-assistant | voice-assistant | HomelabApp | M | PVC |

joshhawleygop (found during Phase 1, fixed 2026-10-27 in `3dba336`): the site
was down Oct 11–27. `build.py`'s kustomization template had no `resources:`
entry, so the Oct 11 bot rebuild dropped the hand-added `homelabapp.yaml` and
Argo pruned the HomelabApp. Separately, the Oct 6 chart→HomelabApp conversion
dropped `httproute.publicHostnames`. Both are restored; the template now carries
`resources:` itself.

### apps/media (ApplicationSet `media-apps-kustomize`, ns `media`)

| App | Type | Risk | Notes |
|---|---|---|---|
| media-base | kustomize (`apps/media/base`) | M | shared PVCs (`smb-media-claim`, `sab-downloads-pvc`); every *arr depends on it |
| agregarr, concertarr, navidrome, prowlarr, radarr, readarr, seerr, sonarr, sportarr, subwave, tautulli | HomelabApp | M | PVCs |
| aonsoku, hdhr2xmltv, ollama, ytdl-triage | HomelabApp | L | |
| bazarr, dispatcharr, jellyfin, kometa, lidarr, recyclarr, sabnzbd, youtarr, ersatztv | kustomize | M | initContainers / StatefulSet / CronJob; several namespace-less (§4.3) |

### core/

| Component | Namespace | Wave | Source | Risk | Phase | Notes |
|---|---|---|---|---|---|---|
| cilium | kube-system | -4 | `cilium@1.20.4` + kustomize | H | 4 | CNI; maintenance window |
| argocd | argocd | -2 | `argo-cd@10.10.3` + kustomize | — | 5 | removed, not migrated |
| cert-manager | cert-manager | -2 | `cert-manager@v1.21.2` + kustomize | H | 4 | caBundle drift ignore |
| kube-vip | kube-system | -2 | kustomize | H | 4 | control-plane VIP |
| cloudnative-pg | cnpg-system | -1 | `cloudnative-pg@0.29.1`, `plugin-barman-cloud@0.8.1` + kustomize | H | 4 | two HelmReleases |
| csi-driver-nfs | kube-system | -1 | `csi-driver-nfs@4.13.6` | H | 4 | |
| csi-driver-smb | default | -1 | kustomize | H | 4 | |
| longhorn | longhorn-system | -1 | `longhorn@1.13.0` + kustomize | H | 4 | `Replace=true` node objects |
| metallb-system | metallb-system | -1 | `metallb@0.16.1` + kustomize | H | 4 | |
| node-tuning | kube-system | -1 | kustomize | M | 4 | |
| shared-postgres | postgres | -1 | kustomize | H | 4 | 6 hook Jobs |
| tailscale-operator | tailscale | -1 | `tailscale-operator@1.104.2` + kustomize | M | 3 | |
| timescale | timescale | -1 | kustomize | M | 3 | 1 hook Job |
| alloy | loki | 0 | `alloy@1.14.3` | L | 3 | |
| kro | kro-system | 0 | `kro@0.9.4` (OCI) | H | 4 | CRD must never be pruned |
| kyverno | kyverno | 0 | `kyverno@3.9.2` + kustomize | H | 4 | admission webhooks |
| loki | loki | 0 | `loki@7.5.0` | L | 3 | |
| prometheus | prometheus | 0 | `kube-prometheus-stack@92.3.1` + 2 kustomize paths | M | 4 | CRDs every ServiceMonitor needs |
| vpa | vpa | 0 | `vpa@5.1.0` | L | 3 | |
| goldilocks | goldilocks | 1 | `goldilocks@11.2.0` | L | 3 | |
| goldilocks-assets | goldilocks | 1 | kustomize | L | 3 | |
| kro-definitions | kro-system | 1 | kustomize | H | 4 | the HomelabApp RGD |
| traefik | default | 1 | `traefik@41.7.1` + kustomize | H | 4 | all ingress; assets dir has a PVC |
| trivy-operator | trivy-system | 1 | `trivy-operator@0.37.0` + kustomize | L | 3 | |
| blackbox-exporter | prometheus | — | `prometheus-blackbox-exporter@11.20.0` + kustomize | L | 3 | |
| cloudflared-tunnel | default | — | kustomize | M | 3 | public ingress path |
| csi-driver-nfs-assets | default | — | kustomize | L | 3 | |
| dozzle | default | — | kustomize | L | 3 | |
| external-dns | external-dns | — | `external-dns@1.23.0` + kustomize | M | 3 | |
| garage | garage | — | kustomize | M | 3 | 1 hook Job; PVC |
| headlamp | kube-system | — | `headlamp@0.46.0` | L | 3 | add Flux plugin here |
| kubernetes-mcp-server | kubernetes-mcp-server | — | `kubernetes-mcp-server@0.1.0` (OCI) + kustomize | L | 3 | |
| monitoring | default | — | kustomize | L | 3 | manual sync → suspended |
| pocketid | pocketid | — | kustomize | M | 3 | SSO for everything |
| reloader | reloader | — | `reloader@2.2.18` | L | 3 | |
| system-upgrade | default | — | kustomize | M | 3 | k3s upgrade Plans |

### cronjobs/

One suspended Kustomization (`cronjobs/`: github-backup, lake-info). Deploy with
`flux resume ks cronjobs && flux suspend ks cronjobs`.
