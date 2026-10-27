# Argo CD → Flux migration

**Status: Phase 1 complete (2026-10-27).** Flux v2.10.0 installed and
self-managing. On Flux: `iss-in-space`, `joshhawleygop`, `spoolman`,
`tronbyt-server`. Verified: in-place adoption (UIDs unchanged, no pod
restarts), drift correction, HomelabApp health via CEL, namespace-less objects,
a Flux-created Namespace, `prune: disabled` keeping an object that left git,
and error alerts reaching ntfy. The Flux Operator now manages Flux (§1); web UI live at https://flux.x.pmcd.io with PocketID SSO (§4.9). Phases 2a–2d done 2026-10-27; 63 apps on Flux. **Phase 2 complete 2026-10-27**: every app except gitea (Phase 4) is on Flux (65 apps, 2 of them HelmReleases). **Phase 3 complete 2026-10-27**: 9 plain core components and 11 core charts are on Flux under `clusters/home/core/` (garage and timescale hook Jobs converted, §4.4). **Phase 4 started 2026-10-27**: kro, kro-definitions, traefik, prometheus, kyverno, cert-manager, shared-postgres and cnpg are on Flux. Argo keeps 12 Applications. Next: metallb-system.

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
  kustomization.yaml      # root, synced by the operator-created flux-system Kustomization
  flux-operator.yaml      # Flux Operator HelmRelease (+ web UI values, SSO via valuesFrom)
  flux-instance.yaml      # FluxInstance "flux": the Flux install itself (version, components, sync)
  flux-web.yaml           # web UI: OnePasswordItem for the OIDC client + RBAC binding
  notifications.yaml      # Flux errors -> Alertmanager -> n8n -> ntfy
  apps/                   # one Kustomization per app (name + path; shared settings patched in)
  # to come: core/ Kustomizations + HelmReleases, a ResourceSet for apps/media, cronjobs
```

The app directories (`apps/<name>/`, `core/<name>/`) don't change. Only the
`apps/<name>.yaml` / `core/<name>.yaml` Application files are replaced by
Kustomization or HelmRelease objects.

### Flux Operator (since 2026-10-27)

Flux is installed and upgraded by the [Flux Operator](https://fluxoperator.dev)
from the `FluxInstance` in `clusters/home/flux-instance.yaml`; bump
`spec.distribution.version` to upgrade. The operator is itself a Flux
HelmRelease, so each can repair the other. Both objects are
`prune: disabled`: deleting the FluxInstance uninstalls Flux, and uninstalling
the operator chart deletes its CRDs (and so the FluxInstance).

The operator creates the root `GitRepository` and `Kustomization`, both named
`flux-system`, from `spec.sync`: in-cluster Gitea URL, anonymous (the repo is
public), `refs/heads/main`, path `clusters/home`. Every Kustomization uses
`sourceRef: flux-system`.

It also brings `ResourceSet` (a template generator, ApplicationSet's
counterpart, used for `apps/media` in Phase 2c) and the web UI (§4.9).

**"Gitea is down" fallback.** Change the sync URL in git *and* live; otherwise
the first reconcile from GitHub re-applies `flux-instance.yaml` with the Gitea URL:

```bash
sed -i 's#http://gitea.gitea.svc.cluster.local:3000/patrickjmcd/k8s-argo.git#https://github.com/patrickjmcd/k8s-argo.git#' clusters/home/flux-instance.yaml
git commit -am "flux: sync from GitHub while Gitea is down" && git push github main
kubectl -n flux-system patch fluxinstance flux --type merge \
  -p '{"spec":{"sync":{"url":"https://github.com/patrickjmcd/k8s-argo.git"}}}'
```

**`.sourceignore` (repo root).** The operator's GitRepository uses Flux's
built-in ignore list, which drops images and archives (`*.png`, `*.jpg`,
`*.zip`, ...). A file a kustomization needs (e.g. joshhawleygop's `og.png`)
must be re-included there with a `!` pattern, or the build fails with "no
such file or directory".

### History: how Flux was installed

Phase 1 used `flux install --export` committed under `clusters/home/flux-system/`
(not `flux bootstrap gitea`, which wants a deploy key and Gitea's public
hostname), applied by hand once. The same day it was handed to the operator in
four commits, none of which restarted a controller:
1. mark the exported manifests `prune: disabled` (commonAnnotations) and add
   the operator HelmRelease;
2. drop `flux-system` from `clusters/home/kustomization.yaml` (34 objects
   "skipped", not pruned);
3. add the FluxInstance (same version line, components, network policy);
   the operator takes over the `flux-system` Kustomization in place;
4. point app Kustomizations at the operator's `flux-system` GitRepository,
   delete the export from git, and `kubectl delete` the old `k8s-argo`
   GitRepository.

Still to do: a Gitea push webhook to a notification-controller `Receiver` so
commits reconcile immediately rather than on the 1m poll.

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
| `ApplicationSet` (list generator) | Flux Operator `ResourceSet` | `inputs` list + a Kustomization template |
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

0. **Pod-template leftovers** (§4.10): if a plain Deployment/DaemonSet/
   StatefulSet in the app carries `kubectl.kubernetes.io/restartedAt`, copy it
   into git or accept one restart.
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

**Proven on nats (2026-10-27), the recipe for every chart app:**
1. Render exactly what Argo rendered: `helm template <release> <repo>/<chart>
   --version <v> -n <ns> -f <values>` (values from the Application's
   `valuesObject`/value files), dropping `helm.sh/hook` objects.
2. Add the live `restartedAt` values (§4.10) through the chart's pod-template
   values (nats: `podTemplate.merge` / `natsBox.podTemplate.merge`).
3. `kubectl diff --server-side --field-manager=helm-controller --force-conflicts
   -f render.yaml` must show nothing but managedFields. Fix values until it does.
4. HelmRelease with `releaseName`, `targetNamespace`, `storageNamespace` set to
   the app namespace (otherwise the release name becomes `<ns>-<name>`),
   `install.serverSideApply: true`, `upgrade.serverSideApply: enabled`,
   `driftDetection.mode: enabled`, and **`install/upgrade.remediation.retries:
   0` with `remediateLastFailure: false`**: remediating a failed first install
   would uninstall the adopted objects. Take-ownership is helm-controller's
   default in v2.10 (`disableTakeOwnership` exists), so no manual
   `meta.helm.sh/*` labelling was needed.
5. The app's extra kustomize path becomes a separate `<app>-extras`
   Kustomization.

Result for nats: `helm list` shows `nats` rev 1 deployed, all 14 objects kept
their UIDs, no pod restarted. helm-controller leaves the `kubectl-rollout`
manager on the pod template (unlike kustomize-controller), so step 2 is about
template equality, not field cleanup.

Older notes, still relevant for the core charts:


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

**Name clash (fixed 2026-10-27):** `kubectl get cluster` used to resolve to a leftover Rancher CAPI CRD (`clusters.cluster.x-k8s.io`) instead of CNPG. The five empty `cluster.x-k8s.io` CRDs were deleted. The `*.cattle.io` CRDs that remain belong to k3s (addons, etcd snapshots, HelmChart) and system-upgrade-controller (Plans), so keep them. `clusters.postgresql.cnpg.io` still works in scripts.

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
| `shared-postgres-backup-grants` | ✅ `core/postgres/jobs/` (Kustomization `shared-postgres-jobs`) |
| `{dispatcharr,n8n,outline,reedme}-ownership-fix` | ✅ `core/postgres/jobs/` |
| `homeassistant-schema` | ✅ `core/timescale/jobs/` (Kustomization `timescale-jobs`) |
| `garage-bootstrap` | ✅ `core/garage/jobs/` (Kustomization `garage-jobs`) |

Move each into its own sub-directory (e.g. `core/postgres/jobs/`) with a
Kustomization that `dependsOn` the parent and sets `force: true`, so a changed
Job (immutable spec) is deleted and recreated. That is the
`BeforeHookCreation` equivalent. Unlike Argo hooks, they no longer re-run on
every sync, only when the Job spec changes. Confirm each one is idempotent and
that this is acceptable (the ownership fixes are one-shot repairs and arguably
can be deleted).

**Done for garage and timescale (2026-10-27).** Two changes made it work:
- **Drop `ttlSecondsAfterFinished`.** Once the TTL deletes a finished Job, Flux
  sees it missing and recreates it, so the Job would run every TTL+interval.
- **Hash the script ConfigMap.** Move the script into `jobs/` and generate its
  ConfigMap with the default name hash. Editing the script then changes the
  Job spec, and the Job re-runs. With a fixed name, a new bucket in
  `bootstrap.py` would never be applied.

Both Jobs ran once on handoff and completed in about 5s.

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
CRD exists. **Done 2026-10-27** in each app's own file (31 apps plus
`media-base`, which every media app already depends on), not in the shared
patch: a patch would replace a file's own `dependsOn` list. A new HomelabApp
under `clusters/home/apps/` needs the same two lines.

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
**Done 2026-10-27:** the `flux` manager is enabled for `clusters/**.yaml`
(HelmRelease versions resolved through their HelmRepository), and the
kubernetes-mcp-server image regex follows its values to
`clusters/home/core/kubernetes-mcp-server.yaml`. The OCI override
rules went with kro (2026-10-27). Still to do: drop the `argocd` manager once
no chart Application remains, and check the first Renovate run after this
change actually lists the HelmRelease charts in its dependency dashboard.

### 4.10 `kubectl rollout restart` leftovers restart workloads on handoff

`kubectl rollout restart` writes `kubectl.kubernetes.io/restartedAt` into a pod
template under a kubectl field manager. Flux takes over kubectl-managed fields
and drops the ones not in git, so the first Flux apply removes the annotation,
which changes the template and rolls the workload once. Seen on batch 2b-1
(2026-10-27): the five adsb Deployments, dreeve and hivemq each restarted
(~30 s, healthy after). HomelabApp Deployments are kro's, not Flux's, so they
are never affected.

Before handing over anything that must not restart, list the templates that
carry it:

```bash
kubectl get deploy,ds,sts -A -o json | jq -r '.items[]
  | select(.spec.template.metadata.annotations["kubectl.kubernetes.io/restartedAt"] != null)
  | select((.metadata.ownerReferences // []) | map(.kind) | index("HomelabApp") | not)
  | "\(.kind)/\(.metadata.namespace)/\(.metadata.name)"'
```

To avoid the restart, copy the live value into git as a pod annotation (the
chart's `podAnnotations`, or the manifest's `spec.template.metadata.annotations`)
in the disarm commit, so Flux's template matches the live one. As of
2026-10-27 this covers, among others, DaemonSets `cilium`, `kube-vip-ds`,
`longhorn-manager`, `metallb-speaker`, `csi-smb-node`, `csi-nfs-node`,
`node-exporter`, and Deployments `pihole` (cluster DNS), `gitea`,
`cert-manager*`, `kyverno-*`, `cloudflared`, `pocketid`, the 1Password Connect
pair, and media `bazarr`, `dispatcharr`, `lidarr`, `sabnzbd`. Not affected:
objects Argo doesn't manage (k3s's coredns/metrics-server/local-path, and the
Prometheus/Alertmanager StatefulSets owned by prometheus-operator).

**HelmReleases don't need this.** helm-controller applies with SSA and never
takes over the `kubectl-rollout` manager's fields, so a live `restartedAt` that
isn't in the chart values stays and nothing rolls. The §4.2 server-side diff
(as `helm-controller`) shows this: it is empty even when the live template
carries the annotation. Phase 3b adopted 11 charts this way, 6 of them with
`restartedAt` on the template, and no pod restarted. Copying the value into
values (as done for nats/immich) is harmless but unnecessary.

Consequence after the migration: `kubectl rollout restart` on a Flux-managed
*plain* Deployment/DaemonSet is undone on the next reconcile, which rolls it a
second time. Restart those by deleting their pods instead (HomelabApps are
unaffected; restart those however you like).

### 4.9 Web UI

The operator serves the Flux web UI on port 9080; `clusters/home/flux-operator.yaml`
exposes it at `https://flux.x.pmcd.io` (internal `traefik-gateway`) with
PocketID OIDC (`issuerURL: https://auth.pmcd.dev`, callback
`/oauth2/callback`). The client ID/secret come from the 1Password item
`flux-web` (`scripts/create-flux-web-item.sh`) through the HelmRelease's
`valuesFrom`. The UI impersonates the user's `email` and `groups` claims, so
access is plain RBAC: `clusters/home/flux-web.yaml` binds the PocketID group
`argocd_admin` (Argo CD's admin group) to the operator's `flux-web-admin`
ClusterRole. Rename the group at Phase 5. Live since 2026-10-27; the server takes a
few seconds to start after an operator restart (Traefik returns 502 until then). Radar (Skyhook) was considered and
skipped: it overlaps Headlamp; it can run locally against a kubeconfig if its
topology view is ever wanted.

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
| 2a stateless ✅ 2026-10-27 | glance, go2rtc, homebox-companion, homelable, homelable-mcp, nut-exporter, pihole-exporter, pve-exporter, unifi-mcp, unpoller, reedme, kaneo, stubs |
| 2b stateful (PVC) ✅ 2026-10-27 | adsb, dreeve, hivemq, homebox, homebridge, homelable-backend, mealie, minecraft, n8n, ntfy, outline, romm, sparkyfitness, sparkyfitness-server, speech-to-phrase, tracearr, unifi-toolkit, voice-assistant |
| 2c media ✅ 2026-10-27 | Done in one batch: ResourceSet `media` (`clusters/home/apps/media.yaml`) generates `media-<app>` Kustomizations depending on `media-base`. Disarm took two commits (ApplicationSet template + media-base first, while the parent `media` app could still apply them; then the parent). After the handoff the orphaned ApplicationSet (with `preserveResourcesOnDeletion`) and media-base Application were deleted by hand. 86 objects, UIDs unchanged; copying `restartedAt` into git (§4.10) kept bazarr, dispatcharr(-celery), lidarr and sabnzbd from restarting |
| 2d special ✅ 2026-10-27 | renovate; ia-ignition (kept at `replicas: 0`, now enforced by Flux instead of Argo's manual sync; not suspended); pihole (cluster DNS; `restartedAt` copied into git, no restart, DNS answered throughout). pihole had been applied client-side by Argo (no ServerSideApply); the SSA takeover was clean |
| 2e Helm ✅ 2026-10-27 | nats, immich: HelmReleases adopted with the §4.2 recipe, no restarts, UIDs unchanged. immich's CNPG Cluster/Database/ObjectStore/ScheduledBackup are prune-disabled; its chart values now live in the HelmRelease |

Hold until Phase 4: **gitea** (source of truth for both Argo and Flux).

### Phase 3: `core/` low risk

**3a ✅ 2026-10-27.** Kustomize-only: cloudflared-tunnel, dozzle, csi-driver-nfs-assets,
goldilocks-assets, pocketid, system-upgrade, garage (+ bootstrap job, §4.4),
timescale (+ schema job), monitoring (only a Namespace, so a normal Kustomization).

**3b ✅ 2026-10-27.** HelmRelease, low blast radius: reloader, goldilocks, vpa, headlamp, loki,
alloy, blackbox-exporter, external-dns, trivy-operator,
kubernetes-mcp-server, tailscale-operator. All 11 via the §4.2 recipe
(script-generated from the Argo `valuesObject`s, empty server-side diff each);
136 objects kept their UIDs and no pod restarted. headlamp's
`sessionTTL: ~` was dropped: Argo's Helm had ignored the null and the live pod
runs the chart default, while Helm 4 honours it and would have removed the
flag. goldilocks `dependsOn` vpa (CRDs). Extras are `<app>-extras`
Kustomizations; HelmRepositories in `clusters/home/core/helm-repositories.yaml`.

### Phase 4: `core/` high risk, one per session with a fresh backup

Order: kro → kro-definitions → traefik → prometheus → kyverno → cert-manager →
shared-postgres (+ jobs) → cnpg → metallb-system → csi-driver-smb / csi-driver-nfs →
node-tuning → kube-vip → longhorn → cilium → gitea.

Per component:
- Before: take a Longhorn backup or CNPG backup if it's stateful; keep the
  `helm template` diff from §4.2 clean.
- ✅ **kro** (2026-10-27): the RGD and CRD must survive. The RGD carries
  `prune: disabled`. The `HomelabApp` CRD is generated by kro, not applied from
  git, so nothing prunes it; kro runs without `--allow-crd-deletion` and the
  chart's own CRDs sit in `crds/`, which Helm never deletes. Adopted as a
  HelmRelease (OCI HelmRepository `kro`, empty server-side diff), kro-definitions
  as a Kustomization. 101 objects (46 HomelabApps and their Deployments) kept
  their UIDs, and no pod restarted; the kro Deployment's generation moved
  without a new ReplicaSet. Pre-handoff backup of the RGD, CRDs, instances and
  GraphRevisions was dumped to YAML. The Renovate `overridePackageName` rules
  for Argo's OCI charts were dropped with it.
- ✅ **traefik** (2026-10-27): HelmRelease plus `traefik-extras`
  (`core/traefik-assets`; its unused SMB PV/PVC are prune-guarded). Render
  with the cluster's API versions (`--api-versions` for each entry of
  `kubectl api-versions`, as Argo and helm-controller do), or the chart's
  ServiceMonitor template fails. The Argo Application had no
  `ServerSideApply=true`, so its objects also carried a client-side
  `kubectl.kubernetes.io/last-applied-configuration` annotation; it was
  removed along with Argo's managedFields. 41 objects kept their UIDs; both
  pods, the `.200` LoadBalancer IP, both Gateways and all 85 HTTPRoutes were
  untouched. The chart's CRDs (25, traefik.io/hub.traefik.io) are in `crds/`;
  the Gateway API CRDs are not the chart's.
- ✅ **prometheus** (2026-10-27): HelmRelease `kube-prometheus-stack` plus
  `prometheus-extras` and `grafana-extras`. 166 objects kept their UIDs;
  Prometheus, Alertmanager, the operator, kube-state-metrics and node-exporter
  did not restart. Lessons:
  - **Charts that use `lookup` render differently under `helm template`.** The
    Grafana chart reuses its existing admin Secret via `lookup`; Argo's
    `helm template` can't look anything up, so every Argo render generated a
    new random password and rolled Grafana (generation 388, five ReplicaSets
    in the day before handoff). helm-controller does run `lookup`. To diff the
    way it renders, use `helm install --dry-run=server --take-ownership
    --skip-crds` and diff its `MANIFEST:` section (`helm template
    --dry-run=server` fails Helm's ownership check instead). The secret
    checksum still changed once, so Grafana restarted once at handoff
    (`Recreate` strategy, RWO PVC, no deadlock).
  - The extras' plain `json-exporter` Deployment rolled once: it carried a
    `restartedAt` (§4.10); only the chart's workloads had been checked. Check
    the extras directories too.
  - Grafana's PVC is a chart object, so it now has `helm.sh/resource-policy:
    keep`. The Prometheus TSDB claim is the operator's (volumeClaimTemplate) on
    the prune-guarded `prometheus-usb` local PV.
  - `core/grafana-assets` had no namespace on its dashboard ConfigMaps; it now
    sets `namespace: prometheus`. Kustomize's namespace transformer also
    renames Namespace objects, so the empty, unused `grafana` Namespace was
    dropped from git. It is still in the cluster, unmanaged.
  - The chart's admission-webhook hook Jobs ran on install and completed.
  - **Lost a hand-set namespace label.** The live `prometheus` Namespace had
    `prometheus: enabled` (set by hand, never in git), which the Prometheus
    CR's `serviceMonitorNamespaceSelector` needs for the chart's own
    ServiceMonitors. Flux's apply of `namespace.yaml` dropped it: 55 targets
    (kubelet, apiserver, node-exporter, kube-state-metrics, ...) disappeared
    for ~20 minutes, firing `KubeAPIDown` and `BeelinkUsbDriveUnmounted`
    (an `absent()` on node-exporter data). Fixed in git. **Before each
    remaining handoff, compare the live labels of every Namespace the
    component's manifests declare with git** (`kubectl get ns <ns>
    --show-labels`).
- ✅ **kyverno** (2026-10-27): HelmRelease plus `kyverno-extras` (the 6
  ClusterPolicies). The chart ships its CRDs as **templates**, not in `crds/`,
  so a Helm uninstall would delete them and every policy; all 22 now carry
  `helm.sh/resource-policy: keep` (values `crds.annotations` and
  `kyverno-api.annotations`, one per subchart). Its only install-time
  `lookup` (`validate.yaml`, a v2 upgrade check) passes. No install hooks
  (only post-upgrade and pre-delete). 76 objects and kyverno's 10 generated
  webhook configurations kept their UIDs, no pod restarted, and a server-side
  dry-run pod still gets the tolerations and node affinity.
- ✅ **cert-manager** (2026-10-27): HelmRelease (release name
  `cert-manager`; the Argo Application set none) plus `cert-manager-extras`.
  Its CRDs are templates but `crds.keep` (default true) already marks them
  `keep`. The post-install `startupapicheck` hook Job ran once on adoption and
  passed. Server-side diff empty, with and without `lookup`. The
  OnePasswordItem relied on Argo's namespace default and now sets
  `cert-manager` itself (a kustomization-wide `namespace:` would have moved
  the Certificates out of `default`). Applied client-side by Argo, so 46
  `last-applied-configuration` annotations were removed along with 70
  managedFields entries. Pods, webhook CA bundle and the served certificates
  unchanged.
- ✅ **shared-postgres** (2026-10-27): Kustomization `shared-postgres`
  (`core/postgres`, empty server-side diff) plus `shared-postgres-jobs`. On-demand
  B2 base backup `20261027T125502` first (3 min). Prune guards come from one
  kustomize patch in `core/postgres/kustomization.yaml` (Cluster, 26
  Databases, ObjectStore, ScheduledBackup, SMB PV/PVC). The 5 Jobs ran once on
  adoption. The grants Job had never worked: Kubernetes turns `$$` in a
  container `command` into `$` (the same `$(VAR)` expansion as env refs), so
  its `DO $$` block failed, and psql exited 0 anyway. Now written `$$$$` with
  `ON_ERROR_STOP=1`. The `backup` role it grants is unused (the pg_dump
  CronJob runs as the superuser). Cluster, pods, PVCs and primary unchanged.
  Two CronJobs carry old node affinity/tolerations applied by hand through
  the Kubernetes MCP server (Apply manager, not in git); Flux leaves those.
- ✅ **cnpg** (2026-10-27): HelmReleases `cloudnative-pg` (operator) and
  `plugin-barman-cloud` (`dependsOn` the operator and cert-manager), plus
  `cnpg-extras` (the ClusterImageCatalog, now prune-guarded). All 12 CRDs
  already carry `keep`. **Argo rendered both charts as one release named
  after the Application (`cloudnative-pg`)**, and Helm can't hold two
  releases with one name. The plugin got its own release name with
  `fullnameOverride: cloudnative-pg-plugin-barman-cloud` (object names
  unchanged) and a `postRenderers` JSON patch that keeps
  `app.kubernetes.io/instance: cloudnative-pg` in its Deployment and Service
  selectors; the Deployment selector is immutable, so changing it would have
  meant deleting and recreating the Deployment. Only metadata labels changed.
  35 objects kept their UIDs; neither operator pod nor any of the 9 Postgres
  pods restarted; a forced WAL switch on shared-postgres archived to B2.
  No other remaining Application has two charts. longhorn has no
  `releaseName`, so its release is the Application name `longhorn`.
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

Kyverno values still say Argo: `config.excludeNamespaces: [argocd]` keeps the
GitOps engine's own pods out of a fail-closed webhook, and `webhookLabels`
stamps `app.kubernetes.io/managed-by: argocd` on its generated webhooks.
Switch both to `flux-system` / `flux` here.

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
| adsb | adsb | kustomize | M | ✅ Flux (2b-1); PVC |
| dreeve | default | kustomize | M | ✅ Flux (2b-1); PVC; namespace-less objects |
| gitea | gitea | kustomize | H | source of truth; Phase 4, last |
| glance | default | HomelabApp | L | ✅ Flux (2a) |
| go2rtc | default | HomelabApp | L | ✅ Flux (2a) |
| hivemq | hivemq | kustomize | M | ✅ Flux (2b-1); PVC |
| homebox | default | HomelabApp | M | ✅ Flux (2b-1); PVC |
| homebox-companion | default | HomelabApp | L | ✅ Flux (2a) |
| homebridge | homebridge | HomelabApp | M | ✅ Flux (2b); PVC |
| homelable | homelable | HomelabApp | L | ✅ Flux (2a) |
| homelable-backend | homelable | HomelabApp | M | ✅ Flux (2b-1); PVC |
| homelable-mcp | homelable | HomelabApp | L | ✅ Flux (2a) |
| ia-ignition | default | kustomize | M | ✅ Flux (2d); replicas 0 enforced; manual sync → suspended; PVC |
| immich | immich | Helm `immich@0.12.0` + kustomize | M | ✅ Flux (2e, HelmRelease adopted; values inlined, Renovate rule moved); §4.2; PVC |
| iss-in-space | default | HomelabApp | L | ✅ Flux (Phase 1) |
| joshhawleygop | default | HomelabApp | L | ✅ Flux (Phase 1); outage fixed (see below) |
| kaneo | kaneo | HomelabApp | L | ✅ Flux (2a) |
| mealie | default | HomelabApp | M | ✅ Flux (2b); PVC |
| media | media | app-of-apps | M | parent of ApplicationSet + media-base; remove last in 2c |
| minecraft | minecraft | kustomize | M | ✅ Flux (2b-1); PVC |
| n8n | default | HomelabApp | M | ✅ Flux (2b); PVC; DB |
| nats | nats | Helm `nats@2.15.1` + kustomize | M | ✅ Flux (2e, HelmRelease adopted); §4.2 |
| ntfy | ntfy | HomelabApp | M | ✅ Flux (2b); PVC; alert path, so confirm alerts afterwards |
| nut-exporter | default | HomelabApp | L | ✅ Flux (2a) |
| outline | default | HomelabApp | M | ✅ Flux (2b); PVC; `markdown-export.yaml` has `Prune=false` |
| pihole | pihole | kustomize | M | ✅ Flux (2d); cluster DNS; namespace-less objects |
| pihole-exporter | pihole | HomelabApp | L | ✅ Flux (2a) |
| pve-exporter | prometheus | HomelabApp | L | ✅ Flux (2a) |
| reedme | default | HomelabApp | L | ✅ Flux (2a); DB |
| renovate | renovate | kustomize | L | ✅ Flux (2d); update its config in the same phase (§4.8) |
| romm | romm | HomelabApp | M | ✅ Flux (2b-1); PVC |
| sparkyfitness | sparkyfitness | HomelabApp | M | ✅ Flux (2b);  |
| sparkyfitness-server | sparkyfitness | HomelabApp | M | ✅ Flux (2b); PVC |
| speech-to-phrase | voice-assistant | HomelabApp | M | ✅ Flux (2b); PVC |
| spoolman | spoolman | HomelabApp | L | ✅ Flux (Phase 1); no PVC; Namespace manifest added |
| stubs | default | HomelabApp | L | ✅ Flux (2a) |
| tracearr | tracearr | HomelabApp | M | ✅ Flux (2b-1); PVC |
| tronbyt-server | default | HomelabApp | M | ✅ Flux (Phase 1); PVC prune-disabled |
| unifi-mcp | default | HomelabApp | L | ✅ Flux (2a) |
| unifi-toolkit | default | HomelabApp | M | ✅ Flux (2b-1); PVC |
| unpoller | default | HomelabApp | L | ✅ Flux (2a) |
| voice-assistant | voice-assistant | HomelabApp | M | ✅ Flux (2b); PVC |

joshhawleygop (found during Phase 1, fixed 2026-10-27 in `3dba336`): the site
was down Oct 11–27. `build.py`'s kustomization template had no `resources:`
entry, so the Oct 11 bot rebuild dropped the hand-added `homelabapp.yaml` and
Argo pruned the HomelabApp. Separately, the Oct 6 chart→HomelabApp conversion
dropped `httproute.publicHostnames`. Both are restored; the template now carries
`resources:` itself.

### apps/media (ApplicationSet `media-apps-kustomize`, ns `media`)

| App | Type | Risk | Notes |
|---|---|---|---|
| media-base | kustomize (`apps/media/base`) | M | ✅ Flux (2c); shared PVCs (`smb-media-claim`, `sab-downloads-pvc`); every *arr depends on it |
| agregarr, concertarr, navidrome, prowlarr, radarr, readarr, seerr, sonarr, sportarr, subwave, tautulli | HomelabApp | M | ✅ Flux (2c); PVCs |
| aonsoku, hdhr2xmltv, ollama, ytdl-triage | HomelabApp | L | ✅ Flux (2c); |
| bazarr, dispatcharr, jellyfin, kometa, lidarr, recyclarr, sabnzbd, youtarr, ersatztv | kustomize | M | ✅ Flux (2c); initContainers / StatefulSet / CronJob; several namespace-less (§4.3) |

### core/

| Component | Namespace | Wave | Source | Risk | Phase | Notes |
|---|---|---|---|---|---|---|
| cilium | kube-system | -4 | `cilium@1.20.4` + kustomize | H | 4 | CNI; maintenance window |
| argocd | argocd | -2 | `argo-cd@10.10.3` + kustomize | — | 5 | removed, not migrated |
| ✅ cert-manager | cert-manager | -2 | `cert-manager@v1.21.2` + kustomize | H | 4 | caBundle drift ignore |
| kube-vip | kube-system | -2 | kustomize | H | 4 | control-plane VIP |
| ✅ cloudnative-pg | cnpg-system | -1 | `cloudnative-pg@0.29.1`, `plugin-barman-cloud@0.8.1` + kustomize | H | 4 | two HelmReleases |
| csi-driver-nfs | kube-system | -1 | `csi-driver-nfs@4.13.6` | H | 4 | |
| csi-driver-smb | default | -1 | kustomize | H | 4 | |
| longhorn | longhorn-system | -1 | `longhorn@1.13.0` + kustomize | H | 4 | `Replace=true` node objects |
| metallb-system | metallb-system | -1 | `metallb@0.16.1` + kustomize | H | 4 | |
| node-tuning | kube-system | -1 | kustomize | M | 4 | |
| ✅ shared-postgres | postgres | -1 | kustomize | H | 4 | 6 hook Jobs |
| ✅ tailscale-operator | tailscale | -1 | `tailscale-operator@1.104.2` + kustomize | M | 3 | |
| ✅ timescale | timescale | -1 | kustomize | M | 3 | 1 hook Job |
| ✅ alloy | loki | 0 | `alloy@1.14.3` | L | 3 | |
| ✅ kro | kro-system | 0 | `kro@0.9.4` (OCI) | H | 4 | CRD must never be pruned |
| ✅ kyverno | kyverno | 0 | `kyverno@3.9.2` + kustomize | H | 4 | admission webhooks |
| ✅ loki | loki | 0 | `loki@7.5.0` | L | 3 | |
| ✅ prometheus | prometheus | 0 | `kube-prometheus-stack@92.3.1` + 2 kustomize paths | M | 4 | CRDs every ServiceMonitor needs |
| ✅ vpa | vpa | 0 | `vpa@5.1.0` | L | 3 | |
| ✅ goldilocks | goldilocks | 1 | `goldilocks@11.2.0` | L | 3 | |
| ✅ goldilocks-assets | goldilocks | 1 | kustomize | L | 3 | |
| ✅ kro-definitions | kro-system | 1 | kustomize | H | 4 | the HomelabApp RGD |
| ✅ traefik | default | 1 | `traefik@41.7.1` + kustomize | H | 4 | all ingress; assets dir has a PVC |
| ✅ trivy-operator | trivy-system | 1 | `trivy-operator@0.37.0` + kustomize | L | 3 | |
| ✅ blackbox-exporter | prometheus | — | `prometheus-blackbox-exporter@11.20.0` + kustomize | L | 3 | |
| ✅ cloudflared-tunnel | default | — | kustomize | M | 3 | public ingress path |
| ✅ csi-driver-nfs-assets | default | — | kustomize | L | 3 | |
| ✅ dozzle | default | — | kustomize | L | 3 | |
| ✅ external-dns | external-dns | — | `external-dns@1.23.0` + kustomize | M | 3 | |
| ✅ garage | garage | — | kustomize | M | 3 | 1 hook Job; PVC |
| ✅ headlamp | kube-system | — | `headlamp@0.46.0` | L | 3 | add Flux plugin here |
| ✅ kubernetes-mcp-server | kubernetes-mcp-server | — | `kubernetes-mcp-server@0.1.0` (OCI) + kustomize | L | 3 | |
| ✅ monitoring | default | — | kustomize | L | 3 | just the Namespace; normal (not suspended) on Flux |
| ✅ pocketid | pocketid | — | kustomize | M | 3 | SSO for everything |
| ✅ reloader | reloader | — | `reloader@2.2.18` | L | 3 | |
| ✅ system-upgrade | default | — | kustomize | M | 3 | k3s upgrade Plans |

### cronjobs/

One suspended Kustomization (`cronjobs/`: github-backup, lake-info). Deploy with
`flux resume ks cronjobs && flux suspend ks cronjobs`.
