#!/usr/bin/env bash
# Usage: scripts/drop-stale-helm-manager.sh <release> [kinds...]
# Removes the managedFields entry of manager "helm" (a hand-run helm CLI
# install, pre-Flux) from every object of the given Helm release, leaving
# helm-controller as the only owner of the fields it applies. Without this,
# a field the chart later drops or changes stays owned by "helm", so
# server-side apply keeps it and merges it with the new value -- how the
# connect 2.4 upgrade failed (Oct 2026). Changes no spec; nothing restarts.
set -euo pipefail
rel=${1:?release name}; shift
kinds=("${@:-deploy ds sts cm svc sa role rolebinding secret clusterrole clusterrolebinding}")
read -r -a kinds <<<"${kinds[*]}"
n=0
for k in "${kinds[@]}"; do
  kubectl get "$k" -A --show-managed-fields -o json | jq -r --arg r "$rel" '
    .items[] | select(.metadata.annotations["meta.helm.sh/release-name"] == $r)
    | (.metadata.managedFields | map(.manager) | index("helm")) as $i
    | select($i != null) | "\(.metadata.namespace // "-") \(.metadata.name) \($i)"' |
  while read -r ns name i; do
    nsarg=(); [ "$ns" != - ] && nsarg=(-n "$ns")
    kubectl "${nsarg[@]}" patch "$k" "$name" --type=json -p \
      "[{\"op\":\"test\",\"path\":\"/metadata/managedFields/$i/manager\",\"value\":\"helm\"},{\"op\":\"remove\",\"path\":\"/metadata/managedFields/$i\"}]" >/dev/null \
      && echo "cleaned $k $ns/$name"
  done
done
