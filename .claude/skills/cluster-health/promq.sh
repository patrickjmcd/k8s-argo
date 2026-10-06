#!/bin/bash
# Instant PromQL query through the apiserver service proxy (no port-forward).
# usage: promq.sh '<promql>'   -> raw Prometheus API JSON on stdout
q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$1")
kubectl get --raw "/api/v1/namespaces/prometheus/services/kube-prometheus-stack-prometheus:9090/proxy/api/v1/query?query=$q"
