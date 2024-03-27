envsubst < configmap.template.yaml > configmap.yaml
kubectl apply -f configmap.yaml
rm configmap.yaml