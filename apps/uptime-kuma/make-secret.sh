envsubst < secret.template.yaml > secret.yaml
kubectl delete -f secret.yaml --ignore-not-found
kubectl apply -f secret.yaml
rm secret.yaml