envsubst < issuer.template.yaml > issuer.yaml
kubectl delete -f issuer.yaml --ignore-not-found
kubectl apply -f issuer.yaml
rm issuer.yaml