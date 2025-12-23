envsubst < config.template.yaml > config.yaml
kubectl delete configmap kometa-config -n media --ignore-not-found
kubectl create configmap kometa-config --from-file=config.yaml -n media
rm config.yaml