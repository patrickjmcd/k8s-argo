envsubst < config.template.yaml > config.yaml
kubectl create configmap recyclarr-config --from-file=config.yaml -n media
rm config.yaml