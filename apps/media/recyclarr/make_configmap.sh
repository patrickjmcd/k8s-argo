envsubst < config.template.yaml > recyclarr.yaml
kubectl create configmap recyclarr-config --from-file=recyclarr.yaml -n media
rm recyclarr.yaml