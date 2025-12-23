envsubst < config.template.yml > config.yml
kubectl delete configmap kometa-config -n media --ignore-not-found
kubectl create configmap kometa-config --from-file=config.yml -n media
rm config.yml