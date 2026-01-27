envsubst < config.template.ini > config.ini
kubectl create configmap bazarr-config --from-file=config.ini -n media
rm config.ini