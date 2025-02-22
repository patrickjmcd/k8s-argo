envsubst < config.template.xml > config.xml
kubectl create configmap radarr-config --from-file=config.xml -n media
rm config.xml