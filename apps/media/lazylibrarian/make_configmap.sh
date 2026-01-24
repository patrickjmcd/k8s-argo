envsubst < config.template.xml > config.xml
kubectl create configmap lazylibrarian-config --from-file=config.xml -n media
rm config.xml
