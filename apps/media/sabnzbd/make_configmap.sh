envsubst < sabnzbd.template.ini > sabnzbd.ini
kubectl delete configmap sabnzbd-config -n media --ignore-not-found
kubectl create configmap sabnzbd-config --from-file=sabnzbd.ini -n media
rm sabnzbd.ini