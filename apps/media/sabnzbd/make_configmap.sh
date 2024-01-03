envsubst < sabnzbd.template.ini > sabnzbd.ini
kubectl create configmap sabnzbd-config --from-file=sabnzbd.ini -n media
rm sabnzbd.ini